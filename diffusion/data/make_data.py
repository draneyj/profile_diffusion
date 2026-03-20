#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
import torch

from ..config import GridConfig, SpeciesConfig
from ..state import CoarseState


_ORDER_DEFAULT_LAMMPS_TYPE = 1


def _parse_list_ints(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.replace(",", " ").split() if x.strip()]


def _parse_list_floats(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.replace(",", " ").split() if x.strip()]


def _numeric_stem(p: Path) -> int:
    try:
        return int(p.stem)
    except ValueError:
        return math.inf


def _compute_grid_dim(extent: float, a: float) -> int:
    if extent <= 0.0:
        raise ValueError(f"Invalid grid extent {extent}")
    n = int(math.floor(extent / a))
    if n <= 0:
        # Avoid zero-sized tensors for tiny boxes.
        n = 1
    return n


def _wrap_periodic_xy(x: np.ndarray, *, xlo: float, xlen: float) -> np.ndarray:
    # Map x into [xlo, xlo + xlen) using modulo.
    return xlo + np.mod(x - xlo, xlen)


def _iter_lammps_dump_frames(
    dump_path: str,
) -> Generator[Tuple[int, Tuple[float, float, float, float, float, float], Dict[str, int], np.ndarray], None, None]:
    """
    Yields frames:
      (timestep, (xlo,xhi,ylo,yhi,zlo,zhi), col_indices, atoms_np)
    where atoms_np columns are exactly in the order of the header columns.

    atoms_np has dtype float32.
    """
    dump_path = str(dump_path)
    with open(dump_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line == "ITEM: TIMESTEP":
            if i + 1 >= n_lines:
                break
            timestep = int(lines[i + 1].strip())
            i += 2

            # NUMBER OF ATOMS
            if i >= n_lines or lines[i].strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"Malformed dump {dump_path}: expected NUMBER OF ATOMS after TIMESTEP")
            if i + 1 >= n_lines:
                raise ValueError(f"Malformed dump {dump_path}: missing NUMBER OF ATOMS value")
            n_atoms = int(lines[i + 1].strip())
            i += 2

            # BOX BOUNDS
            if i >= n_lines or not lines[i].strip().startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Malformed dump {dump_path}: expected BOX BOUNDS")
            i += 1
            if i + 2 >= n_lines:
                raise ValueError(f"Malformed dump {dump_path}: truncated BOX BOUNDS")

            def _two_floats_from_bound_line(s: str) -> Tuple[float, float]:
                toks = s.split()
                if len(toks) < 2:
                    raise ValueError(f"Bad BOX BOUNDS line: {s}")
                lo = float(toks[0])
                hi = float(toks[1])
                return lo, hi

            xlo, xhi = _two_floats_from_bound_line(lines[i].strip())
            ylo, yhi = _two_floats_from_bound_line(lines[i + 1].strip())
            zlo, zhi = _two_floats_from_bound_line(lines[i + 2].strip())
            i += 3

            # ATOMS header
            if i >= n_lines or not lines[i].strip().startswith("ITEM: ATOMS"):
                raise ValueError(f"Malformed dump {dump_path}: expected ATOMS header")
            atom_header = lines[i].strip()
            i += 1
            header_cols = atom_header.split()[2:]  # after "ITEM: ATOMS"
            col_indices = {name: idx for idx, name in enumerate(header_cols)}

            required_cols = ["type", "vx", "vy", "vz"]
            # x/y/z can be "x y z" or sometimes "xu yu zu" (unwrapped).
            coord_candidates = [
                ("x", "y", "z"),
                ("xu", "yu", "zu"),
            ]

            # Validate required velocity + type
            for c in required_cols:
                if c not in col_indices:
                    raise ValueError(f"Dump {dump_path} missing required column '{c}' in ATOMS header")

            coord_xyz: Optional[Tuple[str, str, str]] = None
            for cx, cy, cz in coord_candidates:
                if cx in col_indices and cy in col_indices and cz in col_indices:
                    coord_xyz = (cx, cy, cz)
                    break
            if coord_xyz is None:
                raise ValueError(
                    f"Dump {dump_path} missing x/y/z coordinate columns. "
                    f"Need either (x,y,z) or (xu,yu,zu); got header: {atom_header}"
                )

            # Read atom lines into a float array
            atoms_np = np.empty((n_atoms, len(header_cols)), dtype=np.float32)
            for ai in range(n_atoms):
                if i >= n_lines:
                    raise ValueError(f"Malformed dump {dump_path}: truncated ATOMS data")
                parts = lines[i].split()
                i += 1
                if len(parts) != len(header_cols):
                    raise ValueError(
                        f"Malformed dump {dump_path}: expected {len(header_cols)} atom cols, got {len(parts)}"
                    )
                atoms_np[ai, :] = np.array(parts, dtype=np.float32)

            # We return coord_xyz via meta in caller by checking col_indices keys.
            # (Still leaving atoms_np as-is.)
            yield (
                timestep,
                (xlo, xhi, ylo, yhi, zlo, zhi),
                col_indices,
                atoms_np,
            )
        else:
            i += 1


def _atoms_to_state(
    *,
    atoms: np.ndarray,
    col_indices: Dict[str, int],
    species: SpeciesConfig,
    grid: GridConfig,
    order_lammps_type: int,
    locked_bounds: Tuple[float, float, float, float, float, float, int, int, int],
) -> CoarseState:
    """
    locked_bounds:
      (xlo, xhi, ylo, yhi, zlo, zhi, nx, ny, nz)
    where nx,ny,nz are determined from the first frame of a dump.
    """
    xlo, xhi, ylo, yhi, zlo, zhi, nx, ny, nz = locked_bounds
    a = float(grid.lattice_constant_a)

    xlen = xhi - xlo
    ylen = yhi - ylo
    if xlen <= 0.0 or ylen <= 0.0:
        raise ValueError("Invalid x/y bounds for periodic wrapping.")

    # Coordinate column names.
    if "x" in col_indices and "y" in col_indices and "z" in col_indices:
        cx, cy, cz = "x", "y", "z"
    else:
        cx, cy, cz = "xu", "yu", "zu"

    types = atoms[:, col_indices["type"]].astype(np.int64)
    x_abs = atoms[:, col_indices[cx]].astype(np.float64)
    y_abs = atoms[:, col_indices[cy]].astype(np.float64)
    z_abs = atoms[:, col_indices[cz]].astype(np.float64)

    vx = atoms[:, col_indices["vx"]].astype(np.float64)
    vy = atoms[:, col_indices["vy"]].astype(np.float64)
    vz = atoms[:, col_indices["vz"]].astype(np.float64)

    # Bin indices: periodic in x/y, non-periodic in z.
    if grid.periodic_xy:
        x = _wrap_periodic_xy(x_abs, xlo=xlo, xlen=xlen)
        y = _wrap_periodic_xy(y_abs, xlo=ylo, xlen=ylen)
    else:
        x = x_abs
        y = y_abs

    # Determine z mask based on fixed nz derived from the first frame.
    z_min = zlo
    z_max = zlo + nz * a
    if grid.ignore_atoms_outside_z:
        z_mask = (z_abs >= z_min) & (z_abs < z_max)
    else:
        z_mask = np.ones((len(z_abs),), dtype=bool)

    # Convert to integer cell indices for all atoms (after wrapping and z mask).
    ix = np.floor((x - xlo) / a).astype(np.int64)
    iy = np.floor((y - ylo) / a).astype(np.int64)
    iz = np.floor((z_abs - zlo) / a).astype(np.int64)

    # Clip to be safe against floating-point edge cases.
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)

    # Species mapping for counts/momentum/ke.
    s_map = {int(t): s_idx for s_idx, t in enumerate(species.lammps_types)}
    S = int(species.num_species)

    species_idx = np.full((len(types),), -1, dtype=np.int64)
    for t, s_idx in s_map.items():
        species_idx[types == t] = int(s_idx)

    # Prepare feature arrays.
    counts = np.zeros((S, nx, ny, nz), dtype=np.float32)
    momentum = np.zeros((3, nx, ny, nz), dtype=np.float32)
    ke = np.zeros((1, nx, ny, nz), dtype=np.float32)

    # Add counts/momentum/ke for atoms belonging to known species and in z-range.
    valid = (species_idx >= 0) & z_mask
    if np.any(valid):
        for s in range(S):
            m = float(species.masses[s])
            mask_s = valid & (species_idx == s)
            if not np.any(mask_s):
                continue

            xs, ys, zs = ix[mask_s], iy[mask_s], iz[mask_s]
            np.add.at(counts[s], (xs, ys, zs), 1.0)
            np.add.at(momentum[0], (xs, ys, zs), m * vx[mask_s])
            np.add.at(momentum[1], (xs, ys, zs), m * vy[mask_s])
            np.add.at(momentum[2], (xs, ys, zs), m * vz[mask_s])

            # Kinetic energy: 0.5 * m * (v^2).
            v2 = vx[mask_s] ** 2 + vy[mask_s] ** 2 + vz[mask_s] ** 2
            np.add.at(ke[0], (xs, ys, zs), 0.5 * m * v2)

    # Order parameter uses absolute coordinates and only a specific LAMMPS atom type.
    order = np.zeros((1, nx, ny, nz), dtype=np.float32)
    mask_order = (types == int(order_lammps_type)) & z_mask
    if np.any(mask_order):
        xs, ys, zs = ix[mask_order], iy[mask_order], iz[mask_order]

        cos_x = np.cos(8.0 * math.pi * x_abs[mask_order] / a)
        cos_y = np.cos(8.0 * math.pi * y_abs[mask_order] / a)
        cos_z = np.cos(8.0 * math.pi * z_abs[mask_order] / a)

        cos_x_sum = np.zeros((nx, ny, nz), dtype=np.float64)
        cos_y_sum = np.zeros((nx, ny, nz), dtype=np.float64)
        cos_z_sum = np.zeros((nx, ny, nz), dtype=np.float64)
        count_sum = np.zeros((nx, ny, nz), dtype=np.float64)

        np.add.at(cos_x_sum, (xs, ys, zs), cos_x)
        np.add.at(cos_y_sum, (xs, ys, zs), cos_y)
        np.add.at(cos_z_sum, (xs, ys, zs), cos_z)
        np.add.at(count_sum, (xs, ys, zs), 1.0)

        # Avoid divide-by-zero: cells with no order atoms stay at 0.
        valid_counts = count_sum > 0.0
        lambda_x = np.zeros((nx, ny, nz), dtype=np.float64)
        lambda_y = np.zeros((nx, ny, nz), dtype=np.float64)
        lambda_z = np.zeros((nx, ny, nz), dtype=np.float64)

        lambda_x[valid_counts] = cos_x_sum[valid_counts] / count_sum[valid_counts]
        lambda_y[valid_counts] = cos_y_sum[valid_counts] / count_sum[valid_counts]
        lambda_z[valid_counts] = cos_z_sum[valid_counts] / count_sum[valid_counts]

        xi = (lambda_x + lambda_y + lambda_z) / 3.0
        order[0] = xi.astype(np.float32)

    return CoarseState(
        counts=torch.from_numpy(counts),
        momentum=torch.from_numpy(momentum),
        ke=torch.from_numpy(ke),
        order=torch.from_numpy(order),
    )


def make_coarse_state_from_dump(
    dump_path: str,
    *,
    species: SpeciesConfig,
    grid: GridConfig = GridConfig(),
    order_lammps_type: int = _ORDER_DEFAULT_LAMMPS_TYPE,
) -> Tuple[int, CoarseState, dict]:
    frames = make_coarse_states_from_dump(
        dump_path,
        species=species,
        grid=grid,
        order_lammps_type=order_lammps_type,
        max_frames=1,
    )
    if not frames:
        raise ValueError(f"No frames found in dump: {dump_path}")
    return frames[0]


def make_coarse_states_from_dump(
    dump_path: str,
    *,
    species: SpeciesConfig,
    grid: GridConfig = GridConfig(),
    order_lammps_type: int = _ORDER_DEFAULT_LAMMPS_TYPE,
    max_frames: Optional[int] = None,
) -> List[Tuple[int, CoarseState, dict]]:
    """
    Parse many `ITEM: TIMESTEP` blocks inside one LAMMPS dump file.

    Important: nz may be inconsistent across frames inside a dump due to differing box bounds.
    We lock (nx,ny,nz) to the first frame's computed values and ignore atoms outside
    the locked z-range for subsequent frames.
    """
    # First frame determines locked grid dims.
    parsed_frames = []

    fixed_lock: Optional[Tuple[float, float, float, float, float, float, int, int, int]] = None

    for frame_idx, (timestep, bounds, col_indices, atoms) in enumerate(_iter_lammps_dump_frames(dump_path)):
        xlo, xhi, ylo, yhi, zlo, zhi = bounds

        if fixed_lock is None:
            nx = _compute_grid_dim(xhi - xlo, float(grid.lattice_constant_a))
            ny = _compute_grid_dim(yhi - ylo, float(grid.lattice_constant_a))
            nz = _compute_grid_dim(zhi - zlo, float(grid.lattice_constant_a))
            fixed_lock = (xlo, xhi, ylo, yhi, zlo, zhi, nx, ny, nz)

        assert fixed_lock is not None
        state = _atoms_to_state(
            atoms=atoms,
            col_indices=col_indices,
            species=species,
            grid=grid,
            order_lammps_type=order_lammps_type,
            locked_bounds=fixed_lock,
        )
        meta = {"frame_idx": frame_idx, "grid": {"nx": state.grid_shape[0], "ny": state.grid_shape[1], "nz": state.grid_shape[2]}}
        parsed_frames.append((timestep, state, meta))

        if max_frames is not None and len(parsed_frames) >= int(max_frames):
            break

    return parsed_frames


def _pad_state_to_nz(state: CoarseState, *, nx: int, ny: int, common_nz: int) -> CoarseState:
    nz = state.grid_shape[2]
    if nz == common_nz:
        return state
    if state.grid_shape[0] != nx or state.grid_shape[1] != ny:
        raise ValueError(f"Cannot pad state with different nx/ny. Got {state.grid_shape[:2]}, expected {(nx, ny)}")

    pad_n = common_nz - nz
    # Pad along last dim (z): (left, right) = (0, pad_n)
    # torch.nn.functional.pad expects (pad_last_dim_left, pad_last_dim_right, ...).
    import torch.nn.functional as F

    counts = F.pad(state.counts, (0, pad_n), mode="constant", value=0.0)
    momentum = F.pad(state.momentum, (0, pad_n), mode="constant", value=0.0)
    ke = F.pad(state.ke, (0, pad_n), mode="constant", value=0.0)
    order = F.pad(state.order, (0, pad_n), mode="constant", value=0.0)

    return CoarseState(counts=counts, momentum=momentum, ke=ke, order=order)


def _loss_mask_for_state(
    *,
    nx: int,
    ny: int,
    nz_valid: int,
    common_nz: int,
) -> torch.Tensor:
    """
    Returns mask shape: (1,nx,ny,common_nz)
    """
    mask = torch.zeros((1, nx, ny, common_nz), dtype=torch.float32)
    mask[..., :nz_valid] = 1.0
    return mask


def build_dataset(
    *,
    dumps_dir: str,
    out_path: str,
    species: SpeciesConfig,
    grid: GridConfig = GridConfig(),
    stride_k: int = 1,
    max_dump_files: Optional[int] = None,
    max_pairs: Optional[int] = None,
    pad_to_common_nz: bool = False,
    pad_nz_mode: str = "max",
    common_nz: Optional[int] = None,
    mask_loss_padded_cells: bool = False,
    num_workers: int = 1,
    order_lammps_type: int = _ORDER_DEFAULT_LAMMPS_TYPE,
) -> None:
    dumps_dir = str(dumps_dir)
    out_path = str(out_path)

    p = Path(dumps_dir)
    if not p.exists():
        raise FileNotFoundError(f"dumps_dir not found: {dumps_dir}")

    dump_paths = sorted([x for x in p.glob("*.dump")], key=_numeric_stem)
    if max_dump_files is not None:
        dump_paths = dump_paths[: int(max_dump_files)]
    if not dump_paths:
        raise ValueError(f"No .dump files found in {dumps_dir}")

    # Group dumps by their locked (nx, ny), so we can train per-shape without
    # padding x/y (which would break periodicity and batching).
    dump_info: List[Tuple[int, int, int, Path]] = []
    for dp in dump_paths:
        _, st, _ = make_coarse_state_from_dump(
            str(dp),
            species=species,
            grid=grid,
            order_lammps_type=order_lammps_type,
        )
        nx_i, ny_i, nz_i = st.grid_shape
        dump_info.append((nx_i, ny_i, nz_i, dp))

    grouped: Dict[Tuple[int, int], List[Tuple[int, Path]]] = {}
    for nx_i, ny_i, nz_i, dp in dump_info:
        grouped.setdefault((nx_i, ny_i), []).append((nz_i, dp))

    if not grouped:
        raise ValueError(f"No dumps found under {dumps_dir}")

    out_path_p = Path(out_path)

    # If we have only one (nx,ny) group, preserve the old behavior exactly.
    multiple_groups = len(grouped) > 1

    # Track global max_pairs across all generated datasets (if provided).
    remaining_pairs = int(max_pairs) if max_pairs is not None else None

    manifest_datasets: List[dict] = []

    for (nx, ny), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        # Determine nz values for padding decisions in this (nx,ny) group.
        dump_nzs = [nz_i for nz_i, _ in items]

        if not pad_to_common_nz:
            if len(set(dump_nzs)) != 1:
                raise ValueError(
                    f"pad_to_common_nz is disabled but dumps have different nz within xy={(nx, ny)}. "
                    f"Found nz values: {sorted(set(dump_nzs))}"
                )
            common_nz_i = int(dump_nzs[0])
        else:
            if pad_nz_mode == "max":
                common_nz_i = int(max(dump_nzs))
            elif pad_nz_mode == "first":
                common_nz_i = int(dump_nzs[0])
            elif pad_nz_mode == "fixed":
                if common_nz is None:
                    raise ValueError("--pad_nz_mode fixed requires --common_nz")
                common_nz_i = int(common_nz)
            else:
                raise ValueError(f"Invalid pad_nz_mode: {pad_nz_mode}")

        inputs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        loss_masks: List[torch.Tensor] = []

        # Main data loop: build pairs within each dump file.
        for _nz_i, dump_path in items:
            frames = make_coarse_states_from_dump(
                str(dump_path),
                species=species,
                grid=grid,
                order_lammps_type=order_lammps_type,
            )
            if len(frames) <= stride_k:
                continue

            for i in range(0, len(frames) - stride_k):
                if remaining_pairs is not None and len(inputs) >= int(remaining_pairs):
                    break

                _t0, in_state, _m0 = frames[i]
                _t1, tgt_state, _m1 = frames[i + stride_k]

                in_p = (
                    _pad_state_to_nz(in_state, nx=nx, ny=ny, common_nz=common_nz_i)
                    if pad_to_common_nz
                    else in_state
                )
                tgt_p = (
                    _pad_state_to_nz(tgt_state, nx=nx, ny=ny, common_nz=common_nz_i)
                    if pad_to_common_nz
                    else tgt_state
                )

                inputs.append(in_p.as_features())  # (C,nx,ny,common_nz_i)
                targets.append(tgt_p.as_features())

                if mask_loss_padded_cells:
                    tgt_nz = tgt_state.grid_shape[2]
                    loss_masks.append(
                        _loss_mask_for_state(nx=nx, ny=ny, nz_valid=tgt_nz, common_nz=common_nz_i)
                    )

            if remaining_pairs is not None and len(inputs) >= int(remaining_pairs):
                break

        if not inputs:
            # No pairs from this group (e.g. stride too large); skip.
            continue

        inputs_t = torch.stack(inputs, dim=0)
        targets_t = torch.stack(targets, dim=0)

        if remaining_pairs is not None:
            # Update global remaining based on what we actually added.
            remaining_pairs = max(0, int(remaining_pairs) - int(inputs_t.shape[0]))
            if remaining_pairs <= 0:
                # No more pairs needed globally.
                pass

        payload: Dict[str, object] = {
            "inputs": inputs_t,
            "targets": targets_t,
            "metadata": {
                "species": {"lammps_types": list(species.lammps_types), "masses": list(species.masses)},
                "grid": {
                    "a": float(grid.lattice_constant_a),
                    "periodic_xy": bool(grid.periodic_xy),
                    "nx": int(nx),
                    "ny": int(ny),
                    "nz": int(common_nz_i),
                },
                "stride_k": int(stride_k),
                "num_pairs": int(inputs_t.shape[0]),
                "pad_to_common_nz": bool(pad_to_common_nz),
                "pad_nz_mode": str(pad_nz_mode),
                "order_lammps_type": int(order_lammps_type),
            },
        }

        if mask_loss_padded_cells:
            payload["loss_mask"] = torch.stack(loss_masks, dim=0)

        if multiple_groups:
            # Write per-shape dataset.
            suffix = f"_nx{nx}_ny{ny}"
            if out_path_p.suffix == ".pt":
                out_ds_path = out_path_p.with_name(out_path_p.stem + suffix + out_path_p.suffix)
            else:
                out_ds_path = out_path_p / (out_path_p.name + suffix + ".pt")
        else:
            out_ds_path = out_path_p

        out_ds_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(out_ds_path))

        manifest_datasets.append(
            {
                "nx": int(nx),
                "ny": int(ny),
                "nz": int(common_nz_i),
                "dataset_path": str(out_ds_path),
                "num_pairs": int(inputs_t.shape[0]),
            }
        )

        if remaining_pairs is not None and remaining_pairs <= 0:
            break

    if not manifest_datasets:
        raise ValueError(
            "No training pairs produced. "
            f"Check stride_k={stride_k}, dump contents, and max_dump_files/max_pairs."
        )

    # If we generated multiple datasets, write a small manifest next to out_path.
    if multiple_groups:
        manifest_path = out_path_p.with_name(out_path_p.stem + "_by_xy_manifest.json")
        manifest_obj = {
            "out_path_base": str(out_path_p),
            "generated_datasets": manifest_datasets,
            "pad_to_common_nz": bool(pad_to_common_nz),
            "pad_nz_mode": str(pad_nz_mode),
            "mask_loss_padded_cells": bool(mask_loss_padded_cells),
        }
        import json

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS dump files into a trainable diffusion dataset (inputs/targets pairs)."
    )
    parser.add_argument("--dumps_dir", type=str, default="data/dumps", help="Directory containing *.dump files.")
    parser.add_argument("--out_path", type=str, default="data/processed/dataset.pt", help="Output dataset.pt path.")
    parser.add_argument("--stride_k", type=int, default=1, help="Pairing: frame t -> frame t + stride_k.")

    parser.add_argument("--species_types", type=str, default="1,3", help="Comma/space-separated LAMMPS atom types.")
    parser.add_argument("--masses", type=str, default="12.011,39.948", help="Comma/space-separated masses per species.")
    parser.add_argument("--a", type=float, default=3.5657157, help="Lattice constant a.")

    parser.add_argument("--order_lammps_type", type=int, default=_ORDER_DEFAULT_LAMMPS_TYPE)

    parser.add_argument("--max_dump_files", type=int, default=None)
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional cap on total number of training pairs.")

    parser.add_argument("--pad_to_common_nz", action="store_true", help="Pad each dump's states along z to a common nz.")
    parser.add_argument(
        "--pad_nz_mode",
        type=str,
        default="max",
        choices=["max", "first", "fixed"],
        help="How to select the common nz when padding.",
    )
    parser.add_argument("--common_nz", type=int, default=None, help="Used only with --pad_nz_mode fixed.")
    parser.add_argument("--mask_loss_padded_cells", action="store_true", help="Save a loss_mask to ignore padded cells.")

    parser.add_argument("--num_workers", type=int, default=1, help="Reserved for future parallelism.")

    args = parser.parse_args()

    species = SpeciesConfig(
        lammps_types=_parse_list_ints(args.species_types),
        masses=_parse_list_floats(args.masses),
    )
    grid = GridConfig(lattice_constant_a=float(args.a), periodic_xy=True, ignore_atoms_outside_z=True)

    build_dataset(
        dumps_dir=args.dumps_dir,
        out_path=args.out_path,
        species=species,
        grid=grid,
        stride_k=int(args.stride_k),
        max_dump_files=args.max_dump_files,
        max_pairs=args.max_pairs,
        pad_to_common_nz=bool(args.pad_to_common_nz),
        pad_nz_mode=str(args.pad_nz_mode),
        common_nz=args.common_nz,
        mask_loss_padded_cells=bool(args.mask_loss_padded_cells),
        num_workers=int(args.num_workers),
        order_lammps_type=int(args.order_lammps_type),
    )
    if Path(args.out_path).suffix == ".pt":
        print(f"Done preprocessing dataset(s) into base: {args.out_path}")
    else:
        print(f"Done preprocessing dataset(s) into: {args.out_path}")


if __name__ == "__main__":
    main()

