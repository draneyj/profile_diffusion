#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Allow `python scripts/preprocess_dumps.py` to import the local `diffusion` package.
ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT_DIR))

from diffusion.config import GridConfig, SpeciesConfig
from diffusion.data.make_data import make_coarse_states_from_dump


def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _sorted_dump_paths(dumps_dir: str) -> List[Path]:
    p = Path(dumps_dir)
    dump_paths = sorted([x for x in p.glob("*.dump")], key=lambda x: int(x.stem))
    return dump_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess LAMMPS dump files (each may contain many TIMESTEP frames) into per-frame tensors (data/preprocessed)."
    )
    parser.add_argument("--dumps_dir", type=str, default="data/dumps")
    parser.add_argument("--out_dir", type=str, default="data/preprocessed")
    parser.add_argument("--a", type=float, default=3.5657157, help="Lattice constant")
    parser.add_argument(
        "--species_types",
        type=str,
        default="1,3",
        help="Comma-separated LAMMPS atom type IDs corresponding to contiguous species indices.",
    )
    parser.add_argument(
        "--masses",
        type=str,
        default="12.011,39.948",
        help="Comma-separated masses (same length/order as --species_types).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to preprocess total across all dumps (for testing).",
    )
    parser.add_argument(
        "--max_dump_files",
        type=int,
        default=None,
        help="Optional cap on number of dump files to preprocess (for testing).",
    )
    parser.add_argument("--device_seed", type=int, default=0)
    parser.add_argument(
        "--order_lammps_type",
        type=int,
        default=1,
        help="LAMMPS atom type used for computing the order parameter (default: 1 = carbon).",
    )
    args = parser.parse_args()

    species = SpeciesConfig(
        lammps_types=_parse_list_ints(args.species_types),
        masses=_parse_list_floats(args.masses),
    )
    grid = GridConfig(lattice_constant_a=args.a, periodic_xy=True, ignore_atoms_outside_z=True)

    dumps_dir = str(Path(args.dumps_dir).resolve())
    out_dir = Path(args.out_dir).resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    dump_paths = _sorted_dump_paths(dumps_dir)
    if args.max_dump_files is not None:
        dump_paths = dump_paths[: args.max_dump_files]
    if not dump_paths:
        raise ValueError(f"No .dump files found in {dumps_dir}")

    manifest: Dict[str, object] = {
        "dumps_dir": dumps_dir,
        "out_dir": str(out_dir),
        "a": args.a,
        "species": asdict(species),
        "grid": asdict(grid),
        "momentum": {"representation": "unit_direction", "eps": 1e-8},
        "frames": [],
    }

    reference_grid_shape: Optional[Tuple[int, int, int]] = None

    frames_processed = 0

    for path in dump_paths:
        dump_stem = int(path.stem)
        print(f"[dump {dump_stem}] Preprocessing {path.name} ...")

        frames = make_coarse_states_from_dump(
            str(path),
            species=species,
            grid=grid,
            order_lammps_type=args.order_lammps_type,
        )
        # Determine grid dims from first frame.
        nx, ny, nz = frames[0][1].grid_shape
        if reference_grid_shape is None:
            reference_grid_shape = (nx, ny, nz)
        elif (nx, ny, nz) != reference_grid_shape:
            raise ValueError(
                f"Inconsistent grid shapes across frames: got {(nx, ny, nz)} for {path}, expected {reference_grid_shape}"
            )

        for frame_idx, (timestep, state, _meta) in enumerate(frames):
            if args.max_frames is not None and frames_processed >= args.max_frames:
                break

            frame_out_path = frames_dir / f"{dump_stem:06d}_frame{frame_idx:06d}.pt"
            torch.save(
                {
                    "dump_stem": dump_stem,
                    "frame_index": frame_idx,
                    "timestep": timestep,
                    "counts": state.counts,
                    "momentum": state.momentum,
                    "ke": state.ke,
                    "order": state.order,
                    "grid": {"nx": nx, "ny": ny, "nz": nz, "a": args.a},
                },
                str(frame_out_path),
            )

            manifest["frames"].append(
                {
                    "dump_stem": dump_stem,
                    "dump_path": str(path),
                    "frame_index": frame_idx,
                    "frame_path": str(frame_out_path),
                    "timestep": timestep,
                }
            )
            frames_processed += 1

        if args.max_frames is not None and frames_processed >= args.max_frames:
            break

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Wrote {manifest_path}")


if __name__ == "__main__":
    main()

