from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import GridConfig, SpeciesConfig
from .data.make_data import make_coarse_states_from_dump


def _parse_list_ints(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.replace(",", " ").split(" ") if x.strip()]


def _parse_list_floats(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.replace(",", " ").split(" ") if x.strip()]


def _avg_over_xy_torch(field_xyz_torch, *, x_dim: int = 1, y_dim: int = 2) -> np.ndarray:
    """
    field_xyz_torch: torch tensor
    returns: numpy array (nz,)
    """
    # mean over nx and ny dimensions to get per-z profile
    import torch

    prof = field_xyz_torch.mean(dim=(x_dim, y_dim))  # (nz,)
    return prof.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a raw LAMMPS dump as a heatmap of time vs z (avg over x,y)."
    )
    parser.add_argument("--dump_path", type=str, required=True, help="Path to a LAMMPS dump file (*.dump)")
    parser.add_argument(
        "--field",
        type=str,
        choices=["order", "ke", "counts", "momentum"],
        default="order",
        help="What to visualize.",
    )
    parser.add_argument(
        "--species_types",
        type=str,
        default="1,3",
        help="Comma/space-separated LAMMPS atom types for each species index.",
    )
    parser.add_argument(
        "--masses",
        type=str,
        default="12.011,39.948",
        help="Comma/space-separated masses matching --species_types order.",
    )
    parser.add_argument("--a", type=float, default=3.5657157, help="Lattice constant for z and order parameter.")

    parser.add_argument(
        "--order_lammps_type",
        type=int,
        default=1,
        help="LAMMPS atom type used for order parameter (default: 1=carbon).",
    )
    parser.add_argument("--species_index", type=int, default=0, help="Species index used for --field=counts.")

    # Time axis
    parser.add_argument("--dt_fs", type=float, default=5.0, help="Timestep dt in femtoseconds used to label time.")
    parser.add_argument("--t0_ps", type=float, default=0.0, help="Optional offset added to time labels (ps).")

    # z axis
    parser.add_argument("--z_sign", type=float, default=-1.0, help="Use z = z_sign * a * z_index (default: -a*z_index).")

    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on number of frames processed.")
    parser.add_argument("--out_path", type=str, default=None, help="Output PNG path (default: auto).")
    parser.add_argument(
        "--z_on_x",
        action="store_true",
        help="If set, plot x=z and y=t. Default plots x=t and y=z.",
    )
    parser.add_argument(
        "--origin_lower",
        action="store_true",
        help="(Deprecated) matplotlib origin='lower'. Kept for backwards compatibility.",
    )
    # Default behavior: origin="lower" to match the physical dump z-direction.
    # Use --origin_upper to flip vertically.
    parser.add_argument(
        "--origin_upper",
        action="store_true",
        help="matplotlib origin='upper' (flip vertically).",
    )

    args = parser.parse_args()

    species = SpeciesConfig(lammps_types=_parse_list_ints(args.species_types), masses=_parse_list_floats(args.masses))
    grid = GridConfig(lattice_constant_a=args.a, periodic_xy=True, ignore_atoms_outside_z=True)

    frames = make_coarse_states_from_dump(
        args.dump_path,
        species=species,
        grid=grid,
        order_lammps_type=args.order_lammps_type,
    )

    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError(f"No frames found in dump: {args.dump_path}")

    # Build matrix (T,nz) where each row corresponds to a frame time, columns correspond to z-index.
    T = len(frames)
    nz = int(frames[0][1].counts.shape[-1])
    profiles = np.zeros((T, nz), dtype=np.float64)
    times_ps = np.zeros((T,), dtype=np.float64)

    for i, (timestep, state, _meta) in enumerate(frames):
        import torch

        # Use parsed ITEM TIMESTEP as the MD step index; label time as timestep*dt_fs.
        times_ps[i] = args.t0_ps + float(timestep) * float(args.dt_fs) / 1000.0

        if args.field == "order":
            # state.order: (1,nx,ny,nz)
            profiles[i] = _avg_over_xy_torch(state.order[0], x_dim=0, y_dim=1)  # mean over nx,ny => (nz,)
        elif args.field == "ke":
            profiles[i] = _avg_over_xy_torch(state.ke[0], x_dim=0, y_dim=1)
        elif args.field == "counts":
            # state.counts: (S,nx,ny,nz)
            profiles[i] = _avg_over_xy_torch(state.counts[args.species_index], x_dim=0, y_dim=1)
        elif args.field == "momentum":
            # state.momentum: (3,nx,ny,nz) => magnitude (nx,ny,nz)
            p_mag = torch.sqrt((state.momentum**2).sum(dim=0))
            profiles[i] = _avg_over_xy_torch(p_mag, x_dim=0, y_dim=1)
        else:
            raise ValueError(f"Unsupported field {args.field}")

    # Physical z labels (z = z_sign * a * z_index).
    z_vals = args.z_sign * args.a * np.arange(nz, dtype=np.float64)

    # Output file naming.
    if args.out_path is None:
        out_dir = str(Path(args.dump_path).parent.parent / "processed")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(args.dump_path).stem
        args.out_path = os.path.join(out_dir, f"{stem}_vis_tz_{args.field}.png")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    origin = "upper" if args.origin_upper else "lower"

    # Choose axis orientation (default: x=t, y=z).
    if not args.z_on_x:
        # x=t, y=z => transpose to (nz,T)
        im = plt.imshow(
            profiles.T,
            aspect="auto",
            origin=origin,
            extent=[times_ps.min(), times_ps.max(), z_vals.min(), z_vals.max()],
        )
        plt.xlabel("t (ps)")
        plt.ylabel("z value")
    else:
        # x=z, y=t => (T,nz) as-is
        im = plt.imshow(
            profiles,
            aspect="auto",
            origin=origin,
            extent=[z_vals.min(), z_vals.max(), times_ps.min(), times_ps.max()],
        )
        plt.xlabel("z value")
        plt.ylabel("t (ps)")

    title = {
        "order": "Order parameter xi",
        "ke": "Kinetic energy ke",
        "counts": f"Species {args.species_index} density (counts/cell)",
        "momentum": "Momentum magnitude",
    }[args.field]
    plt.title(f"{title}\n(time vs z), avg over x,y")

    plt.colorbar(im, label=args.field)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=150)
    print(f"Wrote heatmap: {args.out_path}")


if __name__ == "__main__":
    main()

