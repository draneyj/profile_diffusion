from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        out[k] = data[k]
    return out


def _avg_over_xy(field_xyz: np.ndarray) -> np.ndarray:
    """
    field_xyz: (..., nx, ny, nz) or (nx,ny,nz)
    Returns: (..., nz) averaged over x,y dims (axis=-3,-2).
    """
    return field_xyz.mean(axis=(-3, -2))


def _compute_field_profile(payload: dict, field: str, *, species_index: int) -> tuple[np.ndarray, str]:
    """
    Returns:
    - profile_tz: (T, nz)
    - title: str
    """
    if field == "order":
        arr = payload["order"]  # (T,1,nx,ny,nz)
        a_xyz = arr[:, 0]  # (T,nx,ny,nz)
        profile_tz = _avg_over_xy(a_xyz)  # (T,nz)
        return profile_tz, "Order parameter xi (avg over x,y)"

    if field == "ke":
        arr = payload["ke"]  # (T,1,nx,ny,nz)
        a_xyz = arr[:, 0]  # (T,nx,ny,nz)
        profile_tz = _avg_over_xy(a_xyz)
        return profile_tz, "Kinetic energy ke (avg over x,y)"

    if field == "counts":
        arr = payload["counts"]  # (T,S,nx,ny,nz)
        a_xyz = arr[:, species_index]  # (T,nx,ny,nz)
        profile_tz = _avg_over_xy(a_xyz)
        return profile_tz, f"Species {species_index} density (counts/cell) (avg over x,y)"

    if field == "momentum":
        arr = payload["momentum"]  # (T,3,nx,ny,nz)
        # magnitude: sqrt(sum_j p_j^2)
        p_xyz = arr  # (T,3,nx,ny,nz)
        p_mag = np.sqrt((p_xyz**2).sum(axis=1))  # (T,nx,ny,nz)
        profile_tz = _avg_over_xy(p_mag)
        return profile_tz, "Momentum magnitude (avg over x,y)"

    raise ValueError(f"Unsupported field={field}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize rollout as time-vs-z heatmap (avg over x,y).")
    parser.add_argument("--rollout_path", type=str, required=True, help="Path to .npz produced by diffusion/infer_random.py")
    parser.add_argument("--field", type=str, choices=["order", "ke", "counts", "momentum"], default="order")
    parser.add_argument("--species_index", type=int, default=0, help="Used for --field=counts.")
    parser.add_argument(
        "--a",
        type=float,
        default=None,
        help="Lattice constant for physical z axis labels. If omitted, tries rollout metadata/dataset metadata.",
    )
    parser.add_argument(
        "--z_sign",
        type=float,
        default=-1.0,
        help="Physical scaling sign for z: z = z_sign * a * z_index (default: -a*z_index).",
    )
    parser.add_argument(
        "--dt_fs",
        type=float,
        default=5.0,
        help="Time step per rollout prediction in femtoseconds (fs). Used to label x-axis in ps (default: 5 fs).",
    )
    parser.add_argument(
        "--t0_ps",
        type=float,
        default=0.0,
        help="Time offset in ps added to rollout times (default assumes rollout starts at 0 ps).",
    )
    parser.add_argument("--out_path", type=str, default=None, help="Output PNG path (default: auto).")
    parser.add_argument("--origin_lower", action="store_true", help="Use matplotlib origin='lower'.")
    args = parser.parse_args()

    # Headless plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _load_npz(args.rollout_path)
    profile_tz, title = _compute_field_profile(payload, args.field, species_index=int(args.species_index))

    # profile_tz shape: (T,nz) where T is "time" index in the rollout.
    T, nz = profile_tz.shape

    # Determine lattice constant `a` for physical z labels.
    a_val = args.a
    if a_val is None:
        if "lattice_constant_a" in payload:
            a_val = float(payload["lattice_constant_a"])
        elif "dataset_path" in payload:
            try:
                import torch

                ds = torch.load(str(payload["dataset_path"]), map_location="cpu")
                grid_meta = ds.get("metadata", {}).get("grid", {})
                a_val = float(grid_meta.get("a", 3.5657157))
            except Exception:
                a_val = 3.5657157
        else:
            a_val = 3.5657157

    # Physical z: z = z_sign * a * z_index.
    z_vals = args.z_sign * a_val * np.arange(nz, dtype=np.float64)

    # Physical time labels in ps: 1 ps = 1000 fs.
    t_vals_ps = args.t0_ps + (np.arange(T, dtype=np.float64) * float(args.dt_fs) / 1000.0)
    if args.out_path is None:
        out_dir = str(Path(args.rollout_path).parent)
        stem = Path(args.rollout_path).stem
        args.out_path = os.path.join(out_dir, f"{stem}_vis_tz_{args.field}.png")

    plt.figure(figsize=(7, 4))
    origin = "lower" if args.origin_lower else "upper"
    # profile_tz is (T, nz). Transpose so we plot x=t and y=z.
    im = plt.imshow(profile_tz.T, aspect="auto", origin=origin)
    plt.title(f"{title}\nt on x-axis, z on y-axis")
    plt.xlabel("t (ps)")
    plt.ylabel("z value")
    # Set sparse ticks in physical coordinates for readability.
    num_ticks = min(6, nz)
    if num_ticks >= 2:
        tick_indices = np.linspace(0, nz - 1, num_ticks).round().astype(int)
        tick_indices = np.unique(tick_indices)
        tick_labels = [f"{z_vals[i]:.3g}" for i in tick_indices]
        plt.yticks(tick_indices, tick_labels)

    # X ticks in ps.
    num_xticks = min(6, T)
    if num_xticks >= 2:
        x_tick_indices = np.linspace(0, T - 1, num_xticks).round().astype(int)
        x_tick_indices = np.unique(x_tick_indices)
        x_tick_labels = [f"{t_vals_ps[i]:.3g}" for i in x_tick_indices]
        plt.xticks(x_tick_indices, x_tick_labels)
    plt.colorbar(im, label=args.field)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=150)
    print(f"Wrote heatmap: {args.out_path}")


if __name__ == "__main__":
    main()

