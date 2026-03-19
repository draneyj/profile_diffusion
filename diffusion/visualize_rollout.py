from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    # np.load returns an NpzFile; convert to dict of arrays/scalars
    out = {}
    for k in data.files:
        out[k] = data[k]
    return out


def _parse_species_indices(s: str) -> list[int]:
    # Accept "0,1,2" or "0 1 2"
    s = s.strip().replace(",", " ")
    parts = [p for p in s.split(" ") if p.strip()]
    return [int(p) for p in parts]


def _field_to_2d(field: str, payload: dict, *, species_index: int, step_index: int) -> tuple[np.ndarray, str]:
    # Rollout arrays are expected to be saved by diffusion/infer_random.py:
    # - order: (T,1,nx,ny,nz)
    # - ke:    (T,1,nx,ny,nz)
    # - counts:(T,S,nx,ny,nz)
    # - momentum: (T,3,nx,ny,nz)
    if field == "order":
        arr = payload["order"]  # (T,1,nx,ny,nz)
        a = arr[step_index, 0]  # (nx,ny,nz)
        a2d = a.mean(axis=0)  # avg over x -> (ny,nz)
        title = "Order parameter (xi), avg over x"
        return a2d, title

    if field == "ke":
        arr = payload["ke"]  # (T,1,nx,ny,nz)
        a = arr[step_index, 0]  # (nx,ny,nz)
        a2d = a.mean(axis=0)
        title = "Kinetic energy (ke), avg over x"
        return a2d, title

    if field == "counts":
        arr = payload["counts"]  # (T,S,nx,ny,nz)
        a = arr[step_index, species_index]  # (nx,ny,nz)
        a2d = a.mean(axis=0)
        title = f"Species {species_index} density (counts/cell), avg over x"
        return a2d, title

    if field == "momentum":
        arr = payload["momentum"]  # (T,3,nx,ny,nz)
        # Momentum magnitude per cell: sqrt(sum_j p_j^2)
        p = arr[step_index]  # (3,nx,ny,nz)
        p_mag = np.sqrt((p**2).sum(axis=0))  # (nx,ny,nz)
        a2d = p_mag.mean(axis=0)  # (ny,nz)
        title = "Momentum magnitude, avg over x"
        return a2d, title

    raise ValueError(f"Unknown field={field}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a rollout as a 2D y-z map (avg over x).")
    parser.add_argument("--rollout_path", type=str, required=True, help="Path to .npz produced by diffusion/infer_random.py")
    parser.add_argument("--field", type=str, choices=["order", "ke", "counts", "momentum"], default="order")
    parser.add_argument("--species_index", type=int, default=0, help="Species index for --field=counts")
    parser.add_argument(
        "--species_indices",
        type=str,
        default=None,
        help="Comma/space-separated list of species indices to visualize for --field=counts (e.g. '0,1').",
    )
    parser.add_argument(
        "--all_species",
        action="store_true",
        help="If --field=counts, visualize all species (writes one PNG per species).",
    )
    parser.add_argument("--step_index", type=str, default="-1", help="Timestep index to visualize; use -1 for last")
    parser.add_argument("--out_path", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    # Ensure headless safety.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _load_npz(args.rollout_path)
    # Infer T from one of the fields.
    key = args.field if args.field != "counts" else "counts"
    arr = payload[key]
    T = int(arr.shape[0])

    step_index_int = int(args.step_index)
    if step_index_int == -1:
        step_index_int = T - 1
    if not (0 <= step_index_int < T):
        raise ValueError(f"--step_index must be in [-1,{T-1}] got {args.step_index}")

    if args.field == "counts":
        # counts is (T,S,nx,ny,nz)
        S = int(payload["counts"].shape[1])
        if args.all_species:
            species_list = list(range(S))
        elif args.species_indices is not None:
            species_list = _parse_species_indices(args.species_indices)
        else:
            species_list = [int(args.species_index)]

        if args.out_path is not None and len(species_list) != 1:
            raise ValueError("--out_path can only be set when plotting a single species.")

        for sp in species_list:
            a2d, title = _field_to_2d(args.field, payload, species_index=sp, step_index=step_index_int)  # (ny,nz)

            if args.out_path is None:
                out_dir = str(Path(args.rollout_path).parent)
                stem = Path(args.rollout_path).stem
                out_path = os.path.join(out_dir, f"{stem}_vis_counts_species{sp}_step{step_index_int}.png")
            else:
                out_path = args.out_path

            plt.figure(figsize=(6, 4))
            im = plt.imshow(a2d.T, origin="lower", aspect="auto")  # y on x-axis, z on y-axis
            plt.title(f"{title} (step {step_index_int})")
            plt.xlabel("y")
            plt.ylabel("z")
            plt.colorbar(im, label="counts")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print(f"Wrote visualization: {out_path}")
    else:
        a2d, title = _field_to_2d(
            args.field, payload, species_index=int(args.species_index), step_index=step_index_int
        )  # (ny,nz)

        # Prepare output name.
        if args.out_path is None:
            out_dir = str(Path(args.rollout_path).parent)
            stem = Path(args.rollout_path).stem
            out_path = os.path.join(out_dir, f"{stem}_vis_{args.field}_step{step_index_int}.png")
        else:
            out_path = args.out_path

        # Plot.
        plt.figure(figsize=(6, 4))
        im = plt.imshow(a2d.T, origin="lower", aspect="auto")  # transpose so y on x-axis, z on y-axis
        plt.title(f"{title} (step {step_index_int})")
        plt.xlabel("y")
        plt.ylabel("z")
        plt.colorbar(im, label=args.field)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Wrote visualization: {out_path}")


if __name__ == "__main__":
    main()

