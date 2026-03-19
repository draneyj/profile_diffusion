from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .cli_utils import seed_all
from .state import CoarseState


def _load_dataset(path: str) -> dict:
    # dataset.pt is saved via torch.save; it will load tensors when torch is available.
    import torch

    payload = torch.load(path, map_location="cpu")
    for k in ["inputs", "targets", "metadata"]:
        if k not in payload:
            raise ValueError(f"Dataset payload missing key '{k}': {path}")
    return payload


def _parse_species_indices(s: str) -> list[int]:
    s = s.strip().replace(",", " ")
    parts = [p for p in s.split(" ") if p.strip()]
    return [int(p) for p in parts]


def _avg_over_x(field_xyz: np.ndarray) -> np.ndarray:
    """
    field_xyz: (nx,ny,nz) -> average over x => (ny,nz)
    """
    return field_xyz.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dataset samples as 2D y-z maps (avg over x).")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--from_dataset", type=str, choices=["inputs", "targets"], default="inputs")
    parser.add_argument("--sample_index", type=int, default=-1, help="Sample index; -1 picks random.")
    parser.add_argument("--sample_seed", type=int, default=0)

    parser.add_argument("--field", type=str, choices=["order", "counts"], default="order")
    parser.add_argument("--species_index", type=int, default=0, help="Used for --field=counts")
    parser.add_argument(
        "--species_indices",
        type=str,
        default=None,
        help="Comma/space-separated list for --field=counts (e.g. '0,1').",
    )
    parser.add_argument("--all_species", action="store_true", help="If --field=counts, plot all species.")

    parser.add_argument(
        "--mask_padded_cells",
        action="store_true",
        help="If dataset has `loss_mask`, mask padded z-cells out of the visualization.",
    )

    parser.add_argument("--out_path", type=str, default=None, help="If set, write a single PNG to this path.")
    args = parser.parse_args()

    # Import torch only inside the function so matplotlib can be used headlessly without torch.
    payload = _load_dataset(args.dataset_path)
    inputs = payload["inputs"]
    targets = payload["targets"]
    metadata = payload["metadata"]

    species_meta = metadata.get("species", {})
    masses = species_meta.get("masses", None)
    if masses is None:
        raise ValueError("Dataset metadata missing metadata.species.masses")
    num_species = int(len(masses))

    N = int(inputs.shape[0])
    seed_all(args.sample_seed)

    if args.sample_index < 0:
        # torch.randint would require torch import; use numpy RNG for index selection.
        rng = np.random.default_rng(args.sample_seed)
        idx = int(rng.integers(0, N))
    else:
        idx = int(args.sample_index)
        if not (0 <= idx < N):
            raise ValueError(f"--sample_index out of range: {idx} not in [0,{N})")

    feats = inputs[idx] if args.from_dataset == "inputs" else targets[idx]  # (C,nx,ny,nz)
    state = CoarseState.from_features(feats, num_species=num_species)

    loss_mask = payload.get("loss_mask", None)
    z_mask_2d = None
    if args.mask_padded_cells and loss_mask is not None:
        # loss_mask: (N,1,nx,ny,nz)
        mask = loss_mask[idx]  # (1,nx,ny,nz)
        # average over x => (ny,nz); padded z-cells should be 0 across all x.
        mask_np = mask[0].numpy()  # (nx,ny,nz)
        z_mask_2d = mask_np.mean(axis=0)  # (ny,nz)

    # Ensure headless plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.field == "order":
        # state.order: (1,nx,ny,nz)
        order_xyz = state.order[0].detach().cpu().numpy()
        a2d = _avg_over_x(order_xyz)  # (ny,nz)
        if z_mask_2d is not None:
            a2d = np.where(z_mask_2d > 0.5, a2d, np.nan)
        title = "Order parameter (xi), avg over x"
        fields_to_plot = [(None, a2d, title)]

    elif args.field == "counts":
        S = num_species
        if args.all_species:
            species_list = list(range(S))
        elif args.species_indices is not None:
            species_list = _parse_species_indices(args.species_indices)
        else:
            species_list = [int(args.species_index)]

        fields_to_plot = []
        for sp in species_list:
            counts_xyz = state.counts[sp].detach().cpu().numpy()  # (nx,ny,nz)
            a2d = _avg_over_x(counts_xyz)  # (ny,nz)
            if z_mask_2d is not None:
                a2d = np.where(z_mask_2d > 0.5, a2d, np.nan)
            title = f"Species {sp} density (counts/cell), avg over x"
            fields_to_plot.append((sp, a2d, title))
    else:
        raise ValueError(f"Unsupported field: {args.field}")

    out_paths = []
    if args.out_path is not None:
        if len(fields_to_plot) != 1:
            raise ValueError("--out_path can only be set when plotting a single field/species.")
        out_paths = [args.out_path]
    else:
        out_dir = str(Path(args.dataset_path).parent)
        stem = Path(args.dataset_path).stem
        out_paths = [
            os.path.join(out_dir, f"{stem}_vis_{args.field}_idx{idx}_from{args.from_dataset}" + (f"_species{sp}" if sp is not None else "") + ".png")
            for (sp, _a2d, _title) in fields_to_plot
        ]

    for (sp, a2d, title), out_path in zip(fields_to_plot, out_paths):
        plt.figure(figsize=(6, 4))
        im = plt.imshow(a2d.T, origin="lower", aspect="auto")  # y on x-axis, z on y-axis
        plt.title(f"{title} (idx {idx})")
        plt.xlabel("y")
        plt.ylabel("z")
        plt.colorbar(im, label=args.field)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Wrote visualization: {out_path}")


if __name__ == "__main__":
    main()

