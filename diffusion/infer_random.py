from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .cli_utils import add_device_arg, parse_device, seed_all
from .models.option_i import OptionIModel
from .models.option_ii import OptionIIModel
from .models.option_iii import OptionIIIModel
from .state import CoarseState


def _load_checkpoint(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint missing 'model_state_dict': {path}")
    return payload


def _instantiate_from_checkpoint(option: str, ckpt: dict, device: torch.device):
    metadata = ckpt.get("metadata", {})
    species_meta = metadata.get("species", {})
    masses = species_meta.get("masses", None)
    if masses is None:
        raise ValueError("Checkpoint metadata missing metadata.species.masses")
    num_species = len(masses)

    hidden_channels = int(ckpt.get("hidden_channels", 64))
    num_refine_steps = int(ckpt.get("num_refine_steps", 1))
    noise_std = float(ckpt.get("noise_std", 0.1))
    soft_transfer = bool(ckpt.get("soft_transfer", True))

    if option == "i":
        model = OptionIModel(
            num_species=num_species,
            hidden_channels=hidden_channels,
            num_refine_steps=num_refine_steps,
            noise_std=noise_std,
        )
    elif option == "ii":
        model = OptionIIModel(
            num_species=num_species,
            hidden_channels=hidden_channels,
            soft_transfer=soft_transfer,
        )
    elif option == "iii":
        model = OptionIIIModel(
            num_species=num_species,
            hidden_channels=hidden_channels,
        )
    else:
        raise ValueError(f"Unknown option={option}")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    if option == "ii":
        # For large-grid scalability, prefer soft transfer + rounding at the end.
        model.force_soft_transfer_eval = True
        model.soft_round_outputs = True

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference starting from a random coarse-grained state sampled from a dataset."
    )
    parser.add_argument("--option", type=str, choices=["i", "ii", "iii"], required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--start_from", type=str, choices=["inputs", "targets"], default="inputs")
    parser.add_argument("--sample_seed", type=int, default=0)

    add_device_arg(parser)
    parser.add_argument("--out_path", type=str, default=None)

    args = parser.parse_args()

    seed_all(args.sample_seed)
    device = parse_device(args).device

    ds = torch.load(args.dataset_path, map_location="cpu")
    inputs = ds["inputs"]
    targets = ds["targets"]
    metadata = ds.get("metadata", {})
    species_meta = metadata.get("species", {})
    masses = species_meta.get("masses", None)
    if masses is None:
        raise ValueError("Dataset metadata missing metadata.species.masses")
    num_species = len(masses)

    grid_meta = metadata.get("grid", {}) if isinstance(metadata, dict) else {}
    a_val = float(grid_meta.get("a", 3.5657157))
    nz_val = int(inputs.shape[-1])

    ckpt = _load_checkpoint(args.checkpoint)
    model = _instantiate_from_checkpoint(args.option, ckpt, device)

    start_tensor = inputs if args.start_from == "inputs" else targets
    n = int(start_tensor.shape[0])
    idx = int(torch.randint(low=0, high=n, size=(1,)).item())
    start_features = start_tensor[idx]  # (C,nx,ny,nz)

    current = CoarseState.from_features(start_features, num_species=num_species)

    if args.out_path is None:
        ckpt_stem = Path(args.checkpoint).stem
        args.out_path = f"data/processed/rollout_random_option{args.option}_{ckpt_stem}_idx{idx}.npz"

    rollout_counts = []
    rollout_momentum = []
    rollout_ke = []
    rollout_order = []

    with torch.no_grad():
        for step in range(args.num_steps + 1):
            rollout_counts.append(current.counts.detach().cpu().numpy())
            rollout_momentum.append(current.momentum.detach().cpu().numpy())
            rollout_ke.append(current.ke.detach().cpu().numpy())
            rollout_order.append(current.order.detach().cpu().numpy())

            if step == args.num_steps:
                break

            pred = model.predict_next(current.to(device))

            # Keep `current` unbatched so shapes remain stack-compatible across steps.
            counts = pred.counts.detach().cpu()
            momentum = pred.momentum.detach().cpu()
            ke = pred.ke.detach().cpu()
            order = pred.order.detach().cpu()
            if counts.dim() == 5:
                counts = counts[0]
                momentum = momentum[0]
                ke = ke[0]
                order = order[0]

            current = CoarseState(counts=counts, momentum=momentum, ke=ke, order=order)

    np.savez_compressed(
        args.out_path,
        option=args.option,
        dataset_path=args.dataset_path,
        checkpoint=args.checkpoint,
        start_from=args.start_from,
        sample_index=idx,
        num_steps=args.num_steps,
        lattice_constant_a=a_val,
        grid_nz=nz_val,
        counts=np.stack(rollout_counts, axis=0),
        momentum=np.stack(rollout_momentum, axis=0),
        ke=np.stack(rollout_ke, axis=0),
        order=np.stack(rollout_order, axis=0),
    )
    print(f"Wrote rollout to {args.out_path}")


if __name__ == "__main__":
    main()

