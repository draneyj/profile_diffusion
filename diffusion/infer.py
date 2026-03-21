from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .cli_utils import add_device_arg, parse_device, seed_all
from .config import GridConfig, SpeciesConfig
from .models.option_i import OptionIModel
from .models.option_ii import OptionIIModel
from .models.option_iii import OptionIIIModel
from .state import CoarseState
from .data.make_data import make_coarse_state_from_dump


def _load_checkpoint(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint missing model_state_dict: {path}")
    return payload


def _instantiate_model(option: str, ckpt: dict, device: torch.device):
    metadata = ckpt.get("metadata", {})
    species_info = metadata.get("species", {})
    masses = species_info.get("masses", None)
    if masses is None:
        raise ValueError("Checkpoint metadata missing species.masses")
    num_species = len(masses)

    model_kwargs = dict(
        num_species=num_species,
        hidden_channels=ckpt.get("hidden_channels", 64),
    )

    if option == "i":
        model = OptionIModel(
            num_species=num_species,
            hidden_channels=model_kwargs["hidden_channels"],
            num_refine_steps=ckpt.get("num_refine_steps", 1),
            noise_std=ckpt.get("noise_std", 0.1),
        )
    elif option == "ii":
        model = OptionIIModel(
            num_species=num_species,
            hidden_channels=model_kwargs["hidden_channels"],
            soft_transfer=True,
        )
    elif option == "iii":
        model = OptionIIIModel(
            num_species=num_species,
            hidden_channels=model_kwargs["hidden_channels"],
        )
    else:
        raise ValueError(f"Unknown option: {option}")
    model.to(device)
    model.eval()
    if option == "ii":
        # Large-grid inference: prefer soft transfer (then rounding) for scalability.
        model.force_soft_transfer_eval = True
        model.soft_round_outputs = True
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-step inference/rollout with a trained model.")
    parser.add_argument("--option", type=str, choices=["i", "ii", "iii"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--initial_dump", type=str, required=True, help="Path to a LAMMPS dump file.")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    seed_all(args.seed)
    device = parse_device(args).device

    ckpt = _load_checkpoint(args.checkpoint)

    metadata = ckpt.get("metadata", {})
    species_meta = metadata.get("species", {})
    masses = species_meta.get("masses", None)
    lammps_types = species_meta.get("lammps_types", None)
    if masses is None or lammps_types is None:
        raise ValueError("Checkpoint metadata must include species.lammps_types and species.masses")

    grid_meta = metadata.get("grid", {})
    a = float(grid_meta.get("a", GridConfig().lattice_constant_a))

    species = SpeciesConfig(lammps_types=list(lammps_types), masses=list(map(float, masses)))
    grid = GridConfig(lattice_constant_a=a, periodic_xy=True)

    _, state, _ = make_coarse_state_from_dump(args.initial_dump, species=species, grid=grid)

    model = _instantiate_model(args.option, ckpt, device)

    if args.out_path is None:
        ckpt_stem = Path(args.checkpoint).stem
        args.out_path = f"data/processed/rollout_{args.option}_{ckpt_stem}.npz"

    # Store rollout arrays per step.
    rollout_counts: List[np.ndarray] = []
    rollout_momentum: List[np.ndarray] = []
    rollout_ke: List[np.ndarray] = []
    rollout_order: List[np.ndarray] = []

    current = CoarseState(
        counts=state.counts,
        momentum=state.momentum,
        ke=state.ke,
        order=state.order,
    )

    # Convert to model's preferred shape:
    # Our models accept both batched/unbatched; we'll pass unbatched.
    with torch.no_grad():
        for step in range(args.num_steps + 1):
            # Record current.
            rollout_counts.append(current.counts.detach().cpu().numpy())
            rollout_momentum.append(current.momentum.detach().cpu().numpy())
            rollout_ke.append(current.ke.detach().cpu().numpy())
            rollout_order.append(current.order.detach().cpu().numpy())

            if step == args.num_steps:
                break

            pred = model.predict_next(current.to(device))
            # Move back to CPU for storage.
            current = CoarseState(
                counts=pred.counts.detach().cpu(),
                momentum=pred.momentum.detach().cpu(),
                ke=pred.ke.detach().cpu(),
                order=pred.order.detach().cpu(),
            )

    np.savez_compressed(
        args.out_path,
        counts=np.stack(rollout_counts, axis=0),
        momentum=np.stack(rollout_momentum, axis=0),
        ke=np.stack(rollout_ke, axis=0),
        order=np.stack(rollout_order, axis=0),
        option=args.option,
        checkpoint=args.checkpoint,
        initial_dump=args.initial_dump,
        num_steps=args.num_steps,
        lattice_constant_a=a,
        lammps_types=lammps_types,
    )
    print(f"Wrote rollout to {args.out_path}")


if __name__ == "__main__":
    main()

