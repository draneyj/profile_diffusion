from __future__ import annotations

import argparse
import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .cli_utils import add_device_arg, add_common_io_args, parse_device, seed_all
from .config import GridConfig
from .models.option_i import OptionIModel
from .models.option_ii import OptionIIModel
from .state import CoarseState


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    a,b: (B,C,nx,ny,nz)
    mask: (B,1,nx,ny,nz) with 1 for valid cells, 0 for padded cells.
    """
    se = (a - b) ** 2
    weighted = se * mask  # broadcasts over channel dim
    denom = mask.sum() * a.shape[1]
    if float(denom.item()) <= 0.0:
        return torch.mean(se)
    return weighted.sum() / denom


def load_dataset(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    for k in ["inputs", "targets", "metadata"]:
        if k not in payload:
            raise ValueError(f"Invalid dataset payload missing key '{k}' at {path}")
    return payload


def features_batch_to_state(features: torch.Tensor, *, num_species: int) -> CoarseState:
    """
    features: (B,C,nx,ny,nz)
    """

    if features.dim() != 5:
        raise ValueError(f"Expected features batch dim 5, got shape {tuple(features.shape)}")
    return CoarseState.from_features(features, num_species=num_species)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Option I or Option II diffusion models.")
    parser.add_argument("--option", type=str, choices=["i", "ii"], required=True)
    parser.add_argument("--dataset_path", type=str, default="data/processed/dataset.pt")
    parser.add_argument(
        "--dataset_paths",
        type=str,
        default=None,
        help="Comma-separated list of dataset.pt files. If provided, overrides --dataset_path and trains with interleaved batches from each dataset.",
    )
    parser.add_argument(
        "--shape_balance_mode",
        type=str,
        default="proportional",
        choices=["proportional", "equal"],
        help="How to interleave batches across multiple datasets when --dataset_paths is set.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of samples used for validation.")
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=5,
        help="Save checkpoints only every N epochs (final epoch is always saved).",
    )
    parser.add_argument("--split_seed", type=int, default=None, help="Seed for train/val split (defaults to --seed).")

    # Model hyperparameters.
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_refine_steps", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--soft_transfer", action="store_true", help="Use soft transfer for Option II training")
    parser.add_argument("--hard_eval", action="store_true", help="Force hard integer transfer even in train() (debug).")

    add_device_arg(parser)
    add_common_io_args(parser)

    args = parser.parse_args()
    seed_all(args.seed)
    device_cfg = parse_device(args)

    dataset_paths: list[str]
    if args.dataset_paths is not None:
        dataset_paths = [p.strip() for p in args.dataset_paths.split(",") if p.strip()]
        if not dataset_paths:
            raise ValueError("--dataset_paths was provided but parsed to an empty list")
    else:
        dataset_paths = [args.dataset_path]

    # Load datasets, split into train/val per dataset, and interleave training batches.
    val_fraction = float(args.val_fraction)
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"--val_fraction must be in [0,1). Got {val_fraction}")

    split_seed = args.split_seed if args.split_seed is not None else args.seed

    per_dataset = []
    num_species: int | None = None
    metadata_first: dict | None = None

    for path in dataset_paths:
        payload = load_dataset(path)
        inputs = payload["inputs"]  # (N,C,nx,ny,nz)
        targets = payload["targets"]
        metadata = payload["metadata"]
        loss_mask = payload.get("loss_mask", None)

        species_info = metadata.get("species", None)
        if not species_info or "masses" not in species_info:
            raise ValueError(f"Dataset metadata missing species masses: {path}")
        masses = species_info["masses"]
        ds_num_species = len(masses)
        if num_species is None:
            num_species = ds_num_species
            metadata_first = metadata
        elif ds_num_species != num_species:
            raise ValueError(
                f"Inconsistent num_species across datasets: got {ds_num_species} vs expected {num_species} in {path}"
            )

        N = int(inputs.shape[0])
        val_size = int(N * val_fraction)
        g = torch.Generator().manual_seed(int(split_seed))
        perm = torch.randperm(N, generator=g)
        if val_size > 0:
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
        else:
            train_idx = perm
            val_idx = None

        inputs_train = inputs[train_idx]
        targets_train = targets[train_idx]
        if loss_mask is not None:
            loss_mask_train = loss_mask[train_idx]
            ds_train = TensorDataset(inputs_train, targets_train, loss_mask_train)
        else:
            ds_train = TensorDataset(inputs_train, targets_train)

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False)

        dl_val = None
        num_val_samples = 0
        if val_idx is not None and N - int(val_idx.shape[0]) > 0:
            inputs_val = inputs[val_idx]
            targets_val = targets[val_idx]
            num_val_samples = int(inputs_val.shape[0])
            if num_val_samples > 0:
                if loss_mask is not None:
                    loss_mask_val = loss_mask[val_idx]
                    ds_val = TensorDataset(inputs_val, targets_val, loss_mask_val)
                else:
                    ds_val = TensorDataset(inputs_val, targets_val)
                dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

        per_dataset.append(
            {
                "path": path,
                "metadata": metadata,
                "dl_train": dl_train,
                "dl_val": dl_val,
                "num_train_samples": int(inputs_train.shape[0]),
                "num_val_samples": int(num_val_samples),
                "has_loss_mask": loss_mask is not None,
            }
        )

    assert num_species is not None

    if args.option == "i":
        model: nn.Module = OptionIModel(
            num_species=num_species,
            hidden_channels=args.hidden_channels,
            num_refine_steps=args.num_refine_steps,
            noise_std=args.noise_std,
        )
    else:
        model = OptionIIModel(
            num_species=num_species,
            hidden_channels=args.hidden_channels,
            soft_transfer=args.soft_transfer or (not args.hard_eval),
        )

    model.to(device_cfg.device)
    model.train()

    # Force hard transfer if requested (debug).
    if args.option == "ii" and args.hard_eval:
        model.soft_transfer = False

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Training interleave weights across datasets (shape balance).
    train_sizes = torch.tensor([d["num_train_samples"] for d in per_dataset], dtype=torch.float64)
    if train_sizes.sum().item() <= 0:
        raise ValueError("All datasets have 0 training samples after split")
    if args.shape_balance_mode == "equal":
        probs = torch.ones_like(train_sizes) / float(len(train_sizes))
    else:
        probs = train_sizes / train_sizes.sum()

    # Approximate "one epoch" as covering the combined training sample count once.
    total_train_samples = int(train_sizes.sum().item())
    steps_per_epoch = max(1, int(math.ceil(total_train_samples / float(args.batch_size))))

    for epoch in range(1, args.epochs + 1):
        running_train_loss_sum = 0.0
        running_train_n = 0

        train_iters = [iter(d["dl_train"]) for d in per_dataset]

        # Deterministic-ish choice per epoch.
        g_choice = torch.Generator().manual_seed(int(args.seed) + int(epoch) * 10007)

        for _step in range(steps_per_epoch):
            ds_idx = int(torch.multinomial(probs, num_samples=1, generator=g_choice).item())
            try:
                batch = next(train_iters[ds_idx])
            except StopIteration:
                train_iters[ds_idx] = iter(per_dataset[ds_idx]["dl_train"])
                batch = next(train_iters[ds_idx])

            if len(batch) == 3:
                xb, yb, mb = batch
                mb = mb.to(device_cfg.device)
            else:
                xb, yb = batch

            xb = xb.to(device_cfg.device)
            yb = yb.to(device_cfg.device)

            current = features_batch_to_state(xb, num_species=num_species)
            target = features_batch_to_state(yb, num_species=num_species)

            pred = model.predict_next(current, target_state=target)

            if len(batch) == 3:
                loss_counts = _masked_mse(pred.counts, target.counts, mb)
                loss_momentum = _masked_mse(pred.momentum, target.momentum, mb)
                loss_ke = _masked_mse(pred.ke, target.ke, mb)
                loss_order = _masked_mse(pred.order, target.order, mb)
            else:
                loss_counts = _mse(pred.counts, target.counts)
                loss_momentum = _mse(pred.momentum, target.momentum)
                loss_ke = _mse(pred.ke, target.ke)
                loss_order = _mse(pred.order, target.order)
            loss = loss_counts + loss_momentum + loss_ke + loss_order

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(xb.shape[0])
            running_train_loss_sum += float(loss.item()) * bs
            running_train_n += bs

        avg_train = running_train_loss_sum / max(1, running_train_n)

        avg_val = None
        if any(d["dl_val"] is not None for d in per_dataset):
            model.eval()
            running_val_loss_sum = 0.0
            running_val_n = 0
            with torch.no_grad():
                for ds_idx, d in enumerate(per_dataset):
                    if d["dl_val"] is None:
                        continue
                    dl_val = d["dl_val"]
                    for batch in dl_val:
                        if len(batch) == 3:
                            xb, yb, mb = batch
                            mb = mb.to(device_cfg.device)
                        else:
                            xb, yb = batch
                        xb = xb.to(device_cfg.device)
                        yb = yb.to(device_cfg.device)

                        current = features_batch_to_state(xb, num_species=num_species)
                        target = features_batch_to_state(yb, num_species=num_species)
                        pred = model.predict_next(current, target_state=target)

                        if len(batch) == 3:
                            loss_counts = _masked_mse(pred.counts, target.counts, mb)
                            loss_momentum = _masked_mse(pred.momentum, target.momentum, mb)
                            loss_ke = _masked_mse(pred.ke, target.ke, mb)
                            loss_order = _masked_mse(pred.order, target.order, mb)
                        else:
                            loss_counts = _mse(pred.counts, target.counts)
                            loss_momentum = _mse(pred.momentum, target.momentum)
                            loss_ke = _mse(pred.ke, target.ke)
                            loss_order = _mse(pred.order, target.order)
                        loss = loss_counts + loss_momentum + loss_ke + loss_order
                        bs = int(xb.shape[0])
                        running_val_loss_sum += float(loss.item()) * bs
                        running_val_n += bs

            avg_val = running_val_loss_sum / max(1, running_val_n)
            model.train()

        # Checkpoint cadence: save only every N epochs plus always the final epoch.
        save_every = int(args.save_every_n_epochs)
        should_save = (save_every <= 1) or (epoch % save_every == 0) or (epoch == args.epochs)
        if should_save:
            ckpt_path = os.path.join(out_dir, f"checkpoint_option{args.option}_epoch{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "option": args.option,
                    "num_species": num_species,
                    "metadata": metadata_first if metadata_first is not None else {},
                    "hidden_channels": args.hidden_channels,
                    "num_refine_steps": args.num_refine_steps,
                    "noise_std": args.noise_std,
                    "soft_transfer": bool(args.soft_transfer or (not args.hard_eval)),
                },
                ckpt_path,
            )

        if avg_val is None:
            print(f"[epoch {epoch}] train_loss={avg_train:.6f}")
        else:
            print(f"[epoch {epoch}] train_loss={avg_train:.6f} val_loss={avg_val:.6f}")


if __name__ == "__main__":
    main()

