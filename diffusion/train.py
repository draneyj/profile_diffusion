from __future__ import annotations

import argparse
import csv
import os
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .cli_utils import add_device_arg, add_common_io_args, parse_device, seed_all
from .config import GridConfig
from .models.option_i import OptionIModel
from .models.option_ii import OptionIIModel
from .models.option_iii import OptionIIIModel
from .state import CoarseState


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def _target_rms(t: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Detached RMS of target for loss scaling (per batch)."""
    if mask is None:
        return torch.sqrt(torch.mean(t * t) + 1e-20)
    se = (t * t) * mask
    denom = mask.sum() * t.shape[1]
    return torch.sqrt(se.sum() / denom.clamp(min=1.0) + 1e-20)


def _scaled_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    rms_floor: float,
) -> torch.Tensor:
    """
    MSE(pred, target) divided by max(RMS(target)^2, rms_floor^2).
    Keeps gradients on pred/target difference only (scale detached from target).
    """
    rms = _target_rms(target, mask).detach()
    scale_sq = torch.clamp(rms * rms, min=float(rms_floor) ** 2)
    if mask is None:
        return _mse(pred, target) / scale_sq
    return _masked_mse(pred, target, mask) / scale_sq


def compute_state_loss(
    pred: CoarseState,
    target: CoarseState,
    mask: Optional[torch.Tensor],
    *,
    loss_balance: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (loss_counts, loss_momentum, loss_ke, loss_order, loss_total).
    loss_balance: 'none' = raw mean MSE per field, summed; 'rms' = MSE scaled by target RMS per field (recommended for ii/iii).
    """
    if loss_balance == "none":
        if mask is None:
            lc = _mse(pred.counts, target.counts)
            lm = _mse(pred.momentum, target.momentum)
            lk = _mse(pred.ke, target.ke)
            lo = _mse(pred.order, target.order)
        else:
            lc = _masked_mse(pred.counts, target.counts, mask)
            lm = _masked_mse(pred.momentum, target.momentum, mask)
            lk = _masked_mse(pred.ke, target.ke, mask)
            lo = _masked_mse(pred.order, target.order, mask)
    elif loss_balance == "rms":
        # Floors avoid blowing up scale when a channel is near-zero everywhere.
        lc = _scaled_mse(pred.counts, target.counts, mask, rms_floor=1.0)
        # Momentum channels are unit directions (or zero); typical RMS ≤ 1.
        lm = _scaled_mse(pred.momentum, target.momentum, mask, rms_floor=0.25)
        lk = _scaled_mse(pred.ke, target.ke, mask, rms_floor=1e-3)
        lo = _scaled_mse(pred.order, target.order, mask, rms_floor=0.25)
    else:
        raise ValueError(f"Unknown loss_balance: {loss_balance!r}")
    total = lc + lm + lk + lo
    return lc, lm, lk, lo, total


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
    parser = argparse.ArgumentParser(description="Train Option I/II/III diffusion models.")
    parser.add_argument("--option", type=str, choices=["i", "ii", "iii"], required=True)
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
    parser.add_argument(
        "--loss_balance",
        type=str,
        default="auto",
        choices=["auto", "none", "rms"],
        help="auto: for option ii/iii use RMS-normalized MSE per field (counts/momentum/ke/order); "
        "none: raw mean MSE summed (momentum/KE often dominate scale).",
    )
    parser.add_argument(
        "--flux_reg_weight",
        type=float,
        default=0.0,
        help="If > 0, add flux_reg_weight * L2-style penalty on constrained/projected face fluxes "
        "(Option II/III; Option I contributes 0). Encourages smaller transfers for rollout stability.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of samples used for validation.")
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=5,
        help="Save checkpoints only every N epochs (final epoch is always saved).",
    )
    parser.add_argument(
        "--checkpoint_stem",
        type=str,
        default=None,
        help="Filename stem for saved checkpoints (default: checkpoint_option<opt>). "
        "Files are written as <output_dir>/{stem}_epoch{N}.pt. Only the base name is used (no directories).",
    )
    parser.add_argument("--split_seed", type=int, default=None, help="Seed for train/val split (defaults to --seed).")
    parser.add_argument(
        "--learning_curve_path",
        type=str,
        default=None,
        help="Optional CSV path for per-epoch learning curve. Defaults to <output_dir>/learning_curve_option<opt>.csv.",
    )

    # Model hyperparameters.
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_refine_steps", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--soft_transfer", action="store_true", help="Use soft transfer for Option II training")
    parser.add_argument("--hard_eval", action="store_true", help="Force hard integer transfer even in train() (debug).")

    add_device_arg(parser)
    add_common_io_args(parser)

    args = parser.parse_args()
    if args.loss_balance == "auto":
        loss_balance: str = "rms" if args.option in ("ii", "iii") else "none"
    else:
        loss_balance = str(args.loss_balance)

    if args.checkpoint_stem is None:
        checkpoint_stem = f"checkpoint_option{args.option}"
    else:
        checkpoint_stem = Path(str(args.checkpoint_stem).strip()).name
        if not checkpoint_stem or checkpoint_stem in (".", ".."):
            raise ValueError(
                "--checkpoint_stem must be a non-empty base name (directories are ignored; set paths via --output_dir)."
            )

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
    elif args.option == "ii":
        model = OptionIIModel(
            num_species=num_species,
            hidden_channels=args.hidden_channels,
            soft_transfer=args.soft_transfer or (not args.hard_eval),
        )
    else:
        model = OptionIIIModel(
            num_species=num_species,
            hidden_channels=args.hidden_channels,
        )

    model.to(device_cfg.device)
    model.train()

    # Force hard transfer if requested (debug).
    if args.option == "ii" and args.hard_eval:
        model.soft_transfer = False

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    learning_curve_path = (
        args.learning_curve_path
        if args.learning_curve_path is not None
        else os.path.join(out_dir, f"learning_curve_option{args.option}.csv")
    )
    os.makedirs(os.path.dirname(learning_curve_path) or ".", exist_ok=True)
    curve_header = ["epoch", "train_loss", "val_loss"]
    if loss_balance == "rms":
        curve_header += ["train_c", "train_m", "train_ke", "train_o", "val_c", "val_m", "val_ke", "val_o"]
    with open(learning_curve_path, "w", newline="", encoding="utf-8") as f_curve:
        writer = csv.writer(f_curve)
        writer.writerow(curve_header)

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

    flux_reg_w = float(args.flux_reg_weight)
    use_flux_reg = flux_reg_w > 0.0

    for epoch in range(1, args.epochs + 1):
        running_train_loss_sum = 0.0
        running_train_n = 0
        running_tc = running_tm = running_tke = running_to = 0.0
        running_train_flux_reg = 0.0

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

            if use_flux_reg:
                pred, flux_reg = model.predict_next(current, target_state=target, return_flux_reg=True)
            else:
                pred = model.predict_next(current, target_state=target)

            mb_opt: Optional[torch.Tensor] = mb if len(batch) == 3 else None
            loss_counts, loss_momentum, loss_ke, loss_order, loss = compute_state_loss(
                pred, target, mb_opt, loss_balance=loss_balance
            )
            if use_flux_reg:
                loss = loss + flux_reg_w * flux_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(xb.shape[0])
            running_train_loss_sum += float(loss.item()) * bs
            running_train_n += bs
            running_tc += float(loss_counts.item()) * bs
            running_tm += float(loss_momentum.item()) * bs
            running_tke += float(loss_ke.item()) * bs
            running_to += float(loss_order.item()) * bs
            if use_flux_reg:
                running_train_flux_reg += float(flux_reg.item()) * bs

        avg_train = running_train_loss_sum / max(1, running_train_n)
        avg_tc = running_tc / max(1, running_train_n)
        avg_tm = running_tm / max(1, running_train_n)
        avg_tke = running_tke / max(1, running_train_n)
        avg_to = running_to / max(1, running_train_n)
        avg_train_flux_reg = running_train_flux_reg / max(1, running_train_n) if use_flux_reg else 0.0

        avg_val = None
        avg_vc = avg_vm = avg_vke = avg_vo = 0.0
        avg_val_flux_reg = 0.0
        if any(d["dl_val"] is not None for d in per_dataset):
            model.eval()
            running_val_loss_sum = 0.0
            running_val_n = 0
            running_vc = running_vm = running_vke = running_vo = 0.0
            running_val_flux_reg = 0.0
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
                        if use_flux_reg:
                            pred, flux_reg_v = model.predict_next(
                                current, target_state=target, return_flux_reg=True
                            )
                        else:
                            pred = model.predict_next(current, target_state=target)

                        mb_val: Optional[torch.Tensor] = mb if len(batch) == 3 else None
                        loss_counts, loss_momentum, loss_ke, loss_order, loss = compute_state_loss(
                            pred, target, mb_val, loss_balance=loss_balance
                        )
                        if use_flux_reg:
                            loss = loss + flux_reg_w * flux_reg_v
                        bs = int(xb.shape[0])
                        running_val_loss_sum += float(loss.item()) * bs
                        running_val_n += bs
                        running_vc += float(loss_counts.item()) * bs
                        running_vm += float(loss_momentum.item()) * bs
                        running_vke += float(loss_ke.item()) * bs
                        running_vo += float(loss_order.item()) * bs
                        if use_flux_reg:
                            running_val_flux_reg += float(flux_reg_v.item()) * bs

            avg_val = running_val_loss_sum / max(1, running_val_n)
            avg_vc = running_vc / max(1, running_val_n)
            avg_vm = running_vm / max(1, running_val_n)
            avg_vke = running_vke / max(1, running_val_n)
            avg_vo = running_vo / max(1, running_val_n)
            avg_val_flux_reg = running_val_flux_reg / max(1, running_val_n) if use_flux_reg else 0.0
            model.train()

        # Checkpoint cadence: save only every N epochs plus always the final epoch.
        save_every = int(args.save_every_n_epochs)
        should_save = (save_every <= 1) or (epoch % save_every == 0) or (epoch == args.epochs)
        if should_save:
            ckpt_path = os.path.join(out_dir, f"{checkpoint_stem}_epoch{epoch}.pt")
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
                    "loss_balance": loss_balance,
                    "flux_reg_weight": flux_reg_w,
                    "checkpoint_stem": checkpoint_stem,
                },
                ckpt_path,
            )

        flux_msg = ""
        if use_flux_reg:
            flux_msg = f" flux_reg={avg_train_flux_reg:.6g} (w*flux={flux_reg_w * avg_train_flux_reg:.6g})"
            if avg_val is not None:
                flux_msg += f" val_flux_reg={avg_val_flux_reg:.6g}"

        if avg_val is None:
            if loss_balance == "rms":
                print(
                    f"[epoch {epoch}] train_loss={avg_train:.6f} "
                    f"(c={avg_tc:.4f} m={avg_tm:.4f} ke={avg_tke:.4f} o={avg_to:.4f}){flux_msg}",
                    flush=True,
                )
            else:
                print(f"[epoch {epoch}] train_loss={avg_train:.6f}{flux_msg}", flush=True)
        else:
            if loss_balance == "rms":
                print(
                    f"[epoch {epoch}] train_loss={avg_train:.6f} val_loss={avg_val:.6f} "
                    f"train[c,m,ke,o]={avg_tc:.4f},{avg_tm:.4f},{avg_tke:.4f},{avg_to:.4f} "
                    f"val[c,m,ke,o]={avg_vc:.4f},{avg_vm:.4f},{avg_vke:.4f},{avg_vo:.4f}{flux_msg}",
                    flush=True,
                )
            else:
                print(f"[epoch {epoch}] train_loss={avg_train:.6f} val_loss={avg_val:.6f}{flux_msg}", flush=True)
        with open(learning_curve_path, "a", newline="", encoding="utf-8") as f_curve:
            writer = csv.writer(f_curve)
            if loss_balance == "rms":
                if avg_val is None:
                    row = [
                        epoch,
                        f"{avg_train:.10f}",
                        "",
                        f"{avg_tc:.10f}",
                        f"{avg_tm:.10f}",
                        f"{avg_tke:.10f}",
                        f"{avg_to:.10f}",
                        "",
                        "",
                        "",
                        "",
                    ]
                else:
                    row = [
                        epoch,
                        f"{avg_train:.10f}",
                        f"{avg_val:.10f}",
                        f"{avg_tc:.10f}",
                        f"{avg_tm:.10f}",
                        f"{avg_tke:.10f}",
                        f"{avg_to:.10f}",
                        f"{avg_vc:.10f}",
                        f"{avg_vm:.10f}",
                        f"{avg_vke:.10f}",
                        f"{avg_vo:.10f}",
                    ]
            else:
                row = [epoch, f"{avg_train:.10f}", "" if avg_val is None else f"{avg_val:.10f}"]
            writer.writerow(row)


if __name__ == "__main__":
    main()

