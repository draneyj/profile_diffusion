from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for PyTorch (e.g. cpu, cuda, mps).",
    )


def parse_device(args: argparse.Namespace) -> DeviceConfig:
    requested = str(getattr(args, "device", "cpu"))
    if requested == "cuda" and torch.cuda.is_available():
        return DeviceConfig(device=torch.device("cuda"))
    if requested == "mps" and torch.backends.mps.is_available():
        return DeviceConfig(device=torch.device("mps"))
    return DeviceConfig(device=torch.device("cpu"))


def add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="data/processed")


def seed_all(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

