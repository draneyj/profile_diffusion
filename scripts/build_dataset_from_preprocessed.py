#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT_DIR))

from diffusion.state import CoarseState
from diffusion.types import stack_state_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training dataset tensors from preprocessed frames.")
    parser.add_argument("--preprocessed_dir", type=str, default="data/preprocessed")
    parser.add_argument("--out_path", type=str, default="data/preprocessed/dataset.pt")
    parser.add_argument("--stride_k", type=int, default=1)
    parser.add_argument("--max_pairs", type=int, default=None)
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir).resolve()
    manifest_path = preprocessed_dir / "manifest.json"
    frames_dir = preprocessed_dir / "frames"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest["frames"]
    if not frames:
        raise ValueError("manifest.json has no frames")

    # Determine num_species from first frame.
    first_path = Path(frames[0]["frame_path"])
    first = torch.load(str(first_path), map_location="cpu")
    num_species = int(first["counts"].shape[0])

    # Pair only within each dump file's internal frame index order.
    by_dump: Dict[int, List[dict]] = {}
    for fr in frames:
        dump_stem = int(fr["dump_stem"])
        by_dump.setdefault(dump_stem, []).append(fr)

    for dump_stem in by_dump:
        by_dump[dump_stem].sort(key=lambda x: int(x.get("frame_index", 0)))

    inputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    # Build pairs across all dump files, but never across dump boundaries.
    for dump_stem, frs in by_dump.items():
        if len(frs) <= args.stride_k:
            continue
        for i in range(0, len(frs) - args.stride_k):
            if args.max_pairs is not None and len(inputs) >= args.max_pairs:
                break

            f_in_path = Path(frs[i]["frame_path"])
            f_tgt_path = Path(frs[i + args.stride_k]["frame_path"])

            in_payload = torch.load(str(f_in_path), map_location="cpu")
            tgt_payload = torch.load(str(f_tgt_path), map_location="cpu")

            in_state = CoarseState(
                counts=in_payload["counts"],
                momentum=in_payload["momentum"],
                ke=in_payload["ke"],
                order=in_payload["order"],
            )
            tgt_state = CoarseState(
                counts=tgt_payload["counts"],
                momentum=tgt_payload["momentum"],
                ke=tgt_payload["ke"],
                order=tgt_payload["order"],
            )

            inputs.append(in_state.as_features())
            targets.append(tgt_state.as_features())
        if args.max_pairs is not None and len(inputs) >= args.max_pairs:
            break

    if not inputs:
        raise ValueError(
            f"No training pairs produced from preprocessed frames with stride_k={args.stride_k}. "
            f"Check your manifest and ensure each dump has enough frames."
        )

    inputs_t = torch.stack(inputs, dim=0)  # (N,C,nx,ny,nz)
    targets_t = torch.stack(targets, dim=0)

    payload = {
        "inputs": inputs_t,
        "targets": targets_t,
        "metadata": {
            **{k: v for k, v in manifest.items() if k not in ["frames"]},
            "stride_k": args.stride_k,
            "num_pairs": inputs_t.shape[0],
            "num_species": num_species,
        },
    }

    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    print(f"Wrote dataset: {out_path}")


if __name__ == "__main__":
    main()

