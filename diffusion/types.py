from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


TensorShape = Tuple[int, ...]


@dataclass
class CoarseStateTensors:
    """
    Coarse-grained representation stored as tensors.

    Shapes (for a single simulation/grid):
    - counts:  (S, nx, ny, nz)           float32 or int-like
    - momentum: (3, nx, ny, nz) unit direction of cell momentum (or zero)
    - ke:       (1, nx, ny, nz)
    - order:    (1, nx, ny, nz)
    """

    counts: torch.Tensor
    momentum: torch.Tensor
    ke: torch.Tensor
    order: torch.Tensor

    def to(self, device: torch.device) -> "CoarseStateTensors":
        self.counts = self.counts.to(device)
        self.momentum = self.momentum.to(device)
        self.ke = self.ke.to(device)
        self.order = self.order.to(device)
        return self

    @property
    def num_species(self) -> int:
        return int(self.counts.shape[0])

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        nx, ny, nz = self.counts.shape[1:]
        return int(nx), int(ny), int(nz)


def stack_state_features(state: CoarseStateTensors) -> torch.Tensor:
    """
    Stack state tensors into a single feature tensor.

    Output shape: (C, nx, ny, nz) where
    C = S (counts) + 3 (momentum) + 1 (ke) + 1 (order)
    """

    return torch.cat([state.counts, state.momentum, state.ke, state.order], dim=0)


def unstack_state_features(
    features: torch.Tensor, *, num_species: int
) -> CoarseStateTensors:
    """
    Inverse of `stack_state_features`.

    features shape: (S + 3 + 1 + 1, nx, ny, nz)
    """

    s = num_species
    if features.dim() != 4:
        raise ValueError(f"Expected features dim=4, got {features.dim()}")
    counts = features[0:s]
    momentum = features[s : s + 3]
    ke = features[s + 3 : s + 4]
    order = features[s + 4 : s + 5]
    return CoarseStateTensors(counts=counts, momentum=momentum, ke=ke, order=order)


def ensure_batch_grid(features: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is (B, C, nx, ny, nz).

    Accepts either (C, nx, ny, nz) or (B, C, nx, ny, nz).
    """
    if features.dim() == 4:
        return features.unsqueeze(0)
    if features.dim() != 5:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(features.shape)}")
    return features

