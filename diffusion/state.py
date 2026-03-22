from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .types import CoarseStateTensors, stack_state_features, unstack_state_features


def normalize_momentum_direction(momentum: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Per spatial cell, replace momentum with a unit vector in its direction, or zeros if ||p|| < eps.

    Expected shapes:
    - (3, nx, ny, nz) unbatched
    - (B, 3, nx, ny, nz) batched

    Coarse-grained datasets store **momentum as this unit direction** (magnitude lives in KE / counts).
    """

    if momentum.dim() == 4:
        mag_sq = (momentum * momentum).sum(dim=0, keepdim=True)
    elif momentum.dim() == 5:
        mag_sq = (momentum * momentum).sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"normalize_momentum_direction: expected (3,...) or (B,3,...), got {tuple(momentum.shape)}")
    mag = torch.sqrt(mag_sq.clamp(min=eps * eps))
    unit = momentum / mag
    return torch.where(mag_sq > (eps * eps), unit, torch.zeros_like(momentum))


@dataclass
class CoarseState:
    """
    Unified coarse-grained state container used by both model options.

    Tensor shapes for a single grid:
    - counts:  (S, nx, ny, nz)
    - momentum: (3, nx, ny, nz) — **unit direction** of total cell momentum (or zero if negligible)
    - ke:       (1, nx, ny, nz)
    - order:    (1, nx, ny, nz)
    """

    counts: torch.Tensor
    momentum: torch.Tensor
    ke: torch.Tensor
    order: torch.Tensor

    @property
    def num_species(self) -> int:
        # Unbatched: (S, nx, ny, nz)
        # Batched:   (B, S, nx, ny, nz)
        if self.counts.dim() == 4:
            return int(self.counts.shape[0])
        if self.counts.dim() == 5:
            return int(self.counts.shape[1])
        raise ValueError(f"Unexpected counts tensor shape: {tuple(self.counts.shape)}")

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        if self.counts.dim() == 4:
            nx, ny, nz = self.counts.shape[1:]
            return int(nx), int(ny), int(nz)
        if self.counts.dim() == 5:
            nx, ny, nz = self.counts.shape[2:]
            return int(nx), int(ny), int(nz)
        raise ValueError(f"Unexpected counts tensor shape: {tuple(self.counts.shape)}")

    def to(self, device: torch.device) -> "CoarseState":
        self.counts = self.counts.to(device)
        self.momentum = self.momentum.to(device)
        self.ke = self.ke.to(device)
        self.order = self.order.to(device)
        return self

    def clone(self) -> "CoarseState":
        return CoarseState(
            counts=self.counts.clone(),
            momentum=self.momentum.clone(),
            ke=self.ke.clone(),
            order=self.order.clone(),
        )

    def detach(self) -> "CoarseState":
        return CoarseState(
            counts=self.counts.detach(),
            momentum=self.momentum.detach(),
            ke=self.ke.detach(),
            order=self.order.detach(),
        )

    def as_tensors(self) -> CoarseStateTensors:
        # CoarseStateTensors is unbatched by design.
        if self.counts.dim() != 4:
            raise ValueError("as_tensors() requires unbatched state tensors")
        return CoarseStateTensors(counts=self.counts, momentum=self.momentum, ke=self.ke, order=self.order)

    def as_features(self) -> torch.Tensor:
        """
        Return model features: (C, nx, ny, nz) with
        C = S + 3 + 1 + 1.
        """

        if self.counts.dim() == 4:
            return stack_state_features(self.as_tensors())  # (C,nx,ny,nz)
        if self.counts.dim() == 5:
            # Batched: concatenate channels along dim=1
            return torch.cat([self.counts, self.momentum, self.ke, self.order], dim=1)  # (B,C,nx,ny,nz)
        raise ValueError(f"Unexpected counts tensor shape: {tuple(self.counts.shape)}")

    @classmethod
    def from_features(cls, features: torch.Tensor, *, num_species: int) -> "CoarseState":
        """
        features shape: (S+3+1+1, nx, ny, nz)
        """
        if features.dim() == 4:
            t = unstack_state_features(features, num_species=num_species)
            return cls(counts=t.counts, momentum=t.momentum, ke=t.ke, order=t.order)
        if features.dim() == 5:
            # (B,C,nx,ny,nz)
            s = num_species
            counts = features[:, 0:s]
            momentum = features[:, s : s + 3]
            ke = features[:, s + 3 : s + 4]
            order = features[:, s + 4 : s + 5]
            return cls(counts=counts, momentum=momentum, ke=ke, order=order)
        raise ValueError(f"Unexpected features tensor shape: {tuple(features.shape)}")

    @classmethod
    def from_tensors(cls, t: CoarseStateTensors) -> "CoarseState":
        return cls(counts=t.counts, momentum=t.momentum, ke=t.ke, order=t.order)

