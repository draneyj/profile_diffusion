from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..state import CoarseState
from .option_ii import (
    FACE_MINUS_X,
    FACE_MINUS_Y,
    FACE_MINUS_Z,
    FACE_PLUS_X,
    FACE_PLUS_Y,
    FACE_PLUS_Z,
    shift_src_to_dst,
)


class OptionIIIModel(nn.Module):
    """
    Option III: constrained hybrid flux model.

    Constrained parameterization:
    - Species outflow fraction per cell: sigmoid -> [0,1]
    - Face split per species: softmax over 6 faces
    - KE outflow fraction per cell: sigmoid -> [0,1]
    - KE face split: softmax over 6 faces
    - Momentum face fluxes: signed (tanh-bounded)

    Projection/correction:
    - Per-cell scaling alpha in [0,1] to prevent negative species counts and KE.
    """

    def __init__(
        self,
        *,
        num_species: int,
        hidden_channels: int = 64,
        eps: float = 1e-8,
        momentum_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_species = int(num_species)
        self.hidden_channels = int(hidden_channels)
        self.eps = float(eps)
        self.momentum_scale = float(momentum_scale)

        # Eval behavior: keep soft transfers but optionally round counts.
        self.hard_round_counts_eval: bool = True

        self.num_features = self.num_species + 3 + 1 + 1

        # Head layout:
        # [atom_out_frac(S), atom_face_logits(6*S), ke_out_frac(1), ke_face_logits(6), mom_face_raw(6*3)]
        out_ch = self.num_species + (6 * self.num_species) + 1 + 6 + 18
        self.head = nn.Sequential(
            nn.Conv3d(self.num_features, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, out_ch, kernel_size=1),
        )
        self.order_head = nn.Sequential(
            nn.Conv3d(self.num_features, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def _predict_raw_fluxes(self, state: CoarseState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - atom_out_face: (B,6,S,nx,ny,nz), nonnegative
        - ke_out_face:   (B,6,nx,ny,nz), nonnegative
        - mom_face:      (B,6,3,nx,ny,nz), signed
        """
        counts = state.counts
        momentum = state.momentum
        ke = state.ke
        features = state.as_features()
        if features.dim() == 4:
            features = features.unsqueeze(0)
            counts = counts.unsqueeze(0)
            momentum = momentum.unsqueeze(0)
            ke = ke.unsqueeze(0)

        # Conv3d convention (B,C,nz,ny,nx).
        x = features.permute(0, 1, 4, 3, 2)
        y = self.head(x).permute(0, 1, 4, 3, 2)  # (B,C,nx,ny,nz)

        s = self.num_species
        idx = 0

        atom_out_frac = torch.sigmoid(y[:, idx : idx + s])  # (B,S,nx,ny,nz)
        idx += s

        atom_face_logits = y[:, idx : idx + 6 * s].reshape(-1, 6, s, *y.shape[-3:])
        idx += 6 * s
        atom_face_prob = torch.softmax(atom_face_logits, dim=1)  # (B,6,S,nx,ny,nz)

        ke_out_frac = torch.sigmoid(y[:, idx : idx + 1])  # (B,1,nx,ny,nz)
        idx += 1

        ke_face_logits = y[:, idx : idx + 6].reshape(-1, 6, *y.shape[-3:])
        idx += 6
        ke_face_prob = torch.softmax(ke_face_logits, dim=1)  # (B,6,nx,ny,nz)

        mom_face_raw = y[:, idx : idx + 18].reshape(-1, 6, 3, *y.shape[-3:])

        # Total out per species and KE.
        atom_out_total = counts * atom_out_frac  # (B,S,nx,ny,nz)
        atom_out_face = atom_out_total.unsqueeze(1) * atom_face_prob  # (B,6,S,nx,ny,nz)

        ke_out_total = ke[:, 0] * ke_out_frac[:, 0]  # (B,nx,ny,nz)
        ke_out_face = ke_out_total.unsqueeze(1) * ke_face_prob  # (B,6,nx,ny,nz)

        # Signed momentum face flux, scaled by local momentum magnitude.
        p_mag = torch.sqrt(torch.sum(momentum * momentum, dim=1, keepdim=True) + self.eps)  # (B,1,nx,ny,nz)
        mom_scale = self.momentum_scale * p_mag.unsqueeze(1)  # (B,1,1,nx,ny,nz)
        mom_face = torch.tanh(mom_face_raw) * mom_scale

        return atom_out_face, ke_out_face, mom_face

    @staticmethod
    def flux_regularization_projected(
        atom_out_face: torch.Tensor,
        ke_out_face: torch.Tensor,
        mom_face: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean squared projected face fluxes (atoms, KE, momentum) — encourages smaller transfers.
        """
        return (
            (atom_out_face * atom_out_face).mean()
            + (ke_out_face * ke_out_face).mean()
            + (mom_face * mom_face).mean()
        ) / 3.0

    def _project_fluxes(
        self,
        state: CoarseState,
        atom_out_face: torch.Tensor,
        ke_out_face: torch.Tensor,
        mom_face: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scale outgoing fluxes per cell by alpha in [0,1] so counts/KE stay nonnegative.
        """
        counts = state.counts
        ke = state.ke[:, 0] if state.ke.dim() == 5 else state.ke[0]
        if counts.dim() == 4:
            counts = counts.unsqueeze(0)
            ke = ke.unsqueeze(0)

        atom_out_sum = atom_out_face.sum(dim=1)  # (B,S,nx,ny,nz)
        scale_atoms = torch.ones_like(atom_out_sum)
        mask = atom_out_sum > self.eps
        scale_atoms[mask] = counts[mask] / atom_out_sum[mask]
        scale_atoms = scale_atoms.min(dim=1).values  # (B,nx,ny,nz)

        ke_out_sum = ke_out_face.sum(dim=1)  # (B,nx,ny,nz)
        scale_ke = torch.ones_like(ke_out_sum)
        mask_ke = ke_out_sum > self.eps
        scale_ke[mask_ke] = ke[mask_ke] / ke_out_sum[mask_ke]

        alpha = torch.clamp(torch.minimum(scale_atoms, scale_ke), min=0.0, max=1.0)  # (B,nx,ny,nz)
        alpha_face = alpha.unsqueeze(1)

        atom_out_face = atom_out_face * alpha_face.unsqueeze(2)
        ke_out_face = ke_out_face * alpha_face
        mom_face = mom_face * alpha_face.unsqueeze(2)
        return atom_out_face, ke_out_face, mom_face

    def _apply_fluxes(
        self,
        state: CoarseState,
        atom_out_face: torch.Tensor,
        ke_out_face: torch.Tensor,
        mom_face: torch.Tensor,
    ) -> CoarseState:
        counts = state.counts
        momentum = state.momentum
        ke = state.ke
        order = state.order

        if counts.dim() == 4:
            counts = counts.unsqueeze(0)
            momentum = momentum.unsqueeze(0)
            ke = ke.unsqueeze(0)
            order = order.unsqueeze(0)

        # Outgoing totals
        atom_out_sum = atom_out_face.sum(dim=1)  # (B,S,nx,ny,nz)
        ke_out_sum = ke_out_face.sum(dim=1, keepdim=True)  # (B,1,nx,ny,nz)
        mom_out_sum = mom_face.sum(dim=1)  # (B,3,nx,ny,nz)

        # Incoming from neighbors
        atom_in = torch.zeros_like(counts)
        ke_in = torch.zeros_like(ke)
        mom_in = torch.zeros_like(momentum)
        for face in (FACE_PLUS_X, FACE_MINUS_X, FACE_PLUS_Y, FACE_MINUS_Y, FACE_PLUS_Z, FACE_MINUS_Z):
            atom_in = atom_in + shift_src_to_dst(atom_out_face[:, face], face=face)
            ke_in = ke_in + shift_src_to_dst(ke_out_face[:, face].unsqueeze(1), face=face)
            mom_in = mom_in + shift_src_to_dst(mom_face[:, face], face=face)

        counts_next = torch.clamp(counts - atom_out_sum + atom_in, min=0.0)
        ke_next = torch.clamp(ke - ke_out_sum + ke_in, min=0.0)
        momentum_next = momentum - mom_out_sum + mom_in

        # Predict order delta from next-state features.
        next_tmp = CoarseState(counts=counts_next, momentum=momentum_next, ke=ke_next, order=order)
        feat = next_tmp.as_features().permute(0, 1, 4, 3, 2)  # (B,C,nz,ny,nx)
        order_delta = self.order_head(feat).permute(0, 1, 4, 3, 2)
        order_next = torch.clamp(order + order_delta, min=-1.0, max=1.0)

        return CoarseState(counts=counts_next, momentum=momentum_next, ke=ke_next, order=order_next)

    def predict_next(
        self,
        current_state: CoarseState,
        *,
        target_state: Optional[CoarseState] = None,
        return_flux_reg: bool = False,
    ) -> Union[CoarseState, Tuple[CoarseState, torch.Tensor]]:
        input_batched = current_state.counts.dim() == 5
        if not input_batched:
            current_state = CoarseState(
                counts=current_state.counts.unsqueeze(0),
                momentum=current_state.momentum.unsqueeze(0),
                ke=current_state.ke.unsqueeze(0),
                order=current_state.order.unsqueeze(0),
            )

        atom_out_face, ke_out_face, mom_face = self._predict_raw_fluxes(current_state)
        atom_out_face, ke_out_face, mom_face = self._project_fluxes(current_state, atom_out_face, ke_out_face, mom_face)
        flux_reg: Optional[torch.Tensor] = None
        if return_flux_reg:
            flux_reg = self.flux_regularization_projected(atom_out_face, ke_out_face, mom_face)
        next_state = self._apply_fluxes(current_state, atom_out_face, ke_out_face, mom_face)

        if (not self.training) and self.hard_round_counts_eval:
            next_state.counts = torch.clamp(next_state.counts, min=0.0).round()
            next_state.ke = torch.clamp(next_state.ke, min=0.0)

        if not input_batched:
            next_state = CoarseState(
                counts=next_state.counts[0],
                momentum=next_state.momentum[0],
                ke=next_state.ke[0],
                order=next_state.order[0],
            )

        if return_flux_reg:
            assert flux_reg is not None
            return next_state, flux_reg
        return next_state

