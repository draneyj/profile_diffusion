from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..state import CoarseState, normalize_momentum_direction


def _pad_xy_periodic_z_zero(x: torch.Tensor, pad_xy: int = 1, pad_z: int = 1) -> torch.Tensor:
    """
    x: (B,C,D=z,H=y,W=x)
    Pads:
    - x (W) and y (H) with circular padding
    - z (D) with zeros
    """
    x_padded_xy = F.pad(x, (pad_xy, pad_xy, pad_xy, pad_xy, 0, 0), mode="circular")
    x_padded = F.pad(x_padded_xy, (0, 0, 0, 0, pad_z, pad_z), mode="constant", value=0.0)
    return x_padded


class OptionIVModel(nn.Module):
    """
    Option IV: direct next-state predictor using 3x3x3 local neighborhoods.

    Unlike Option I, this predicts the next state in one pass (no iterative diffusion refinement).
    """

    def __init__(
        self,
        *,
        num_species: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.num_species = int(num_species)
        self.num_features = self.num_species + 3 + 1 + 1

        self.net = nn.Sequential(
            nn.Conv3d(self.num_features, hidden_channels, kernel_size=3, padding=0),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=0),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, self.num_features, kernel_size=1, padding=0),
        )

    def _features_to_conv3d(self, features: torch.Tensor) -> torch.Tensor:
        return features.permute(0, 1, 4, 3, 2)  # (B,C,nz,ny,nx)

    def _conv3d_to_features(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 4, 3, 2)  # (B,C,nx,ny,nz)

    @torch.no_grad()
    def _postprocess_eval(self, state: CoarseState) -> CoarseState:
        state.counts = torch.clamp(state.counts, min=0.0).round()
        state.ke = torch.clamp(state.ke, min=0.0)
        state.order = torch.clamp(state.order, min=-1.0, max=1.0)
        state.momentum = normalize_momentum_direction(state.momentum)
        return state

    def predict_next(
        self,
        current_state: CoarseState,
        *,
        target_state: Optional[CoarseState] = None,
        return_flux_reg: bool = False,
    ) -> Union[CoarseState, Tuple[CoarseState, torch.Tensor]]:
        # target_state is accepted for API compatibility with train/infer code.
        features = current_state.as_features()
        if features.dim() == 4:
            features = features.unsqueeze(0)

        x = self._features_to_conv3d(features)
        # Two 3x3x3 valid convolutions shrink by 4, so pad by 2 to preserve shape.
        x = _pad_xy_periodic_z_zero(x, pad_xy=2, pad_z=2)
        y = self.net(x)
        pred_features = self._conv3d_to_features(y)

        pred = CoarseState.from_features(pred_features, num_species=self.num_species)
        pred = CoarseState(
            counts=torch.clamp(pred.counts, min=0.0),
            momentum=normalize_momentum_direction(pred.momentum),
            ke=torch.clamp(pred.ke, min=0.0),
            order=torch.clamp(pred.order, min=-1.0, max=1.0),
        )

        if not self.training:
            pred = self._postprocess_eval(pred)

        if return_flux_reg:
            z = torch.zeros((), device=pred.counts.device, dtype=pred.counts.dtype)
            return pred, z
        return pred
