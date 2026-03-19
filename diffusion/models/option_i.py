from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..state import CoarseState


def _pad_xy_periodic_z_zero(x: torch.Tensor, pad_xy: int = 1, pad_z: int = 1) -> torch.Tensor:
    """
    x: (B,C,D=z,H=y,W=x)
    Pads:
    - x (W) and y (H) with circular padding
    - z (D) with zeros
    """

    # Pad x and y circularly first.
    x_padded_xy = F.pad(x, (pad_xy, pad_xy, pad_xy, pad_xy, 0, 0), mode="circular")
    # Then pad z with zeros.
    x_padded = F.pad(x_padded_xy, (0, 0, 0, 0, pad_z, pad_z), mode="constant", value=0.0)
    return x_padded


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=0)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class OptionIModel(nn.Module):
    """
    Option I: diffusion-style CNN that refines a noisy next-state estimate.

    Unified API:
    - `predict_next(current_state, target_state=None)`:
        * if `target_state` is provided (training): starts from target + noise and refines
        * else (inference): starts from current_state and refines
    """

    def __init__(
        self,
        *,
        num_species: int,
        hidden_channels: int = 64,
        num_refine_steps: int = 1,
        noise_std: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_species = int(num_species)
        self.num_refine_steps = int(num_refine_steps)
        self.noise_std = float(noise_std)
        self.eps = float(eps)

        self.num_features = self.num_species + 3 + 1 + 1  # S + momentum(3) + ke(1) + order(1)
        in_ch = 2 * self.num_features  # concat(cond, noisy_estimate)

        # Number of spatial-shrinking Conv3d(kernel=3,padding=0) blocks.
        # Each Conv3d shrinks spatial dims by 2, so total shrink is 2 * num_blocks.
        self.num_shrinking_conv_blocks = 3

        self.net = nn.Sequential(
            ConvBlock(in_ch, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv3d(hidden_channels, self.num_features, kernel_size=1, padding=0),
        )

    def _features_to_conv3d(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B,C,nx,ny,nz) -> (B,C,nz,ny,nx)
        """

        return features.permute(0, 1, 4, 3, 2)

    def _conv3d_to_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,nz,ny,nx) -> (B,C,nx,ny,nz)
        """

        return x.permute(0, 1, 4, 3, 2)

    def denoise_step(self, cond_features: torch.Tensor, x_est: torch.Tensor) -> torch.Tensor:
        """
        cond_features, x_est: (B,C,nx,ny,nz)
        returns x_next: (B,C,nx,ny,nz)
        """

        # Periodic x/y, zero z padding before each conv.
        x = torch.cat([cond_features, x_est], dim=1)  # (B,2C,nx,ny,nz)
        x = self._features_to_conv3d(x)  # (B,2C,nz,ny,nx)

        # We stack 3 Conv3d(kernel=3,padding=0) blocks, so we must pad enough
        # to compensate for the total shrink (6 voxels).
        x = _pad_xy_periodic_z_zero(
            x,
            pad_xy=self.num_shrinking_conv_blocks,
            pad_z=self.num_shrinking_conv_blocks,
        )
        # Since we manually padded, conv3d kernels are padding=0.
        delta = self.net(x)  # (B,C,nz,ny,nx)
        x_next = x_est + self._conv3d_to_features(delta)
        return x_next

    @torch.no_grad()
    def _postprocess_eval(self, state: CoarseState) -> CoarseState:
        # In eval: counts should be non-negative and (approximately) integer.
        if state.counts.dim() == 5:
            state.counts = torch.clamp(state.counts, min=0.0).round()
        else:
            state.counts = torch.clamp(state.counts, min=0.0).round()
        # Keep KE non-negative; order is bounded by the cosine combination.
        state.ke = torch.clamp(state.ke, min=0.0)
        state.order = torch.clamp(state.order, min=-1.0, max=1.0)
        return state

    def predict_next(
        self,
        current_state: CoarseState,
        *,
        target_state: Optional[CoarseState] = None,
    ) -> CoarseState:
        """
        Returns predicted next state.
        """

        cond_features = current_state.as_features()  # (B,C,nx,ny,nz) or (C,nx,ny,nz)
        if cond_features.dim() == 4:
            cond_features = cond_features.unsqueeze(0)

        if target_state is not None:
            x_target = target_state.as_features()
            if x_target.dim() == 4:
                x_target = x_target.unsqueeze(0)
            noise = torch.randn_like(x_target) * self.noise_std
            x_est = x_target + noise
        else:
            # Inference start: use current state as a crude initialization.
            x_est = cond_features.clone()

        for _ in range(self.num_refine_steps):
            x_est = self.denoise_step(cond_features, x_est)

        # Convert features -> state.
        pred = CoarseState.from_features(x_est, num_species=self.num_species)

        if not self.training:
            pred = self._postprocess_eval(pred)
        return pred

