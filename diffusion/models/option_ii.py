from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..state import CoarseState


FACE_PLUS_X = 0
FACE_MINUS_X = 1
FACE_PLUS_Y = 2
FACE_MINUS_Y = 3
FACE_PLUS_Z = 4
FACE_MINUS_Z = 5

FACE_NAMES = {
    FACE_PLUS_X: "+x",
    FACE_MINUS_X: "-x",
    FACE_PLUS_Y: "+y",
    FACE_MINUS_Y: "-y",
    FACE_PLUS_Z: "+z",
    FACE_MINUS_Z: "-z",
}


def rotate_vec_to_face_normal(vec: torch.Tensor, face: int) -> torch.Tensor:
    """
    Rotate a vector into a face-normal aligned coordinate frame.

    vec shape: (B,3,nx,ny,nz) or (B,S,3,nx,ny,nz)
    Returns same shape.
    """

    if vec.dim() == 5:
        # (B,3,nx,ny,nz)
        vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
        if face == FACE_PLUS_X:
            return torch.stack([vx, vy, vz], dim=1)
        if face == FACE_MINUS_X:
            return torch.stack([-vx, vy, vz], dim=1)
        if face == FACE_PLUS_Y:
            return torch.stack([vy, vx, vz], dim=1)
        if face == FACE_MINUS_Y:
            return torch.stack([-vy, vx, vz], dim=1)
        if face == FACE_PLUS_Z:
            return torch.stack([vz, vx, vy], dim=1)
        if face == FACE_MINUS_Z:
            return torch.stack([-vz, vx, vy], dim=1)
        raise ValueError(f"Unknown face {face}")

    if vec.dim() == 6:
        # (B,S,3,nx,ny,nz)
        vx, vy, vz = vec[:, :, 0], vec[:, :, 1], vec[:, :, 2]
        if face == FACE_PLUS_X:
            return torch.stack([vx, vy, vz], dim=2)
        if face == FACE_MINUS_X:
            return torch.stack([-vx, vy, vz], dim=2)
        if face == FACE_PLUS_Y:
            return torch.stack([vy, vx, vz], dim=2)
        if face == FACE_MINUS_Y:
            return torch.stack([-vy, vx, vz], dim=2)
        if face == FACE_PLUS_Z:
            return torch.stack([vz, vx, vy], dim=2)
        if face == FACE_MINUS_Z:
            return torch.stack([-vz, vx, vy], dim=2)
        raise ValueError(f"Unknown face {face}")

    raise ValueError(f"Unsupported vec dim={vec.dim()}")


def rotate_vec_from_face_normal(rot: torch.Tensor, face: int) -> torch.Tensor:
    """
    Inverse transform for `rotate_vec_to_face_normal`.
    """

    if rot.dim() == 5:
        # (B,3,nx,ny,nz) where rot[0] = normal component (with sign)
        r0, r1, r2 = rot[:, 0], rot[:, 1], rot[:, 2]
        if face == FACE_PLUS_X:
            return torch.stack([r0, r1, r2], dim=1)
        if face == FACE_MINUS_X:
            return torch.stack([-r0, r1, r2], dim=1)
        if face == FACE_PLUS_Y:
            # rot = [m_y, m_x, m_z]
            return torch.stack([r1, r0, r2], dim=1)
        if face == FACE_MINUS_Y:
            # rot = [-m_y, m_x, m_z]
            return torch.stack([r1, -r0, r2], dim=1)
        if face == FACE_PLUS_Z:
            # rot = [m_z, m_x, m_y]
            return torch.stack([r1, r2, r0], dim=1)
        if face == FACE_MINUS_Z:
            # rot = [-m_z, m_x, m_y]
            return torch.stack([r1, r2, -r0], dim=1)
        raise ValueError(f"Unknown face {face}")

    if rot.dim() == 6:
        r0, r1, r2 = rot[:, :, 0], rot[:, :, 1], rot[:, :, 2]
        if face == FACE_PLUS_X:
            return torch.stack([r0, r1, r2], dim=2)
        if face == FACE_MINUS_X:
            return torch.stack([-r0, r1, r2], dim=2)
        if face == FACE_PLUS_Y:
            return torch.stack([r1, r0, r2], dim=2)
        if face == FACE_MINUS_Y:
            return torch.stack([r1, -r0, r2], dim=2)
        if face == FACE_PLUS_Z:
            return torch.stack([r1, r2, r0], dim=2)
        if face == FACE_MINUS_Z:
            return torch.stack([r1, r2, -r0], dim=2)
        raise ValueError(f"Unknown face {face}")

    raise ValueError(f"Unsupported rot dim={rot.dim()}")


def shift_src_to_dst(t: torch.Tensor, *, face: int) -> torch.Tensor:
    """
    Shift a (B,*,nx,ny,nz) tensor from source cell coordinates to
    destination cell coordinates for the directed face.

    x and y are periodic; z is non-periodic.
    """

    # Assume the grid dims are the last 3 dims.
    if t.dim() < 4:
        raise ValueError("Expected tensor with at least 4 dims (B,...,nx,ny,nz)")

    # Identify grid dims: (..., nx, ny, nz) are last 3 dims.
    nx_dim = -3
    ny_dim = -2
    nz_dim = -1

    if face == FACE_PLUS_X:
        return torch.roll(t, shifts=1, dims=nx_dim)
    if face == FACE_MINUS_X:
        return torch.roll(t, shifts=-1, dims=nx_dim)
    if face == FACE_PLUS_Y:
        return torch.roll(t, shifts=1, dims=ny_dim)
    if face == FACE_MINUS_Y:
        return torch.roll(t, shifts=-1, dims=ny_dim)
    if face == FACE_PLUS_Z:
        out = torch.zeros_like(t)
        out[..., 1:] = t[..., :-1]
        return out
    if face == FACE_MINUS_Z:
        out = torch.zeros_like(t)
        out[..., :-1] = t[..., 1:]
        return out
    raise ValueError(f"Unknown face {face}")


def shift_dst_to_src(t: torch.Tensor, *, face: int) -> torch.Tensor:
    """
    Inverse shift (destination view -> source aligned tensor) for gathering neighbor features.
    For our implementation it is simpler to directly build neighbors with roll/slicing; this helper
    is not used currently.
    """

    raise NotImplementedError


class OptionIIModel(nn.Module):
    """
    Option II: face-wise flux prediction + constrained transfer.

    - Uses periodicity in x/y.
    - Non-periodic in z (no wrap; faces out of domain are effectively ignored).

    Training:
    - uses differentiable soft transfer by default.
    Inference:
    - uses hard integer transfers for atom fluxes.
    """

    def __init__(
        self,
        *,
        num_species: int,
        hidden_channels: int = 64,
        soft_transfer: bool = True,
        num_faces: int = 6,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_species = int(num_species)
        self.hidden_channels = int(hidden_channels)
        self.soft_transfer = bool(soft_transfer)
        self.num_faces = int(num_faces)
        self.eps = float(eps)
        # If True, `predict_next()` will use soft transfer even when the module is in eval()
        # mode (i.e. self.training == False). This is meant for large-grid inference where the
        # exact integer transfer is too expensive.
        self.force_soft_transfer_eval: bool = False
        # If True, after soft transfer (including forced-eval soft), clamp/round counts to
        # keep outputs physically plausible as cell occupancies.
        self.soft_round_outputs: bool = True

        # Per-cell state feature channels.
        self.num_features = self.num_species + 3 + 1 + 1  # S + momentum(3) + ke(1) + order(1)

        # Face input concatenates src and dst cell features with rotated momentum components.
        in_ch_face = 2 * self.num_features
        out_ch_face = 4 + 5 * self.num_species  # force_mom(3)+force_ke(1)+ atom_flux(S)+ material_mom(S*3)+ material_ke(S)

        self.face_mlp = nn.Sequential(
            nn.Conv3d(in_ch_face, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, out_ch_face, kernel_size=1),
        )

        # Separate head for predicting order at next timestep (flux transfer doesn't define it directly).
        self.order_head = nn.Sequential(
            nn.Conv3d(self.num_features, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def _build_face_neighbor_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Gather destination-cell features for each directed face.

        features: (B,C,nx,ny,nz)
        Returns 6 tensors each of shape (B,C,nx,ny,nz) giving destination features
        for face in FACE order [+x,-x,+y,-y,+z,-z].
        """

        # x dim is -3, y dim is -2, z dim is -1 in (B,C,nx,ny,nz)
        x_dim, y_dim, z_dim = -3, -2, -1

        dst_px = torch.roll(features, shifts=-1, dims=x_dim)  # dst at (i+1) -> for source at i
        dst_mx = torch.roll(features, shifts=1, dims=x_dim)
        dst_py = torch.roll(features, shifts=-1, dims=y_dim)
        dst_my = torch.roll(features, shifts=1, dims=y_dim)

        # z is non-periodic: for +z face, destination for source at z=k is at z=k+1.
        dst_pz = torch.zeros_like(features)
        if features.shape[z_dim] >= 2:
            dst_pz[..., :-1] = features[..., 1:]
            # dst_pz at z=nz-1 stays 0

        dst_mz = torch.zeros_like(features)
        if features.shape[z_dim] >= 2:
            dst_mz[..., 1:] = features[..., :-1]
            # dst_mz at z=0 stays 0

        return dst_px, dst_mx, dst_py, dst_my, dst_pz, dst_mz

    def _rotate_state_momentum_for_face(self, momentum: torch.Tensor, face: int) -> torch.Tensor:
        return rotate_vec_to_face_normal(momentum, face)

    def _extract_state_components(self, state: CoarseState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return state.counts, state.momentum, state.ke, state.order

    def _predict_fluxes(self, state: CoarseState) -> dict:
        """
        Predict directed fluxes for each face from current state.

        Returns dict with tensors:
        - atom_flux: (B,F,S,nx,ny,nz) >= 0
        - material_momentum_flux: (B,F,S,3,nx,ny,nz) in face-rotated coordinates
        - material_ke_flux: (B,F,S,nx,ny,nz) >=0
        - force_momentum_flux: (B,F,3,nx,ny,nz) in face-rotated coordinates
        - force_ke_flux: (B,F,nx,ny,nz) >=0
        """

        counts, momentum, ke, order = self._extract_state_components(state)
        # Assemble features channels in global coordinates (order doesn't rotate).
        features = state.as_features()  # (B,C,nx,ny,nz)
        if features.dim() == 4:
            features = features.unsqueeze(0)

        # Gather destination features (for counts/ke/order and for momentum rotation too).
        dst_px, dst_mx, dst_py, dst_my, dst_pz, dst_mz = self._build_face_neighbor_features(features)

        dst_features_by_face = [dst_px, dst_mx, dst_py, dst_my, dst_pz, dst_mz]

        # Split src features into components for easier rotation.
        s = self.num_species
        src_counts = features[:, 0:s]
        src_mom = features[:, s : s + 3]  # (B,3,nx,ny,nz)
        src_ke = features[:, s + 3 : s + 4]  # (B,1,nx,ny,nz)
        src_order = features[:, s + 4 : s + 5]  # (B,1,nx,ny,nz)

        face_fluxes = {
            "atom_flux": [],
            "material_momentum_flux": [],
            "material_ke_flux": [],
            "force_momentum_flux": [],
            "force_ke_flux": [],
        }

        # Evaluate shared face MLP for each directed face.
        for face in range(6):
            dst_feat = dst_features_by_face[face]
            dst_counts = dst_feat[:, 0:s]
            dst_mom = dst_feat[:, s : s + 3]
            dst_ke = dst_feat[:, s + 3 : s + 4]
            dst_order = dst_feat[:, s + 4 : s + 5]

            # Rotate momentum into face-normal aligned coordinates.
            src_mom_rot = self._rotate_state_momentum_for_face(src_mom, face)  # (B,3,nx,ny,nz)
            dst_mom_rot = self._rotate_state_momentum_for_face(dst_mom, face)

            # Material momentum flux is predicted per species, but input uses total momentum per cell.
            # We include rotated total momentum for both src and dst.

            # Face input: concat(src_comp, dst_comp) in channel dim.
            face_in = torch.cat(
                [
                    src_counts,
                    src_mom_rot,
                    src_ke,
                    src_order,
                    dst_counts,
                    dst_mom_rot,
                    dst_ke,
                    dst_order,
                ],
                dim=1,
            )  # (B,2C,nx,ny,nz)

            # MLP uses Conv3d with channels-first; our tensor is (B,in_ch,nx,ny,nz) which matches Conv3d expecting (B,C,D,H,W).
            # Treat x=nx as D, y=ny as H, z=nz as W for Conv3d consistency. This is just a learned embedding;
            # transfer logic will interpret the fluxes via the same topology we use for neighbors.
            # To keep it consistent, we permute to (B,in_ch,nz,ny,nx) -> Conv3d (D=z,H=y,W=x).
            face_in_conv = face_in.permute(0, 1, 4, 3, 2)  # (B,in_ch,nz,ny,nx)
            face_out_conv = self.face_mlp(face_in_conv)  # (B,out_ch,nz,ny,nx)
            face_out = face_out_conv.permute(0, 1, 4, 3, 2)  # (B,out_ch,nx,ny,nz)

            # Parse channels.
            # out layout:
            # [force_mom(3), force_ke(1), atom_flux(S), material_mom(S*3), material_ke(S)]
            force_mom_rot = face_out[:, 0:3]  # (B,3,nx,ny,nz)
            force_ke = face_out[:, 3:4].squeeze(1)  # (B,nx,ny,nz)

            atom_flux = face_out[:, 4 : 4 + s]  # (B,S,nx,ny,nz)
            material_mom_flat = face_out[:, 4 + s : 4 + s + 3 * s]  # (B,3S,nx,ny,nz)
            material_mom_rot = material_mom_flat.view(-1, s, 3, *face_out.shape[2:])  # (B,S,3,nx,ny,nz)
            material_ke = face_out[:, 4 + s + 3 * s : 4 + s + 3 * s + s]  # (B,S,nx,ny,nz)

            # Enforce non-negative for atom/KE flux magnitudes.
            atom_flux = F.softplus(atom_flux)
            force_ke = F.softplus(force_ke)
            material_ke = F.softplus(material_ke)

            face_fluxes["atom_flux"].append(atom_flux)
            face_fluxes["material_momentum_flux"].append(material_mom_rot)
            face_fluxes["material_ke_flux"].append(material_ke)
            face_fluxes["force_momentum_flux"].append(force_mom_rot)
            face_fluxes["force_ke_flux"].append(force_ke)

        # Stack across faces.
        atom_flux = torch.stack(face_fluxes["atom_flux"], dim=1)  # (B,F,S,nx,ny,nz)
        material_momentum_flux = torch.stack(face_fluxes["material_momentum_flux"], dim=1)  # (B,F,S,3,nx,ny,nz)
        material_ke_flux = torch.stack(face_fluxes["material_ke_flux"], dim=1)  # (B,F,S,nx,ny,nz)
        force_momentum_flux = torch.stack(face_fluxes["force_momentum_flux"], dim=1)  # (B,F,3,nx,ny,nz)
        force_ke_flux = torch.stack(face_fluxes["force_ke_flux"], dim=1)  # (B,F,nx,ny,nz)

        return {
            "atom_flux": atom_flux,
            "material_momentum_flux": material_momentum_flux,
            "material_ke_flux": material_ke_flux,
            "force_momentum_flux": force_momentum_flux,
            "force_ke_flux": force_ke_flux,
        }

    def _predict_order_next(self, state: CoarseState) -> torch.Tensor:
        """
        Predict order at the next timestep.

        Returns order_next: (B,1,nx,ny,nz)
        """
        features = state.as_features()
        if features.dim() == 4:
            features = features.unsqueeze(0)
        # Convert to Conv3d convention (B,C,nz,ny,nx).
        x = features.permute(0, 1, 4, 3, 2)
        delta = self.order_head(x)  # (B,1,nz,ny,nx)
        delta = delta.permute(0, 1, 4, 3, 2)  # (B,1,nx,ny,nz)
        return state.order + delta

    def _constrained_scale(
        self,
        state: CoarseState,
        fluxes: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Scale all outgoing fluxes from each source cell equally so that:
        - atom counts do not go negative
        - kinetic energy does not go negative
        """

        counts, momentum, ke, order = self._extract_state_components(state)
        if counts.dim() != 5:
            raise ValueError("_constrained_scale expects batched state")

        atom_flux = fluxes["atom_flux"]
        material_momentum_flux = fluxes["material_momentum_flux"]
        material_ke_flux = fluxes["material_ke_flux"]
        force_momentum_flux = fluxes["force_momentum_flux"]
        force_ke_flux = fluxes["force_ke_flux"]

        # Total predicted outgoing atom flux by species.
        atom_out_sum = atom_flux.sum(dim=1)  # (B,S,nx,ny,nz)

        # Atom constraint: scale <= counts for each species.
        scale_atoms_per_species = torch.ones_like(atom_out_sum)
        mask = atom_out_sum > self.eps
        scale_atoms_per_species[mask] = counts[mask] / atom_out_sum[mask]
        scale_atoms_all = scale_atoms_per_species.min(dim=1).values  # (B,nx,ny,nz)

        # KE constraint: total KE out = force_ke + sum_species(material_ke)
        force_ke_out = force_ke_flux.sum(dim=1)  # (B,nx,ny,nz)
        material_ke_out = material_ke_flux.sum(dim=1).sum(dim=1)  # (B,nx,ny,nz)
        ke_out_total = force_ke_out + material_ke_out  # (B,nx,ny,nz)
        scale_ke = torch.ones_like(ke_out_total)
        mask2 = ke_out_total > self.eps
        # ke has shape (B,1,nx,ny,nz)
        ke_scalar = ke[:, 0]
        scale_ke[mask2] = ke_scalar[mask2] / ke_out_total[mask2]

        scale = torch.minimum(scale_atoms_all, scale_ke)  # (B,nx,ny,nz)
        scale = torch.clamp(scale, min=0.0, max=1.0)

        # Apply to all outgoing fluxes equally.
        scale_face = scale.unsqueeze(1)  # (B,1,nx,ny,nz)
        atom_flux = atom_flux * scale_face.unsqueeze(2)  # (B,F,S,nx,ny,nz)
        material_momentum_flux = material_momentum_flux * scale_face.unsqueeze(2).unsqueeze(3)  # (B,F,S,3,nx,ny,nz)
        material_ke_flux = material_ke_flux * scale_face.unsqueeze(2)  # (B,F,S,nx,ny,nz)
        force_momentum_flux = force_momentum_flux * scale_face.unsqueeze(2)  # (B,F,3,nx,ny,nz)
        force_ke_flux = force_ke_flux * scale_face  # (B,F,nx,ny,nz)

        scaled = {
            "atom_flux": atom_flux,
            "material_momentum_flux": material_momentum_flux,
            "material_ke_flux": material_ke_flux,
            "force_momentum_flux": force_momentum_flux,
            "force_ke_flux": force_ke_flux,
        }
        return scale, scaled

    @staticmethod
    def flux_regularization_scaled(scaled: dict) -> torch.Tensor:
        """
        Mean squared magnitude of constrained (scaled) face fluxes — encourages smaller transfers for stability.
        One scalar per batch element is averaged into a single loss term via .mean().
        """
        parts = []
        for v in scaled.values():
            parts.append((v * v).mean())
        return torch.stack(parts).mean()

    def _soft_transfer(self, state: CoarseState, fluxes: dict, scaled: Optional[dict] = None) -> CoarseState:
        counts, momentum, ke, order = self._extract_state_components(state)
        # Ensure batched.
        if counts.dim() != 5:
            raise ValueError("_soft_transfer expects batched state")

        # Constrained scaling to prevent negative counts/KE.
        if scaled is None:
            _, scaled = self._constrained_scale(state, fluxes)
        atom_flux = scaled["atom_flux"]  # (B,F,S,nx,ny,nz)
        material_mom_rot = scaled["material_momentum_flux"]  # (B,F,S,3,nx,ny,nz)
        material_ke = scaled["material_ke_flux"]  # (B,F,S,nx,ny,nz)
        force_mom_rot = scaled["force_momentum_flux"]  # (B,F,3,nx,ny,nz)
        force_ke = scaled["force_ke_flux"]  # (B,F,nx,ny,nz)

        # Gate material fluxes by atom_flux (if atom_flux == 0, material flux ignored).
        gate = atom_flux / (atom_flux + self.eps)  # (B,F,S,nx,ny,nz)
        material_mom_rot = material_mom_rot * gate.unsqueeze(3)  # (B,F,S,3,nx,ny,nz)
        material_ke = material_ke * gate  # (B,F,S,nx,ny,nz)

        # Convert rotated momentum fluxes back to global.
        # We'll compute in face loop to keep code readable.
        B, F, S, nx, ny, nz = atom_flux.shape

        out_counts = atom_flux.sum(dim=1)  # (B,S,nx,ny,nz)

        out_force_mom_global = torch.zeros((B, 3, nx, ny, nz), device=counts.device, dtype=counts.dtype)
        out_material_mom_global = torch.zeros_like(out_force_mom_global)
        out_ke_force = force_ke.sum(dim=1).unsqueeze(1)  # (B,1,nx,ny,nz)
        out_ke_material = material_ke.sum(dim=1).sum(dim=1).unsqueeze(1)  # (B,1,nx,ny,nz)

        in_counts = torch.zeros_like(counts)
        in_mom = torch.zeros_like(momentum)
        in_ke = torch.zeros_like(ke)

        for face in range(6):
            atom_face = atom_flux[:, face]  # (B,S,nx,ny,nz)
            in_counts = in_counts + shift_src_to_dst(atom_face, face=face)

            force_mom_face_rot = force_mom_rot[:, face]  # (B,3,nx,ny,nz)
            force_mom_face_global = rotate_vec_from_face_normal(force_mom_face_rot, face)  # (B,3,nx,ny,nz)
            out_force_mom_global = out_force_mom_global + force_mom_face_global
            in_mom = in_mom + shift_src_to_dst(force_mom_face_global, face=face)

            material_mom_face_rot = material_mom_rot[:, face]  # (B,S,3,nx,ny,nz)
            # Convert each vector: rotate expects (B,S,3,...) works.
            material_mom_face_global = rotate_vec_from_face_normal(material_mom_face_rot, face)  # (B,S,3,...)
            material_mom_total_global = material_mom_face_global.sum(dim=1)  # (B,3,nx,ny,nz)
            out_material_mom_global = out_material_mom_global + material_mom_total_global
            in_mom = in_mom + shift_src_to_dst(material_mom_total_global, face=face)

            # KE (scalars)
            force_ke_face = force_ke[:, face]  # (B,nx,ny,nz)
            in_ke = in_ke + shift_src_to_dst(force_ke_face.unsqueeze(1), face=face)

            material_ke_face = material_ke[:, face].sum(dim=1)  # (B,nx,ny,nz)
            in_ke = in_ke + shift_src_to_dst(material_ke_face.unsqueeze(1), face=face)

        counts_next = counts - out_counts + in_counts
        momentum_next = momentum - out_force_mom_global - out_material_mom_global + in_mom
        ke_next = ke - out_ke_force - out_ke_material + in_ke

        # Keep KE/counts numerically non-negative.
        counts_next = torch.clamp(counts_next, min=0.0)
        ke_next = torch.clamp(ke_next, min=0.0)

        order_next = self._predict_order_next(CoarseState(counts=counts_next, momentum=momentum_next, ke=ke_next, order=order))
        order_next = torch.clamp(order_next, min=-1.0, max=1.0)

        return CoarseState(counts=counts_next, momentum=momentum_next, ke=ke_next, order=order_next)

    @torch.no_grad()
    def _hard_transfer(self, state: CoarseState, fluxes: dict) -> CoarseState:
        counts, momentum, ke, order = self._extract_state_components(state)
        if counts.dim() != 5:
            raise ValueError("_hard_transfer expects batched state")

        # Constrained scaling to prevent negative counts/KE.
        _, scaled = self._constrained_scale(state, fluxes)
        atom_flux = scaled["atom_flux"]  # (B,F,S,nx,ny,nz)
        material_mom_rot = scaled["material_momentum_flux"]  # (B,F,S,3,nx,ny,nz)
        material_ke = scaled["material_ke_flux"]  # (B,F,S,nx,ny,nz)
        force_mom_rot = scaled["force_momentum_flux"]  # (B,F,3,nx,ny,nz)
        force_ke = scaled["force_ke_flux"]  # (B,F,nx,ny,nz)

        B, F, S, nx, ny, nz = atom_flux.shape

        # Integerize available counts for atom transfer.
        available_atoms = torch.clamp(torch.round(counts), min=0).to(torch.int64)  # (B,S,nx,ny,nz)

        # Realized integer atom transfer per face.
        realized_atoms = torch.zeros_like(atom_flux, dtype=torch.int64)  # (B,F,S,nx,ny,nz)

        # Greedy allocation per source cell/species.
        for b in range(B):
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        for sp in range(S):
                            avail = int(available_atoms[b, sp, ix, iy, iz].item())
                            if avail <= 0:
                                continue
                            fluxes = atom_flux[b, :, sp, ix, iy, iz].detach().cpu().numpy().tolist()  # length 6
                            total_flux = float(sum(fluxes))
                            # Realize an integer number based on total predicted flux.
                            realized_total = min(avail, int(round(total_flux)))
                            if realized_total <= 0:
                                continue
                            # Greedy choose highest remaining flux, subtracting 1 each allocation.
                            fluxes_work = fluxes[:]
                            alloc = [0] * 6
                            for _ in range(realized_total):
                                dest = int(max(range(6), key=lambda j: fluxes_work[j]))
                                alloc[dest] += 1
                                fluxes_work[dest] -= 1.0
                            realized_atoms[b, :, sp, ix, iy, iz] = torch.tensor(
                                alloc, dtype=torch.int64, device=realized_atoms.device
                            )

        # Realized atoms out per face/species -> counts out.
        atom_out = realized_atoms.sum(dim=1).to(counts.dtype)  # (B,S,nx,ny,nz)

        # Ratio for scaling material momentum/KE fluxes with realized atoms.
        atom_pred = atom_flux.clamp(min=0.0)  # (B,F,S,nx,ny,nz)
        ratio = torch.zeros_like(atom_pred)
        mask = atom_pred > self.eps
        ratio[mask] = realized_atoms[mask].to(counts.dtype) / atom_pred[mask]

        # Effective material fluxes after integerization.
        material_mom_eff_rot = material_mom_rot * ratio.unsqueeze(3)  # (B,F,S,3,nx,ny,nz)
        material_ke_eff = material_ke * ratio  # (B,F,S,nx,ny,nz)

        out_force_mom_global = torch.zeros((B, 3, nx, ny, nz), device=counts.device, dtype=counts.dtype)
        out_material_mom_global = torch.zeros_like(out_force_mom_global)

        out_ke_force = force_ke.sum(dim=1).unsqueeze(1)  # (B,1,nx,ny,nz)
        out_ke_material = material_ke_eff.sum(dim=1).sum(dim=1).unsqueeze(1)  # (B,1,nx,ny,nz)

        # Destination deltas:
        in_counts = torch.zeros_like(counts)
        in_mom = torch.zeros_like(momentum)
        in_ke = torch.zeros_like(ke)

        for face in range(6):
            # Counts
            atom_face_real = realized_atoms[:, face].to(counts.dtype)  # (B,S,nx,ny,nz)
            in_counts = in_counts + shift_src_to_dst(atom_face_real, face=face)

            # Force momentum
            force_mom_face_rot = force_mom_rot[:, face]  # (B,3,nx,ny,nz)
            force_mom_face_global = rotate_vec_from_face_normal(force_mom_face_rot, face)
            out_force_mom_global = out_force_mom_global + force_mom_face_global
            in_mom = in_mom + shift_src_to_dst(force_mom_face_global, face=face)

            # Material momentum (sum over species)
            mat_mom_face_rot = material_mom_eff_rot[:, face]  # (B,S,3,nx,ny,nz)
            mat_mom_face_global = rotate_vec_from_face_normal(mat_mom_face_rot, face)  # (B,S,3,...)
            mat_mom_total_global = mat_mom_face_global.sum(dim=1)  # (B,3,nx,ny,nz)
            out_material_mom_global = out_material_mom_global + mat_mom_total_global
            in_mom = in_mom + shift_src_to_dst(mat_mom_total_global, face=face)

            # KE
            force_ke_face = force_ke[:, face]  # (B,nx,ny,nz)
            in_ke = in_ke + shift_src_to_dst(force_ke_face.unsqueeze(1), face=face)
            mat_ke_face = material_ke_eff[:, face].sum(dim=1)  # (B,nx,ny,nz)
            in_ke = in_ke + shift_src_to_dst(mat_ke_face.unsqueeze(1), face=face)

        counts_next = counts - atom_out + in_counts
        momentum_next = momentum - out_force_mom_global - out_material_mom_global + in_mom
        ke_next = ke - out_ke_force - out_ke_material + in_ke

        counts_next = torch.clamp(counts_next, min=0.0)
        ke_next = torch.clamp(ke_next, min=0.0)

        order_next = self._predict_order_next(CoarseState(counts=counts_next, momentum=momentum_next, ke=ke_next, order=order))
        order_next = torch.clamp(order_next, min=-1.0, max=1.0)

        return CoarseState(counts=counts_next, momentum=momentum_next, ke=ke_next, order=order_next)

    def predict_next(
        self,
        current_state: CoarseState,
        *,
        target_state: Optional[CoarseState] = None,
        return_flux_reg: bool = False,
    ) -> Union[CoarseState, Tuple[CoarseState, torch.Tensor]]:
        """
        Predict next state using flux transfer.

        Note: `target_state` is unused (Option II doesn't require it for prediction),
        but it is accepted for interface compatibility with Option I.

        If ``return_flux_reg`` is True, returns ``(next_state, flux_reg)`` where ``flux_reg`` is a scalar
        tensor (mean squared constrained flux magnitudes) for optional regularization during training.
        """

        input_batched = current_state.counts.dim() == 5
        if not input_batched:
            # Add batch dimension.
            current_state = CoarseState(
                counts=current_state.counts.unsqueeze(0),
                momentum=current_state.momentum.unsqueeze(0),
                ke=current_state.ke.unsqueeze(0),
                order=current_state.order.unsqueeze(0),
            )

        fluxes = self._predict_fluxes(current_state)

        use_soft = self.soft_transfer and (self.training or self.force_soft_transfer_eval)
        flux_reg: Optional[torch.Tensor] = None
        scaled_pre: Optional[dict] = None
        if return_flux_reg:
            _, scaled_pre = self._constrained_scale(current_state, fluxes)
            flux_reg = self.flux_regularization_scaled(scaled_pre)

        if use_soft:
            next_state = self._soft_transfer(
                current_state,
                fluxes,
                scaled=scaled_pre if return_flux_reg else None,
            )
        else:
            next_state = self._hard_transfer(current_state, fluxes)

        if use_soft and self.soft_round_outputs and (self.force_soft_transfer_eval or not self.training):
            # For soft transfer we may produce fractional counts; for evaluation/inference we
            # round counts to whole occupancies and clamp energies/order into safe ranges.
            next_state.counts = torch.clamp(next_state.counts, min=0.0).round()
            next_state.ke = torch.clamp(next_state.ke, min=0.0)
            next_state.order = torch.clamp(next_state.order, min=-1.0, max=1.0)

        if not input_batched:
            # Squeeze batch dimension back to unbatched.
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

