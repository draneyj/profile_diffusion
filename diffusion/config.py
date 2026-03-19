from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class SpeciesConfig:
    """
    Mapping between LAMMPS atom `type` integers and the model's contiguous
    species indices [0..S-1].
    """

    # LAMMPS atom type IDs in order of species index.
    lammps_types: List[int]
    # Mass per species index (same length as lammps_types).
    masses: List[float]

    def __post_init__(self) -> None:
        if len(self.lammps_types) == 0:
            raise ValueError("SpeciesConfig.lammps_types must be non-empty")
        if len(self.lammps_types) != len(self.masses):
            raise ValueError("SpeciesConfig.masses must match lammps_types length")

    @property
    def num_species(self) -> int:
        return len(self.lammps_types)


@dataclass(frozen=True)
class GridConfig:
    """
    Coarse-graining grid parameters.

    Notes on boundary conditions (per the project spec):
    - Periodic boundary conditions in x and y.
    - Non-periodic in z: atoms mapped outside the chosen z-range are ignored.
    """

    lattice_constant_a: float = 3.5657157
    periodic_xy: bool = True
    # If True, use `floor(box_extent/a)` for each axis to determine grid dims.
    # If False, require exact divisibility.
    use_floor_dims: bool = True

    # Optional: restrict to atoms in z-range [z_min, z_min + nz*a)
    # derived from the computed grid.
    ignore_atoms_outside_z: bool = True

    def order_parameter_a(self) -> float:
        return self.lattice_constant_a


@dataclass(frozen=True)
class DataPairingConfig:
    """
    How training samples are paired: frame t -> frame t + stride_k.
    """

    stride_k: int = 1


@dataclass(frozen=True)
class ModelConfig:
    num_refine_steps: int = 1
    # For Option I diffusion denoising: stddev of Gaussian noise added to target.
    noise_std: float = 0.1
    # For Option II training: use differentiable soft transfer.
    soft_transfer: bool = True
    # Shared output clamping ranges (optional).
    eps: float = 1e-8

