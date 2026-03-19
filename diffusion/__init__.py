"""
Coarse-grained diffusion models for MD simulation rollouts.

This package contains:
- LAMMPS dump preprocessing (`diffusion/data/make_data.py`)
- Two interchangeable model options (Option I and Option II)
- Training (`diffusion/train.py`) and inference (`diffusion/infer.py`) entrypoints
"""

from .config import SpeciesConfig, GridConfig

__all__ = ["SpeciesConfig", "GridConfig"]

