import math
import os
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as e:
    raise unittest.SkipTest("PyTorch is required to run diffusion tests") from e

from diffusion.config import GridConfig, SpeciesConfig
from diffusion.data.make_data import make_coarse_state_from_dump
from diffusion.models.option_ii import OptionIIModel, DIRECTIONS_26
from diffusion.state import CoarseState


class TestOrderParameter(unittest.TestCase):
    def test_order_parameter_matches_formula(self) -> None:
        # Use a small box with nx=2,ny=2,nz=1 based on a.
        a = 3.5657157
        species = SpeciesConfig(lammps_types=[1], masses=[12.0])
        grid = GridConfig(lattice_constant_a=a, periodic_xy=True)

        x1 = 0.1 * a
        x2 = 0.2 * a
        y = 0.3 * a
        z = 0.4 * a

        # Both atoms fall into cell ix=0,iy=0,iz=0.
        dump = f"""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 {2*a}
0.0 {2*a}
0.0 {1*a}
ITEM: ATOMS id type x y z vx vy vz fx fy fz
1 1 {x1} {y} {z} 0 0 0 0 0 0
2 1 {x2} {y} {z} 0 0 0 0 0 0
"""

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "0.dump")
            with open(path, "w", encoding="utf-8") as f:
                f.write(dump)

            _, state, _ = make_coarse_state_from_dump(path, species=species, grid=grid)
            # state.order shape: (1,nx,ny,nz)
            xi_model = float(state.order[0, 0, 0, 0].item())

        lambda_x = (math.cos(8.0 * math.pi * x1 / a) + math.cos(8.0 * math.pi * x2 / a)) / 2.0
        lambda_y = math.cos(8.0 * math.pi * y / a)
        lambda_z = math.cos(8.0 * math.pi * z / a)
        xi_expected = (lambda_x + lambda_y + lambda_z) / 3.0

        self.assertAlmostEqual(xi_model, xi_expected, places=5)


class TestPeriodicWrapping(unittest.TestCase):
    def test_periodic_x_y_wrapping_for_binning(self) -> None:
        a = 3.5657157
        species = SpeciesConfig(lammps_types=[1], masses=[12.0])
        grid = GridConfig(lattice_constant_a=a, periodic_xy=True)

        x_wrap = 2 * a + 0.1 * a  # should wrap back to ix=0 when nx=2 and grid extent is 2*a
        x_same_cell = 0.1 * a
        y = 0.1 * a
        z = 0.2 * a

        dump = f"""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 {2*a}
0.0 {2*a}
0.0 {1*a}
ITEM: ATOMS id type x y z vx vy vz fx fy fz
1 1 {x_wrap} {y} {z} 0 0 0 0 0 0
2 1 {x_same_cell} {y} {z} 0 0 0 0 0 0
"""

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "0.dump")
            with open(path, "w", encoding="utf-8") as f:
                f.write(dump)

            _, state, _ = make_coarse_state_from_dump(path, species=species, grid=grid)
            # state.counts shape: (S,nx,ny,nz)
            c00 = float(state.counts[0, 0, 0, 0].item())
            c10 = float(state.counts[0, 1, 0, 0].item())

        self.assertEqual(c00, 2.0)
        self.assertEqual(c10, 0.0)


class TestOptionIIHardTransfer(unittest.TestCase):
    def test_hard_transfer_non_negative_constraints(self) -> None:
        torch.manual_seed(0)

        # Small grid: nx=2, ny=1, nz=1, single species.
        nx, ny, nz = 2, 1, 1
        B = 1
        S = 1

        model = OptionIIModel(num_species=S, hidden_channels=8, soft_transfer=False)
        # Make order head deterministic: order_next = order.
        for p in model.order_head.parameters():
            p.data.zero_()

        counts = torch.tensor([[[[[1.0]], [[0.0]]]]])  # (B,S,nx,ny,nz) -> shape manip below
        # Rebuild with explicit dims:
        counts = torch.zeros((B, S, nx, ny, nz), dtype=torch.float32)
        counts[0, 0, 0, 0, 0] = 1.0
        momentum = torch.zeros((B, 3, nx, ny, nz), dtype=torch.float32)
        ke = torch.zeros((B, 1, nx, ny, nz), dtype=torch.float32)
        ke[0, 0, :, 0, 0] = 1.0
        order = torch.zeros((B, 1, nx, ny, nz), dtype=torch.float32)

        state = CoarseState(counts=counts, momentum=momentum, ke=ke, order=order)

        # Predicted fluxes: send too much atoms/energy out of +x neighbors.
        D = len(DIRECTIONS_26)
        idx_plus_x = DIRECTIONS_26.index((1, 0, 0))

        atom_flux = torch.zeros((B, D, S, nx, ny, nz), dtype=torch.float32)
        force_ke_flux = torch.zeros((B, D, nx, ny, nz), dtype=torch.float32)
        material_momentum_flux = torch.zeros((B, D, S, 3, nx, ny, nz), dtype=torch.float32)
        material_ke_flux = torch.zeros((B, D, S, nx, ny, nz), dtype=torch.float32)
        force_momentum_flux = torch.zeros((B, D, 3, nx, ny, nz), dtype=torch.float32)

        # For both cells, predict 2 atoms out on +x face (will be scaled down because counts are 1.0).
        atom_flux[:, idx_plus_x, 0, :, 0, 0] = 2.0
        # Predict KE flux out larger than available ke (=1.0) to force KE scaling.
        force_ke_flux[:, idx_plus_x, :, 0, 0] = 2.0

        fluxes = {
            "atom_flux": atom_flux,
            "material_momentum_flux": material_momentum_flux,
            "material_ke_flux": material_ke_flux,
            "force_momentum_flux": force_momentum_flux,
            "force_ke_flux": force_ke_flux,
        }

        next_state = model._hard_transfer(state, fluxes)

        self.assertTrue(torch.all(next_state.counts >= 0.0).item())
        self.assertTrue(torch.all(next_state.ke >= 0.0).item())

        # Total counts should remain integers and not exceed initial total.
        total_counts = next_state.counts.sum().item()
        self.assertGreaterEqual(total_counts, 0.0)
        self.assertLessEqual(total_counts, counts.sum().item() + 1e-5)


if __name__ == "__main__":
    unittest.main()

