"""Tests for the one-pass Darcy Galerkin correction."""

from __future__ import annotations

import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import raw_darcy_residual
from .two_level import galerkin_correct, tile_pou_basis


def test_tile_pou_constant_modes_sum_to_one() -> None:
    cover = CartesianCover(19, 23, 8, 2, 3, 5)
    basis = tile_pou_basis(
        cover, degree=0, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert basis.shape == (cover.tile_count, 19, 23)
    torch.testing.assert_close(basis.sum(0), torch.ones(19, 23), atol=2.0e-6, rtol=0.0)


def test_galerkin_correction_reduces_energy_and_annihilates_coarse_residual() -> None:
    torch.manual_seed(7)
    height = width = 16
    cover = CartesianCover(height, width, 8, 2)
    mask = torch.ones(2, 1, height, width)
    mask[:, :, 6:10, 7:9] = 0.0
    batch = {
        "mask": mask,
        "logk": 0.2 * torch.randn(2, 1, height, width),
        "source": torch.randn(2, 1, height, width) * mask,
        "bc_value": torch.zeros(2, 1, height, width),
    }
    preliminary = 0.1 * torch.randn(2, 1, height, width) * mask
    basis = tile_pou_basis(
        cover, degree=1, device=torch.device("cpu"), dtype=torch.float32,
    )
    corrected, diagnostics = galerkin_correct(preliminary, batch, basis)
    coefficient = torch.pow(10.0, batch["logk"])

    def energy_objective(field: torch.Tensor) -> torch.Tensor:
        residual = raw_darcy_residual(
            field, coefficient, batch["source"], mask, batch["bc_value"],
        )
        homogeneous = raw_darcy_residual(
            field, coefficient, torch.zeros_like(field), mask, torch.zeros_like(field),
        )
        rhs = homogeneous - residual
        return 0.5 * (field * homogeneous).flatten(1).sum(1) - (field * rhs).flatten(1).sum(1)

    assert torch.all(energy_objective(corrected) <= energy_objective(preliminary) + 2.0e-4)
    corrected_residual = raw_darcy_residual(
        corrected, coefficient, batch["source"], mask, batch["bc_value"],
    )[:, 0]
    active_basis = basis[None] * mask
    coarse_residual = torch.einsum("bmhw,bhw->bm", active_basis, corrected_residual)
    assert coarse_residual.abs().amax() < 2.0e-2
    assert torch.all(diagnostics.rank > 0)

