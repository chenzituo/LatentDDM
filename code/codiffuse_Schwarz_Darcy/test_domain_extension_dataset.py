from __future__ import annotations

import h5py
import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover
from codiffuse_Schwarz_Darcy.adapter import (
    DarcyAdapter,
    DarcyStatistics,
    H5DarcyDataset,
    batch_to_transition,
    tiled_sni_face_conditions,
)
from codiffuse_Schwarz_Darcy.domain_extension_dataset import (
    DarcyDomainExtensionConfig,
    generate_domain_extension_hdf5,
)
from codiffuse_Schwarz_Darcy.evaluate import residual_rms


def _adapter() -> DarcyAdapter:
    return DarcyAdapter(
        DarcyStatistics(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1),
        use_discrete_residual=False,
    )


def test_fixed_dx_extension_schema_and_reader(tmp_path) -> None:
    config = DarcyDomainExtensionConfig.smoke(blocks=4, seed=31)
    path = generate_domain_extension_hdf5(tmp_path / "extension.h5", config)
    with h5py.File(path, "r") as handle:
        assert handle.attrs["scaling_mode"] == "fixed_grid_spacing_physical_growth"
        assert int(handle.attrs["grid_size"]) == 16
        assert np.isclose(handle.attrs["grid_spacing"], 2.0 / 12.0)
        assert np.isclose(handle.attrs["physical_extent"], 8.0 / 3.0)
        assert set(handle) == {"train", "val", "test"}
        assert handle["train/solution"].shape == (2, 1, 16, 16)
        assert np.isfinite(handle["train/solution"][:]).all()

    sample = H5DarcyDataset(path, "validation")[0]
    assert torch.isclose(sample["grid_spacing"], torch.tensor(2.0 / 12.0))
    assert torch.isclose(sample["physical_extent"], torch.tensor(8.0 / 3.0))
    batch = {
        key: value[None] if isinstance(value, torch.Tensor) else [value]
        for key, value in sample.items()
    }
    assert float(residual_rms(batch["solution"], batch)) < 2.0e-5


def test_local_source_scale_stays_fixed_as_domain_grows() -> None:
    adapter = _adapter()
    values = []
    for blocks in (3, 4, 8):
        grid = blocks * 8
        spacing = 2.0 / (3 * 8)
        batch = {
            "mask": torch.ones(1, 1, grid, grid),
            "logk": torch.zeros(1, 1, grid, grid),
            "source": torch.ones(1, 1, grid, grid),
            "bc_mask": torch.zeros(1, 1, grid, grid),
            "bc_value": torch.zeros(1, 1, grid, grid),
            "sdf": -torch.ones(1, 1, grid, grid),
            "normals": torch.zeros(1, 2, grid, grid),
            "grid_spacing": torch.tensor([spacing]),
            "physical_extent": torch.tensor([grid * spacing]),
        }
        cover = CartesianCover(grid, grid, core=8, halo=2)
        conditions = tiled_sni_face_conditions(
            adapter, batch, torch.zeros_like(batch["source"]), cover,
        )
        # Adapter source channel 2 is normalized by unit statistics.
        values.append(float(conditions.physical[:, 2].max()))
        assert np.isclose(conditions.dx, spacing)
    np.testing.assert_allclose(values, values[0], rtol=1.0e-6, atol=1.0e-7)


def test_global_operator_receives_physical_target_coordinates() -> None:
    grid = 8
    extent = 8.0 / 3.0
    shape = (1, 1, grid, grid)
    batch = {
        "mask": torch.ones(shape),
        "logk": torch.zeros(shape),
        "source": torch.ones(shape),
        "bc_mask": torch.zeros(shape),
        "bc_value": torch.zeros(shape),
        "solution": torch.zeros(shape),
        "sdf": -torch.ones(shape),
        "normals": torch.zeros(1, 2, grid, grid),
        "grid_spacing": torch.tensor([extent / grid]),
        "physical_extent": torch.tensor([extent]),
    }
    transition = batch_to_transition(batch, torch.zeros(shape))
    condition = _adapter().conditions(transition, torch.ones(1), 1.0)
    x = condition.spatial[0, 9]
    y = condition.spatial[0, 10]
    expected_edge_center = 0.5 * extent - 0.5 * extent / grid
    assert torch.isclose(x[0, -1], torch.tensor(expected_edge_center))
    assert torch.isclose(x[0, 0], torch.tensor(-expected_edge_center))
    assert torch.isclose(y[-1, 0], torch.tensor(expected_edge_center))
