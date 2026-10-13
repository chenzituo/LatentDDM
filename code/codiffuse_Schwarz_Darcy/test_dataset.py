from __future__ import annotations

import json

import h5py
import numpy as np

from codiffuse_Schwarz_Darcy.adapter import H5DarcyDataset
from codiffuse_Schwarz_Darcy.dataset import (
    DarcyDatasetConfig,
    boundary_normals,
    generate_hdf5,
    signed_distance,
)
from codiffuse_Schwarz_Darcy.physics import (
    aitken_relaxation,
    assemble_darcy_system,
    interface_value_flux_jump,
    optimal_energy_relaxation,
    relative_energy_error,
    relative_residual,
    solve_darcy,
)


def test_darcy_system_is_symmetric_positive_and_solved() -> None:
    mask = np.ones((12, 12), dtype=bool)
    mask[4:7, 5:8] = False
    y, x = np.meshgrid(np.linspace(0, 1, 12), np.linspace(0, 1, 12), indexing="ij")
    kappa = 0.2 + x + 0.4 * y
    source = 1.0 + x
    boundary = 0.1 + 0.2 * y
    solution, system = solve_darcy(mask, kappa, source, boundary)
    difference = system.matrix - system.matrix.T
    assert np.linalg.norm(difference.data) < 1.0e-12
    assert np.min(np.linalg.eigvalsh(system.matrix.toarray())) > 0.0
    assert relative_residual(solution, system) < 1.0e-10
    assert relative_energy_error(solution, solution, system) < 1.0e-12


def test_geometry_channels_have_expected_sign_and_normals() -> None:
    mask = np.zeros((24, 24), dtype=bool)
    mask[4:20, 5:19] = True
    sdf = signed_distance(mask, 2.0 / 24)
    bc = mask & (
        ~np.roll(mask, 1, 0) | ~np.roll(mask, -1, 0)
        | ~np.roll(mask, 1, 1) | ~np.roll(mask, -1, 1)
    )
    normals = boundary_normals(sdf, bc, 2.0 / 24)
    assert np.all(sdf[mask] < 0.0)
    assert np.all(sdf[~mask] >= 0.0)
    magnitude = np.sqrt((normals * normals).sum(0))
    assert np.allclose(magnitude[bc], 1.0, atol=1.0e-6)
    assert np.all(magnitude[~bc] == 0.0)


def test_relaxation_and_interface_diagnostics() -> None:
    mask = np.ones((8, 8), dtype=bool)
    system = assemble_darcy_system(mask, np.ones_like(mask), np.ones_like(mask), np.zeros_like(mask))
    current = np.zeros((8, 8))
    exact, _ = solve_darcy(mask, np.ones_like(mask), np.ones_like(mask), np.zeros_like(mask))
    omega = optimal_energy_relaxation(current, exact, system)
    assert np.isclose(omega, 1.0)
    assert 0.05 <= aitken_relaxation(exact, 0.5 * exact, 1.0) <= 1.5

    base = np.add.outer(np.arange(6.0), 2.0 * np.arange(6.0))
    patches = np.stack((base, base))
    report = interface_value_flux_jump(
        patches, [(0, 0), (0, 0)], np.ones_like(patches, dtype=bool),
        np.ones_like(patches), dx=1.0,
    )
    assert report["overlap_points"] == 36
    assert report["value_jump_rms"] == 0.0
    assert report["normal_flux_jump_rms"] == 0.0


def test_smoke_hdf5_schema(tmp_path) -> None:
    path = generate_hdf5(tmp_path / "smoke.h5", DarcyDatasetConfig.smoke(seed=9))
    with h5py.File(path, "r") as handle:
        assert (
            handle.attrs["schema_version"]
            == "darcy_complex_cartesian_v2_face_dirichlet"
        )
        for split, count in (("train", 6), ("val", 2), ("test", 2)):
            group = handle[split]
            assert group["mask"].shape == (count, 1, 16, 16)
            assert group["normals"].shape == (count, 2, 16, 16)
            assert group["dirichlet_face_mask"].shape == (count, 4, 16, 16)
            assert group["dirichlet_face_value"].shape == (count, 4, 16, 16)
            assert group["dirichlet_face_role"].shape == (count, 4, 16, 16)
            for key in (
                "mask", "logk", "source", "bc_mask", "bc_value",
                "solution", "sdf", "normals", "family", "seed", "bc_role",
                "dirichlet_face_mask", "dirichlet_face_value",
                "dirichlet_face_role",
            ):
                assert key in group
            face_mask = np.asarray(group["dirichlet_face_mask"], dtype=bool)
            face_role = np.asarray(group["dirichlet_face_role"])
            assert np.all(face_role[~face_mask] == 0)
            assert set(np.unique(face_role)).issubset({0, 1, 2})
        assert set(handle["domains"]) == {"A", "B", "C", "D"}
        normalization = json.loads(handle.attrs["normalization"])
        assert normalization["solution"]["std"] > 0.0
        solution = np.asarray(handle["train/solution"][0, 0])
        assert np.isfinite(solution).all()


def test_v1_cellwise_hdf5_remains_loadable(tmp_path) -> None:
    path = generate_hdf5(tmp_path / "legacy.h5", DarcyDatasetConfig.smoke(seed=13))
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema_version"] = "darcy_complex_cartesian_v1"
        for key in (
            "dirichlet_face_mask", "dirichlet_face_value", "dirichlet_face_role",
        ):
            del handle[f"train/{key}"]
    dataset = H5DarcyDataset(path, "train")
    assert not dataset.has_face_dirichlet
    sample = dataset[0]
    assert "bc_value" in sample
    assert "dirichlet_face_value" not in sample


def test_training_only_v2_never_writes_test_or_domains(tmp_path) -> None:
    path = generate_hdf5(
        tmp_path / "trainval_only.h5", DarcyDatasetConfig.smoke(seed=21),
        training_only=True,
    )
    with h5py.File(path, "r") as handle:
        assert set(handle.keys()) == {"train", "val"}
        assert bool(handle.attrs["training_only"])
