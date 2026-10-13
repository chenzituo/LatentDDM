from __future__ import annotations

import torch

from codiffuse_Schwarz.core import (
    CoDiffuseSchwarz,
    ToyConfig,
    coordinate_grid,
    generate_dataset,
    patch_cores,
    predict_patch_velocity,
    relative_residual,
    sample_flow,
)


def main() -> None:
    torch.manual_seed(19)
    config = ToyConfig(height=16, width=16, patch_size=8, halo=2, solve_iters=80)
    data = generate_dataset(2, config, seed=19, batch_size=2)
    assert data["field"].shape == (2, 1, 16, 16)
    assert len(patch_cores(16, 16, 8)) == 4
    assert torch.all(data["field"] * (1.0 - data["mask"]) == 0.0)

    model = CoDiffuseSchwarz(width=16, depth=2, m_dim=8, heads=2)
    time = torch.tensor([0.25, 0.75])
    coords = coordinate_grid(16, 16).expand(2, -1, -1, -1)
    velocity = predict_patch_velocity(
        model,
        data["field"],
        data["mask"],
        data["source"],
        data["reaction"],
        coords,
        time,
        patch_size=8,
        halo=2,
        connected=True,
    )
    assert velocity.shape == data["field"].shape
    assert torch.isfinite(velocity).all()
    averaged_velocity = predict_patch_velocity(
        model,
        data["field"],
        data["mask"],
        data["source"],
        data["reaction"],
        coords,
        time,
        patch_size=8,
        halo=2,
        connected=True,
        assembly="average",
    )
    assert averaged_velocity.shape == data["field"].shape
    assert torch.isfinite(averaged_velocity).all()

    eos_model = CoDiffuseSchwarz(width=16, depth=2, m_dim=8, heads=2, use_eos=True)
    assert eos_model.eos is not None
    residual = data["source"] * 0.0
    full_m = eos_model.eos.extensive(
        data["field"], residual, data["source"], data["mask"], coords, normalizer=16 * 16
    )
    patch_m = []
    for y0, y1, x0, x1 in patch_cores(16, 16, 8):
        patch_m.append(
            eos_model.eos.extensive(
                data["field"][:, :, y0:y1, x0:x1],
                residual[:, :, y0:y1, x0:x1],
                data["source"][:, :, y0:y1, x0:x1],
                data["mask"][:, :, y0:y1, x0:x1],
                coords[:, :, y0:y1, x0:x1],
                normalizer=16 * 16,
            )
        )
    assert torch.allclose(full_m, torch.stack(patch_m).sum(dim=0), atol=1.0e-6)

    result = sample_flow(
        model,
        data["mask"],
        data["source"],
        data["reaction"],
        data["field"],
        config,
        mode="schwarz",
        inference_steps=2,
        initial_noise=torch.randn_like(data["field"]),
    )
    assert result.field.shape == data["field"].shape
    assert len(result.trajectory) == 2
    assert result.trajectory[1]["time"] == 0.0
    assert torch.all(result.field * (1.0 - data["mask"]) == 0.0)
    assert torch.isfinite(relative_residual(result.field, data["source"], data["mask"], data["reaction"])).all()
    print("all CoDiffuse-Schwarz toy tests passed")


if __name__ == "__main__":
    main()
