from __future__ import annotations

import torch

from codiffuse_Schwarz.amr_flow import (
    AMRFlowModel,
    TwoLevelAMRConfig,
    build_amr_layouts,
    coverage_masks,
    layout_costs,
    predict_fine_patch_velocity,
    sample_amr_flow,
)
from codiffuse_Schwarz.core import ToyConfig, coordinate_grid, generate_dataset


def main() -> None:
    torch.manual_seed(23)
    toy = ToyConfig(height=32, width=32, solve_iters=80)
    data = generate_dataset(8, toy, seed=23, batch_size=8)
    config = TwoLevelAMRConfig(tag_threshold=0.48, halo=2)
    layouts = build_amr_layouts(data["mask"], data["source"], data["reaction"], config)

    assert len(layouts) == data["field"].shape[0]
    assert all(layout.fine_boxes for layout in layouts)
    for layout in layouts:
        for y0, y1, x0, x1 in layout.fine_boxes:
            assert 0 <= y0 < y1 <= config.fine_height
            assert 0 <= x0 < x1 <= config.fine_width
            assert all(value % config.blocking == 0 for value in (y0, y1, x0, x1))

    coverage = coverage_masks(layouts, (config.fine_height, config.fine_width), torch.device("cpu"))
    assert torch.all((coverage == 0.0) | (coverage == 1.0))
    assert float(coverage.mean()) < 0.9
    costs = layout_costs(layouts, config)
    assert costs["stored_fraction_vs_uniform_fine"] < 1.0
    assert costs["processed_fraction_vs_uniform"] < 1.0

    model = AMRFlowModel(width=16, depth=2)
    time = torch.full((data["field"].shape[0],), 0.5)
    coords = coordinate_grid(32, 32).expand(data["field"].shape[0], -1, -1, -1)
    velocity, written = predict_fine_patch_velocity(
        model,
        data["field"],
        data["mask"],
        data["source"],
        data["reaction"],
        coords,
        time,
        layouts,
        config.halo,
    )
    assert velocity.shape == data["field"].shape
    assert torch.isfinite(velocity).all()
    assert torch.equal(written.bool(), coverage.bool())

    result = sample_amr_flow(
        model,
        data["mask"],
        data["source"],
        data["reaction"],
        data["field"],
        torch.randn_like(data["field"]),
        layouts,
        config,
        steps=2,
    )
    assert result.field.shape == data["field"].shape
    assert len(result.trajectory) == 2
    assert torch.isfinite(result.field).all()
    assert torch.all(result.field * (1.0 - data["mask"]) == 0.0)
    print("all adaptive AMR flow tests passed")


if __name__ == "__main__":
    main()
