from __future__ import annotations

import torch

from .adapter import PhysicalTransition
from .cover import CartesianCover
from .direct import direct_endpoint_loss, predict_direct, predict_direct_multilevel
from .example_diffusion import make_example
from .experimental_flow import (
    CoDiffuseOperator,
    LocalFlowOperator,
    combined_loss,
    integrate_flow,
    pnp_multilevel_sample,
    pnp_schwarz_sample,
)
from .hierarchy import build_two_level_hierarchy
from .model import (
    DirectEndpointOperator,
    DirectEndpointRegressor,
    NeuralEndpointProjector,
)
from .schwarz import PreparedLevel


def small_flow_model(adapter) -> CoDiffuseOperator:
    physical_channels = adapter.spatial_condition_channels + 1
    return CoDiffuseOperator(
        LocalFlowOperator(
            adapter.state_channels, physical_channels,
            adapter.scalar_condition_dim, width=16, depth=1, heads=4,
        ),
        NeuralEndpointProjector(
            adapter.state_channels, physical_channels,
            adapter.scalar_condition_dim, width=16, depth=1,
        ),
    )


def small_direct_model(adapter) -> DirectEndpointOperator:
    physical_channels = adapter.spatial_condition_channels + 1
    return DirectEndpointOperator(
        DirectEndpointRegressor(
            adapter.state_channels, physical_channels,
            adapter.scalar_condition_dim, width=16, depth=1,
        ),
        NeuralEndpointProjector(
            adapter.state_channels, physical_channels,
            adapter.scalar_condition_dim, width=16, depth=1,
            cell_attention=False,
        ),
    )


def test_shifted_cover_round_trip_for_partial_boundary_cells() -> None:
    field = torch.randn(2, 5, 35, 51)
    cover = CartesianCover(35, 51, core=16, halo=3, offset_y=7, offset_x=11)
    patches = cover.tile(field)
    torch.testing.assert_close(cover.assemble_ras(patches, 2), field)
    torch.testing.assert_close(cover.assemble_weighted(patches, 2), field)
    assert patches.shape == (2 * cover.tile_count, 5, 22, 22)


def test_endpoint_projector_is_identity_at_initialization() -> None:
    projector = NeuralEndpointProjector(2, 6, 3, width=16, depth=1)
    endpoint = torch.randn(8, 2, 20, 20)
    physical = torch.randn(8, 6, 20, 20)
    physical[:, -1] = 1.0
    scalar = torch.randn(8, 3)
    torch.testing.assert_close(
        projector(endpoint, physical, scalar, tile_count=4), endpoint
    )


def test_direct_endpoint_loss_backpropagates() -> None:
    adapter, transition = make_example(torch.device("cpu"))
    cover = CartesianCover(64, 64, core=16, halo=4, offset_y=3, offset_x=9)
    level = PreparedLevel(transition, adapter, cover)
    model = small_direct_model(adapter)
    report = direct_endpoint_loss(model, level)
    report.total.backward()
    assert torch.isfinite(report.total)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_direct_endpoint_prediction_is_deterministic_residual() -> None:
    adapter, transition = make_example(torch.device("cpu"))
    level = PreparedLevel(
        transition, adapter, CartesianCover(64, 64, 16, 4, 5, 7)
    )
    model = small_direct_model(adapter)
    first = predict_direct(model, level)
    second = predict_direct(model, level)
    torch.testing.assert_close(first.endpoint, second.endpoint)
    # Zero-initialized residual heads predict the physical identity map.
    torch.testing.assert_close(first.prediction, transition.current)
    assert first.selected_tiles.all()
    assert first.overlap_residual.shape == ()


def test_direct_multilevel_execution_is_causal_and_sparse() -> None:
    adapter, transition = make_example(torch.device("cpu"))
    hierarchy = build_two_level_hierarchy(
        adapter, transition, core=8, halo=2, factor=2,
        fine_fraction=.1, seed=13,
    )
    model = small_direct_model(adapter)
    sample = predict_direct_multilevel(model, hierarchy)
    assert sample.endpoint.shape == transition.current.shape
    assert sample.prediction.shape == transition.current.shape
    assert 0 < int(sample.selected_tiles.sum()) < sample.selected_tiles.numel()

    altered = PhysicalTransition(
        transition.previous, transition.current,
        torch.randn_like(transition.target), transition.dt,
        transition.payload, transition.metadata,
    )
    altered_hierarchy = build_two_level_hierarchy(
        adapter, altered, core=8, halo=2, factor=2,
        fine_fraction=.1, seed=13,
    )
    altered_sample = predict_direct_multilevel(model, altered_hierarchy)
    torch.testing.assert_close(sample.prediction, altered_sample.prediction)
    torch.testing.assert_close(sample.selected_tiles, altered_sample.selected_tiles)


def test_experimental_flow_and_projection_loss_backpropagate() -> None:
    adapter, transition = make_example(torch.device("cpu"))
    cover = CartesianCover(64, 64, core=16, halo=4, offset_y=3, offset_x=9)
    level = PreparedLevel(transition, adapter, cover)
    model = small_flow_model(adapter)
    report = combined_loss(model, level)
    report.total.backward()
    assert torch.isfinite(report.total)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_experimental_same_level_flow_and_pnp_shapes() -> None:
    adapter, transition = make_example(torch.device("cpu"))
    level = PreparedLevel(
        transition, adapter, CartesianCover(64, 64, 16, 4, 5, 7)
    )
    model = small_flow_model(adapter)
    integrated = integrate_flow(
        model, level, torch.randn_like(transition.current),
        steps=2, schwarz_iterations=2,
    )
    assert integrated.endpoint.shape == transition.current.shape
    assert integrated.overlap_residuals.numel() == 4
    pnp = pnp_schwarz_sample(
        model, level, torch.randn(3, *transition.current.shape)
    )
    assert pnp.endpoint.shape == transition.current.shape
    assert pnp.overlap_residuals.shape == (3,)
    assert torch.isfinite(pnp.endpoint).all()


def test_experimental_multilevel_pnp_is_system_agnostic() -> None:
    adapter, transition = make_example(torch.device("cpu"))
    hierarchy = build_two_level_hierarchy(
        adapter, transition, core=16, halo=3, factor=2,
        fine_fraction=.4, seed=13,
    )
    model = small_flow_model(adapter)
    sample = pnp_multilevel_sample(
        model, hierarchy, torch.randn(3, *transition.current.shape)
    )
    assert sample.endpoint.shape == transition.current.shape
    assert sample.overlap_residuals.shape == (3,)
    assert 0.0 < float(hierarchy.refinement_mask.mean()) <= 1.0
