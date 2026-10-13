from __future__ import annotations

import pytest
import torch

from codiffuse_Schwarz.tile_communication import (
    CenteredResidualFieldEncoder, RelativeTileAttention,
    ResidualFieldCodeLift, ResidualKernelAttention, ResidualKernelCodeMap,
    StateGatedResidualKernelAttention, StateGatedResidualKernelCodeMap,
    TileGridCommunicator, _relative_edge_features,
)


@pytest.mark.parametrize(
    ("dimension", "shape", "spatial"),
    ((2, (2, 3), (8, 8)), (3, (2, 2, 3), (5, 5, 5))),
)
def test_zero_initialized_communicator_is_identity(
    dimension: int, shape: tuple[int, ...], spatial: tuple[int, ...],
) -> None:
    tiles = torch.randn(2 * torch.tensor(shape).prod().item(), 8, *spatial)
    scalar = torch.randn(tiles.shape[0], 3)
    model = TileGridCommunicator(8, 3, dimension, heads=2)
    torch.testing.assert_close(model(tiles, scalar, shape), tiles)


def test_relative_communication_has_global_gradient() -> None:
    model = TileGridCommunicator(8, 1, 2, heads=2, zero_init=False)
    tiles = torch.randn(6, 8, 4, 4, requires_grad=True)
    scalar = torch.zeros(6, 1)
    output = model(tiles, scalar, (2, 3))
    output[-1].sum().backward()
    assert tiles.grad is not None
    assert float(tiles.grad[0].abs().sum()) > 0.0


def test_no_layout_means_local_identity() -> None:
    model = TileGridCommunicator(8, 2, 3, heads=2, zero_init=False)
    tiles = torch.randn(3, 8, 4, 4, 4)
    scalar = torch.randn(3, 2)
    assert model(tiles, scalar, None) is tiles


def test_incompatible_grid_is_rejected() -> None:
    model = TileGridCommunicator(8, 2, 2, heads=2)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(5, 8, 4, 4), torch.randn(5, 2), (2, 2))


def test_radius_mask_blocks_remote_tile_but_global_attention_sees_it() -> None:
    torch.manual_seed(7)
    model = RelativeTileAttention(8, 2, 2)
    base = torch.zeros(1, 5, 8)
    remote_changed = base.clone()
    remote_changed[0, 4] = torch.tensor(
        [1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -2.0],
    )

    radius_base = model(base, (1, 5), max_graph_radius=1)
    radius_changed = model(remote_changed, (1, 5), max_graph_radius=1)
    torch.testing.assert_close(radius_changed[:, 0], radius_base[:, 0])

    global_base = model(base, (1, 5))
    global_changed = model(remote_changed, (1, 5))
    assert not torch.allclose(global_changed[:, 0], global_base[:, 0])


def test_negative_communication_radius_is_rejected() -> None:
    model = RelativeTileAttention(8, 2, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        model(torch.randn(1, 4, 8), (2, 2), max_graph_radius=-1)


def test_tile_unit_position_is_consistent_when_cover_is_extended() -> None:
    reference = torch.zeros(1)
    small = _relative_edge_features((1, 3), reference, "tile_units")
    large = _relative_edge_features((1, 5), reference, "tile_units")
    torch.testing.assert_close(small[0, 1], large[0, 1], rtol=0.0, atol=0.0)

    small_normalized = _relative_edge_features(
        (1, 3), reference, "cover_normalized",
    )
    large_normalized = _relative_edge_features(
        (1, 5), reference, "cover_normalized",
    )
    assert not torch.equal(small_normalized[0, 1], large_normalized[0, 1])


def test_tile_unit_mode_reaches_state_gated_attention() -> None:
    model = StateGatedResidualKernelCodeMap(
        context_width=7, residual_width=5, token_width=12,
        code_width=4, heads=3, dimension=2, value_rank=2,
        position_mode="tile_units",
    )
    assert model.global_kernel.position_mode == "tile_units"
    context = torch.randn(1, 6, 7)
    residual = torch.randn(1, 6, 5)
    output = model(context, residual, (2, 3))
    assert output.shape == (1, 6, 4)


@pytest.mark.parametrize(
    ("dimension", "grid_shape"), ((2, (2, 3)), (3, (2, 2, 2))),
)
def test_residual_kernel_code_is_linear_and_zero_preserving(
    dimension: int, grid_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(19)
    count = int(torch.tensor(grid_shape).prod())
    model = ResidualKernelCodeMap(
        context_width=7, residual_width=5, token_width=12,
        code_width=4, heads=3, dimension=dimension, zero_init=False,
    ).double()
    context = torch.randn(2, count, 7, dtype=torch.double)
    first = torch.randn(2, count, 5, dtype=torch.double)
    second = torch.randn(2, count, 5, dtype=torch.double)
    scale = -0.37

    zero = model(context, torch.zeros_like(first), grid_shape)
    assert torch.count_nonzero(zero) == 0
    torch.testing.assert_close(
        model(context, first + second, grid_shape),
        model(context, first, grid_shape) + model(context, second, grid_shape),
        rtol=1.0e-12, atol=1.0e-12,
    )
    torch.testing.assert_close(
        model(context, scale * first, grid_shape),
        scale * model(context, first, grid_shape),
        rtol=1.0e-12, atol=1.0e-12,
    )


def test_residual_kernel_radius_blocks_remote_right_hand_side() -> None:
    torch.manual_seed(23)
    model = ResidualKernelAttention(
        context_width=6, residual_width=4, width=8, heads=2, dimension=2,
    )
    context = torch.randn(1, 5, 6)
    base = torch.zeros(1, 5, 4)
    changed = base.clone()
    changed[:, -1] = torch.tensor([1.0, -2.0, 0.5, 0.25])

    local_base = model(context, base, (1, 5), max_graph_radius=1)
    local_changed = model(context, changed, (1, 5), max_graph_radius=1)
    torch.testing.assert_close(local_base[:, 0], local_changed[:, 0])
    global_changed = model(context, changed, (1, 5))
    assert not torch.allclose(global_changed[:, 0], local_base[:, 0])


@pytest.mark.parametrize(
    ("dimension", "grid_shape"), ((2, (2, 3)), (3, (2, 2, 2))),
)
def test_state_gated_residual_kernel_is_linear_and_zero_preserving(
    dimension: int, grid_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(29)
    count = int(torch.tensor(grid_shape).prod())
    model = StateGatedResidualKernelCodeMap(
        context_width=7, residual_width=5, token_width=12,
        code_width=4, heads=3, dimension=dimension, value_rank=3,
        zero_init=False,
    ).double()
    torch.nn.init.normal_(
        model.global_kernel.state_value_coefficients.weight, std=0.1,
    )
    context = torch.randn(2, count, 7, dtype=torch.double)
    first = torch.randn(2, count, 5, dtype=torch.double)
    second = torch.randn(2, count, 5, dtype=torch.double)
    scale = -0.41

    assert torch.count_nonzero(
        model(context, torch.zeros_like(first), grid_shape)
    ) == 0
    torch.testing.assert_close(
        model(context, first + second, grid_shape),
        model(context, first, grid_shape) + model(context, second, grid_shape),
        rtol=1.0e-12, atol=1.0e-12,
    )
    torch.testing.assert_close(
        model(context, scale * first, grid_shape),
        scale * model(context, first, grid_shape),
        rtol=1.0e-12, atol=1.0e-12,
    )


def test_state_gated_attention_is_exact_identity_lift_with_live_gradient() -> None:
    torch.manual_seed(31)
    base = ResidualKernelAttention(
        context_width=6, residual_width=5, width=8, heads=2, dimension=2,
    )
    lifted = StateGatedResidualKernelAttention(
        context_width=6, residual_width=5, width=8, heads=2, dimension=2,
        value_rank=3,
    )
    incompatible = lifted.load_state_dict(base.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "state_value_coefficients.weight", "state_value_basis",
    }
    context = torch.randn(2, 6, 6)
    residual = torch.randn(2, 6, 5)
    expected = base(context, residual, (2, 3))
    actual = lifted(context, residual, (2, 3))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    actual.square().mean().backward()
    coefficient_gradient = lifted.state_value_coefficients.weight.grad
    assert coefficient_gradient is not None
    assert torch.isfinite(coefficient_gradient).all()
    assert torch.count_nonzero(coefficient_gradient) > 0

    # Deployment uses the exact parent floating-point path while the gate is
    # closed.  This matters on large CUDA covers, where evaluating and adding
    # a numerically zero dynamic GEMM need not be bitwise neutral.
    base.eval(); lifted.eval()
    with torch.no_grad():
        expected = base(context, residual, (2, 3))
        actual = lifted(context, residual, (2, 3))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_state_gated_value_changes_with_source_state_only_when_residual_exists() -> None:
    torch.manual_seed(37)
    model = StateGatedResidualKernelAttention(
        context_width=6, residual_width=5, width=8, heads=2, dimension=2,
        value_rank=3,
    )
    torch.nn.init.normal_(model.state_value_coefficients.weight, std=0.2)
    context = torch.randn(1, 6, 6)
    changed_context = context.clone()
    changed_context[:, -1, 0] += 2.0
    residual = torch.zeros(1, 6, 5)
    residual[:, -1] = torch.randn(5)
    first = model._state_projected_value(context, residual)
    second = model._state_projected_value(changed_context, residual)
    assert not torch.allclose(first[:, -1], second[:, -1])
    assert torch.count_nonzero(
        model._state_projected_value(changed_context, torch.zeros_like(residual))
    ) == 0


@pytest.mark.parametrize(
    ("dimension", "grid_shape", "spatial"),
    ((2, (2, 3), (9, 11)), (3, (2, 2, 2), (7, 8, 9))),
)
def test_centered_residual_field_lift_is_exactly_zero_at_compatibility(
    dimension: int,
    grid_shape: tuple[int, ...],
    spatial: tuple[int, ...],
) -> None:
    torch.manual_seed(41)
    batch = 2
    count = int(torch.tensor(grid_shape).prod())
    encoder = CenteredResidualFieldEncoder(
        field_channels=3, token_width=7, dimension=dimension,
        hidden_width=5, bins_per_axis=2,
    )
    zero = torch.zeros(batch * count, 3, *spatial)
    assert torch.count_nonzero(encoder(zero)) == 0

    lift = ResidualFieldCodeLift(
        field_channels=3, sketch_width=7, token_width=12,
        code_width=4, heads=3, dimension=dimension,
        hidden_width=5, bins_per_axis=2, zero_init=False,
    )
    logits = torch.randn(batch, 3, count, count)
    weight = logits.softmax(-1)
    assert torch.count_nonzero(lift(weight, zero, grid_shape)) == 0


def test_residual_field_lift_reuses_one_kernel_and_has_live_gradient() -> None:
    torch.manual_seed(43)
    grid_shape = (2, 3)
    batch, heads, count = 2, 3, 6
    lift = ResidualFieldCodeLift(
        field_channels=3, sketch_width=8, token_width=12,
        code_width=5, heads=heads, dimension=2,
        hidden_width=6, bins_per_axis=2,
    )
    weight = torch.randn(batch, heads, count, count).softmax(-1)
    defect = torch.randn(batch * count, 3, 9, 11)
    # A closed lift is an exact identity addition.
    output = lift(weight, defect, grid_shape)
    assert torch.count_nonzero(output) == 0
    output.square().mean().backward()
    # Squared zero output has a zero derivative.  A supervised residual target
    # supplies the live first gradient used by the actual identity-lift run.
    lift.zero_grad(set_to_none=True)
    target = torch.randn_like(output)
    torch.nn.functional.mse_loss(lift(weight, defect, grid_shape), target).backward()
    gradient = lift.code_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_residual_field_lift_accepts_centered_auxiliary_coordinates() -> None:
    torch.manual_seed(47)
    grid_shape = (2, 3)
    batch, heads, count = 2, 2, 6
    lift = ResidualFieldCodeLift(
        field_channels=3, sketch_width=7, token_width=12,
        code_width=9, heads=heads, dimension=2,
        hidden_width=6, bins_per_axis=2, auxiliary_residual_width=4,
        zero_init=False,
    )
    weight = torch.randn(batch, heads, count, count).softmax(-1)
    defect = torch.randn(batch * count, 3, 7, 9)
    auxiliary = torch.randn(batch, count, 4)
    output = lift(
        weight, defect, grid_shape, auxiliary_residual=auxiliary,
    )
    assert output.shape == (batch, count, 9)
    zero = lift(
        weight, torch.zeros_like(defect), grid_shape,
        auxiliary_residual=torch.zeros_like(auxiliary),
    )
    assert torch.count_nonzero(zero) == 0
    with pytest.raises(ValueError, match="auxiliary residual"):
        lift(weight, defect, grid_shape)
