from __future__ import annotations

import torch

from .field_token_composer import DarcyFieldTokenComposer
from .residual_port_composer import LinearResidualResponseChart
from .train_residual_port_chart import FrozenContext
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover


def test_diagonal_graph_recovers_exact_local_core_field() -> None:
    torch.manual_seed(71)
    cover = CartesianCover(8, 8, core=4, halo=1)
    batch, tiles, size = 2, cover.tile_count, cover.tile_size
    context = FrozenContext(
        local=torch.randn(batch * tiles, 1, size, size),
        physical=torch.randn(batch * tiles, 5, size, size),
        scalar=torch.randn(batch * tiles, 3),
        valid=torch.ones(batch * tiles, 1, size, size),
        target_normalized=torch.randn(batch, 1, 8, 8),
        target_tiles=torch.randn(batch * tiles, 1, size, size),
        oracle_code=torch.randn(batch, tiles, 4),
        cover=cover,
    )
    chart = LinearResidualResponseChart(1, 5, 3, 4, response_width=8)
    composer = DarcyFieldTokenComposer(
        5, 3, 4, token_width=16, heads=4, communication_depth=2,
    ).eval()
    diagonal = torch.eye(tiles, dtype=torch.bool)
    output = composer(
        chart, context, torch.ones(batch, 1, 8, 8), graph_mask=diagonal,
    )
    expected = cover.assemble_ras(context.local, batch)
    assert torch.count_nonzero(output.latent.correction_code) == 0
    torch.testing.assert_close(output.assembled, expected, rtol=0.0, atol=0.0)


def test_spatial_ports_are_exact_parent_at_initialization() -> None:
    torch.manual_seed(72)
    cover = CartesianCover(8, 8, core=4, halo=1)
    batch, tiles, size = 1, cover.tile_count, cover.tile_size
    context = FrozenContext(
        local=torch.randn(batch * tiles, 1, size, size),
        physical=torch.randn(batch * tiles, 5, size, size),
        scalar=torch.randn(batch * tiles, 3),
        valid=torch.ones(batch * tiles, 1, size, size),
        target_normalized=torch.randn(batch, 1, 8, 8),
        target_tiles=torch.randn(batch * tiles, 1, size, size),
        oracle_code=torch.randn(batch, tiles, 4),
        cover=cover,
    )
    chart = LinearResidualResponseChart(1, 5, 3, 4, response_width=8)
    base = DarcyFieldTokenComposer(
        5, 3, 4, token_width=16, heads=4, communication_depth=2,
    ).eval()
    child = DarcyFieldTokenComposer(
        5, 3, 4, token_width=16, heads=4, communication_depth=2,
        spatial_port_bins=2,
    ).eval()
    incompatible = child.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(name.startswith("spatial_port_") for name in incompatible.missing_keys)
    mask = torch.ones(batch, 1, 8, 8)
    parent_output = base(chart, context, mask)
    child_output = child(chart, context, mask)
    torch.testing.assert_close(
        child_output.assembled, parent_output.assembled, rtol=0.0, atol=0.0,
    )


def test_trained_spatial_ports_keep_diagonal_graph_exact() -> None:
    torch.manual_seed(73)
    cover = CartesianCover(8, 8, core=4, halo=1)
    batch, tiles, size = 1, cover.tile_count, cover.tile_size
    context = FrozenContext(
        local=torch.randn(batch * tiles, 1, size, size),
        physical=torch.randn(batch * tiles, 5, size, size),
        scalar=torch.randn(batch * tiles, 3),
        valid=torch.ones(batch * tiles, 1, size, size),
        target_normalized=torch.randn(batch, 1, 8, 8),
        target_tiles=torch.randn(batch * tiles, 1, size, size),
        oracle_code=torch.randn(batch, tiles, 4),
        cover=cover,
    )
    chart = LinearResidualResponseChart(1, 5, 3, 4, response_width=8)
    composer = DarcyFieldTokenComposer(
        5, 3, 4, token_width=16, heads=4, communication_depth=2,
        spatial_port_bins=2,
    ).eval()
    with torch.no_grad():
        for parameter in composer.spatial_port_communication.parameters():
            parameter.normal_()
    diagonal = torch.eye(tiles, dtype=torch.bool)
    output = composer(
        chart, context, torch.ones(batch, 1, 8, 8), graph_mask=diagonal,
    )
    expected = cover.assemble_ras(context.local, batch)
    assert torch.count_nonzero(output.latent.correction_code) == 0
    torch.testing.assert_close(output.assembled, expected, rtol=0.0, atol=0.0)
