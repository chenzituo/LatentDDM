"""Contract tests for the end-to-end Darcy point/anchor baseline."""

import torch

from .masked_anchor_operator import MaskedAnchorOperator


def _model() -> MaskedAnchorOperator:
    return MaskedAnchorOperator(
        5, 2, width=32, depth=2, heads=4, anchors=8, mlp_ratio=2,
    )


def test_shape_mask_and_gradient() -> None:
    model = _model()
    spatial = torch.randn(2, 5, 12, 10)
    scalar = torch.randn(2, 2)
    mask = torch.ones(2, 1, 12, 10)
    mask[:, :, :3, :4] = 0
    output = model(spatial, scalar, mask)
    assert output.shape == (2, 1, 12, 10)
    assert torch.count_nonzero(output[:, :, :3, :4]) == 0
    output.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_invalid_padding_cannot_change_active_prediction() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    spatial = torch.randn(1, 5, 8, 8)
    scalar = torch.randn(1, 2)
    mask = torch.ones(1, 1, 8, 8)
    mask[:, :, :2] = 0
    changed = spatial.clone()
    changed[:, :, :2] = 1.0e4 * torch.randn_like(changed[:, :, :2])
    with torch.no_grad():
        reference = model(spatial, scalar, mask)
        perturbed = model(changed, scalar, mask)
    active = mask.bool().expand_as(reference)
    assert torch.allclose(reference[active], perturbed[active], atol=1e-6, rtol=1e-6)
