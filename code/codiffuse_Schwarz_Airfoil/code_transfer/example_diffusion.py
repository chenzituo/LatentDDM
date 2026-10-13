"""Minimal example adapter for a variable-coefficient diffusion transition."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .adapter import ConditionBatch, PhysicalTransition, PhysicsAdapter
from .direct import direct_endpoint_loss, predict_direct_multilevel
from .hierarchy import build_two_level_hierarchy
from .model import (
    DirectEndpointOperator,
    DirectEndpointRegressor,
    NeuralEndpointProjector,
)


Tensor = torch.Tensor


class DiffusionAdapter(PhysicsAdapter):
    """Example only: applications should estimate normalization from train data."""

    state_channels = 1
    spatial_condition_channels = 7
    scalar_condition_dim = 2

    def __init__(self, delta_scale: float = .05) -> None:
        self.delta_scale = delta_scale

    def conditions(
        self, transition: PhysicalTransition, endpoint_phase: Tensor, level_scale: float,
    ) -> ConditionBatch:
        coefficient = transition.payload["coefficient"]
        source = transition.payload["source"]
        batch, _, height, width = transition.current.shape
        y = torch.linspace(-1, 1, height, device=transition.current.device)
        x = torch.linspace(-1, 1, width, device=transition.current.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy))[None].expand(batch, -1, -1, -1)
        spatial = torch.cat((
            transition.current,
            (transition.current - transition.previous) / self.delta_scale,
            coefficient, source, coordinates,
            torch.ones_like(source) * endpoint_phase[:, None, None, None],
        ), dim=1)
        scalars = torch.stack((transition.dt, torch.full_like(transition.dt, level_scale)), 1)
        return ConditionBatch(spatial, scalars)

    def normalized_target(self, transition: PhysicalTransition) -> Tensor:
        if transition.target is None:
            raise ValueError("training target is unavailable")
        return (transition.target - transition.current) / self.delta_scale

    def apply_endpoint(
        self, transition: PhysicalTransition, normalized_endpoint: Tensor,
    ) -> Tensor:
        return transition.current + self.delta_scale * normalized_endpoint

    def refinement_indicator(self, transition: PhysicalTransition) -> Tensor:
        state = transition.current
        laplacian = (
            -4 * state + torch.roll(state, 1, -1) + torch.roll(state, -1, -1)
            + torch.roll(state, 1, -2) + torch.roll(state, -1, -2)
        ).abs()
        return laplacian + transition.payload["source"].abs()


def make_example(device: torch.device) -> tuple[DiffusionAdapter, PhysicalTransition]:
    size = 64
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device), indexing="ij",
    )
    current = torch.exp(-12 * (x.square() + y.square()))[None, None]
    previous = torch.exp(-13 * (x.square() + y.square()))[None, None]
    coefficient = (0.5 + 0.25 * torch.sin(3 * x) * torch.cos(2 * y))[None, None]
    source = torch.exp(-80 * ((x - .35).square() + (y + .2).square()))[None, None]
    # A real dataset supplies this target. This explicit step is only a compact
    # executable demonstration of the adapter contract.
    laplacian = (
        -4 * current + torch.roll(current, 1, -1) + torch.roll(current, -1, -1)
        + torch.roll(current, 1, -2) + torch.roll(current, -1, -2)
    )
    target = current + .02 * (coefficient * laplacian + source)
    transition = PhysicalTransition(
        previous, current, target, torch.tensor([.02], device=device),
        {"coefficient": coefficient, "source": source},
    )
    return DiffusionAdapter(), transition


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter, transition = make_example(device)
    hierarchy = build_two_level_hierarchy(
        adapter, transition, core=8, halo=2, fine_fraction=.10, seed=7
    )
    proposal = DirectEndpointRegressor(
        adapter.state_channels, adapter.spatial_condition_channels + 1,
        adapter.scalar_condition_dim, width=32, depth=2,
    )
    projector = NeuralEndpointProjector(
        adapter.state_channels, adapter.spatial_condition_channels + 1,
        adapter.scalar_condition_dim, width=32, depth=2, cell_attention=False,
    )
    model = DirectEndpointOperator(proposal, projector).to(device)
    report = direct_endpoint_loss(model, hierarchy.fine)
    report.total.backward()
    sample = predict_direct_multilevel(model, hierarchy)
    print({
        "loss": float(report.total.detach()),
        "prediction_shape": tuple(sample.prediction.shape),
        "fine_tile_fraction": float(sample.selected_tiles.float().mean()),
        "overlap_residual": float(sample.overlap_residual),
    })


if __name__ == "__main__":
    main()
