"""Full-domain deterministic MPP baseline with CoDiffuse conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from codiffuse_Schwarz_Airfoil.coupled_trajectory import ControlledTransition
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics
from codiffuse_Schwarz_Airfoil.model import MPPFlow


Tensor = torch.Tensor

MPP_INPUT_CHANNELS = {
    "physical": (
        "q_n_u", "q_n_v", "q_n_p", "sdf_n", "sdf_np1",
        "wall_u_n", "wall_v_n", "wall_u_np1", "wall_v_np1",
        "x", "y", "history_du", "history_dv", "history_dp", "valid",
    ),
    "scalar": (
        "absolute_phase", "thickness", "pivot", "angle_n", "omega_n",
        "angle_np1", "omega_np1", "accel_n", "accel_np1", "dt_over_0p01",
        "clean_endpoint_time", "dt_over_dx", "viscous_step_scale",
        "domain_center_x", "domain_center_y",
    ),
}


@dataclass(frozen=True)
class FullDomainCover:
    """One rectangular domain token satisfying ``ControlledTransition``."""

    height: int
    width: int
    core: int = 32
    tile_count: int = 1

    def tile(self, field: Tensor, **_: object) -> Tensor:
        if field.shape[-2:] != (self.height, self.width):
            raise ValueError("field and full-domain cover shapes differ")
        return field

    def valid_tiles(
        self, batch: int, device: torch.device, dtype: torch.dtype,
    ) -> Tensor:
        return torch.ones(batch, 1, self.height, self.width, device=device, dtype=dtype)

    def centers(
        self, x0: float, y0: float, dx: float, device: torch.device,
    ) -> Tensor:
        return torch.tensor(
            [[x0 + .5 * self.width * dx, y0 + .5 * self.height * dx]],
            device=device, dtype=torch.float32,
        )


def build_fullfield_mpp(
    statistics: FieldStatistics,
    *,
    physical_channels: int = 15,
    scalar_dim: int = 15,
    width: int = 256,
    depth: int = 10,
    heads: int = 8,
    transport_skip: bool = True,
    device: torch.device | None = None,
) -> tuple[MPPFlow, dict[str, object]]:
    """Construct a parameter-matched deterministic version of local MPPFlow."""
    if physical_channels != 15 or scalar_dim != 15:
        raise ValueError("airfoil MPP requires 15 physical and 15 scalar channels")
    model = MPPFlow(
        physical_channels=physical_channels,
        scalar_dim=scalar_dim,
        width=width,
        depth=depth,
        heads=heads,
        transport_skip=transport_skip,
    )
    model.set_endpoint_statistics(
        statistics.state_mean, statistics.state_std,
        statistics.delta_std, statistics.delta_mean,
    )
    if device is not None:
        model = model.to(device)
    return model, {
        "physical_channels": physical_channels,
        "scalar_dim": scalar_dim,
        "width": width,
        "depth": depth,
        "heads": heads,
        "transport_skip": transport_skip,
    }


def load_fullfield_mpp(checkpoint: dict, device: torch.device) -> MPPFlow:
    statistics = FieldStatistics.from_json(checkpoint["statistics"])
    model, _ = build_fullfield_mpp(
        statistics, device=device, **checkpoint["model_args"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def mpp_conditions(
    batch: dict[str, Tensor], dataset, statistics: FieldStatistics,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    q_n = batch["q_n"]
    cover = FullDomainCover(q_n.shape[-2], q_n.shape[-1])
    transition = ControlledTransition(batch, statistics, dataset, cover, device)
    return transition.endpoint_conditions()


def normalized_mpp_delta(
    model: MPPFlow,
    batch: dict[str, Tensor],
    dataset,
    statistics: FieldStatistics,
    device: torch.device,
) -> Tensor:
    """Predict one normalized physical increment with no tiling or stitching."""
    physical, scalar = mpp_conditions(batch, dataset, statistics, device)
    flow_state = torch.zeros(
        physical.shape[0], 3, *physical.shape[-2:],
        device=device, dtype=physical.dtype,
    )
    flow_time = torch.ones(
        physical.shape[0], device=device, dtype=physical.dtype,
    )
    return model(flow_state, physical, scalar, flow_time)
