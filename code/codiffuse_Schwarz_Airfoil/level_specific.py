"""Resolution-specific local operators for a Cartesian AMR hierarchy."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from codiffuse_Schwarz_Airfoil.model import FiLMResidualBlock
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import load_pretrained


class LevelSpecificOperators(torch.nn.Module):
    """One cell-shared operator per AMR spacing, without global field layers."""

    def __init__(
        self, coarse_operator: torch.nn.Module, fine_operator: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.coarse_operator = coarse_operator
        self.fine_operator = fine_operator

    def set_endpoint_statistics(self, *args) -> None:
        self.coarse_operator.set_endpoint_statistics(*args)
        self.fine_operator.set_endpoint_statistics(*args)

    @property
    def gradient_checkpointing(self) -> bool:
        return bool(self.coarse_operator.gradient_checkpointing)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, enabled: bool) -> None:
        self.coarse_operator.gradient_checkpointing = enabled
        self.fine_operator.gradient_checkpointing = enabled


class GuardSpecificOperators(torch.nn.Module):
    """Specialize exterior coarse cells while preserving the inner hierarchy."""

    def __init__(
        self, guard_operator: torch.nn.Module, shared_operator: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.guard_operator = guard_operator
        self.shared_operator = shared_operator

    def set_endpoint_statistics(self, *args) -> None:
        self.guard_operator.set_endpoint_statistics(*args)
        self.shared_operator.set_endpoint_statistics(*args)

    @property
    def gradient_checkpointing(self) -> bool:
        return bool(self.guard_operator.gradient_checkpointing)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, enabled: bool) -> None:
        self.guard_operator.gradient_checkpointing = enabled
        self.shared_operator.gradient_checkpointing = enabled


class GuardedInterfaceOperators(torch.nn.Module):
    """Compose an exterior L4 expert with an inner geometry-interface EoS."""

    def __init__(
        self,
        guard_operator: torch.nn.Module,
        interface_operator: torch.nn.Module,
    ) -> None:
        super().__init__()
        if not hasattr(interface_operator, "correct_surface_level"):
            raise ValueError("inner operator must provide a surface-level EoS")
        self.guard_operator = guard_operator
        self.interface_operator = interface_operator

    @property
    def shared_operator(self):
        return self.interface_operator.shared_operator

    def correct_surface_level(self, *args, **kwargs):
        return self.interface_operator.correct_surface_level(*args, **kwargs)

    def set_endpoint_statistics(self, *args) -> None:
        self.guard_operator.set_endpoint_statistics(*args)
        self.interface_operator.set_endpoint_statistics(*args)

    @property
    def gradient_checkpointing(self) -> bool:
        return bool(self.shared_operator.gradient_checkpointing)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, enabled: bool) -> None:
        self.guard_operator.gradient_checkpointing = enabled
        self.interface_operator.gradient_checkpointing = enabled


class GeometrySpecificOperators(torch.nn.Module):
    """Blend a cut-cell endpoint EoS into an otherwise frozen local operator."""

    def __init__(
        self,
        geometry_operator: torch.nn.Module,
        shared_operator: torch.nn.Module,
        geometry_band_cells: float = 8.0,
        body_frame: bool = False,
        tile_routing: bool = False,
    ) -> None:
        super().__init__()
        if geometry_band_cells <= 0.0:
            raise ValueError("geometry band must be positive")
        self.geometry_operator = geometry_operator
        self.shared_operator = shared_operator
        self.geometry_band_cells = geometry_band_cells
        self.body_frame = body_frame
        self.tile_routing = tile_routing

    def set_endpoint_statistics(self, *args) -> None:
        self.geometry_operator.set_endpoint_statistics(*args)
        self.shared_operator.set_endpoint_statistics(*args)

    @property
    def gradient_checkpointing(self) -> bool:
        return bool(self.geometry_operator.gradient_checkpointing)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, enabled: bool) -> None:
        self.geometry_operator.gradient_checkpointing = enabled
        self.shared_operator.gradient_checkpointing = enabled

    @property
    def endpoint_learned_projection(self) -> bool:
        return bool(self.shared_operator.endpoint_learned_projection)

    @property
    def endpoint_eos_direct(self) -> bool:
        return bool(self.shared_operator.endpoint_eos_direct)

    @property
    def projection_attention(self):
        return self.shared_operator.projection_attention

    @property
    def projection_spatial_attention(self):
        return self.shared_operator.projection_spatial_attention

    def geometry_weight(self, physical: torch.Tensor) -> torch.Tensor:
        if physical.shape[1] < 5:
            raise ValueError("geometry-role routing requires two SDF channels")
        # ControlledTransition normalizes SDF by 4 * core * dx, core=32.
        distance_cells = 128.0 * torch.minimum(
            physical[:, 3:4].abs(), physical[:, 4:5].abs(),
        )
        swept = torch.exp(-distance_cells / self.geometry_band_cells)
        solid = ((physical[:, 3:4] < 0.0) | (physical[:, 4:5] < 0.0)).to(
            swept.dtype,
        )
        return torch.maximum(swept, solid) * physical[:, -1:]

    def geometry_tile_role(self, physical: torch.Tensor) -> torch.Tensor:
        """Route a complete Cartesian cell when its valid support meets the body."""
        weight = self.geometry_weight(physical)
        # geometry_weight is exp(-distance / band).  A 1/e cutoff therefore
        # means actual band support, rather than the long exponential tail.
        return (weight.amax(dim=(-2, -1), keepdim=True) >= .367879).to(weight)

    def body_frame_conditions(
        self,
        physical: torch.Tensor,
        scalar: torch.Tensor,
        motion_physical: torch.Tensor,
        motion_scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pull the source state to the destination pose for cut-cell evolution.

        The returned transport is in normalized-delta units and is used to
        construct the registered input state.  The geometry operator still
        returns the complete Eulerian increment: directly adding rigid
        transport would overwrite the small Brinkman wall mismatch needed by
        the force diagnostic.
        """
        if motion_physical.shape[1] < 15:
            raise ValueError("body-frame routing requires two-step airfoil fields")
        transport = self.shared_operator._prescribed_motion_increment(
            motion_physical, motion_scalar,
        )
        state_ratio = (
            self.shared_operator.endpoint_delta_std.to(motion_physical)
            / self.shared_operator.endpoint_state_std.to(motion_physical)
                .clamp_min(1.0e-12)
        )
        canonical = motion_physical.clone()
        canonical[:, :3] = canonical[:, :3] + transport * state_ratio

        # In the destination body frame the geometry and wall are stationary.
        # Destination wall velocity remains explicit in channels 5:9.  The
        # source/destination scalar pair is retained so angular acceleration
        # and displacement remain observable to the EoS.
        canonical[:, 3:4] = canonical[:, 4:5]
        canonical[:, 5:7] = canonical[:, 7:9]
        return canonical, scalar, transport

    def correct_endpoint_fields(self, *args, **kwargs):
        physical = args[1] if len(args) > 1 else kwargs["physical"]
        shared, shared_vorticity = self.shared_operator.correct_endpoint_fields(
            *args, **kwargs,
        )
        if self.body_frame:
            motion_physical = kwargs.get("motion_physical")
            motion_scalar = kwargs.get("motion_scalar")
            if len(args) > 5:
                motion_physical, motion_scalar = args[5], args[6]
            if motion_physical is None or motion_scalar is None:
                raise ValueError("body-frame EoS requires endpoint motion conditions")
            canonical, canonical_scalar, transport = self.body_frame_conditions(
                physical, args[2] if len(args) > 2 else kwargs["scalar"],
                motion_physical, motion_scalar,
            )
            geometry_args = list(args)
            if len(geometry_args) > 1:
                geometry_args[1] = canonical
                geometry_args[2] = canonical_scalar
                geometry, geometry_vorticity = (
                    self.geometry_operator.correct_endpoint_fields(
                        *geometry_args, **kwargs,
                    )
                )
            else:
                geometry_kwargs = dict(kwargs)
                geometry_kwargs["physical"] = canonical
                geometry_kwargs["scalar"] = canonical_scalar
                geometry, geometry_vorticity = (
                    self.geometry_operator.correct_endpoint_fields(
                        **geometry_kwargs,
                    )
                )
            weight = self.geometry_tile_role(motion_physical).to(shared)
        else:
            geometry, geometry_vorticity = (
                self.geometry_operator.correct_endpoint_fields(*args, **kwargs)
            )
            weight = (
                self.geometry_tile_role(physical)
                if self.tile_routing else self.geometry_weight(physical)
            ).to(shared)
        result = shared + weight * (geometry - shared)
        if shared_vorticity is None or geometry_vorticity is None:
            vorticity = shared_vorticity
        else:
            vorticity = shared_vorticity + weight[:, 0] * (
                geometry_vorticity - shared_vorticity
            )
        return result, vorticity

    def correct_endpoint(
        self, endpoint: torch.Tensor, physical: torch.Tensor, scalar: torch.Tensor,
    ) -> torch.Tensor:
        return self.correct_endpoint_fields(endpoint, physical, scalar)[0]


class GeometryResidualOperators(torch.nn.Module):
    """Small moving-cut-cell residual around a frozen shared endpoint EoS."""

    def __init__(
        self,
        shared_operator: torch.nn.Module,
        *,
        width: int = 64,
        depth: int = 4,
        geometry_band_cells: float = 8.0,
        preserve_destination_solid: bool = True,
    ) -> None:
        super().__init__()
        if min(width, depth) <= 0 or geometry_band_cells <= 0.0:
            raise ValueError("geometry adapter dimensions and band must be positive")
        self.shared_operator = shared_operator
        self.geometry_width = width
        self.geometry_depth = depth
        self.geometry_band_cells = geometry_band_cells
        self.preserve_destination_solid = preserve_destination_solid
        self.geometry_stem = nn.Conv2d(
            shared_operator.physical_channels + 6, width, kernel_size=5, padding=2,
        )
        self.geometry_condition = nn.Sequential(
            nn.Linear(shared_operator.scalar_dim + 6, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.geometry_blocks = nn.ModuleList(
            FiLMResidualBlock(width, (1, 2, 4)[index % 3])
            for index in range(depth)
        )
        self.geometry_output = nn.Conv2d(width, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.geometry_output.weight)
        nn.init.zeros_(self.geometry_output.bias)
        self.gradient_checkpointing = False

    def set_endpoint_statistics(self, *args) -> None:
        self.shared_operator.set_endpoint_statistics(*args)

    @property
    def endpoint_learned_projection(self) -> bool:
        return bool(self.shared_operator.endpoint_learned_projection)

    @property
    def endpoint_eos_direct(self) -> bool:
        return bool(self.shared_operator.endpoint_eos_direct)

    @property
    def projection_attention(self):
        return self.shared_operator.projection_attention

    @property
    def projection_spatial_attention(self):
        return self.shared_operator.projection_spatial_attention

    def geometry_weight(self, physical: torch.Tensor) -> torch.Tensor:
        distance_cells = 128.0 * torch.minimum(
            physical[:, 3:4].abs(), physical[:, 4:5].abs(),
        )
        swept = torch.exp(-distance_cells / self.geometry_band_cells)
        if self.preserve_destination_solid:
            destination_fluid = (physical[:, 4:5] >= 0.0).to(swept.dtype)
            swept = swept * destination_fluid
        return swept * physical[:, -1:]

    def correct_endpoint_fields(self, *args, **kwargs):
        endpoint = args[0] if args else kwargs["endpoint"]
        physical = args[1] if len(args) > 1 else kwargs["physical"]
        scalar = args[2] if len(args) > 2 else kwargs["scalar"]
        motion_physical = kwargs.get("motion_physical")
        motion_scalar = kwargs.get("motion_scalar")
        if len(args) > 5:
            motion_physical, motion_scalar = args[5], args[6]
        shared, vorticity = self.shared_operator.correct_endpoint_fields(
            *args, **kwargs,
        )
        normalized, statistics = self.shared_operator.revin_previous_field(physical)
        if motion_physical is None or motion_scalar is None:
            transport = torch.zeros_like(endpoint)
        else:
            transport = self.shared_operator._prescribed_motion_increment(
                motion_physical, motion_scalar,
            )
        condition = self.geometry_condition(torch.cat((scalar, statistics), dim=1))
        hidden = self.geometry_stem(torch.cat(
            (normalized, shared, transport), dim=1,
        ))
        for block in self.geometry_blocks:
            hidden = block(hidden, condition)
        correction = 0.5 * torch.tanh(self.geometry_output(F.silu(hidden)))
        result = shared + self.geometry_weight(physical).to(shared) * correction
        return result, vorticity

    def correct_endpoint(
        self, endpoint: torch.Tensor, physical: torch.Tensor, scalar: torch.Tensor,
    ) -> torch.Tensor:
        return self.correct_endpoint_fields(endpoint, physical, scalar)[0]


def clone_single_level_checkpoint(
    checkpoint: dict, device: torch.device,
) -> tuple[LevelSpecificOperators, dict]:
    coarse, spec = load_pretrained(checkpoint, device, history_steps=2)
    fine = deepcopy(coarse)
    return LevelSpecificOperators(coarse, fine), dict(spec) | {
        "level_specific_operators": True,
    }


def clone_guard_operator_checkpoint(
    checkpoint: dict, device: torch.device,
) -> tuple[GuardSpecificOperators, dict]:
    shared, spec = load_pretrained(checkpoint, device, history_steps=2)
    guard = deepcopy(shared)
    return GuardSpecificOperators(guard, shared), dict(spec) | {
        "guard_specific_operator": True,
    }


def clone_geometry_operator_checkpoint(
    checkpoint: dict,
    device: torch.device,
    geometry_band_cells: float = 8.0,
    body_frame: bool = False,
) -> tuple[GeometrySpecificOperators, dict]:
    shared, spec = load_pretrained(checkpoint, device, history_steps=2)
    geometry = deepcopy(shared)
    return GeometrySpecificOperators(
        geometry, shared, geometry_band_cells, body_frame,
    ), dict(spec) | {
        "geometry_specific_operator": True,
        "geometry_band_cells": geometry_band_cells,
        "geometry_body_frame": body_frame,
    }


def clone_geometry_residual_checkpoint(
    checkpoint: dict,
    device: torch.device,
    *,
    width: int = 64,
    depth: int = 4,
    geometry_band_cells: float = 8.0,
    preserve_destination_solid: bool = True,
) -> tuple[GeometryResidualOperators, dict]:
    shared, spec = load_pretrained(checkpoint, device, history_steps=2)
    model = GeometryResidualOperators(
        shared, width=width, depth=depth,
        geometry_band_cells=geometry_band_cells,
        preserve_destination_solid=preserve_destination_solid,
    ).to(device)
    return model, dict(spec) | {
        "geometry_residual_operator": True,
        "geometry_width": width,
        "geometry_depth": depth,
        "geometry_band_cells": geometry_band_cells,
        "geometry_preserve_destination_solid": preserve_destination_solid,
    }


def load_geometry_residual_checkpoint(
    checkpoint: dict, device: torch.device,
) -> tuple[GeometryResidualOperators, dict]:
    spec = dict(checkpoint.get("model_spec", {}))
    if not bool(spec.get("geometry_residual_operator", False)):
        return clone_geometry_residual_checkpoint(checkpoint, device)
    prefix = "shared_operator."
    shared_state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint["model"].items() if key.startswith(prefix)
    }
    if not shared_state:
        raise ValueError("geometry residual checkpoint is missing shared operator")
    payload = dict(checkpoint)
    payload["model"] = shared_state
    payload["model_spec"] = {
        key: value for key, value in spec.items()
        if key not in (
            "geometry_residual_operator", "geometry_width", "geometry_depth",
            "geometry_band_cells", "geometry_preserve_destination_solid",
        )
    }
    shared, _ = load_pretrained(payload, device, history_steps=2)
    model = GeometryResidualOperators(
        shared, width=int(spec.get("geometry_width", 64)),
        depth=int(spec.get("geometry_depth", 4)),
        geometry_band_cells=float(spec.get("geometry_band_cells", 8.0)),
        preserve_destination_solid=bool(spec.get(
            "geometry_preserve_destination_solid", True,
        )),
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model"])
    return model, spec


def load_geometry_specific_checkpoint(
    checkpoint: dict, device: torch.device,
) -> tuple[GeometrySpecificOperators, dict]:
    spec = dict(checkpoint.get("model_spec", {}))
    if not bool(spec.get("geometry_specific_operator", False)):
        return clone_geometry_operator_checkpoint(checkpoint, device)
    models = []
    for name in ("geometry_operator", "shared_operator"):
        prefix = f"{name}."
        state = {
            key.removeprefix(prefix): value
            for key, value in checkpoint["model"].items()
            if key.startswith(prefix)
        }
        if not state:
            raise ValueError(f"geometry-specific checkpoint is missing {name}")
        payload = dict(checkpoint)
        payload["model"] = state
        payload["model_spec"] = {
            key: value for key, value in spec.items()
            if key not in (
                "geometry_specific_operator", "geometry_band_cells",
                "geometry_body_frame",
            )
        }
        model, _ = load_pretrained(payload, device, history_steps=2)
        models.append(model)
    return GeometrySpecificOperators(
        *models, geometry_band_cells=float(spec.get("geometry_band_cells", 8.0)),
        body_frame=bool(spec.get("geometry_body_frame", False)),
    ), spec


def load_guard_specific_checkpoint(
    checkpoint: dict, device: torch.device,
) -> tuple[GuardSpecificOperators, dict]:
    spec = dict(checkpoint.get("model_spec", {}))
    if not bool(spec.get("guard_specific_operator", False)):
        return clone_guard_operator_checkpoint(checkpoint, device)
    models = []
    for name in ("guard_operator", "shared_operator"):
        prefix = f"{name}."
        state = {
            key.removeprefix(prefix): value
            for key, value in checkpoint["model"].items()
            if key.startswith(prefix)
        }
        if not state:
            raise ValueError(f"guard-specific checkpoint is missing {name}")
        payload = dict(checkpoint)
        payload["model"] = state
        payload["model_spec"] = {
            key: value for key, value in spec.items()
            if key != "guard_specific_operator"
        }
        model, _ = load_pretrained(payload, device, history_steps=2)
        models.append(model)
    return GuardSpecificOperators(*models), spec


def load_level_specific_checkpoint(
    checkpoint: dict, device: torch.device,
) -> tuple[LevelSpecificOperators, dict]:
    spec = dict(checkpoint.get("model_spec", {}))
    if not bool(spec.get("level_specific_operators", False)):
        return clone_single_level_checkpoint(checkpoint, device)
    state = checkpoint["model"]
    models = []
    for level in ("coarse_operator", "fine_operator"):
        prefix = f"{level}."
        level_state = {
            key.removeprefix(prefix): value
            for key, value in state.items() if key.startswith(prefix)
        }
        if not level_state:
            raise ValueError(f"level-specific checkpoint is missing {level}")
        payload = dict(checkpoint)
        payload["model"] = level_state
        payload["model_spec"] = {
            key: value for key, value in spec.items()
            if key != "level_specific_operators"
        }
        model, _ = load_pretrained(payload, device, history_steps=2)
        models.append(model)
    return LevelSpecificOperators(*models), spec
