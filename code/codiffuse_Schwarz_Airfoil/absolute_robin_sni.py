"""Absolute Robin local time-step operator and Schwarz sweep for Airfoil."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from src.neural_eos.abupt import DynamicABUPTOperator

from .airfoil_flux_conditioned_sni import (
    directional_conservative_flux_2d,
    directional_face_roles_2d,
)
from .airfoil_sni import _FiLMBlock2D, _groups
from .airfoil_trace_sni import (
    assemble_subdomains_2d,
    neighboring_face_mask_2d,
)
from .pressure_coarse_correction import overlap_pressure_constant_correction_2d


Tensor = torch.Tensor
CONTRACT = "airfoil_absolute_robin_local_step_v1"


def absolute_robin_port_2d(
    state: Tensor,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    flux_scale: Tensor,
    dx: float,
    viscosity: Tensor | float,
    impedance: Tensor | tuple[float, float] = (1.0, 1.0),
    face_width: int = 1,
    neighbor_faces: Tensor | None = None,
    valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return four absolute Robin blocks for normalized ``(u,v,p)``.

    Conservative port order is ``(mass,momentum-x,momentum-y)`` and its trace
    companion is ``(p,u,v)``.  The pressure--mass pairing retains the elliptic
    pressure coupling while the momentum rows carry velocity impedance.
    """
    if state.ndim != 4 or state.shape[1] != 3:
        raise ValueError("Airfoil Robin state must have shape B,3,H,W")
    mean = torch.as_tensor(
        state_mean, device=state.device, dtype=state.dtype,
    ).reshape(1, 3, 1, 1)
    std = torch.as_tensor(
        state_std, device=state.device, dtype=state.dtype,
    ).reshape(1, 3, 1, 1)
    if bool(torch.any(std <= 0.0)):
        raise ValueError("state standard deviations must be positive")
    scale = torch.as_tensor(
        flux_scale, device=state.device, dtype=state.dtype,
    ).flatten()
    if scale.numel() != 12 or bool(torch.any(scale <= 0.0)):
        raise ValueError("flux scale must contain 12 positive values")
    z = torch.as_tensor(
        impedance, device=state.device, dtype=state.dtype,
    ).flatten()
    if z.numel() == 2:
        z = torch.stack((z[0], z[1], z[1]))
    if z.numel() != 3 or bool(torch.any(z < 0.0)):
        raise ValueError("impedance must contain mass and momentum values")

    physical = state * std + mean
    flux, roles = directional_conservative_flux_2d(
        physical, dx=dx, viscosity=viscosity, face_width=face_width,
    )
    if neighbor_faces is not None:
        if neighbor_faces.shape != (state.shape[0], 4, 1, 1):
            raise ValueError("neighbor face mask must have shape B,4,1,1")
        roles = roles * neighbor_faces
    if valid is not None:
        if valid.shape != state[:, :1].shape:
            raise ValueError("valid mask must have shape B,1,H,W")
        roles = roles * valid
    trace = torch.cat((state[:, 2:3], state[:, :2]), dim=1)
    blocks = []
    for face in range(4):
        role = roles[:, face : face + 1]
        normalized_flux = flux[:, 3 * face : 3 * (face + 1)] \
            / scale[3 * face : 3 * (face + 1)].reshape(1, 3, 1, 1)
        blocks.append((normalized_flux + z.reshape(1, 3, 1, 1) * trace) * role)
    return torch.cat(blocks, dim=1), roles


class AbsoluteRobinTileSolver2D(nn.Module):
    """Predict a complete local next state from local history and Robin data."""

    def __init__(
        self,
        *,
        physical_channels: int = 13,
        scalar_dim: int = 4,
        width: int = 96,
        depth: int = 8,
        residual_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if min(physical_channels, width, depth) <= 0 or scalar_dim < 0:
            raise ValueError("invalid Airfoil Robin local-operator dimensions")
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        if len(residual_scale) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in residual_scale
        ):
            raise ValueError("Airfoil residual scale must contain three positives")
        # One physical step is much smaller than the state distribution.  The
        # network therefore works in the train-set increment coordinates and
        # the public solver still returns the standardized complete state.
        self.register_buffer(
            "residual_scale",
            torch.tensor(residual_scale, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.stem = nn.Conv2d(
            physical_channels + 12 + 4, width, 5, padding=2,
        )
        cycle = (1, 2, 4, 1, 2, 4)
        self.blocks = nn.ModuleList(
            _FiLMBlock2D(width, scalar_dim, cycle[index % len(cycle)])
            for index in range(depth)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        robin_port: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        if physical.ndim != 4 or physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition has the wrong shape")
        shape = physical.shape[-2:]
        if robin_port.shape != (physical.shape[0], 12, *shape):
            raise ValueError("Robin port must have four three-channel blocks")
        if face_roles.shape != (physical.shape[0], 4, *shape):
            raise ValueError("face roles have the wrong shape")
        if scalar.shape != (physical.shape[0], self.scalar_dim):
            raise ValueError("scalar condition has the wrong shape")
        hidden = self.stem(torch.cat((physical, robin_port, face_roles), dim=1))
        for block in self.blocks:
            hidden = block(hidden, scalar)
        # The first three strict-local physical channels are q_n.  A temporal
        # local solve advances that admissible initial state; it is not a
        # residual chart around a separately learned whole-step endpoint.
        return (
            physical[:, :3] + self.output(hidden) * self.residual_scale.to(hidden)
        ) * physical[:, -1:]


class AbsoluteRobinPointTileSolver2D(nn.Module):
    """Strict-local point Transformer with the same Robin BVP contract.

    Every crop is expressed in its own ``[-1,1]^2`` chart.  Consequently two
    crops with identical local fields, scalar parameters, and Robin data
    produce identical outputs: neither a tile index nor an absolute coordinate
    enters the model.  The AB-UPT attention backbone is deliberately the same
    family as the strongest whole-domain Point baseline, while its inputs and
    outputs remain those of one reusable local BVP solver.
    """

    def __init__(
        self,
        *,
        physical_channels: int = 13,
        scalar_dim: int = 4,
        d_model: int = 320,
        surface_anchors: int = 256,
        volume_anchors: int = 512,
        geometry_anchors: int = 128,
        depth: int = 5,
        decoder_depth: int = 3,
        heads: int = 8,
        robin_channels: int = 12,
        projection_channels: int = 0,
        interface_anchor_fraction: float = 0.0,
        port_input_scale: float = 1.0,
        residual_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if min(
            physical_channels, robin_channels, d_model, surface_anchors, volume_anchors,
            geometry_anchors, depth, decoder_depth, heads,
        ) <= 0 or scalar_dim < 0:
            raise ValueError("invalid point-local Airfoil operator dimensions")
        if len(residual_scale) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in residual_scale
        ):
            raise ValueError("Airfoil residual scale must contain three positives")
        if projection_channels not in (0, 2):
            raise ValueError("Point projection channels must be zero or two")
        self.physical_channels = int(physical_channels)
        self.scalar_dim = int(scalar_dim)
        self.robin_channels = int(robin_channels)
        self.projection_channels = int(projection_channels)
        self.d_model = int(d_model)
        self.surface_anchors = int(surface_anchors)
        self.volume_anchors = int(volume_anchors)
        self.geometry_anchors = int(geometry_anchors)
        self.depth = int(depth)
        self.decoder_depth = int(decoder_depth)
        self.heads = int(heads)
        if not 0.0 <= interface_anchor_fraction <= 1.0:
            raise ValueError("interface-anchor fraction must lie in [0,1]")
        self.interface_anchor_fraction = float(interface_anchor_fraction)
        if not math.isfinite(port_input_scale) or port_input_scale <= 0.0:
            raise ValueError("Point port input scale must be finite and positive")
        self.port_input_scale = float(port_input_scale)
        self.register_buffer(
            "residual_scale",
            torch.tensor(residual_scale, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        # physical + four Robin blocks + four face-role masks + broadcast
        # physical scalars + two strictly relative coordinates.
        in_channels = (
            self.physical_channels + self.robin_channels
            + 4 + self.scalar_dim + 2
        )
        coordinate_channels = (in_channels - 2, in_channels - 1)
        face_role_channels = tuple(
            range(
                self.physical_channels + self.robin_channels,
                self.physical_channels + self.robin_channels + 4,
            )
        )
        self.operator = DynamicABUPTOperator(
            in_channels=in_channels,
            d_model=self.d_model,
            surface_anchors=self.surface_anchors,
            volume_anchors=self.volume_anchors,
            geometry_anchors=self.geometry_anchors,
            depth=self.depth,
            decoder_depth=self.decoder_depth,
            heads=self.heads,
            out_channels=3 + self.projection_channels,
            # q_n(3), local sdf_n(1), local sdf_np1(1), ...
            sdf_channel=4,
            coordinate_channels=coordinate_channels,
            surface_mask_channels=face_role_channels,
            surface_mask_fraction=self.interface_anchor_fraction,
        )

    def architecture_signature(self) -> dict[str, int]:
        """Return the AB-UPT capacity coordinates used by global baselines.

        Input channels differ deliberately between a whole-domain transition
        and a local Robin BVP.  The attention width, anchor budgets, processor
        depth, decoder depth, and head count are the architecture/capacity
        coordinates that must match in a controlled local-versus-global
        comparison.
        """
        return {
            "d_model": self.d_model,
            "surface_anchors": self.surface_anchors,
            "volume_anchors": self.volume_anchors,
            "geometry_anchors": self.geometry_anchors,
            "depth": self.depth,
            "decoder_depth": self.decoder_depth,
            "heads": self.heads,
        }

    @property
    def architecture_variant(self) -> str:
        """Name the parameter-neutral boundary-conditioned AB-UPT policy."""
        return (
            "boundary_geometry_abupt_v1"
            if self.interface_anchor_fraction > 0.0
            else "strict_local_abupt_v1"
        )

    @staticmethod
    def _relative_coordinates(reference: Tensor) -> Tensor:
        height, width = reference.shape[-2:]
        y = torch.linspace(
            -1.0, 1.0, height, device=reference.device,
            dtype=reference.dtype,
        )
        x = torch.linspace(
            -1.0, 1.0, width, device=reference.device,
            dtype=reference.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0)[None].expand(
            reference.shape[0], -1, -1, -1,
        )

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        robin_port: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        if physical.ndim != 4 or physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition has the wrong shape")
        shape = physical.shape[-2:]
        if robin_port.shape != (
            physical.shape[0], self.robin_channels, *shape,
        ):
            raise ValueError(
                f"Robin port must have {self.robin_channels} channels"
            )
        if face_roles.shape != (physical.shape[0], 4, *shape):
            raise ValueError("face roles have the wrong shape")
        if scalar.shape != (physical.shape[0], self.scalar_dim):
            raise ValueError("scalar condition has the wrong shape")
        scalar_field = scalar[:, :, None, None].expand(-1, -1, *shape)
        condition = torch.cat((
            physical,
            robin_port * self.port_input_scale,
            face_roles,
            scalar_field,
            self._relative_coordinates(physical),
        ), dim=1)
        output = self.operator(condition)
        increment = output[:, :3]
        state = (
            physical[:, :3] + increment * self.residual_scale.to(increment)
        ) * physical[:, -1:]
        if self.projection_channels == 0:
            return state
        # incflo's lagged cell-centred pressure gradient is part of the next
        # time-step state.  It is predicted in its own standardized absolute
        # coordinates rather than in the physical-state residual chart.
        projection = output[:, 3:] * physical[:, -1:]
        return torch.cat((state, projection), dim=1)


def transfer_point_zero_port_function(
    source: AbsoluteRobinPointTileSolver2D,
    target: AbsoluteRobinPointTileSolver2D,
    *,
    port_column_mapping: tuple[int | None, ...] | None = None,
    port_column_scales: tuple[float, ...] | None = None,
) -> dict[str, object]:
    """Transfer a local Point endpoint while changing its port dimension.

    All trained computation is copied.  The only shape-dependent tensors are
    the two input embeddings.  Their physical, face-role, scalar, and relative-
    coordinate columns are copied with the appropriate offset.  New port
    columns are either zero or initialized from declared source columns.
    Either choice preserves the source's zero-port Local/Core function
    exactly; a semantic column map can additionally reuse learned boundary
    sensitivity before native-port training.
    """
    if source.physical_channels != target.physical_channels \
            or source.scalar_dim != target.scalar_dim:
        raise ValueError("Point zero-port transfer requires identical local inputs")
    if source.projection_channels != target.projection_channels:
        raise ValueError(
            "Point zero-port transfer requires identical projection outputs"
        )
    if source.architecture_signature() != target.architecture_signature():
        raise ValueError("Point zero-port transfer requires identical capacity")
    if source.interface_anchor_fraction != target.interface_anchor_fraction:
        raise ValueError("Point zero-port transfer requires identical anchor policy")
    if port_column_mapping is None:
        port_column_mapping = (None,) * target.robin_channels
    if len(port_column_mapping) != target.robin_channels:
        raise ValueError("Point port mapping must cover every target channel")
    if any(
        index is not None and not 0 <= index < source.robin_channels
        for index in port_column_mapping
    ):
        raise ValueError("Point port mapping contains an invalid source channel")
    if port_column_scales is None:
        port_column_scales = (1.0,) * target.robin_channels
    if len(port_column_scales) != target.robin_channels or any(
        not math.isfinite(float(value)) for value in port_column_scales
    ):
        raise ValueError("Point port scales must be finite and channel matched")

    source_state = source.state_dict()
    target_state = target.state_dict()
    embedding_keys = {
        "operator.surface_embedding.0.weight",
        "operator.volume_embedding.0.weight",
    }
    mismatched = {
        key for key in target_state
        if key not in source_state or source_state[key].shape != target_state[key].shape
    }
    # Changing the number of port channels changes only the two input
    # embeddings.  A semantic retargeting with the same channel count (for
    # example conservative-flux -> full-state Dirichlet) has no shape
    # mismatch, but still needs the column-wise remap below.
    expected_mismatched = (
        embedding_keys
        if source.robin_channels != target.robin_channels else set()
    )
    if mismatched != expected_mismatched:
        raise ValueError(
            "unexpected Point transfer shape mismatch: "
            f"{sorted(mismatched)}"
        )
    copied_parameters = 0
    for key, target_value in target_state.items():
        if key not in embedding_keys:
            target_state[key] = source_state[key].to(target_value)
            copied_parameters += int(target_value.numel())
            continue
        source_value = source_state[key].to(target_value)
        mapped = torch.zeros_like(target_value)
        physical = target.physical_channels
        source_tail = physical + source.robin_channels
        target_tail = physical + target.robin_channels
        if source_value.shape[1] - source_tail \
                != target_value.shape[1] - target_tail:
            raise ValueError("Point transfer non-port feature dimensions differ")
        mapped[:, :physical] = source_value[:, :physical]
        for target_index, source_index in enumerate(port_column_mapping):
            if source_index is None:
                continue
            mapped[:, physical + target_index] = (
                source_value[:, physical + source_index]
                * float(port_column_scales[target_index])
            )
            copied_parameters += int(mapped[:, physical + target_index].numel())
        mapped[:, target_tail:] = source_value[:, source_tail:]
        target_state[key] = mapped
        copied_parameters += int(mapped[:, :physical].numel())
        copied_parameters += int(mapped[:, target_tail:].numel())
    target.load_state_dict(target_state, strict=True)
    total_parameters = sum(value.numel() for value in target_state.values())
    return {
        "source_robin_channels": source.robin_channels,
        "target_robin_channels": target.robin_channels,
        "copied_parameter_count": copied_parameters,
        "target_parameter_count": total_parameters,
        "copied_parameter_fraction": copied_parameters / total_parameters,
        "port_column_mapping": list(port_column_mapping),
        "port_column_scales": [float(value) for value in port_column_scales],
        "new_port_embedding_columns_initialized_to_zero": all(
            index is None for index in port_column_mapping
        ),
        "zero_port_function_preserved": True,
    }


def transfer_point_physical_history_function(
    source: AbsoluteRobinPointTileSolver2D,
    target: AbsoluteRobinPointTileSolver2D,
    *,
    added_history_channels: int = 3,
) -> dict[str, object]:
    """Insert zero-initialized history channels before the validity mask.

    Strict-local Airfoil conditions always store the crop-validity mask in the
    final physical channel.  A two-level endpoint inserts ``q_n-q_(n-1)``
    immediately before that mask.  This transfer copies every trained weight,
    moves the old validity column to its new semantic position, shifts every
    port/role/scalar/coordinate column, and initializes only the new history
    columns to zero.  Consequently the target computes the source function
    exactly until those history columns are trained.
    """
    if added_history_channels <= 0:
        raise ValueError("Point history transfer needs positive added channels")
    if target.physical_channels != (
        source.physical_channels + int(added_history_channels)
    ):
        raise ValueError(
            "Point history transfer target must add the declared channels"
        )
    if source.scalar_dim != target.scalar_dim \
            or source.robin_channels != target.robin_channels:
        raise ValueError(
            "Point history transfer requires identical scalar and port inputs"
        )
    if source.projection_channels != target.projection_channels:
        raise ValueError(
            "Point history transfer requires identical projection outputs"
        )
    if source.architecture_signature() != target.architecture_signature():
        raise ValueError("Point history transfer requires identical capacity")
    if source.interface_anchor_fraction != target.interface_anchor_fraction:
        raise ValueError("Point history transfer requires identical anchor policy")

    source_state = source.state_dict()
    target_state = target.state_dict()
    embedding_keys = {
        "operator.surface_embedding.0.weight",
        "operator.volume_embedding.0.weight",
    }
    mismatched = {
        key for key in target_state
        if key not in source_state or source_state[key].shape != target_state[key].shape
    }
    if mismatched != embedding_keys:
        raise ValueError(
            "unexpected Point history-transfer shape mismatch: "
            f"{sorted(mismatched)}"
        )

    copied_parameters = 0
    source_valid = source.physical_channels - 1
    target_valid = target.physical_channels - 1
    history_start = source_valid
    history_stop = target_valid
    for key, target_value in target_state.items():
        if key not in embedding_keys:
            target_state[key] = source_state[key].to(target_value)
            copied_parameters += int(target_value.numel())
            continue
        source_value = source_state[key].to(target_value)
        mapped = torch.zeros_like(target_value)
        # Physical fields preceding validity keep their indices.  The new
        # history columns remain zero and the validity column moves after them.
        mapped[:, :source_valid] = source_value[:, :source_valid]
        mapped[:, target_valid] = source_value[:, source_valid]
        # Every non-physical condition follows the expanded physical block.
        mapped[:, target.physical_channels:] = source_value[
            :, source.physical_channels:
        ]
        target_state[key] = mapped
        copied_parameters += int(mapped[:, :source_valid].numel())
        copied_parameters += int(mapped[:, target_valid].numel())
        copied_parameters += int(mapped[:, target.physical_channels:].numel())
    target.load_state_dict(target_state, strict=True)
    total_parameters = sum(value.numel() for value in target_state.values())
    return {
        "source_physical_channels": source.physical_channels,
        "target_physical_channels": target.physical_channels,
        "added_history_channels": int(added_history_channels),
        "source_valid_column": source_valid,
        "target_history_columns": list(range(history_start, history_stop)),
        "target_valid_column": target_valid,
        "copied_parameter_count": copied_parameters,
        "target_parameter_count": total_parameters,
        "copied_parameter_fraction": copied_parameters / total_parameters,
        "new_history_embedding_columns_initialized_to_zero": True,
        "zero_history_function_preserved": True,
    }


def build_absolute_robin_tile_solver(
    *,
    backbone: str = "cnn",
    physical_channels: int = 13,
    scalar_dim: int = 4,
    width: int = 96,
    depth: int = 8,
    point_d_model: int = 320,
    point_surface_anchors: int = 256,
    point_volume_anchors: int = 512,
    point_geometry_anchors: int = 128,
    point_depth: int = 5,
    point_decoder_depth: int = 3,
    point_heads: int = 8,
    point_robin_channels: int = 12,
    point_projection_channels: int = 0,
    point_interface_anchor_fraction: float = 0.0,
    point_port_input_scale: float = 1.0,
    residual_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> nn.Module:
    """Construct either local backbone behind one unchanged BVP interface."""
    if backbone == "cnn":
        return AbsoluteRobinTileSolver2D(
            physical_channels=physical_channels,
            scalar_dim=scalar_dim,
            width=width,
            depth=depth,
            residual_scale=residual_scale,
        )
    if backbone == "point":
        return AbsoluteRobinPointTileSolver2D(
            physical_channels=physical_channels,
            scalar_dim=scalar_dim,
            d_model=point_d_model,
            surface_anchors=point_surface_anchors,
            volume_anchors=point_volume_anchors,
            geometry_anchors=point_geometry_anchors,
            depth=point_depth,
            decoder_depth=point_decoder_depth,
            heads=point_heads,
            robin_channels=point_robin_channels,
            projection_channels=point_projection_channels,
            interface_anchor_fraction=point_interface_anchor_fraction,
            port_input_scale=point_port_input_scale,
            residual_scale=residual_scale,
        )
    raise ValueError(f"unknown Airfoil local backbone: {backbone}")


def absolute_robin_solver_from_payload(
    payload: dict[str, object],
    device: torch.device | str | None = None,
) -> nn.Module:
    """Rebuild a local solver while retaining old CNN checkpoint support."""
    saved = dict(payload["args"])
    solver = build_absolute_robin_tile_solver(
        backbone=str(payload.get("backbone", saved.get("backbone", "cnn"))),
        physical_channels=int(saved.get("physical_channels", 13)),
        scalar_dim=int(saved.get("scalar_dim", 4)),
        width=int(saved.get("width", 96)),
        depth=int(saved.get("depth", 8)),
        point_d_model=int(saved.get("point_d_model", 320)),
        point_surface_anchors=int(saved.get("point_surface_anchors", 256)),
        point_volume_anchors=int(saved.get("point_volume_anchors", 512)),
        point_geometry_anchors=int(saved.get("point_geometry_anchors", 128)),
        point_depth=int(saved.get("point_depth", 5)),
        point_decoder_depth=int(saved.get("point_decoder_depth", 3)),
        point_heads=int(saved.get("point_heads", 8)),
        point_robin_channels=int(saved.get("point_robin_channels", 12)),
        point_projection_channels=int(payload.get("projection_channels", 0)),
        point_interface_anchor_fraction=float(saved.get(
            "point_interface_anchor_fraction", 0.0,
        )),
        point_port_input_scale=float(saved.get("point_port_input_scale", 1.0)),
        residual_scale=tuple(payload.get(
            "residual_scale_state_coordinates", (1.0, 1.0, 1.0),
        )),
    )
    if device is not None:
        solver = solver.to(device)
    solver.load_state_dict(payload["solver"], strict=True)
    return solver


@dataclass(frozen=True)
class AbsoluteRobinSweep2D:
    field: Tensor
    local_candidates: Tensor
    imposed_port: Tensor
    next_port: Tensor
    interface_rms: Tensor
    port_defect_rms: Tensor
    realization_rms: Tensor
    pressure_reference_mean: Tensor | None = None
    pressure_overlap_rms_before: Tensor | None = None
    pressure_overlap_rms_after: Tensor | None = None


def absolute_robin_sni_sweep_2d(
    solver: nn.Module,
    physical_tiles: Tensor,
    scalars: Tensor,
    current: Tensor,
    cover,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    flux_scale: Tensor,
    dx: float,
    viscosity: Tensor | float,
    impedance: Tensor | tuple[float, float] = (1.0, 1.0),
    relaxation: float = 0.5,
    face_width: int = 1,
    chunk_tiles: int | None = None,
    neighbor_faces: Tensor | None = None,
    imposed_port: Tensor | None = None,
    assembly: str = "normalized_overlap",
    pressure_coarse_correction: bool = False,
    pressure_reference_mean: Tensor | None = None,
) -> AbsoluteRobinSweep2D:
    """Apply one parallel Robin-control sweep at a fixed physical step.

    The physical history stays fixed. ``imposed_port`` carries the absolute
    artificial-boundary state between sweeps; when omitted it is initialized
    from ``current`` (normally :math:`q^n`).  Relaxation acts on this control,
    not on the assembled physical field.
    """
    if not 0.0 < relaxation <= 1.5:
        raise ValueError("relaxation must lie in (0,1.5]")
    current_tiles = cover.tile(current, padding_mode="replicate")
    if physical_tiles.shape[0] != current_tiles.shape[0]:
        raise ValueError("physical tiles and current cover disagree")
    if neighbor_faces is None:
        neighbor_faces = neighboring_face_mask_2d(
            cover.grid_shape,
            batch=current.shape[0],
            device=current.device,
            dtype=current.dtype,
        )
    initialized_port, roles = absolute_robin_port_2d(
        current_tiles,
        state_mean=state_mean,
        state_std=state_std,
        flux_scale=flux_scale,
        dx=dx,
        viscosity=viscosity,
        impedance=impedance,
        face_width=face_width,
        neighbor_faces=neighbor_faces,
        valid=physical_tiles[:, -1:],
    )
    port = initialized_port if imposed_port is None else imposed_port
    if port.shape != initialized_port.shape:
        raise ValueError("imposed Robin port has the wrong shape")
    active = roles.repeat_interleave(3, dim=1)
    port = port * active
    chunk = current_tiles.shape[0] if chunk_tiles is None else int(chunk_tiles)
    if chunk <= 0:
        raise ValueError("chunk_tiles must be positive")
    candidates = []
    for start in range(0, current_tiles.shape[0], chunk):
        stop = min(start + chunk, current_tiles.shape[0])
        candidates.append(solver(
            physical_tiles[start:stop],
            scalars[start:stop],
            port[start:stop],
            roles[start:stop],
        ))
    local = torch.cat(candidates)
    pressure_overlap_before = None
    pressure_overlap_after = None
    if pressure_coarse_correction:
        pressure_correction = overlap_pressure_constant_correction_2d(
            local, cover, current.shape[0],
            valid_tiles=physical_tiles[:, -1:],
            reference_mean=pressure_reference_mean,
        )
        local = pressure_correction.patches
        pressure_reference_mean = pressure_correction.reference_mean
        pressure_overlap_before = pressure_correction.overlap_rms_before
        pressure_overlap_after = pressure_correction.overlap_rms_after
    proposal = assemble_subdomains_2d(
        cover, local, current.shape[0], assembly,
    )
    # Use the unique RAS core owner for transmission.  A normalized overlap
    # field contains part of the receiving tile itself and would turn the
    # Robin refresh into self-feedback rather than neighbor communication.
    exchange_field = assemble_subdomains_2d(
        cover, local, current.shape[0], "disjoint_core",
    )
    valid = assemble_subdomains_2d(
        cover, physical_tiles[:, -1:], current.shape[0], assembly,
    )
    field = proposal * valid
    consensus = cover.tile(exchange_field * valid, padding_mode="replicate")
    collar = (roles.sum(1, keepdim=True) > 0.0).to(local)
    interface_rms = (
        ((local - consensus).square() * collar).sum()
        / (collar.sum() * local.shape[1]).clamp_min(1.0)
    ).sqrt()
    realized_port, _ = absolute_robin_port_2d(
        local,
        state_mean=state_mean,
        state_std=state_std,
        flux_scale=flux_scale,
        dx=dx,
        viscosity=viscosity,
        impedance=impedance,
        face_width=face_width,
        neighbor_faces=neighbor_faces,
        valid=physical_tiles[:, -1:],
    )
    realization_rms = (
        ((realized_port - port).square() * active).sum()
        / active.sum().clamp_min(1.0)
    ).sqrt()
    unrestricted_next_port, _ = absolute_robin_port_2d(
        consensus,
        state_mean=state_mean,
        state_std=state_std,
        flux_scale=flux_scale,
        dx=dx,
        viscosity=viscosity,
        impedance=impedance,
        face_width=face_width,
        neighbor_faces=neighbor_faces,
        valid=physical_tiles[:, -1:],
    )
    port_residual = (unrestricted_next_port - port) * active
    port_defect_rms = (
        port_residual.square().sum() / active.sum().clamp_min(1.0)
    ).sqrt()
    next_port = (port + relaxation * port_residual) * active
    return AbsoluteRobinSweep2D(
        field, local, port, next_port, interface_rms, port_defect_rms,
        realization_rms, pressure_reference_mean,
        pressure_overlap_before, pressure_overlap_after,
    )
