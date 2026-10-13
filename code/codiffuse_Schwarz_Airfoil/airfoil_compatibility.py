"""One-message post-endpoint compatibility codes for the Airfoil case.

This module implements the Section 3 order of operations literally:

1. a frozen local operator produces one endpoint on every Cartesian unity;
2. one pooled token per unity participates in one globally receptive,
   relative-position message;
3. an explicit local decoder maps the resulting compatibility code to a field
   correction; and
4. the existing caller performs its unchanged partition-of-unity assembly.

The code never consumes the final two Airfoil scalar controls because those are
absolute tile centers.  Geometry-relative position enters only through the
pairwise offsets in :class:`RelativeTileAttention`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from codiffuse_Schwarz.tile_communication import RelativeTileAttention
from codiffuse_Schwarz.residual_port import (
    ResidualPortSpec,
    project_residual_ports,
    scale_port_code,
)


Tensor = torch.Tensor
CodeMode = Literal["zero", "predicted", "shuffled", "oracle"]
PortFunctionalKind = Literal["pressure_traction", "brinkman_force"]
AIRFOIL_RESIDUAL_PORT_SPEC = ResidualPortSpec(
    dimension=2, field_channels=3, polynomial_degree=1,
    maximum_modes_per_face=2, appended_functionals=8,
)
RESIDUAL_PORT_DIM = AIRFOIL_RESIDUAL_PORT_SPEC.code_width


def _statistics_tensor(values: tuple[float, float, float], reference: Tensor) -> Tensor:
    return reference.new_tensor(values)[None, :, None, None]


def airfoil_force_semantics(
    local_endpoint: Tensor,
    physical: Tensor,
    *,
    state_mean: tuple[float, float, float],
    state_std: tuple[float, float, float],
    delta_mean: tuple[float, float, float],
    delta_std: tuple[float, float, float],
) -> Tensor:
    """Construct pressure/upwind/load-bearing channels available in the data.

    ``local_endpoint`` is the normalized ``q_(n+1)-q_n`` prediction.  The
    Airfoil condition contract stores normalized ``q_n`` in channels 0:3,
    source/destination SDF in 3:5, source/destination wall velocity in 5:9,
    and tile validity in the final channel.  The returned seven channels are
    gauge-centered pressure, two pressure-traction proxies, two Brinkman-slip
    (force-bearing) densities, and positive/negative streamwise upwind parts.

    These are semantic neural inputs, not a discrete Navier--Stokes operator.
    In particular, the pressure traction uses the destination-SDF normal and
    the Brinkman terms use the wall velocity already present in the dataset.
    """
    if local_endpoint.ndim != 4 or local_endpoint.shape[1] != 3:
        raise ValueError("Airfoil endpoint must have shape T,3,H,W")
    if physical.ndim != 4 or physical.shape[0] != local_endpoint.shape[0]:
        raise ValueError("physical fields must share the endpoint tile batch")
    if physical.shape[1] < 10 or physical.shape[-2:] != local_endpoint.shape[-2:]:
        raise ValueError("physical fields do not satisfy the Airfoil condition contract")

    valid = physical[:, -1:].to(local_endpoint.dtype).clamp(0.0, 1.0)
    current = (
        physical[:, :3] * _statistics_tensor(state_std, local_endpoint)
        + _statistics_tensor(state_mean, local_endpoint)
    )
    increment = (
        local_endpoint * _statistics_tensor(delta_std, local_endpoint)
        + _statistics_tensor(delta_mean, local_endpoint)
    )
    predicted = current + increment
    pressure = predicted[:, 2:3]
    pressure_mean = (pressure * valid).sum((-2, -1), keepdim=True) / valid.sum(
        (-2, -1), keepdim=True,
    ).clamp_min(1.0)
    pressure = pressure - pressure_mean

    destination_sdf = physical[:, 4:5]
    grad_y, grad_x = torch.gradient(destination_sdf, dim=(-2, -1))
    magnitude = torch.sqrt(grad_x.square() + grad_y.square()).clamp_min(1.0e-6)
    normal_x, normal_y = grad_x / magnitude, grad_y / magnitude
    surface = torch.exp(-8.0 * destination_sdf.abs()) * valid
    traction_x = -pressure * normal_x * surface
    traction_y = -pressure * normal_y * surface

    destination_wall = physical[:, 7:9]
    solid = (destination_sdf <= 0.0).to(local_endpoint.dtype) * valid
    slip_x = (destination_wall[:, 0:1] - predicted[:, 0:1]) * solid
    slip_y = (destination_wall[:, 1:2] - predicted[:, 1:2]) * solid
    upstream = F.relu(predicted[:, 0:1]) * valid
    downstream = F.relu(-predicted[:, 0:1]) * valid
    return torch.tanh(torch.cat((
        pressure * valid, traction_x, traction_y, slip_x, slip_y,
        upstream, downstream,
    ), dim=1))


def _masked_moment(field: Tensor, weight: Tensor) -> Tensor:
    denominator = weight.abs().sum((-2, -1)).clamp_min(1.0)
    return (field * weight).sum((-2, -1)) / denominator


def airfoil_residual_port_code(
    local_endpoint: Tensor,
    target_tiles: Tensor,
    physical: Tensor,
    *,
    state_mean: tuple[float, float, float],
    state_std: tuple[float, float, float],
    delta_mean: tuple[float, float, float],
    delta_std: tuple[float, float, float],
    functional_kind: PortFunctionalKind = "pressure_traction",
    force_area_scale: float = 1.0,
    lift_sign: float = 1.0,
) -> Tensor:
    """Return the shared fixed 32D residual-port coordinate per Airfoil tile.

    The shared contract contributes 24 masked least-squares degree-one
    Legendre face coefficients in fixed ``(y-,y+,x-,x+)`` order.  Eight
    Airfoil functionals append positive/negative streamwise-upwind residual
    bins (three channels each) and pressure-traction load residuals in x/y.
    The target is used only for training labels and oracle evaluation.
    """
    if target_tiles.shape != local_endpoint.shape:
        raise ValueError("target and endpoint tiles must have the same shape")
    if physical.ndim != 4 or physical.shape[0] != local_endpoint.shape[0]:
        raise ValueError("physical fields must share the endpoint tile batch")
    valid = physical[:, -1:].to(local_endpoint).clamp(0.0, 1.0)
    residual = (target_tiles - local_endpoint) * valid

    current = (
        physical[:, :3] * _statistics_tensor(state_std, local_endpoint)
        + _statistics_tensor(state_mean, local_endpoint)
    )
    local_increment = (
        local_endpoint * _statistics_tensor(delta_std, local_endpoint)
        + _statistics_tensor(delta_mean, local_endpoint)
    )
    streamwise = current[:, :1] + local_increment[:, :1]
    positive = F.relu(streamwise) * valid
    negative = F.relu(-streamwise) * valid
    upwind = (
        _masked_moment(residual, positive),
        _masked_moment(residual, negative),
    )

    residual_physical = residual * _statistics_tensor(delta_std, residual)
    pressure_residual = residual_physical[:, 2:3]
    destination_sdf = physical[:, 4:5]
    grad_y, grad_x = torch.gradient(destination_sdf, dim=(-2, -1))
    magnitude = torch.sqrt(grad_x.square() + grad_y.square()).clamp_min(1.0e-6)
    normal_x, normal_y = grad_x / magnitude, grad_y / magnitude
    surface = torch.exp(-8.0 * destination_sdf.abs()) * valid
    traction_x = -pressure_residual * normal_x
    traction_y = -pressure_residual * normal_y
    if functional_kind == "pressure_traction":
        load = (
            _masked_moment(traction_x, surface),
            _masked_moment(traction_y, surface),
        )
    elif functional_kind == "brinkman_force":
        solid = (destination_sdf <= 0.0).to(residual) * valid
        load = (
            force_area_scale * (residual_physical[:, 0:1] * solid).sum((-2, -1)),
            -lift_sign * force_area_scale
            * (residual_physical[:, 1:2] * solid).sum((-2, -1)),
        )
    else:
        raise ValueError(f"unknown Airfoil port functional kind: {functional_kind}")
    functionals = torch.cat((*upwind, *load), dim=1)
    code = project_residual_ports(
        residual, valid, AIRFOIL_RESIDUAL_PORT_SPEC,
        functionals=functionals,
    )
    if code.shape[1] != RESIDUAL_PORT_DIM:
        raise RuntimeError("residual-port schema dimension drifted")
    return code


@dataclass(frozen=True)
class CompatibilityOutput:
    corrected: Tensor
    code: Tensor
    predicted_code: Tensor
    semantics: Tensor


class AirfoilCompatibilityComposer(nn.Module):
    """Pool local endpoints, communicate once, and decode local corrections."""

    semantic_channels = 7
    residual_port_spec = AIRFOIL_RESIDUAL_PORT_SPEC

    def __init__(
        self,
        *,
        field_channels: int = 3,
        scalar_dim: int = 13,
        width: int = 64,
        code_dim: int = 32,
        rank: int = 8,
        heads: int = 4,
        correction_scale: float = 0.1,
        max_graph_radius: int | None = None,
        port_functional_kind: PortFunctionalKind = "pressure_traction",
        state_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        state_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        delta_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        delta_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if (
            field_channels != 3 or min(scalar_dim, width, code_dim, rank, heads) <= 0
            or width % heads or correction_scale <= 0.0
            or (max_graph_radius is not None and max_graph_radius < 0)
            or port_functional_kind not in ("pressure_traction", "brinkman_force")
        ):
            raise ValueError("invalid Airfoil compatibility dimensions")
        self.field_channels = field_channels
        self.scalar_dim = scalar_dim
        self.width = width
        self.code_dim = code_dim
        self.rank = rank
        self.correction_scale = correction_scale
        self.max_graph_radius = max_graph_radius
        self.port_functional_kind = port_functional_kind
        feature_channels = field_channels + self.semantic_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(feature_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, width), width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.SiLU(),
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.SiLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, width), nn.SiLU(), nn.Linear(width, width),
        )
        # Exactly one globally receptive message. Relative offsets, rather than
        # the Airfoil scalar tile-center entries, identify the cover geometry.
        self.global_message = RelativeTileAttention(width, heads, dimension=2)
        self.code_head = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, code_dim),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, width), width),
            nn.SiLU(),
            nn.Conv2d(width, field_channels * rank, 3, padding=1),
        )
        # Bias-free coefficients make z=0 an exact local-endpoint baseline for
        # the lifetime of the model, not only at initialization.
        self.code_coefficients = nn.Linear(
            code_dim, field_channels * rank, bias=False,
        )
        nn.init.zeros_(self.code_coefficients.weight)
        self.register_buffer("port_mean", torch.zeros(code_dim))
        self.register_buffer("port_std", torch.ones(code_dim))
        self.attention_calls = 0
        self.set_statistics(state_mean, state_std, delta_mean, delta_std)

    def set_statistics(
        self,
        state_mean: tuple[float, float, float],
        state_std: tuple[float, float, float],
        delta_mean: tuple[float, float, float],
        delta_std: tuple[float, float, float],
    ) -> None:
        for name, values in (
            ("state_mean", state_mean), ("state_std", state_std),
            ("delta_mean", delta_mean), ("delta_std", delta_std),
        ):
            value = torch.as_tensor(values, dtype=torch.float32)
            if value.shape != (3,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite values")
            if name.endswith("std") and torch.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
            if name in self._buffers:
                self._buffers[name] = value
            else:
                self.register_buffer(name, value)

    def semantic_features(self, local_endpoint: Tensor, physical: Tensor) -> Tensor:
        return airfoil_force_semantics(
            local_endpoint, physical,
            state_mean=tuple(float(x) for x in self.state_mean),
            state_std=tuple(float(x) for x in self.state_std),
            delta_mean=tuple(float(x) for x in self.delta_mean),
            delta_std=tuple(float(x) for x in self.delta_std),
        )

    def set_port_statistics(self, mean: Tensor, std: Tensor) -> None:
        if mean.shape != (self.code_dim,) or std.shape != (self.code_dim,):
            raise ValueError("port statistics have incompatible shape")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("port statistics must be finite")
        self.port_mean.copy_(mean.to(self.port_mean))
        self.port_std.copy_(std.to(self.port_std).clamp_min(1.0e-6))

    def set_port_scale(self, scale: Tensor) -> None:
        """Set train-only RMS scale while preserving the physical zero code."""
        self.set_port_statistics(torch.zeros_like(scale), scale)

    def oracle_residual_port_code(
        self,
        local_endpoint: Tensor,
        target_tiles: Tensor,
        physical: Tensor,
        *,
        standardized: bool = True,
        force_area_scale: float = 1.0,
        lift_sign: float = 1.0,
    ) -> Tensor:
        if self.code_dim != RESIDUAL_PORT_DIM:
            raise ValueError("the identifiable Airfoil port schema requires code_dim=32")
        raw = airfoil_residual_port_code(
            local_endpoint, target_tiles, physical,
            state_mean=tuple(float(x) for x in self.state_mean),
            state_std=tuple(float(x) for x in self.state_std),
            delta_mean=tuple(float(x) for x in self.delta_mean),
            delta_std=tuple(float(x) for x in self.delta_std),
            functional_kind=self.port_functional_kind,
            force_area_scale=force_area_scale,
            lift_sign=lift_sign,
        )
        return scale_port_code(raw, self.port_std) if standardized else raw

    def reset_attention_calls(self) -> None:
        self.attention_calls = 0

    def _tokens(
        self,
        local_endpoint: Tensor,
        semantics: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        if local_endpoint.shape[0] % tile_count:
            raise ValueError("tile batch is incompatible with grid_shape")
        batch = local_endpoint.shape[0] // tile_count
        if scalar.ndim != 2 or scalar.shape[0] != local_endpoint.shape[0]:
            raise ValueError("scalar controls must have shape T,S")
        if scalar.shape[1] < self.scalar_dim:
            raise ValueError("scalar controls omit required case variables")
        features = torch.cat((local_endpoint, semantics), dim=1)
        encoded = self.encoder(features) * valid
        denominator = valid.sum((-2, -1)).clamp_min(1.0)
        mean = encoded.sum((-2, -1)) / denominator
        rms = torch.sqrt(encoded.square().sum((-2, -1)) / denominator + 1.0e-8)
        token = self.pool(torch.cat((mean, rms), dim=1)).reshape(
            batch, tile_count, self.width,
        )
        # Only case controls are consumed. In the retained Airfoil adapter the
        # remaining two scalar entries are absolute tile centers.
        controls = scalar[:, :self.scalar_dim].reshape(
            batch, tile_count, self.scalar_dim,
        ).mean(dim=1)
        return token + self.scalar_encoder(controls)[:, None]

    def predict_code(
        self,
        local_endpoint: Tensor,
        physical: Tensor,
        scalar: Tensor,
        grid_shape: tuple[int, int],
        *,
        valid: Tensor | None = None,
        semantics: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if valid is None:
            valid = physical[:, -1:].to(local_endpoint)
        if valid.shape != (local_endpoint.shape[0], 1, *local_endpoint.shape[-2:]):
            raise ValueError("validity field has incompatible shape")
        if semantics is None:
            semantics = self.semantic_features(local_endpoint, physical)
        if semantics.shape != (
            local_endpoint.shape[0], self.semantic_channels, *local_endpoint.shape[-2:],
        ):
            raise ValueError("semantic fields have incompatible shape")
        tokens = self._tokens(
            local_endpoint, semantics, scalar, valid, grid_shape,
        )
        message = self.global_message(
            tokens, grid_shape, max_graph_radius=self.max_graph_radius,
        )
        self.attention_calls += 1
        return self.code_head(torch.cat((tokens, message), dim=-1)), semantics

    @staticmethod
    def _shuffle_code(code: Tensor, permutation: Tensor | None) -> Tensor:
        batch, tiles = code.shape[:2]
        if permutation is None:
            return code.flip(0) if batch > 1 else code.roll(1, dims=1)
        permutation = permutation.to(device=code.device)
        if permutation.shape == (batch,):
            return code[permutation]
        if permutation.shape == (batch, tiles):
            return torch.gather(
                code, 1, permutation[..., None].expand(-1, -1, code.shape[-1]),
            )
        raise ValueError("shuffle permutation must have shape B or B,N")

    def decode(
        self,
        local_endpoint: Tensor,
        semantics: Tensor,
        code: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        batch = local_endpoint.shape[0] // tile_count
        if code.shape != (batch, tile_count, self.code_dim):
            raise ValueError("compatibility code has incompatible shape")
        features = torch.cat((local_endpoint, semantics), dim=1)
        basis = torch.tanh(self.decoder(features)).reshape(
            batch * tile_count, self.field_channels, self.rank,
            *local_endpoint.shape[-2:],
        )
        coefficients = self.code_coefficients(code).reshape(
            batch * tile_count, self.field_channels, self.rank,
        )
        correction = torch.einsum("tcrhw,tcr->tchw", basis, coefficients)
        correction = self.correction_scale * torch.tanh(correction) * valid
        return local_endpoint + correction

    def forward(
        self,
        local_endpoint: Tensor,
        physical: Tensor,
        scalar: Tensor,
        grid_shape: tuple[int, int],
        *,
        code_mode: CodeMode = "predicted",
        oracle_code: Tensor | None = None,
        shuffle_permutation: Tensor | None = None,
        valid: Tensor | None = None,
        semantics: Tensor | None = None,
    ) -> CompatibilityOutput:
        if code_mode not in ("zero", "predicted", "shuffled", "oracle"):
            raise ValueError(f"unknown compatibility code mode: {code_mode}")
        tile_count = math.prod(grid_shape)
        if local_endpoint.shape[0] % tile_count:
            raise ValueError("tile batch is incompatible with grid_shape")
        batch = local_endpoint.shape[0] // tile_count
        if valid is None:
            valid = physical[:, -1:].to(local_endpoint)
        if semantics is None:
            semantics = self.semantic_features(local_endpoint, physical)
        shape = (batch, tile_count, self.code_dim)
        if code_mode in ("predicted", "shuffled"):
            predicted, semantics = self.predict_code(
                local_endpoint, physical, scalar, grid_shape,
                valid=valid, semantics=semantics,
            )
            code = (
                predicted if code_mode == "predicted"
                else self._shuffle_code(predicted, shuffle_permutation)
            )
        elif code_mode == "zero":
            predicted = local_endpoint.new_zeros(shape)
            code = predicted
        else:
            if oracle_code is None or oracle_code.shape != shape:
                raise ValueError("oracle mode requires a B,N,code_dim oracle_code")
            code = oracle_code.to(local_endpoint)
            predicted = local_endpoint.new_zeros(shape)
        corrected = self.decode(
            local_endpoint, semantics, code, valid, grid_shape,
        )
        return CompatibilityOutput(corrected, code, predicted, semantics)


def fit_broken_space_oracle_code(
    composer: AirfoilCompatibilityComposer,
    local_endpoint: Tensor,
    physical: Tensor,
    scalar: Tensor,
    target_tiles: Tensor,
    grid_shape: tuple[int, int],
    *,
    steps: int = 64,
    learning_rate: float = 0.1,
    valid: Tensor | None = None,
) -> Tensor:
    """Fit an evaluation-only local-chart oracle without changing the decoder."""
    if steps <= 0 or learning_rate <= 0.0 or target_tiles.shape != local_endpoint.shape:
        raise ValueError("oracle fit requires positive settings and matching targets")
    if valid is None:
        valid = physical[:, -1:].to(local_endpoint)
    with torch.no_grad():
        predicted, semantics = composer.predict_code(
            local_endpoint, physical, scalar, grid_shape, valid=valid,
        )
    code = nn.Parameter(predicted.detach().clone())
    optimizer = torch.optim.Adam((code,), lr=learning_rate)
    # This diagnostic must optimize z only.  Temporarily disabling composer
    # gradients makes that contract explicit and avoids accumulating unused
    # decoder/attention gradients during validation.
    requires_grad = [parameter.requires_grad for parameter in composer.parameters()]
    try:
        for parameter in composer.parameters():
            parameter.requires_grad_(False)
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                corrected = composer.decode(
                    local_endpoint.detach(), semantics.detach(), code, valid, grid_shape,
                )
                loss = (
                    (corrected - target_tiles).square() * valid
                ).sum() / (valid.sum().clamp_min(1.0) * local_endpoint.shape[1])
                loss.backward()
                optimizer.step()
    finally:
        for parameter, enabled in zip(composer.parameters(), requires_grad):
            parameter.requires_grad_(enabled)
    return code.detach()


class CompatibilityCodedAirfoilOperator(nn.Module):
    """Freeze a retained endpoint and append the one-message local composer.

    The retained interface/strip operator, when present, remains the existing
    post-POU ``correct_surface_level`` stage.  Only ``composer`` parameters are
    trainable.  Sparse fine execution can call
    :meth:`correct_post_endpoint_compatibility` after filling the inactive
    tiles with its prolonged-coarse baseline.
    """

    def __init__(
        self,
        retained_endpoint: nn.Module,
        composer: AirfoilCompatibilityComposer,
    ) -> None:
        super().__init__()
        self.retained_endpoint = retained_endpoint
        self.composer = composer
        self.code_mode: CodeMode = "predicted"
        self.oracle_code: Tensor | None = None
        self.shuffle_permutation: Tensor | None = None
        self.last_compatibility: CompatibilityOutput | None = None
        for parameter in self.retained_endpoint.parameters():
            parameter.requires_grad_(False)
        self.retained_endpoint.eval()

    @property
    def _local_operator(self) -> nn.Module:
        return getattr(self.retained_endpoint, "shared_operator", self.retained_endpoint)

    def train(self, mode: bool = True):
        super().train(mode)
        self.retained_endpoint.eval()
        self.composer.train(mode)
        return self

    def set_code_override(
        self,
        mode: CodeMode,
        *,
        oracle_code: Tensor | None = None,
        shuffle_permutation: Tensor | None = None,
    ) -> None:
        self.code_mode = mode
        self.oracle_code = oracle_code
        self.shuffle_permutation = shuffle_permutation

    def set_graph_radius(self, radius: int | None) -> None:
        """Select full, self-only (0), or finite-radius tile communication."""
        if radius is not None and radius < 0:
            raise ValueError("graph radius must be nonnegative or None")
        self.composer.max_graph_radius = radius

    def reset_attention_calls(self) -> None:
        self.composer.reset_attention_calls()

    @property
    def attention_calls(self) -> int:
        return self.composer.attention_calls

    def set_endpoint_statistics(self, *args) -> None:
        self._local_operator.set_endpoint_statistics(*args)
        state_mean, state_std, delta_std = args[:3]
        delta_mean = args[3] if len(args) > 3 else (0.0, 0.0, 0.0)
        self.composer.set_statistics(state_mean, state_std, delta_mean, delta_std)

    @property
    def endpoint_learned_projection(self) -> bool:
        return bool(getattr(self._local_operator, "endpoint_learned_projection", False))

    @property
    def endpoint_eos_direct(self) -> bool:
        return bool(getattr(self._local_operator, "endpoint_eos_direct", False))

    @property
    def endpoint_multilevel_projection(self) -> bool:
        return bool(getattr(self._local_operator, "endpoint_multilevel_projection", False))

    @property
    def endpoint_vorticity_residual(self) -> bool:
        return bool(getattr(self._local_operator, "endpoint_vorticity_residual", False))

    def correct_endpoint_fields(
        self,
        endpoint: Tensor,
        physical: Tensor,
        scalar: Tensor,
        tile_count: int | None = None,
        *,
        tile_grid_shape: tuple[int, int] | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        # The frozen endpoint first produces executable local physical fields;
        # no backward graph or optimizer state is allocated for its 13.65M
        # retained parameters.  Composition acts only on those outputs.
        with torch.no_grad():
            local, vorticity = self._local_operator.correct_endpoint_fields(
                endpoint, physical, scalar, tile_count,
                tile_grid_shape=tile_grid_shape, **kwargs,
            )
        if tile_grid_shape is None:
            return local, vorticity
        result = self.composer(
            local, physical, scalar, tile_grid_shape,
            code_mode=self.code_mode, oracle_code=self.oracle_code,
            shuffle_permutation=self.shuffle_permutation,
        )
        self.last_compatibility = result
        return result.corrected, vorticity

    def correct_post_endpoint_compatibility(
        self,
        local_endpoint: Tensor,
        physical: Tensor,
        scalar: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        result = self.composer(
            local_endpoint, physical, scalar, grid_shape,
            code_mode=self.code_mode, oracle_code=self.oracle_code,
            shuffle_permutation=self.shuffle_permutation,
        )
        self.last_compatibility = result
        return result.corrected

    def correct_surface_level(self, *args, **kwargs) -> Tensor:
        if not hasattr(self.retained_endpoint, "correct_surface_level"):
            return args[1]
        return self.retained_endpoint.correct_surface_level(*args, **kwargs)

    def correct_endpoint(self, endpoint: Tensor, physical: Tensor, scalar: Tensor) -> Tensor:
        return self.correct_endpoint_fields(endpoint, physical, scalar)[0]
