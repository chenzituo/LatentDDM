"""Geometry-boundary EoS shared by Cartesian cells and AMR levels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from codiffuse_Schwarz_Airfoil.interface_trace import (
    interface_trace,
    naca_half_thickness,
)
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import load_pretrained


Tensor = torch.Tensor
OFFSETS = (-1.5, -.5, .5, 1.5, 3.0)


class TraceResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, width), width)
        self.conv1 = nn.Conv1d(
            width, width, 5, padding=2 * dilation, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            width, width, 5, padding=2 * dilation, dilation=dilation,
        )

    def forward(self, state: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm(state)))
        return state + self.conv2(F.silu(hidden))


class TraceRasterResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, width), width)
        self.conv1 = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation, bias=False,
        )
        self.conv2 = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation, bias=False,
        )

    def forward(self, state: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm(state)))
        return state + self.conv2(F.silu(hidden))


class InterfaceTraceOperators(nn.Module):
    """Frozen local field operator plus a learned ordered-interface EoS."""

    def __init__(
        self,
        shared_operator: nn.Module,
        *,
        width: int = 64,
        depth: int = 4,
        stations: int = 64,
        solid_velocity_scale: float = 0.0,
        vorticity_latent: bool = False,
        strip_operator: bool = False,
        strip_transport_operator: bool = False,
        strip_normal_samples: int = 24,
        strip_normal_min: float = -3.0,
        strip_normal_max: float = 8.0,
        trace_transport_operator: bool = False,
        interior_force_operator: bool = False,
    ) -> None:
        super().__init__()
        if (
            min(width, depth, stations, strip_normal_samples) <= 0
            or not 0.0 <= solid_velocity_scale <= 1.0
            or strip_normal_min >= strip_normal_max
        ):
            raise ValueError("trace operator dimensions must be positive")
        self.shared_operator = shared_operator
        self.trace_width = width
        self.trace_depth = depth
        self.trace_stations = stations
        self.interface_solid_velocity_scale = solid_velocity_scale
        self.interface_vorticity_latent = vorticity_latent
        self.interface_strip_operator = strip_operator
        self.interface_strip_transport_operator = strip_transport_operator
        self.interface_trace_transport_operator = trace_transport_operator
        self.interface_interior_force_operator = interior_force_operator
        self.strip_normal_samples = strip_normal_samples
        self.strip_normal_min = strip_normal_min
        self.strip_normal_max = strip_normal_max
        self.strip_offsets = tuple(
            float(value) for value in torch.linspace(
                strip_normal_min, strip_normal_max, strip_normal_samples,
            )
        )
        trace_channels = 4 * 2 * len(OFFSETS)
        self.trace_stem = nn.Conv1d(2 * trace_channels + 8, width, 5, padding=2)
        self.trace_blocks = nn.ModuleList(
            TraceResidualBlock(width, (1, 2, 4)[index % 3])
            for index in range(depth)
        )
        output_channels = (4 if vorticity_latent else 3) * 2 * len(OFFSETS)
        self.trace_output = nn.Conv1d(width, output_channels, 3, padding=1)
        nn.init.zeros_(self.trace_output.weight)
        nn.init.zeros_(self.trace_output.bias)
        self.trace_transport_output: nn.Conv1d | None = None
        if trace_transport_operator:
            # A gate, rather than a free correction, explicitly carries the
            # wall-relative velocity profile between prescribed body poses.
            self.trace_transport_output = nn.Conv1d(
                width, 2 * 2 * len(OFFSETS), 3, padding=1,
            )
            nn.init.zeros_(self.trace_transport_output.weight)
            nn.init.zeros_(self.trace_transport_output.bias)
        self.trace_raster_stem: nn.Conv2d | None = None
        self.trace_raster_blocks: nn.ModuleList | None = None
        self.trace_raster_output: nn.Conv2d | None = None
        if vorticity_latent:
            raster_width = max(16, width // 2)
            self.trace_raster_stem = nn.Conv2d(
                4, raster_width, 5, padding=2, bias=False,
            )
            self.trace_raster_blocks = nn.ModuleList(
                TraceRasterResidualBlock(raster_width, (1, 2, 4)[index % 3])
                for index in range(max(2, depth // 2))
            )
            self.trace_raster_output = nn.Conv2d(
                raster_width, 3, 3, padding=1, bias=False,
            )
            nn.init.zeros_(self.trace_raster_output.weight)
        self.strip_stem: nn.Conv2d | None = None
        self.strip_blocks: nn.ModuleList | None = None
        self.strip_output: nn.Conv2d | None = None
        self.strip_transport_output: nn.Conv2d | None = None
        if strip_transport_operator and not strip_operator:
            raise ValueError("strip transport requires the body-strip operator")
        if strip_operator:
            self.strip_stem = nn.Conv2d(16, width, 5, padding=2, bias=False)
            self.strip_blocks = nn.ModuleList(
                TraceRasterResidualBlock(width, (1, 2, 4)[index % 3])
                for index in range(depth)
            )
            self.strip_output = nn.Conv2d(
                width, 3, 3, padding=1, bias=False,
            )
            nn.init.zeros_(self.strip_output.weight)
            if strip_transport_operator:
                self.strip_transport_output = nn.Conv2d(
                    width, 3, 3, padding=1, bias=True,
                )
                nn.init.zeros_(self.strip_transport_output.weight)
                nn.init.zeros_(self.strip_transport_output.bias)
        self.force_stem: nn.Conv1d | None = None
        self.force_blocks: nn.ModuleList | None = None
        self.force_output: nn.Linear | None = None
        if interior_force_operator:
            self.force_stem = nn.Conv1d(
                2 * trace_channels + 8, width, 5, padding=2,
            )
            self.force_blocks = nn.ModuleList(
                TraceResidualBlock(width, (1, 2, 4)[index % 3])
                for index in range(max(2, depth // 2))
            )
            self.force_output = nn.Linear(width, 2)
            nn.init.zeros_(self.force_output.weight)
            nn.init.zeros_(self.force_output.bias)

    def set_endpoint_statistics(self, *args) -> None:
        self.shared_operator.set_endpoint_statistics(*args)

    @property
    def gradient_checkpointing(self) -> bool:
        return bool(self.shared_operator.gradient_checkpointing)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, enabled: bool) -> None:
        self.shared_operator.gradient_checkpointing = enabled

    def _trace_correction(
        self,
        current: Tensor,
        prediction: Tensor,
        motion_n: Tensor,
        motion_np1: Tensor,
        *,
        dx: float,
        window: tuple[float, float, float, float],
        thickness: float,
        pivot_fraction: float,
    ) -> Tensor:
        current_trace = interface_trace(
            current, motion_n, dx=dx, window=window, thickness=thickness,
            pivot_fraction=pivot_fraction, stations=self.trace_stations,
            normal_offsets=OFFSETS,
        )
        predicted_trace = interface_trace(
            prediction, motion_np1, dx=dx, window=window, thickness=thickness,
            pivot_fraction=pivot_fraction, stations=self.trace_stations,
            normal_offsets=OFFSETS,
        )
        scale = prediction.new_tensor((1.0, 1.0, 1.0, 50.0))[
            None, :, None, None, None
        ]
        traces = torch.cat((current_trace / scale, predicted_trace / scale), dim=1)
        traces = traces.flatten(1, 3)
        motion = torch.cat((motion_n[:, :4], motion_np1[:, :4]), dim=1)
        features = torch.cat((
            traces, motion[:, :, None].expand(-1, -1, self.trace_stations),
        ), dim=1)
        hidden = self.trace_stem(features)
        for block in self.trace_blocks:
            hidden = block(hidden)
        channels = 4 if self.interface_vorticity_latent else 3
        correction = .1 * torch.tanh(self.trace_output(F.silu(hidden)))
        correction = correction.reshape(
            prediction.shape[0], channels, 2, len(OFFSETS), self.trace_stations,
        )
        if self.interface_vorticity_latent:
            # Vorticity is roughly fifty times larger than velocity/pressure
            # in this benchmark. Keep the learned trace in balanced units and
            # restore its physical amplitude only at rasterization.
            amplitude = correction.new_tensor((1.0, 1.0, 1.0, 50.0))[
                None, :, None, None, None
            ]
            correction = correction * amplitude
        if self.trace_transport_output is not None:
            gate = .5 * torch.tanh(
                self.trace_transport_output(F.silu(hidden))
            ).reshape(
                prediction.shape[0], 2, 2, len(OFFSETS), self.trace_stations,
            )
            transported_velocity = current_trace[:, :2] - predicted_trace[:, :2]
            correction = torch.cat((
                correction[:, :2] + gate * transported_velocity,
                correction[:, 2:],
            ), dim=1)
        return correction

    @staticmethod
    def _rasterize(
        correction: Tensor,
        sdf: Tensor,
        motion: Tensor,
        *,
        dx: float,
        window: tuple[float, float, float, float],
        thickness: float,
        pivot_fraction: float,
        solid_velocity_scale: float,
    ) -> Tensor:
        batch, _, height, width = sdf.shape
        x = window[0] + (
            torch.arange(width, device=sdf.device, dtype=sdf.dtype) + .5
        ) * dx
        y = window[2] + (
            torch.arange(height, device=sdf.device, dtype=sdf.dtype) + .5
        ) * dx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        angle = motion[:, 0, None, None]
        cosine, sine = angle.cos(), angle.sin()
        x_body = cosine * xx + sine * yy
        y_body = -sine * xx + cosine * yy
        xi = x_body + pivot_fraction

        station_coordinate = 2.0 * (xi - .01) / .98 - 1.0
        sample_grid = torch.stack((
            station_coordinate,
            torch.zeros_like(station_coordinate),
        ), dim=-1)
        sampled_sides = []
        channels = correction.shape[1]
        for side in range(2):
            side_trace = correction[:, :, side].reshape(
                batch, channels * len(OFFSETS), 1, -1,
            )
            sampled = F.grid_sample(
                side_trace, sample_grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            ).reshape(batch, channels, len(OFFSETS), height, width)
            sampled_sides.append(sampled)
        upper = (y_body >= 0.0)[:, None, None]
        sampled = torch.where(upper, sampled_sides[0], sampled_sides[1])
        offsets = sdf.new_tensor(OFFSETS)[None, None, :, None, None]
        normal_coordinate = sdf[:, :, None] / dx
        normal_weight = torch.softmax(
            -2.0 * (normal_coordinate - offsets).square(), dim=2,
        )
        local = (sampled * normal_weight).sum(dim=2)

        xi_clamped = xi.clamp(.01, .99)
        _, slope = naca_half_thickness(xi_clamped, thickness)
        upper_tangent = F.normalize(torch.stack((
            torch.ones_like(slope), slope,
        ), dim=1), dim=1)
        lower_tangent = F.normalize(torch.stack((
            torch.ones_like(slope), -slope,
        ), dim=1), dim=1)
        tangent = torch.where(
            (y_body >= 0.0)[:, None], upper_tangent, lower_tangent,
        )
        upper_normal = torch.stack((-upper_tangent[:, 1], upper_tangent[:, 0]), dim=1)
        lower_normal = torch.stack((lower_tangent[:, 1], -lower_tangent[:, 0]), dim=1)
        normal = torch.where(
            (y_body >= 0.0)[:, None], upper_normal, lower_normal,
        )
        body_u = local[:, 0] * tangent[:, 0] + local[:, 1] * normal[:, 0]
        body_v = local[:, 0] * tangent[:, 1] + local[:, 1] * normal[:, 1]
        lab_u = cosine * body_u - sine * body_v
        lab_v = sine * body_u + cosine * body_v
        gate = (
            (xi >= .01) & (xi <= .99)
        ).to(sdf.dtype) * torch.exp(-.5 * (sdf[:, 0] / (3.0 * dx)).square())
        velocity_gate = gate * torch.where(
            sdf[:, 0] >= 0.0,
            torch.ones_like(sdf[:, 0]),
            torch.full_like(sdf[:, 0], solid_velocity_scale),
        )
        field = torch.stack((
            lab_u * velocity_gate,
            lab_v * velocity_gate,
            local[:, 2] * gate,
        ), dim=1)
        if correction.shape[1] == 4:
            field = torch.cat((field, (local[:, 3] * gate)[:, None]), dim=1)
        return field

    def _decode_raster(self, raster: Tensor, sdf: Tensor, dx: float) -> Tensor:
        if not self.interface_vorticity_latent:
            return raster
        assert self.trace_raster_stem is not None
        assert self.trace_raster_blocks is not None
        assert self.trace_raster_output is not None
        hidden = self.trace_raster_stem(raster)
        for block in self.trace_raster_blocks:
            hidden = block(hidden)
        # The direct first three channels retain the established trace path;
        # the learned residual lets the explicit omega trace alter a locally
        # coherent velocity field. Bias-free layers make zero trace exactly
        # identity preserving.
        geometry_gate = torch.exp(-.5 * (sdf / (3.5 * dx)).square())
        return raster[:, :3] + geometry_gate * .1 * torch.tanh(
            self.trace_raster_output(F.silu(hidden))
        )

    def _strip_correction(
        self,
        current: Tensor,
        prediction: Tensor,
        motion_n: Tensor,
        motion_np1: Tensor,
        *,
        dx: float,
        window: tuple[float, float, float, float],
        thickness: float,
        pivot_fraction: float,
    ) -> Tensor:
        if not self.interface_strip_operator:
            raise RuntimeError("body strip operator is disabled")
        assert self.strip_stem is not None
        assert self.strip_blocks is not None
        assert self.strip_output is not None
        current_strip = interface_trace(
            current, motion_n, dx=dx, window=window, thickness=thickness,
            pivot_fraction=pivot_fraction, stations=self.trace_stations,
            normal_offsets=self.strip_offsets,
        )
        predicted_strip = interface_trace(
            prediction, motion_np1, dx=dx, window=window,
            thickness=thickness, pivot_fraction=pivot_fraction,
            stations=self.trace_stations, normal_offsets=self.strip_offsets,
        )
        scale = prediction.new_tensor((1.0, 1.0, 1.0, 50.0))[
            None, :, None, None, None
        ]
        field = torch.cat((current_strip / scale, predicted_strip / scale), dim=1)
        motion = torch.cat((motion_n[:, :4], motion_np1[:, :4]), dim=1)
        motion = motion[:, :, None, None, None].expand(
            -1, -1, 2, self.strip_normal_samples, self.trace_stations,
        )
        features = torch.cat((field, motion), dim=1).permute(
            0, 2, 1, 3, 4,
        ).reshape(
            -1, 16, self.strip_normal_samples, self.trace_stations,
        )
        hidden = self.strip_stem(features)
        for block in self.strip_blocks:
            hidden = block(hidden)
        activated = F.silu(hidden)
        correction = .1 * torch.tanh(self.strip_output(activated))
        if self.strip_transport_output is not None:
            # Both traces use body-relative velocity coordinates, so their
            # difference transports the attached boundary layer between poses
            # without asking an Eulerian crop to rediscover the rigid motion.
            transported = (
                current_strip[:, :3] - predicted_strip[:, :3]
            ).permute(0, 2, 1, 3, 4).reshape(
                -1, 3, self.strip_normal_samples, self.trace_stations,
            )
            gate = torch.tanh(self.strip_transport_output(activated))
            correction = correction + gate * transported
        batch = prediction.shape[0]
        return correction.reshape(
            batch, 2, 3, self.strip_normal_samples, self.trace_stations,
        ).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _rasterize_strip(
        correction: Tensor,
        sdf: Tensor,
        motion: Tensor,
        *,
        dx: float,
        window: tuple[float, float, float, float],
        thickness: float,
        pivot_fraction: float,
        normal_min: float,
        normal_max: float,
        solid_velocity_scale: float,
    ) -> Tensor:
        batch, _, height, width = sdf.shape
        x = window[0] + (
            torch.arange(width, device=sdf.device, dtype=sdf.dtype) + .5
        ) * dx
        y = window[2] + (
            torch.arange(height, device=sdf.device, dtype=sdf.dtype) + .5
        ) * dx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        angle = motion[:, 0, None, None]
        cosine, sine = angle.cos(), angle.sin()
        x_body = cosine * xx + sine * yy
        y_body = -sine * xx + cosine * yy
        xi = x_body + pivot_fraction
        station_coordinate = 2.0 * (xi - .01) / .98 - 1.0
        normal_coordinate = 2.0 * (
            sdf[:, 0] / dx - normal_min
        ) / (normal_max - normal_min) - 1.0
        sample_grid = torch.stack((station_coordinate, normal_coordinate), dim=-1)
        sampled_sides = [
            F.grid_sample(
                correction[:, :, side], sample_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )
            for side in range(2)
        ]
        local = torch.where(
            (y_body >= 0.0)[:, None], sampled_sides[0], sampled_sides[1],
        )

        xi_clamped = xi.clamp(.01, .99)
        _, slope = naca_half_thickness(xi_clamped, thickness)
        upper_tangent = F.normalize(torch.stack((
            torch.ones_like(slope), slope,
        ), dim=1), dim=1)
        lower_tangent = F.normalize(torch.stack((
            torch.ones_like(slope), -slope,
        ), dim=1), dim=1)
        tangent = torch.where(
            (y_body >= 0.0)[:, None], upper_tangent, lower_tangent,
        )
        upper_normal = torch.stack(
            (-upper_tangent[:, 1], upper_tangent[:, 0]), dim=1,
        )
        lower_normal = torch.stack(
            (lower_tangent[:, 1], -lower_tangent[:, 0]), dim=1,
        )
        normal = torch.where(
            (y_body >= 0.0)[:, None], upper_normal, lower_normal,
        )
        body_u = local[:, 0] * tangent[:, 0] + local[:, 1] * normal[:, 0]
        body_v = local[:, 0] * tangent[:, 1] + local[:, 1] * normal[:, 1]
        lab_u = cosine * body_u - sine * body_v
        lab_v = sine * body_u + cosine * body_v
        chord_gate = ((xi >= .01) & (xi <= .99)).to(sdf.dtype)
        velocity_gate = chord_gate * torch.where(
            sdf[:, 0] >= 0.0,
            torch.ones_like(sdf[:, 0]),
            torch.full_like(
                sdf[:, 0], solid_velocity_scale,
            ),
        )
        field = torch.stack((
            lab_u * velocity_gate,
            lab_v * velocity_gate,
            local[:, 2] * chord_gate,
        ), dim=1)
        if correction.shape[1] == 4:
            field = torch.cat((field, (local[:, 3] * chord_gate)[:, None]), dim=1)
        return field

    def _force_correction(
        self,
        current: Tensor,
        prediction: Tensor,
        motion_n: Tensor,
        motion_np1: Tensor,
        *,
        dx: float,
        window: tuple[float, float, float, float],
        thickness: float,
        pivot_fraction: float,
    ) -> Tensor:
        if not self.interface_interior_force_operator:
            raise RuntimeError("interior-force operator is disabled")
        assert self.force_stem is not None
        assert self.force_blocks is not None
        assert self.force_output is not None
        current_trace = interface_trace(
            current, motion_n, dx=dx, window=window, thickness=thickness,
            pivot_fraction=pivot_fraction, stations=self.trace_stations,
            normal_offsets=OFFSETS,
        )
        predicted_trace = interface_trace(
            prediction, motion_np1, dx=dx, window=window, thickness=thickness,
            pivot_fraction=pivot_fraction, stations=self.trace_stations,
            normal_offsets=OFFSETS,
        )
        scale = prediction.new_tensor((1.0, 1.0, 1.0, 50.0))[
            None, :, None, None, None
        ]
        traces = torch.cat((current_trace / scale, predicted_trace / scale), dim=1)
        traces = traces.flatten(1, 3)
        motion = torch.cat((motion_n[:, :4], motion_np1[:, :4]), dim=1)
        features = torch.cat((
            traces, motion[:, :, None].expand(-1, -1, self.trace_stations),
        ), dim=1)
        hidden = self.force_stem(features)
        for block in self.force_blocks:
            hidden = block(hidden)
        return 5.0 * torch.tanh(self.force_output(F.silu(hidden).mean(dim=-1)))

    @staticmethod
    def _apply_force_correction(
        prediction: Tensor,
        force_correction: Tensor,
        sdf: Tensor,
        *,
        dx: float,
        penalty_time: float,
        rotation_sign: float,
    ) -> Tensor:
        """Write a requested load correction into integrated solid slip."""
        if penalty_time <= 0.0 or rotation_sign == 0.0:
            raise ValueError("invalid Brinkman force convention")
        solid = (sdf <= 0.0).to(prediction.dtype)
        represented_area = (solid.sum(dim=(-2, -1)) * dx**2).clamp_min(dx**2)
        delta_u = (
            penalty_time * force_correction[:, 0:1]
            / (2.0 * represented_area)
        )[:, :, None, None]
        delta_v = (
            -penalty_time * force_correction[:, 1:2]
            / (2.0 * float(rotation_sign) * represented_area)
        )[:, :, None, None]
        correction = torch.cat((
            delta_u * solid, delta_v * solid, torch.zeros_like(solid),
        ), dim=1)
        return prediction + correction

    def correct_surface_level(
        self,
        level: str,
        prediction: Tensor,
        batch: dict[str, Tensor],
        dataset,
    ) -> Tensor:
        del level
        current = batch["q_n"].to(prediction)
        motion_n = batch["motion_n"].to(prediction)
        motion_np1 = batch["motion_np1"].to(prediction)
        thickness = float(dataset.physical_parameters["thickness"])
        pivot = float(dataset.physical_parameters["pivot_fraction"])
        correction = self._trace_correction(
            current, prediction, motion_n, motion_np1, dx=dataset.dx,
            window=dataset.window, thickness=thickness, pivot_fraction=pivot,
        )
        raster = self._rasterize(
            correction, batch["sdf_np1"].to(prediction), motion_np1,
            dx=dataset.dx, window=dataset.window, thickness=thickness,
            pivot_fraction=pivot,
            solid_velocity_scale=self.interface_solid_velocity_scale,
        )
        corrected = prediction + self._decode_raster(
            raster, batch["sdf_np1"].to(prediction), dataset.dx,
        )
        if self.interface_strip_operator:
            strip = self._strip_correction(
                current, corrected, motion_n, motion_np1, dx=dataset.dx,
                window=dataset.window, thickness=thickness,
                pivot_fraction=pivot,
            )
            corrected = corrected + self._rasterize_strip(
                strip, batch["sdf_np1"].to(prediction), motion_np1,
                dx=dataset.dx, window=dataset.window, thickness=thickness,
                pivot_fraction=pivot,
                normal_min=self.strip_normal_min,
                normal_max=self.strip_normal_max,
                solid_velocity_scale=self.interface_solid_velocity_scale,
            )
        if self.interface_interior_force_operator:
            load_correction = self._force_correction(
                current, corrected, motion_n, motion_np1, dx=dataset.dx,
                window=dataset.window, thickness=thickness,
                pivot_fraction=pivot,
            )
            corrected = self._apply_force_correction(
                corrected, load_correction, batch["sdf_np1"].to(prediction),
                dx=dataset.dx, penalty_time=float(dataset.penalty_time),
                rotation_sign=float(dataset.rotation_sign),
            )
        return corrected


def clone_interface_trace_checkpoint(
    checkpoint: dict, device: torch.device,
    *, width: int = 64, depth: int = 4, stations: int = 64,
    solid_velocity_scale: float = 0.0,
    vorticity_latent: bool = False,
    strip_operator: bool = False,
    strip_transport_operator: bool = False,
    strip_normal_samples: int = 24,
    trace_transport_operator: bool = False,
    interior_force_operator: bool = False,
) -> tuple[InterfaceTraceOperators, dict]:
    shared, spec = load_pretrained(checkpoint, device, history_steps=2)
    model = InterfaceTraceOperators(
        shared, width=width, depth=depth, stations=stations,
        solid_velocity_scale=solid_velocity_scale,
        vorticity_latent=vorticity_latent,
        strip_operator=strip_operator,
        strip_transport_operator=strip_transport_operator,
        strip_normal_samples=strip_normal_samples,
        trace_transport_operator=trace_transport_operator,
        interior_force_operator=interior_force_operator,
    ).to(device)
    return model, dict(spec) | {
        "interface_trace_operator": True,
        "interface_operator_width": width,
        "interface_operator_depth": depth,
        "interface_operator_stations": stations,
        "interface_solid_velocity_scale": solid_velocity_scale,
        "interface_vorticity_latent": vorticity_latent,
        "interface_strip_operator": strip_operator,
        "interface_strip_transport_operator": strip_transport_operator,
        "interface_strip_normal_samples": strip_normal_samples,
        "interface_trace_transport_operator": trace_transport_operator,
        "interface_interior_force_operator": interior_force_operator,
    }


def load_interface_trace_checkpoint(
    checkpoint: dict,
    device: torch.device,
    *,
    endpoint_curl_memory: bool = False,
    endpoint_curl_memory_width: int = 32,
    endpoint_curl_memory_depth: int = 2,
    endpoint_tile_communication: bool = False,
) -> tuple[InterfaceTraceOperators, dict]:
    spec = dict(checkpoint.get("model_spec", {}))
    if not bool(spec.get("interface_trace_operator", False)):
        return clone_interface_trace_checkpoint(checkpoint, device)
    prefix = "shared_operator."
    shared_state = {
        key.removeprefix(prefix): value for key, value in checkpoint["model"].items()
        if key.startswith(prefix)
    }
    if not shared_state:
        raise ValueError("interface checkpoint is missing its shared operator")
    payload = dict(checkpoint)
    payload["model"] = shared_state
    payload["model_spec"] = {
        key: value for key, value in spec.items()
        if not key.startswith("interface_")
    }
    shared, shared_spec = load_pretrained(
        payload, device, history_steps=2,
        endpoint_curl_memory=endpoint_curl_memory,
        endpoint_curl_memory_width=endpoint_curl_memory_width,
        endpoint_curl_memory_depth=endpoint_curl_memory_depth,
        endpoint_tile_communication=endpoint_tile_communication,
    )
    model = InterfaceTraceOperators(
        shared, width=int(spec.get("interface_operator_width", 64)),
        depth=int(spec.get("interface_operator_depth", 4)),
        stations=int(spec.get("interface_operator_stations", 64)),
        solid_velocity_scale=float(spec.get("interface_solid_velocity_scale", 1.0)),
        vorticity_latent=bool(spec.get("interface_vorticity_latent", False)),
        strip_operator=bool(spec.get("interface_strip_operator", False)),
        strip_transport_operator=bool(spec.get(
            "interface_strip_transport_operator", False,
        )),
        strip_normal_samples=int(spec.get("interface_strip_normal_samples", 24)),
        strip_normal_min=float(spec.get("interface_strip_normal_min", -3.0)),
        strip_normal_max=float(spec.get("interface_strip_normal_max", 8.0)),
        trace_transport_operator=bool(spec.get(
            "interface_trace_transport_operator", False,
        )),
        interior_force_operator=bool(spec.get(
            "interface_interior_force_operator", False,
        )),
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = set()
    if endpoint_curl_memory and not bool(spec.get("endpoint_curl_memory", False)):
        allowed_missing = {
            key for key in model.state_dict()
            if key.startswith("shared_operator.curl_memory_")
        }
    if endpoint_tile_communication and not bool(
        spec.get("endpoint_tile_communication", False)
    ):
        allowed_missing |= {
            key for key in model.state_dict()
            if key.startswith("shared_operator.endpoint_tile_communicator.")
        }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(
            "incompatible interface checkpoint: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return model, dict(shared_spec) | {
        key: value for key, value in spec.items() if key.startswith("interface_")
    }


def upgrade_interface_vorticity_checkpoint(
    checkpoint: dict, device: torch.device,
) -> tuple[InterfaceTraceOperators, dict]:
    """Add the vorticity-bearing trace decoder without changing predictions."""
    old, spec = load_interface_trace_checkpoint(checkpoint, device)
    if old.interface_vorticity_latent:
        return old, spec
    model = InterfaceTraceOperators(
        old.shared_operator,
        width=old.trace_width,
        depth=old.trace_depth,
        stations=old.trace_stations,
        solid_velocity_scale=old.interface_solid_velocity_scale,
        vorticity_latent=True,
        strip_operator=old.interface_strip_operator,
        strip_transport_operator=old.interface_strip_transport_operator,
        strip_normal_samples=old.strip_normal_samples,
        strip_normal_min=old.strip_normal_min,
        strip_normal_max=old.strip_normal_max,
        trace_transport_operator=old.interface_trace_transport_operator,
        interior_force_operator=old.interface_interior_force_operator,
    ).to(device)
    old_state = {
        key: value for key, value in old.state_dict().items()
        if not key.startswith("trace_output.")
    }
    missing, unexpected = model.load_state_dict(old_state, strict=False)
    if unexpected or any(not key.startswith(
        ("trace_output.", "trace_raster_"),
    ) for key in missing):
        raise RuntimeError(
            "unexpected vorticity upgrade state: "
            f"missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        old_channels = old.trace_output.out_channels
        model.trace_output.weight[:old_channels].copy_(old.trace_output.weight)
        model.trace_output.bias[:old_channels].copy_(old.trace_output.bias)
    return model, dict(spec) | {"interface_vorticity_latent": True}


def upgrade_interface_strip_checkpoint(
    checkpoint: dict, device: torch.device, *, normal_samples: int = 24,
    transport_operator: bool = False,
) -> tuple[InterfaceTraceOperators, dict]:
    """Add a dense body-frame strip while preserving the parent prediction."""
    old, spec = load_interface_trace_checkpoint(checkpoint, device)
    if old.interface_strip_operator and (
        not transport_operator or old.interface_strip_transport_operator
    ):
        return old, spec
    model = InterfaceTraceOperators(
        old.shared_operator,
        width=old.trace_width,
        depth=old.trace_depth,
        stations=old.trace_stations,
        solid_velocity_scale=old.interface_solid_velocity_scale,
        vorticity_latent=old.interface_vorticity_latent,
        strip_operator=True,
        strip_transport_operator=transport_operator,
        strip_normal_samples=normal_samples,
        trace_transport_operator=old.interface_trace_transport_operator,
        interior_force_operator=old.interface_interior_force_operator,
    ).to(device)
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    if unexpected or any(not key.startswith("strip_") for key in missing):
        raise RuntimeError(
            f"unexpected strip upgrade state: missing={missing}, unexpected={unexpected}"
        )
    return model, dict(spec) | {
        "interface_strip_operator": True,
        "interface_strip_transport_operator": transport_operator,
        "interface_strip_normal_samples": normal_samples,
        "interface_strip_normal_min": -3.0,
        "interface_strip_normal_max": 8.0,
    }


def upgrade_interface_force_checkpoint(
    checkpoint: dict, device: torch.device,
) -> tuple[InterfaceTraceOperators, dict]:
    """Add an integrated interior-slip EoS without changing the parent."""
    old, spec = load_interface_trace_checkpoint(checkpoint, device)
    if old.interface_interior_force_operator:
        return old, spec
    model = InterfaceTraceOperators(
        old.shared_operator,
        width=old.trace_width,
        depth=old.trace_depth,
        stations=old.trace_stations,
        solid_velocity_scale=old.interface_solid_velocity_scale,
        vorticity_latent=old.interface_vorticity_latent,
        strip_operator=old.interface_strip_operator,
        strip_transport_operator=old.interface_strip_transport_operator,
        strip_normal_samples=old.strip_normal_samples,
        strip_normal_min=old.strip_normal_min,
        strip_normal_max=old.strip_normal_max,
        trace_transport_operator=old.interface_trace_transport_operator,
        interior_force_operator=True,
    ).to(device)
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    if unexpected or any(not key.startswith("force_") for key in missing):
        raise RuntimeError(
            f"unexpected force upgrade state: missing={missing}, unexpected={unexpected}"
        )
    return model, dict(spec) | {"interface_interior_force_operator": True}


def upgrade_interface_transport_checkpoint(
    checkpoint: dict, device: torch.device,
) -> tuple[InterfaceTraceOperators, dict]:
    """Add body-frame trace transport while preserving the parent exactly."""
    old, spec = load_interface_trace_checkpoint(checkpoint, device)
    if old.interface_trace_transport_operator:
        return old, spec
    model = InterfaceTraceOperators(
        old.shared_operator,
        width=old.trace_width,
        depth=old.trace_depth,
        stations=old.trace_stations,
        solid_velocity_scale=old.interface_solid_velocity_scale,
        vorticity_latent=old.interface_vorticity_latent,
        strip_operator=old.interface_strip_operator,
        strip_transport_operator=old.interface_strip_transport_operator,
        strip_normal_samples=old.strip_normal_samples,
        strip_normal_min=old.strip_normal_min,
        strip_normal_max=old.strip_normal_max,
        trace_transport_operator=True,
        interior_force_operator=old.interface_interior_force_operator,
    ).to(device)
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    if unexpected or any(
        not key.startswith("trace_transport_output.") for key in missing
    ):
        raise RuntimeError(
            "unexpected trace-transport upgrade state: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model, dict(spec) | {"interface_trace_transport_operator": True}
