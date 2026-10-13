"""One-message zero-preserving Airfoil student for q_Gamma + q_momentum."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from codiffuse_Schwarz.tile_communication import (
    ResidualFieldCodeLift, ResidualKernelCodeMap,
    StateGatedResidualKernelCodeMap,
)
from .airfoil_two_level import centered_gradient
from .airfoil_fixed_point_analyzer import compatibility_face_support
from .airfoil_latent_schur_physical_ablation import (
    _components, _decode24, _physical_prediction, _weighted_vector,
)


Tensor = torch.Tensor


class DampedStudentOutput(NamedTuple):
    corrected: Tensor
    code: Tensor
    alpha: Tensor
    compatibility_cycle: Tensor
    tile_momentum_cycle: Tensor
    combined_cycle: Tensor
    message_calls: int
    qmom_analyses: int
    line_search_candidates: int


class CenteredResponseChartLift2D(nn.Module):
    """Add response directions without changing the frozen local endpoint.

    The existing Airfoil chart exposes 32 response coordinates.  This lift
    appends a small set of learned coordinates whose spatial basis is computed
    from the already executable local field and its physical semantics.  The
    coefficient map has no bias, so a zero appended code gives a bitwise-zero
    correction for every local state.  In particular, enriching the response
    chart does not weaken LatentDDM's compatible fixed-point contract.
    """

    def __init__(
        self, *, feature_channels: int, field_channels: int,
        code_width: int, rank: int, width: int,
        correction_scale: float,
    ) -> None:
        super().__init__()
        if min(
            feature_channels, field_channels, code_width, rank, width,
        ) <= 0 or correction_scale <= 0.0:
            raise ValueError("invalid centered response-chart lift dimensions")
        self.field_channels = int(field_channels)
        self.code_width = int(code_width)
        self.rank = int(rank)
        self.correction_scale = float(correction_scale)
        groups = min(8, width)
        while width % groups:
            groups -= 1
        self.basis = nn.Sequential(
            nn.Conv2d(feature_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(
                width, field_channels * rank, 3, padding=1, bias=False,
            ),
        )
        self.coefficients = nn.Linear(
            code_width, field_channels * rank, bias=False,
        )

    def correction(
        self, local: Tensor, semantics: Tensor, code: Tensor, valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        if local.ndim != 4 or local.shape[1] != self.field_channels:
            raise ValueError("response-lift local field has incompatible shape")
        if local.shape[0] % tile_count:
            raise ValueError("response-lift tile batch is incompatible with cover")
        batch = local.shape[0] // tile_count
        if code.shape != (batch, tile_count, self.code_width):
            raise ValueError("response-lift code has incompatible shape")
        if semantics.shape[0] != local.shape[0] \
                or semantics.shape[-2:] != local.shape[-2:]:
            raise ValueError("response-lift semantics have incompatible shape")
        if valid.shape != (local.shape[0], 1, *local.shape[-2:]):
            raise ValueError("response-lift validity has incompatible shape")
        basis = torch.tanh(self.basis(torch.cat((local, semantics), dim=1))).reshape(
            local.shape[0], self.field_channels, self.rank, *local.shape[-2:],
        )
        coefficients = self.coefficients(code).reshape(
            local.shape[0], self.field_channels, self.rank,
        )
        correction = torch.einsum("tcrhw,tcr->tchw", basis, coefficients)
        return self.correction_scale * torch.tanh(correction) * valid


class CenteredPotentialResponseChartLift2D(nn.Module):
    """Add a zero-preserving scalar-potential response direction.

    A communicated code selects local scalar-potential bases.  Their spatial
    gradient is decoded as the two velocity channels, while a separate basis
    supplies the pressure-like channel.  The gradient branch is the Helmholtz
    direction that can cancel divergence without asking the generic response
    chart to discover a pressure-projection geometry from arbitrary vector
    bases.  Zero code is exactly zero, and callers may zero-initialize the
    coefficient map to make this module an exact functional child.
    """

    def __init__(
        self, *, feature_channels: int, field_channels: int,
        code_width: int, rank: int, width: int,
        correction_scale: float,
    ) -> None:
        super().__init__()
        if (
            field_channels != 3
            or min(feature_channels, code_width, rank, width) <= 0
            or correction_scale <= 0.0
        ):
            raise ValueError("invalid potential response-chart dimensions")
        self.field_channels = int(field_channels)
        self.code_width = int(code_width)
        self.rank = int(rank)
        self.correction_scale = float(correction_scale)
        groups = min(8, width)
        while width % groups:
            groups -= 1
        self.basis = nn.Sequential(
            nn.Conv2d(feature_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, 2 * rank, 3, padding=1, bias=False),
        )
        self.coefficients = nn.Linear(code_width, 2 * rank, bias=False)

    def correction(
        self, local: Tensor, semantics: Tensor, code: Tensor, valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        if local.ndim != 4 or local.shape[1] != self.field_channels:
            raise ValueError("potential-response local field has incompatible shape")
        if local.shape[0] % tile_count:
            raise ValueError("potential-response tile batch is incompatible with cover")
        batch = local.shape[0] // tile_count
        if code.shape != (batch, tile_count, self.code_width):
            raise ValueError("potential-response code has incompatible shape")
        if semantics.shape[0] != local.shape[0] \
                or semantics.shape[-2:] != local.shape[-2:]:
            raise ValueError("potential-response semantics have incompatible shape")
        if valid.shape != (local.shape[0], 1, *local.shape[-2:]):
            raise ValueError("potential-response validity has incompatible shape")

        bases = torch.tanh(
            self.basis(torch.cat((local, semantics), dim=1))
        ).reshape(local.shape[0], 2, self.rank, *local.shape[-2:])
        coefficients = self.coefficients(code).reshape(
            local.shape[0], 2, self.rank,
        )
        potential = torch.tanh(torch.einsum(
            "tkrhw,tkr->tkhw", bases, coefficients,
        ))
        grad_x, grad_y = centered_gradient(potential[:, 0], 1.0)
        pressure = potential[:, 1]
        correction = torch.stack((grad_x, grad_y, pressure), dim=1)
        return self.correction_scale * correction * valid


class CenteredStreamfunctionResponseChartLift2D(nn.Module):
    """Add a zero-preserving, discretely solenoidal response direction.

    A communicated code selects local streamfunction bases ``psi`` and the
    velocity response is decoded as ``(d_y psi, -d_x psi)``.  Centered mixed
    differences commute away from crop boundaries, so this branch changes
    transported vorticity without injecting interior divergence.  Together
    with :class:`CenteredPotentialResponseChartLift2D`, it supplies the two
    velocity components of a local Helmholtz response chart.  Pressure is
    untouched; zero code, or a zero-initialized coefficient map, is exactly
    the parent field.
    """

    def __init__(
        self, *, feature_channels: int, field_channels: int,
        code_width: int, rank: int, width: int,
        correction_scale: float,
    ) -> None:
        super().__init__()
        if (
            field_channels != 3
            or min(feature_channels, code_width, rank, width) <= 0
            or correction_scale <= 0.0
        ):
            raise ValueError("invalid streamfunction response-chart dimensions")
        self.field_channels = int(field_channels)
        self.code_width = int(code_width)
        self.rank = int(rank)
        self.correction_scale = float(correction_scale)
        groups = min(8, width)
        while width % groups:
            groups -= 1
        self.basis = nn.Sequential(
            nn.Conv2d(feature_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, rank, 3, padding=1, bias=False),
        )
        self.coefficients = nn.Linear(code_width, rank, bias=False)

    def correction(
        self, local: Tensor, semantics: Tensor, code: Tensor, valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        if local.ndim != 4 or local.shape[1] != self.field_channels:
            raise ValueError(
                "streamfunction-response local field has incompatible shape"
            )
        if local.shape[0] % tile_count:
            raise ValueError(
                "streamfunction-response tile batch is incompatible with cover"
            )
        batch = local.shape[0] // tile_count
        if code.shape != (batch, tile_count, self.code_width):
            raise ValueError("streamfunction-response code has incompatible shape")
        if semantics.shape[0] != local.shape[0] \
                or semantics.shape[-2:] != local.shape[-2:]:
            raise ValueError(
                "streamfunction-response semantics have incompatible shape"
            )
        if valid.shape != (local.shape[0], 1, *local.shape[-2:]):
            raise ValueError(
                "streamfunction-response validity has incompatible shape"
            )

        bases = torch.tanh(
            self.basis(torch.cat((local, semantics), dim=1))
        ).reshape(local.shape[0], self.rank, *local.shape[-2:])
        coefficients = self.coefficients(code).reshape(
            local.shape[0], self.rank,
        )
        streamfunction = torch.tanh(torch.einsum(
            "trhw,tr->thw", bases, coefficients,
        ))
        grad_x, grad_y = centered_gradient(streamfunction, 1.0)
        correction = torch.stack((
            grad_y, -grad_x, torch.zeros_like(streamfunction),
        ), dim=1)
        return self.correction_scale * correction * valid


class CenteredCoarseHelmholtzResponseChartLift2D(nn.Module):
    r"""Decode one smooth graph-scale Helmholtz correction in a single pass.

    The communicated code at every executable local region selects three
    coarse nodal values: a scalar potential, a streamfunction, and pressure.
    They are bilinearly interpolated over the region lattice before spatial
    differentiation.  Consequently neighboring owned cores sample the same
    continuous coarse field instead of differentiating unrelated tile-local
    charts.  The correction is

    ``delta u = grad(phi) + (d_y psi, -d_x psi), delta p = p``.

    This is the learned one-pass counterpart of a coarse DDM correction.  The
    bias-free coefficient map makes a zero message exactly Local/Core, and a
    zero-initialized map makes the complete module an exact functional child
    of an existing checkpoint for every nonzero message.
    """

    def __init__(
        self, *, code_width: int, halo: int, correction_scale: float,
    ) -> None:
        super().__init__()
        if code_width <= 0 or halo < 0 or correction_scale <= 0.0:
            raise ValueError("invalid coarse Helmholtz response configuration")
        self.code_width = int(code_width)
        self.halo = int(halo)
        self.correction_scale = float(correction_scale)
        self.coefficients = nn.Linear(code_width, 3, bias=False)

    @staticmethod
    def _sampling_grid(
        grid_shape: tuple[int, int], tile_shape: tuple[int, int],
        *, halo: int, reference: Tensor,
    ) -> Tensor:
        rows, columns = grid_shape
        height, width = tile_shape
        core_y, core_x = height - 2 * halo, width - 2 * halo
        if min(rows, columns, core_y, core_x) <= 0:
            raise ValueError("coarse Helmholtz response has an invalid cover")
        tile_y, tile_x = torch.meshgrid(
            torch.arange(rows, device=reference.device, dtype=reference.dtype),
            torch.arange(columns, device=reference.device, dtype=reference.dtype),
            indexing="ij",
        )
        local_y = (
            torch.arange(height, device=reference.device, dtype=reference.dtype)
            - halo - 0.5 * (core_y - 1)
        ) / core_y
        local_x = (
            torch.arange(width, device=reference.device, dtype=reference.dtype)
            - halo - 0.5 * (core_x - 1)
        ) / core_x
        coarse_y = tile_y[..., None, None] + local_y[None, None, :, None]
        coarse_x = tile_x[..., None, None] + local_x[None, None, None, :]
        coarse_y = coarse_y.expand(rows, columns, height, width)
        coarse_x = coarse_x.expand(rows, columns, height, width)
        # align_corners=True identifies integer lattice coordinates with the
        # graph nodes.  Degenerate one-region axes have the unique coordinate
        # zero and remain a valid constant coarse response.
        normalized_y = (
            2.0 * coarse_y / (rows - 1) - 1.0
            if rows > 1 else torch.zeros_like(coarse_y)
        )
        normalized_x = (
            2.0 * coarse_x / (columns - 1) - 1.0
            if columns > 1 else torch.zeros_like(coarse_x)
        )
        return torch.stack((normalized_x, normalized_y), dim=-1).reshape(
            rows * columns, height, width, 2,
        )

    def correction(
        self, local: Tensor, code: Tensor, valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        if local.ndim != 4 or local.shape[1] != 3:
            raise ValueError("coarse Helmholtz response expects a 3-channel field")
        if local.shape[0] % tile_count:
            raise ValueError("coarse Helmholtz tile batch is incompatible")
        batch = local.shape[0] // tile_count
        if code.shape != (batch, tile_count, self.code_width):
            raise ValueError("coarse Helmholtz code has incompatible shape")
        if valid.shape != (local.shape[0], 1, *local.shape[-2:]):
            raise ValueError("coarse Helmholtz validity has incompatible shape")

        rows, columns = grid_shape
        coarse = self.coefficients(code).permute(0, 2, 1).reshape(
            batch, 3, rows, columns,
        )
        grid = self._sampling_grid(
            grid_shape, tuple(local.shape[-2:]), halo=self.halo,
            reference=local,
        )[None].expand(batch, -1, -1, -1, -1).reshape(
            batch * tile_count, *local.shape[-2:], 2,
        )
        sampled = F.grid_sample(
            coarse[:, None].expand(-1, tile_count, -1, -1, -1).reshape(
                batch * tile_count, 3, rows, columns,
            ),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        potential = torch.tanh(sampled[:, 0])
        streamfunction = torch.tanh(sampled[:, 1])
        pressure = torch.tanh(sampled[:, 2])
        phi_x, phi_y = centered_gradient(potential, 1.0)
        psi_x, psi_y = centered_gradient(streamfunction, 1.0)
        correction = torch.stack((
            phi_x + psi_y,
            phi_y - psi_x,
            pressure,
        ), dim=1)
        return self.correction_scale * correction * valid


class CenteredConformingHelmholtzResponseChartLift2D(nn.Module):
    r"""Decode a continuous, spatially resolved graph-scale correction.

    Every region proposes potential, streamfunction, and pressure values on a
    small nodal patch.  Coincident nodes on adjacent regions are averaged
    before a *single* global field is sampled back onto every halo-extended
    crop.  This retains the one-pass coarse-DDM interpretation of
    :class:`CenteredCoarseHelmholtzResponseChartLift2D`, while allowing the
    coarse field to resolve more than one smooth mode per region.  The shared
    nodal assembly is conforming: neighboring owned cores are restrictions of
    the same interpolated field, rather than separately decoded tile charts.

    The bias-free coefficient map makes a zero communicated code exactly
    Local/Core.  Zero initialization additionally makes this module an exact
    functional child of an existing checkpoint for arbitrary nonzero codes.
    """

    def __init__(
        self, *, code_width: int, halo: int, subcells: int,
        correction_scale: float,
    ) -> None:
        super().__init__()
        if (
            code_width <= 0 or halo < 0 or subcells <= 1
            or correction_scale <= 0.0
        ):
            raise ValueError("invalid conforming Helmholtz response configuration")
        self.code_width = int(code_width)
        self.halo = int(halo)
        self.subcells = int(subcells)
        self.correction_scale = float(correction_scale)
        nodes_per_patch = (self.subcells + 1) ** 2
        self.coefficients = nn.Linear(
            self.code_width, 3 * nodes_per_patch, bias=False,
        )

    def _global_nodes(
        self, code: Tensor, grid_shape: tuple[int, int],
    ) -> Tensor:
        """Average coincident region proposals onto one conforming lattice."""
        rows, columns = grid_shape
        tile_count = rows * columns
        if code.ndim != 3 or code.shape[1:] != (
            tile_count, self.code_width,
        ):
            raise ValueError("conforming Helmholtz code has incompatible shape")
        batch = code.shape[0]
        patch_nodes = self.subcells + 1
        proposed = self.coefficients(code).reshape(
            batch, tile_count, 3, patch_nodes, patch_nodes,
        ).permute(0, 2, 1, 3, 4).reshape(
            batch, 3, tile_count * patch_nodes**2,
        )

        tile_y, tile_x = torch.meshgrid(
            torch.arange(rows, device=code.device),
            torch.arange(columns, device=code.device),
            indexing="ij",
        )
        local_y, local_x = torch.meshgrid(
            torch.arange(patch_nodes, device=code.device),
            torch.arange(patch_nodes, device=code.device),
            indexing="ij",
        )
        global_columns = columns * self.subcells + 1
        node_y = (
            tile_y[..., None, None] * self.subcells
            + local_y[None, None]
        )
        node_x = (
            tile_x[..., None, None] * self.subcells
            + local_x[None, None]
        )
        flat_index = (node_y * global_columns + node_x).reshape(-1)
        global_rows = rows * self.subcells + 1
        global_count = global_rows * global_columns
        index = flat_index[None, None].expand(batch, 3, -1)
        summed = code.new_zeros(batch, 3, global_count)
        summed.scatter_add_(2, index, proposed)
        multiplicity = code.new_zeros(global_count)
        multiplicity.scatter_add_(
            0, flat_index, torch.ones_like(flat_index, dtype=code.dtype),
        )
        return (summed / multiplicity.clamp_min(1.0)[None, None]).reshape(
            batch, 3, global_rows, global_columns,
        )

    def _sampling_grid(
        self, grid_shape: tuple[int, int], tile_shape: tuple[int, int],
        *, reference: Tensor,
    ) -> Tensor:
        rows, columns = grid_shape
        height, width = tile_shape
        core_y, core_x = height - 2 * self.halo, width - 2 * self.halo
        if min(rows, columns, core_y, core_x) <= 0:
            raise ValueError("conforming Helmholtz response has an invalid cover")
        tile_y, tile_x = torch.meshgrid(
            torch.arange(rows, device=reference.device, dtype=reference.dtype),
            torch.arange(columns, device=reference.device, dtype=reference.dtype),
            indexing="ij",
        )
        # Pixel centers are used in the owned core.  A halo pixel and the
        # corresponding pixel in its neighboring core therefore receive the
        # same global coordinate exactly.
        local_y = (
            torch.arange(height, device=reference.device, dtype=reference.dtype)
            - self.halo + 0.5
        ) / core_y
        local_x = (
            torch.arange(width, device=reference.device, dtype=reference.dtype)
            - self.halo + 0.5
        ) / core_x
        node_y = self.subcells * (
            tile_y[..., None, None] + local_y[None, None, :, None]
        )
        node_x = self.subcells * (
            tile_x[..., None, None] + local_x[None, None, None, :]
        )
        node_y = node_y.expand(rows, columns, height, width)
        node_x = node_x.expand(rows, columns, height, width)
        global_rows = rows * self.subcells + 1
        global_columns = columns * self.subcells + 1
        normalized_y = 2.0 * node_y / (global_rows - 1) - 1.0
        normalized_x = 2.0 * node_x / (global_columns - 1) - 1.0
        return torch.stack((normalized_x, normalized_y), dim=-1).reshape(
            rows * columns, height, width, 2,
        )

    def correction(
        self, local: Tensor, code: Tensor, valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        if local.ndim != 4 or local.shape[1] != 3:
            raise ValueError("conforming Helmholtz response expects 3 channels")
        if local.shape[0] % tile_count:
            raise ValueError("conforming Helmholtz tile batch is incompatible")
        batch = local.shape[0] // tile_count
        if code.shape != (batch, tile_count, self.code_width):
            raise ValueError("conforming Helmholtz code has incompatible shape")
        if valid.shape != (local.shape[0], 1, *local.shape[-2:]):
            raise ValueError("conforming Helmholtz validity is incompatible")

        global_nodes = self._global_nodes(code, grid_shape)
        grid = self._sampling_grid(
            grid_shape, tuple(local.shape[-2:]), reference=local,
        )[None].expand(batch, -1, -1, -1, -1).reshape(
            batch * tile_count, *local.shape[-2:], 2,
        )
        sampled = F.grid_sample(
            global_nodes[:, None].expand(
                -1, tile_count, -1, -1, -1,
            ).reshape(batch * tile_count, 3, *global_nodes.shape[-2:]),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        potential = torch.tanh(sampled[:, 0])
        streamfunction = torch.tanh(sampled[:, 1])
        pressure = torch.tanh(sampled[:, 2])
        phi_x, phi_y = centered_gradient(potential, 1.0)
        psi_x, psi_y = centered_gradient(streamfunction, 1.0)
        correction = torch.stack((
            phi_x + psi_y,
            phi_y - psi_x,
            pressure,
        ), dim=1)
        return self.correction_scale * correction * valid


def complete_residual_drive(model, example) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return inference-observable q_Gamma, q_mom and their shared drive."""
    local, _, physical, _, _, grid_shape, transition, _ = example
    support = compatibility_face_support(
        physical[:, -1:].to(local), grid_shape,
    )
    q_gamma, q_momentum = _components(model, example, local, support)
    return q_gamma.reshape(
        transition.batch_size, transition.cover.tile_count, 24,
    ), q_momentum.reshape(
        transition.batch_size, transition.cover.tile_count, 6,
    ), support, local


def pou_field_defect(example, local: Tensor | None = None) -> Tensor:
    """Return the explicit patchwise mismatch ``R_i A_chi(v)-v_i``."""
    endpoint, _, physical, _, _, _, transition, _ = example
    if local is None:
        local = endpoint
    if local.shape != endpoint.shape:
        raise ValueError("field-defect endpoint must match the example tiles")
    consensus = transition.cover.tile(
        transition.cover.assemble_weighted(local, transition.batch_size),
    )
    return (consensus - local) * physical[:, -1:].to(local)


def pou_transport_load_defect(
    model, example, local: Tensor | None = None,
) -> Tensor:
    """Return the eight centered transport/load coordinates of the POU defect."""
    endpoint, _, physical, _, _, grid_shape, transition, dataset = example
    if local is None:
        local = endpoint
    if local.shape != endpoint.shape:
        raise ValueError("transport/load endpoint must match the example tiles")
    consensus = transition.cover.tile(
        transition.cover.assemble_weighted(local, transition.batch_size),
    )
    # The chart's 32D port is fitted on train-only scales.  Calling the same
    # port functional on the provisional consensus is inference-observable;
    # no target field enters this coordinate.
    code = model.composer.oracle_residual_port_code(
        local, consensus, physical, standardized=True,
        force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
        lift_sign=dataset.rotation_sign,
    ).reshape(
        transition.batch_size, transition.cover.tile_count, 32,
    )
    if code.shape[1] != math.prod(grid_shape):
        raise RuntimeError("transport/load residual does not match the cover")
    return code[..., 24:]


class SharedCompatibilityMomentumKernel(nn.Module):
    """Fixed-context-linear map of the complete residual through one kernel."""

    def __init__(
        self, model, *, beta: float, heads: int = 4,
        drive_scale: Tensor | None = None, state_value_rank: int = 0,
        residual_sketch_width: int = 0,
        residual_sketch_full_chart: bool = False,
        response_lift_rank: int = 0,
        response_lift_global_only: bool = False,
    ) -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if state_value_rank < 0:
            raise ValueError("state_value_rank must be nonnegative")
        if residual_sketch_width < 0:
            raise ValueError("residual_sketch_width must be nonnegative")
        if response_lift_rank < 0:
            raise ValueError("response_lift_rank must be nonnegative")
        if state_value_rank and residual_sketch_width:
            raise ValueError(
                "state-gated and residual-field lifts must be tested separately"
            )
        if residual_sketch_full_chart and not residual_sketch_width:
            raise ValueError("the full-chart lift requires a residual-field sketch")
        if response_lift_rank and not residual_sketch_full_chart:
            raise ValueError(
                "the response-chart lift requires the complete residual sketch"
            )
        if response_lift_global_only and not response_lift_rank:
            raise ValueError(
                "a global-only response lift requires a positive lift rank"
            )
        self.model = model
        self.composer = model.composer
        self.beta = float(beta)
        self.state_value_rank = int(state_value_rank)
        self.residual_sketch_width = int(residual_sketch_width)
        self.residual_sketch_full_chart = bool(residual_sketch_full_chart)
        self.response_lift_rank = int(response_lift_rank)
        self.response_lift_global_only = bool(response_lift_global_only)
        self.response_code_width = (
            32 + self.response_lift_rank
            if self.residual_sketch_full_chart else 24
        )
        self.contract = (
            "airfoil_global_residual_response_chart_one_message_student_v6"
            if self.response_lift_global_only else (
            "airfoil_augmented_response_chart_one_message_student_v5"
            if self.response_lift_rank else (
            "airfoil_complete_residual_full_chart_one_message_student_v4"
            if self.residual_sketch_full_chart else (
                "airfoil_qgamma_qmom_residual_field_one_message_student_v3"
                if self.residual_sketch_width else (
                "airfoil_qgamma_qmom_state_gated_one_message_student_v2"
                if self.state_value_rank
                else "airfoil_qgamma_qmom_one_message_student_v1"
                )
            )
            )
            )
        )
        if drive_scale is None:
            drive_scale = torch.ones(30)
        drive_scale = torch.as_tensor(drive_scale, dtype=torch.float32)
        if drive_scale.shape != (30,) or not torch.isfinite(drive_scale).all() \
                or torch.any(drive_scale <= 0.0):
            raise ValueError("drive_scale must contain 30 finite positive values")
        self.register_buffer("drive_scale", drive_scale)
        width = self.composer.width
        map_type = (
            StateGatedResidualKernelCodeMap
            if self.state_value_rank else ResidualKernelCodeMap
        )
        map_kwargs = (
            {"value_rank": self.state_value_rank}
            if self.state_value_rank else {}
        )
        self.residual_map = map_type(
            context_width=width, residual_width=30, token_width=width,
            code_width=24, heads=heads, dimension=2, zero_init=True,
            **map_kwargs,
        ).to(self.composer.port_std)
        self.residual_sketch = (
            ResidualFieldCodeLift(
                field_channels=len(self.composer.state_mean),
                sketch_width=self.residual_sketch_width,
                token_width=width, code_width=self.response_code_width,
                heads=heads, dimension=2,
                hidden_width=max(12, self.residual_sketch_width // 2),
                bins_per_axis=3,
                auxiliary_residual_width=(
                    8 if self.residual_sketch_full_chart else 0
                ),
                global_only_suffix_width=(
                    self.response_lift_rank
                    if self.response_lift_global_only else 0
                ),
                zero_init=True,
            ).to(self.composer.port_std)
            if self.residual_sketch_width else None
        )
        self.response_lift = (
            CenteredResponseChartLift2D(
                feature_channels=(
                    self.composer.field_channels
                    + self.composer.semantic_channels
                ),
                field_channels=self.composer.field_channels,
                code_width=self.response_lift_rank,
                rank=self.response_lift_rank,
                width=self.composer.width,
                correction_scale=self.composer.correction_scale,
            ).to(self.composer.port_std)
            if self.response_lift_rank else None
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # Context depends on geometry/state, never on the communicated residual.
        # For each frozen context the complete q -> z map remains exactly linear.
        trainable_modules = [
            self.composer.encoder, self.composer.pool,
            self.composer.scalar_encoder, self.residual_map,
        ]
        if self.residual_sketch is not None:
            trainable_modules.append(self.residual_sketch)
        if self.response_lift is not None:
            trainable_modules.append(self.response_lift)
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def context(
        self, local: Tensor, physical: Tensor, scalar: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        valid = physical[:, -1:].to(local)
        semantics = self.composer.semantic_features(local, physical)
        return self.composer._tokens(local, semantics, scalar, valid, grid_shape)

    def code(
        self, local: Tensor, physical: Tensor, scalar: Tensor,
        grid_shape: tuple[int, int], q24: Tensor, qmom6: Tensor,
        *, graph_radius: int | None = None,
        residual_field: Tensor | None = None,
        auxiliary_residual: Tensor | None = None,
    ) -> Tensor:
        if q24.shape[:-1] != qmom6.shape[:-1] or q24.shape[-1] != 24 \
                or qmom6.shape[-1] != 6:
            raise ValueError("complete residual must be B,N,24 plus B,N,6")
        drive = torch.cat((q24, math.sqrt(self.beta) * qmom6), dim=-1)
        drive = drive / self.drive_scale.to(drive)
        context = self.context(local, physical, scalar, grid_shape)
        if self.residual_sketch is None:
            if residual_field is not None or auxiliary_residual is not None:
                raise ValueError("base residual kernel does not accept a field sketch")
            return self.residual_map(
                context, drive, grid_shape, max_graph_radius=graph_radius,
            )
        if residual_field is None:
            raise ValueError("residual-field LatentDDM requires the POU field defect")
        if self.residual_sketch_full_chart:
            if auxiliary_residual is None:
                raise ValueError("full-chart LatentDDM requires transport/load residuals")
        elif auxiliary_residual is not None:
            raise ValueError("24D residual-field LatentDDM has no auxiliary residual")
        # At initialization/evaluation retain the literal accepted-parent path,
        # not merely an algebraically equivalent reconstruction on CUDA.
        if not self.training and not bool(torch.count_nonzero(
            self.residual_sketch.code_head.weight,
        )):
            parent = self.residual_map(
                context, drive, grid_shape, max_graph_radius=graph_radius,
            )
            return (
                torch.cat((
                    parent,
                    parent.new_zeros(
                        *parent.shape[:-1], 8 + self.response_lift_rank,
                    ),
                ), -1)
                if self.residual_sketch_full_chart else parent
            )
        kernel = self.residual_map.global_kernel
        weight = kernel._weights(context, grid_shape, graph_radius)
        message = kernel._apply_weights(weight, drive)
        parent_code = self.residual_map.code_head(torch.cat((
            self.residual_map.local_value(drive), message,
        ), dim=-1))
        if self.residual_sketch_full_chart:
            parent_code = torch.cat((
                parent_code,
                parent_code.new_zeros(
                    *parent_code.shape[:-1], 8 + self.response_lift_rank,
                ),
            ), dim=-1)
        return parent_code + self.residual_sketch(
            weight, residual_field, grid_shape,
            auxiliary_residual=auxiliary_residual,
        )

    def decode(
        self, local: Tensor, physical: Tensor, grid_shape: tuple[int, int],
        code: Tensor,
    ) -> Tensor:
        if code.shape[-1] == 24:
            code = torch.cat((
                code,
                code.new_zeros(
                    *code.shape[:-1], 8 + self.response_lift_rank,
                ),
            ), dim=-1)
        elif code.shape[-1] == 32 and self.response_lift_rank:
            code = torch.cat((
                code,
                code.new_zeros(*code.shape[:-1], self.response_lift_rank),
            ), dim=-1)
        elif code.shape[-1] != self.response_code_width:
            raise ValueError(
                "Airfoil response code has incompatible chart width"
            )
        semantics = self.composer.semantic_features(local, physical)
        corrected = self.composer.decode(
            local, semantics, code[..., :32],
            physical[:, -1:].to(local), grid_shape,
        )
        if self.response_lift is not None:
            corrected = corrected + self.response_lift.correction(
                local, semantics, code[..., 32:],
                physical[:, -1:].to(local), grid_shape,
            )
        return corrected


def _quartic_minimizer(
    q0: Tensor, q1: Tensor, m0: Tensor, m1: Tensor, m2: Tensor,
    beta: float, *, block_relative_tolerance: float = 2.0e-6,
) -> Tensor:
    """Minimize ``||q0+a q1||^2 + beta||m0+a m1+a^2 m2||^2`` on [0,1].

    The Airfoil compatibility residual is affine in a field-space damping
    coordinate, while the retained backward-Euler momentum moments are
    quadratic because convection is bilinear.  Their squared joint objective
    is therefore quartic.  Solving its cubic derivative replaces the former
    17-point grid without inspecting a target or invoking another neural map.
    Candidate points include stationary points of the combined quartic and all
    roots of the separate q_Gamma/momentum nonexpansion boundaries.  Candidates
    that expand either block are rejected.  Alpha=0 is always feasible, so the
    result is block-safe as well as combined-residual minimizing.
    """
    coefficients = torch.stack((
        q0.square().sum(1) + beta * m0.square().sum(1),
        2.0 * (q0 * q1).sum(1) + 2.0 * beta * (m0 * m1).sum(1),
        q1.square().sum(1) + beta * (
            m1.square().sum(1) + 2.0 * (m0 * m2).sum(1)
        ),
        2.0 * beta * (m1 * m2).sum(1),
        beta * m2.square().sum(1),
    ), dim=1)
    q_boundary = torch.stack((
        2.0 * (q0 * q1).sum(1), q1.square().sum(1),
    ), dim=1)
    momentum_boundary = torch.stack((
        2.0 * (m0 * m1).sum(1),
        m1.square().sum(1) + 2.0 * (m0 * m2).sum(1),
        2.0 * (m1 * m2).sum(1), m2.square().sum(1),
    ), dim=1)
    q_baseline = q0.square().sum(1)
    momentum_baseline = m0.square().sum(1)
    alpha = []
    # Root finding is a tiny target-free scalar operation (one cubic per
    # physical case).  Keeping it off the autograd graph is intentional: the
    # damping is an inference safeguard, not a trainable neural component.
    arrays = zip(
        coefficients.detach().double().cpu().numpy(),
        q_boundary.detach().double().cpu().numpy(),
        momentum_boundary.detach().double().cpu().numpy(),
        q_baseline.detach().double().cpu().numpy(),
        momentum_baseline.detach().double().cpu().numpy(),
    )
    for row, q_row, momentum_row, q_base, momentum_base in arrays:
        derivative = np.asarray((4.0 * row[4], 3.0 * row[3],
                                 2.0 * row[2], row[1]), dtype=np.float64)
        candidates = [0.0, 1.0]
        # Combined stationary roots, then the two block-safety boundaries.
        polynomials = (
            derivative,
            np.asarray((q_row[1], q_row[0],
                        -block_relative_tolerance * q_base), dtype=np.float64),
            np.asarray((momentum_row[3], momentum_row[2],
                        momentum_row[1], momentum_row[0],
                        -block_relative_tolerance * momentum_base),
                       dtype=np.float64),
        )
        for polynomial in polynomials:
            nonzero = np.flatnonzero(np.abs(polynomial) > 1.0e-14)
            if nonzero.size:
                roots = np.roots(polynomial[int(nonzero[0]):])
            else:
                roots = ()
            for root in roots:
                if abs(float(root.imag)) <= 1.0e-8 \
                        and -1.0e-8 <= float(root.real) <= 1.0 + 1.0e-8:
                    candidates.append(float(np.clip(root.real, 0.0, 1.0)))
        candidates = list(dict.fromkeys(candidates))
        feasible = []
        for value in candidates:
            q_delta = q_row[0] * value + q_row[1] * value**2
            momentum_delta = sum(
                momentum_row[k - 1] * value**k for k in range(1, 5)
            )
            boundary_slack = 1.0 + 1.0e-8
            if q_delta <= boundary_slack * block_relative_tolerance * max(
                q_base, 1.0e-30,
            ) and momentum_delta <= boundary_slack * block_relative_tolerance * max(
                momentum_base, 1.0e-30,
            ):
                feasible.append(value)
        values = [sum(row[k] * value**k for k in range(5)) for value in feasible]
        alpha.append(feasible[int(np.argmin(values))])
    return q0.new_tensor(alpha)


def actual_polynomial_damping(
    student: SharedCompatibilityMomentumKernel,
    example, proposed: Tensor, q24: Tensor, qmom6: Tensor, support: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Exact quartic line search for the affine/quadratic residual pair."""
    local, _, _, _, _, _, transition, _ = example
    q_zero = q24.reshape(transition.batch_size, -1)
    momentum_zero = qmom6.reshape(transition.batch_size, -1)
    local_view = local.reshape(
        transition.batch_size, transition.cover.tile_count, *local.shape[1:],
    )
    proposed_view = proposed.reshape_as(local_view)
    direction = proposed_view - local_view
    half = (local_view + 0.5 * direction).reshape_as(local)
    q_half, momentum_half = _components(
        student.model, example, half, support,
    )
    q_one, momentum_one = _components(
        student.model, example, proposed, support,
    )
    q_linear = q_one - q_zero
    momentum_half_delta = momentum_half - momentum_zero
    momentum_one_delta = momentum_one - momentum_zero
    momentum_quadratic = 2.0 * momentum_one_delta - 4.0 * momentum_half_delta
    momentum_linear = momentum_one_delta - momentum_quadratic
    alpha = _quartic_minimizer(
        q_zero, q_linear, momentum_zero, momentum_linear,
        momentum_quadratic, student.beta,
    )
    corrected = (
        local_view + alpha[:, None, None, None, None]
        * direction
    ).reshape_as(local)
    # Re-evaluate at the selected scalar to audit the actual implementation,
    # rather than reporting only the fitted polynomial value.
    q_selected, momentum_selected = _components(
        student.model, example, corrected, support,
    )
    total_zero = _weighted_vector(q_zero, momentum_zero, student.beta)
    total_selected = _weighted_vector(
        q_selected, momentum_selected, student.beta,
    )
    # Audit the actual finite-precision residuals.  A rare interpolation/root
    # roundoff violation falls back to the exactly feasible identity.
    safe = (
        q_selected.square().sum(1)
        <= (1.0 + 2.0e-5)**2 * q_zero.square().sum(1).clamp_min(1.0e-30)
    ) & (
        momentum_selected.square().sum(1)
        <= (1.0 + 2.0e-5)**2
        * momentum_zero.square().sum(1).clamp_min(1.0e-30)
    )
    if not bool(torch.all(safe)):
        alpha = torch.where(safe, alpha, torch.zeros_like(alpha))
        safe_tiles = safe[:, None].expand(
            transition.batch_size, transition.cover.tile_count,
        ).reshape(-1)
        corrected = torch.where(
            safe_tiles[:, None, None, None], corrected, local,
        )
        q_selected = torch.where(safe[:, None], q_selected, q_zero)
        momentum_selected = torch.where(
            safe[:, None], momentum_selected, momentum_zero,
        )
        total_selected = torch.where(
            safe[:, None], total_selected, total_zero,
        )

    def ratio(after: Tensor, before: Tensor) -> Tensor:
        return torch.sqrt(
            after.square().sum(1) / before.square().sum(1).clamp_min(1e-12)
        )

    return (
        corrected, alpha, ratio(q_selected, q_zero),
        ratio(momentum_selected, momentum_zero),
        ratio(total_selected, total_zero),
    )


def infer(
    student: SharedCompatibilityMomentumKernel, example, *,
    graph_radius: int | None = None, shuffle: bool = False,
) -> DampedStudentOutput:
    local, _, physical, scalar, _, grid_shape, _, _ = example
    q24, qmom6, support, _ = complete_residual_drive(student.model, example)
    residual_field = (
        pou_field_defect(example, local)
        if getattr(student, "residual_sketch", None) is not None else None
    )
    auxiliary_residual = (
        pou_transport_load_defect(student.model, example, local)
        if getattr(student, "residual_sketch_full_chart", False) else None
    )
    code = student.code(
        local, physical, scalar, grid_shape, q24, qmom6,
        graph_radius=graph_radius, residual_field=residual_field,
        auxiliary_residual=auxiliary_residual,
    )
    if shuffle:
        code = code.flip(0) if code.shape[0] > 1 else code.roll(1, dims=1)
    proposed = student.decode(local, physical, grid_shape, code)
    corrected, alpha, q_cycle, momentum_cycle, combined_cycle = actual_polynomial_damping(
        student, example, proposed, q24, qmom6, support,
    )
    return DampedStudentOutput(
        corrected, code, alpha, q_cycle, momentum_cycle, combined_cycle,
        1, 4, 4,
    )
