"""Fixed-point-preserving one-message analyzer for the Airfoil response chart.

The 32-dimensional chart coordinate is split into a 24-dimensional interface
compatibility code and an eight-dimensional transport/load auxiliary.  The
compatibility branch is one geometry/state-conditioned residual-kernel
application: query/key weights may be nonlinear functions of the provisional
state, while values and the code map are bias-free linear maps of the explicit
POU defect.  Consequently ``d_gamma == 0`` implies ``z_gamma == 0`` exactly.

The auxiliary branch is deliberately named and returned separately.  It may
use state context because it predicts transport and integrated-load
coordinates, and must therefore be ablated independently in experiments.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from codiffuse_Schwarz.reduced_inverse import (
    ReducedInverseProbe, build_reduced_inverse_probe,
    inverse_cycle_relative_square,
)
from codiffuse_Schwarz.tile_communication import ResidualKernelCodeMap
from .airfoil_compatibility import CompatibilityCodedAirfoilOperator
from .airfoil_defect_ports import provisional_defect_port_code


Tensor = torch.Tensor
SplitMode = Literal[
    "zero", "full", "compatibility_only", "auxiliary_only",
    "shuffle_compatibility", "shuffle_auxiliary", "shuffle_all", "oracle",
]


class SplitAnalyzerOutput(NamedTuple):
    corrected: Tensor
    code: Tensor
    compatibility_code: Tensor
    auxiliary_code: Tensor
    defect_code: Tensor
    message_calls: int


class FixedPointSplitAnalyzer(nn.Module):
    """Analyze a label-free POU defect with one residual-kernel message."""

    compatibility_dim = 24
    auxiliary_dim = 8

    def __init__(
        self,
        model: CompatibilityCodedAirfoilOperator,
        *,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if model.composer.code_dim != self.compatibility_dim + self.auxiliary_dim:
            raise ValueError("the fixed-point split requires the shared 32D chart")
        self.model = model
        self.composer = model.composer
        width = self.composer.width
        self.compatibility_map = ResidualKernelCodeMap(
            context_width=width,
            residual_width=self.compatibility_dim,
            token_width=width,
            code_width=self.compatibility_dim,
            heads=heads,
            dimension=2,
            zero_init=True,
        ).to(self.composer.port_std)
        # The auxiliary remains context-conditioned, but context only gates a
        # bias-free linear value of the explicit QoI defect.  The final head is
        # also bias-free and sees only that gated value and z_gamma.  Hence a
        # zero *full* defect makes both code blocks and the decoded correction
        # exactly zero; context alone can never invent a load correction.
        self.auxiliary_value = nn.Linear(
            self.auxiliary_dim, width, bias=False,
        ).to(self.composer.port_std)
        self.auxiliary_gate = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.Sigmoid(),
        ).to(self.composer.port_std)
        self.auxiliary_head = nn.Linear(
            width + self.compatibility_dim, self.auxiliary_dim, bias=False,
        ).to(self.composer.port_std)

        # Freeze the retained endpoint and learned response chart.  Only the
        # state/geometry context analyzer and the two new code branches train.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for module in (
            self.composer.encoder, self.composer.pool, self.composer.scalar_encoder,
            self.compatibility_map, self.auxiliary_value,
            self.auxiliary_gate, self.auxiliary_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    @property
    def analysis_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @staticmethod
    def _shuffle(code: Tensor) -> Tensor:
        return code.flip(0) if code.shape[0] > 1 else code.roll(1, dims=1)

    def _defect(
        self,
        local_endpoint: Tensor,
        consensus_endpoint: Tensor,
        physical: Tensor,
        grid_shape: tuple[int, int],
        *,
        force_area_scale: float,
        lift_sign: float,
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        batch = local_endpoint.shape[0] // tile_count
        return provisional_defect_port_code(
            local_endpoint, consensus_endpoint, physical,
            state_mean=tuple(float(x) for x in self.composer.state_mean),
            state_std=tuple(float(x) for x in self.composer.state_std),
            delta_mean=tuple(float(x) for x in self.composer.delta_mean),
            delta_std=tuple(float(x) for x in self.composer.delta_std),
            port_scale=self.composer.port_std,
            functional_kind=self.composer.port_functional_kind,
            force_area_scale=force_area_scale,
            lift_sign=lift_sign,
        ).reshape(batch, tile_count, -1)

    def predict_split_code(
        self,
        local_endpoint: Tensor,
        consensus_endpoint: Tensor,
        physical: Tensor,
        scalar: Tensor,
        grid_shape: tuple[int, int],
        *,
        graph_radius: int | None = None,
        force_area_scale: float = 1.0,
        lift_sign: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        valid = physical[:, -1:].to(local_endpoint)
        semantics = self.composer.semantic_features(local_endpoint, physical)
        context = self.composer._tokens(
            local_endpoint, semantics, scalar, valid, grid_shape,
        )
        defect = self._defect(
            local_endpoint, consensus_endpoint, physical, grid_shape,
            force_area_scale=force_area_scale, lift_sign=lift_sign,
        )
        # Production consumes q=P_chi(v)-v directly.  The minus sign belongs
        # only to the synthetic probe response in airfoil_probe_invert_loss;
        # negating here would double-negate the learned reduced inverse.
        compatibility = self.compatibility_map(
            context, defect[..., :self.compatibility_dim], grid_shape,
            max_graph_radius=graph_radius,
        )
        auxiliary_value = self.auxiliary_value(
            defect[..., self.compatibility_dim:],
        ) * self.auxiliary_gate(context)
        auxiliary = self.auxiliary_head(torch.cat((
            auxiliary_value, compatibility,
        ), dim=-1))
        return compatibility, auxiliary, defect, semantics

    def forward(
        self,
        local_endpoint: Tensor,
        consensus_endpoint: Tensor,
        physical: Tensor,
        scalar: Tensor,
        grid_shape: tuple[int, int],
        *,
        mode: SplitMode = "full",
        oracle_code: Tensor | None = None,
        graph_radius: int | None = None,
        force_area_scale: float = 1.0,
        lift_sign: float = 1.0,
    ) -> SplitAnalyzerOutput:
        tile_count = int(np.prod(grid_shape))
        batch = local_endpoint.shape[0] // tile_count
        shape = (batch, tile_count, self.composer.code_dim)
        valid = physical[:, -1:].to(local_endpoint)
        semantics = self.composer.semantic_features(local_endpoint, physical)
        zeros_gamma = local_endpoint.new_zeros(
            batch, tile_count, self.compatibility_dim,
        )
        zeros_aux = local_endpoint.new_zeros(batch, tile_count, self.auxiliary_dim)
        defect = local_endpoint.new_zeros(shape)
        calls = 0

        if mode == "zero":
            compatibility, auxiliary = zeros_gamma, zeros_aux
        elif mode == "oracle":
            if oracle_code is None or oracle_code.shape != shape:
                raise ValueError("oracle mode requires a B,N,32 oracle code")
            compatibility = oracle_code[..., :self.compatibility_dim]
            auxiliary = oracle_code[..., self.compatibility_dim:]
        elif mode in (
            "full", "compatibility_only", "auxiliary_only",
            "shuffle_compatibility", "shuffle_auxiliary", "shuffle_all",
        ):
            compatibility, auxiliary, defect, semantics = self.predict_split_code(
                local_endpoint, consensus_endpoint, physical, scalar, grid_shape,
                graph_radius=graph_radius,
                force_area_scale=force_area_scale,
                lift_sign=lift_sign,
            )
            calls = 1
            if mode == "compatibility_only":
                auxiliary = zeros_aux
            elif mode == "auxiliary_only":
                compatibility = zeros_gamma
            elif mode == "shuffle_compatibility":
                compatibility = self._shuffle(compatibility)
            elif mode == "shuffle_auxiliary":
                auxiliary = self._shuffle(auxiliary)
            elif mode == "shuffle_all":
                compatibility = self._shuffle(compatibility)
                auxiliary = self._shuffle(auxiliary)
        else:
            raise ValueError(f"unknown split analyzer mode: {mode}")

        code = torch.cat((compatibility, auxiliary), dim=-1)
        corrected = self.composer.decode(
            local_endpoint, semantics, code, valid, grid_shape,
        )
        return SplitAnalyzerOutput(
            corrected, code, compatibility, auxiliary, defect, calls,
        )


def compatibility_face_support(
    validity: Tensor, grid_shape: tuple[int, int],
) -> Tensor:
    """Return the B,N,24 mask of trace coordinates present on each tile."""
    if validity.ndim != 4 or validity.shape[1] != 1:
        raise ValueError("validity must have shape T,1,H,W")
    tile_count = math.prod(grid_shape)
    if validity.shape[0] % tile_count:
        raise ValueError("validity tile batch is incompatible with grid_shape")
    active_faces = []
    for tensor_axis in (2, 3):
        for index in (0, -1):
            face = validity.select(tensor_axis, index)
            active = (face.flatten(1).sum(1) > 0).to(validity)
            active_faces.append(active[:, None].expand(-1, 6))
    support = torch.cat(active_faces, dim=1)
    return support.reshape(validity.shape[0] // tile_count, tile_count, 24)


def airfoil_probe_invert_loss(
    analyzer: FixedPointSplitAnalyzer,
    local_endpoint: Tensor,
    physical: Tensor,
    scalar: Tensor,
    grid_shape: tuple[int, int],
    cover,
    batch_size: int,
    excitation: Tensor,
    *,
    graph_radius: int | None = None,
    force_area_scale: float = 1.0,
    lift_sign: float = 1.0,
) -> tuple[Tensor, ReducedInverseProbe, Tensor]:
    """Target-free frozen-chart probe--invert loss for the 24D branch.

    The eight transport/load coordinates are exactly zero during probing.  We
    analyze the change in POU consensus defect induced by the frozen chart and
    apply the common sign/masking contract
    ``C(c, -[q(v+L(xi))-q(v)]) = xi``.
    """
    tile_count = math.prod(grid_shape)
    expected = (batch_size, tile_count, analyzer.compatibility_dim)
    if excitation.shape != expected:
        raise ValueError(f"excitation must have shape {expected}")
    valid = physical[:, -1:].to(local_endpoint)
    support = compatibility_face_support(valid, grid_shape)
    supported_excitation = excitation * support
    zero_auxiliary = excitation.new_zeros(
        batch_size, tile_count, analyzer.auxiliary_dim,
    )
    semantics = analyzer.composer.semantic_features(local_endpoint, physical)
    perturbed = analyzer.composer.decode(
        local_endpoint, semantics,
        torch.cat((supported_excitation, zero_auxiliary), dim=-1),
        valid, grid_shape,
    )
    baseline_consensus = cover.tile(
        cover.assemble_weighted(local_endpoint, batch_size),
    )
    perturbed_consensus = cover.tile(
        cover.assemble_weighted(perturbed, batch_size),
    )
    baseline = analyzer._defect(
        local_endpoint, baseline_consensus, physical, grid_shape,
        force_area_scale=force_area_scale, lift_sign=lift_sign,
    )[..., :analyzer.compatibility_dim]
    changed = analyzer._defect(
        perturbed, perturbed_consensus, physical, grid_shape,
        force_area_scale=force_area_scale, lift_sign=lift_sign,
    )[..., :analyzer.compatibility_dim]
    probe = build_reduced_inverse_probe(
        baseline, changed, excitation, supported_coordinates=support,
    )
    context = analyzer.composer._tokens(
        local_endpoint, semantics, scalar, valid, grid_shape,
    )
    predicted = analyzer.compatibility_map(
        context, probe.inverse_rhs.detach(), grid_shape,
        max_graph_radius=graph_radius,
    )
    loss = inverse_cycle_relative_square(
        predicted, probe, supported_coordinates=support,
    ).mean()
    return loss, probe, predicted
