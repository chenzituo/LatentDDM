"""Two-round Airfoil LatentDDM on a frozen absolute-Robin local operator.

The first local solve is exactly the Local/Core endpoint.  A lightweight
relative-geometry Transformer then maps local conditions, tentative solutions,
and their interface defects to a low-order correction of the Robin input.  The
same frozen local operator is evaluated once more and the overlapping fields
are assembled.  No absolute tile coordinate is supplied to either stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from codiffuse_Schwarz.boundary_trace_basis import decode_trace_code_2d
from codiffuse_Schwarz.field_token_composer import (
    CenteredFieldTokenTransformer,
    FieldTokenComposition,
)

from .absolute_robin_sni import (
    absolute_robin_port_2d,
)
from .airfoil_trace_sni import assemble_subdomains_2d


Tensor = torch.Tensor
CONTRACT = "airfoil_absolute_robin_latentddm_v1"


def _groups(width: int) -> int:
    groups = min(8, width)
    while width % groups:
        groups -= 1
    return groups


@dataclass(frozen=True)
class AirfoilRobinLatentOutput:
    """Auditable fields from one fixed two-round composition."""

    field: Tensor
    local_field: Tensor
    final_tiles: Tensor
    local_tiles: Tensor
    initial_port: Tensor
    corrected_port: Tensor
    proposed_neighbor_port: Tensor
    realized_local_port: Tensor
    face_roles: Tensor
    standardized_correction_code: Tensor
    correction_code: Tensor
    communication: FieldTokenComposition


class AirfoilRobinLatentDDM(nn.Module):
    """Predict one Robin correction between two calls to a frozen local map."""

    def __init__(
        self,
        local_solver: nn.Module,
        *,
        width: int = 64,
        heads: int = 4,
        depth: int = 4,
        trace_modes: int = 4,
        spatial_bins: int = 4,
        code_scale: Tensor | None = None,
        code_center: Tensor | None = None,
        communication_mode: str = "centered_delta",
    ) -> None:
        super().__init__()
        if (
            min(width, heads, depth, trace_modes, spatial_bins) <= 0
            or width % heads
        ):
            raise ValueError("invalid Airfoil LatentDDM dimensions")
        self.local_solver = local_solver
        self.local_solver.eval().requires_grad_(False)
        self.width = int(width)
        self.trace_modes = int(trace_modes)
        self.spatial_bins = int(spatial_bins)
        self.port_channels = int(getattr(local_solver, "robin_channels", 12))
        if self.port_channels % 4:
            raise ValueError("the local port must contain four equal face blocks")
        self.channels_per_face = self.port_channels // 4
        self.code_width = 4 * self.channels_per_face * self.trace_modes
        if communication_mode not in (
            "centered_delta", "full_delta", "full_absolute",
        ):
            raise ValueError("unknown Airfoil communication mode")
        self.communication_mode = communication_mode
        # Physical condition, tentative q, three versions of the face port,
        # and four oriented face roles.  Conservative ports have three values
        # per face; native velocity/traction ports have two.
        input_channels = (
            int(local_solver.physical_channels) + 3
            + 3 * self.port_channels + 4
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
        )
        self.bulk_projection = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.SiLU(),
        )
        self.spatial_projection = nn.Sequential(
            nn.LayerNorm(width * spatial_bins**2),
            nn.Linear(width * spatial_bins**2, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width, bias=False),
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(local_solver.scalar_dim, width),
            nn.SiLU(),
            nn.Linear(width, width, bias=False),
        )
        self.communication = CenteredFieldTokenTransformer(
            width, self.code_width, heads, dimension=2, depth=depth,
        )
        scale = torch.ones(self.code_width) if code_scale is None else torch.as_tensor(
            code_scale, dtype=torch.float32,
        ).flatten()
        if scale.numel() != self.code_width or bool(torch.any(scale <= 0.0)):
            raise ValueError("Robin correction scale has the wrong shape")
        self.register_buffer("code_scale", scale)
        center = torch.zeros(self.code_width) if code_center is None else torch.as_tensor(
            code_center, dtype=torch.float32,
        ).flatten()
        if center.numel() != self.code_width or not bool(torch.isfinite(center).all()):
            raise ValueError("Robin code center has the wrong shape")
        # Stored explicitly in the model checkpoint metadata.  Keeping the
        # center non-persistent preserves backward compatibility with the
        # previously selected centered-delta composer.
        self.register_buffer("code_center", center, persistent=False)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def composer_state_dict(self) -> dict[str, Tensor]:
        """Return only the lightweight stage; the local checkpoint is lineage."""
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("local_solver.")
        }

    def load_composer_state_dict(self, state: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        expected_missing = {
            name for name in self.state_dict()
            if name.startswith("local_solver.")
        }
        if set(missing) != expected_missing or unexpected:
            raise ValueError(
                "Airfoil Robin composer checkpoint has incompatible keys: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def train(self, mode: bool = True) -> "AirfoilRobinLatentDDM":
        super().train(mode)
        self.local_solver.eval()
        return self

    def _port_kwargs(self, example: dict[str, object]) -> dict[str, object]:
        configuration = example["configuration"]
        return {
            "state_mean": example["mean"].flatten(),
            "state_std": example["std"].flatten(),
            "flux_scale": example["flux_scale"],
            "dx": example["dx"],
            "viscosity": example["viscosity"],
            "impedance": (
                float(configuration.mass_impedance),
                float(configuration.momentum_impedance),
            ),
            "face_width": int(configuration.face_width),
            "neighbor_faces": example["neighbor_faces"],
            "valid": example["valid_tiles"],
        }

    def _tokens(
        self,
        physical: Tensor,
        scalar: Tensor,
        local: Tensor,
        initial_port: Tensor,
        realized_port: Tensor,
        proposed_port: Tensor,
        roles: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        count = math.prod(grid_shape)
        if local.shape[0] % count:
            raise ValueError("tile batch is incompatible with the cover")
        valid = physical[:, -1:].to(local)
        features = torch.cat((
            physical,
            local,
            initial_port,
            realized_port - initial_port,
            proposed_port - initial_port,
            roles,
        ), dim=1)
        encoded = self.encoder(features) * valid
        denominator = valid.sum((-2, -1)).clamp_min(1.0)
        mean = encoded.sum((-2, -1)) / denominator
        rms = torch.sqrt(
            encoded.square().sum((-2, -1)) / denominator + 1.0e-8,
        )
        bulk = self.bulk_projection(torch.cat((mean, rms), dim=1))
        pooled_valid = F.adaptive_avg_pool2d(valid, self.spatial_bins).clamp_min(
            1.0e-6,
        )
        spatial = F.adaptive_avg_pool2d(
            encoded, self.spatial_bins,
        ) / pooled_valid
        spatial = self.spatial_projection(spatial.flatten(1))
        batch = local.shape[0] // count
        return (
            bulk + spatial + self.scalar_projection(scalar)
        ).reshape(batch, count, self.width)

    def forward(
        self,
        example: dict[str, object],
        *,
        graph_mask: Tensor | None = None,
    ) -> AirfoilRobinLatentOutput:
        physical = example["physical"]
        scalar = example["scalar"]
        cover = example["cover"]
        grid_shape = tuple(int(value) for value in cover.grid_shape)
        current_tiles = cover.tile(
            example["previous"], padding_mode="replicate",
        )
        kwargs = self._port_kwargs(example)
        initial_port, roles = absolute_robin_port_2d(current_tiles, **kwargs)
        local = self.local_solver(physical, scalar, initial_port, roles)
        local_field = assemble_subdomains_2d(
            cover, local, example["previous"].shape[0], "normalized_overlap",
        )
        # A unique RAS owner supplies the neighboring tentative solution used
        # to form an interface defect.  This is token construction, not a
        # second local-operator call.
        exchange = assemble_subdomains_2d(
            cover, local, example["previous"].shape[0], "disjoint_core",
        )
        consensus = cover.tile(exchange, padding_mode="replicate")
        realized_port, _ = absolute_robin_port_2d(local, **kwargs)
        proposed_port, _ = absolute_robin_port_2d(consensus, **kwargs)
        tokens = self._tokens(
            physical, scalar, local, initial_port, realized_port,
            proposed_port, roles, grid_shape,
        )
        communication = self.communication(
            tokens,
            grid_shape=grid_shape,
            graph_mask=graph_mask,
            centered=self.communication_mode == "centered_delta",
        )
        standardized_tokens = (
            communication.correction_code
            if self.communication_mode == "centered_delta" else
            communication.global_code
        )
        standardized = standardized_tokens.reshape(
            local.shape[0], self.code_width,
        )
        active = roles.repeat_interleave(3, dim=1)
        decoded_code = (
            standardized * self.code_scale.to(standardized)
            + self.code_center.to(standardized)
        )
        decoded_port = decode_trace_code_2d(
            decoded_code, roles, channels=3, modes=self.trace_modes,
        )
        if self.communication_mode == "full_absolute":
            correction_code = decoded_code
            corrected_port = decoded_port * active
        else:
            correction_code = decoded_code
            corrected_port = (initial_port + decoded_port) * active
        final_tiles = self.local_solver(
            physical, scalar, corrected_port, roles,
        )
        field = assemble_subdomains_2d(
            cover, final_tiles, example["previous"].shape[0],
            "normalized_overlap",
        )
        return AirfoilRobinLatentOutput(
            field=field,
            local_field=local_field,
            final_tiles=final_tiles,
            local_tiles=local,
            initial_port=initial_port,
            corrected_port=corrected_port,
            proposed_neighbor_port=proposed_port,
            realized_local_port=realized_port,
            face_roles=roles,
            standardized_correction_code=standardized,
            correction_code=correction_code,
            communication=communication,
        )
