"""Local conditional Flow-Matching model for airfoil block increments."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from codiffuse_Schwarz.tile_communication import TileGridCommunicator


Tensor = torch.Tensor


def timestep_embedding(timestep: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=timestep.dtype)
        / max(half - 1, 1)
    )
    angle = timestep[:, None] * frequency[None]
    embedding = torch.cat((angle.sin(), angle.cos()), dim=1)
    return F.pad(embedding, (0, dimension % 2))


class AntiAliasFilter2d(nn.Module):
    """Depthwise binomial low-pass with boundary-safe padding."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        coefficients = torch.tensor((1.0, 4.0, 6.0, 4.0, 1.0))
        kernel = coefficients[:, None] * coefficients[None, :]
        kernel = kernel / kernel.sum()
        self.channels = channels
        self.register_buffer(
            "kernel", kernel[None, None].repeat(channels, 1, 1, 1),
            persistent=False,
        )

    def forward(self, state: Tensor) -> Tensor:
        padded = F.pad(state, (2, 2, 2, 2), mode="replicate")
        return F.conv2d(padded, self.kernel.to(state), groups=self.channels)


class FiLMResidualBlock(nn.Module):
    def __init__(
        self,
        width: int,
        dilation: int = 1,
        spectral_shaping: bool = False,
    ) -> None:
        super().__init__()
        groups = min(16, width)
        while width % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.condition = nn.Linear(width, 2 * width)
        self.anti_alias = AntiAliasFilter2d(width) if spectral_shaping else None
        self.spectral_shaping_gate = (
            nn.Parameter(torch.zeros(())) if spectral_shaping else None
        )

    def shape_spectrum(self, state: Tensor) -> Tensor:
        if self.anti_alias is None:
            return state
        assert self.spectral_shaping_gate is not None
        low_pass = self.anti_alias(state)
        # clamp_min has unit derivative at zero in PyTorch, preserving both
        # exact checkpoint identity and a learnable anti-alias direction.
        gate = 0.75 * torch.tanh(self.spectral_shaping_gate.clamp_min(0.0))
        return state + gate * (low_pass - state)

    def forward(self, state: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm1(state)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(self.shape_spectrum(F.silu(hidden)))
        hidden = self.shape_spectrum(F.silu(self.norm2(hidden)))
        hidden = self.conv2(hidden)
        return state + hidden


class LocalAirfoilFlow(nn.Module):
    """Shared local model; communication is intentionally absent in P0."""

    def __init__(
        self,
        physical_channels: int = 12,
        scalar_dim: int = 15,
        width: int = 96,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.width = width
        dilation_cycle = (1, 1, 2, 2, 4, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.stem = nn.Conv2d(3 + physical_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(width, dilation_cycle[index % len(dilation_cycle)])
            for index in range(depth)
        )
        groups = min(16, width)
        while width % groups:
            groups -= 1
        self.output = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(
        self,
        flow_state: Tensor,
        physical: Tensor,
        scalar: Tensor,
        flow_time: Tensor,
    ) -> Tensor:
        if physical.shape[1] != self.physical_channels:
            raise ValueError(
                f"expected {self.physical_channels} physical channels, got {physical.shape[1]}"
            )
        condition = self.time_mlp(timestep_embedding(flow_time, self.width))
        condition = condition + self.scalar_mlp(scalar)
        hidden = self.stem(torch.cat((flow_state, physical), dim=1))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden) * physical[:, -1:]

    @torch.no_grad()
    def sample(
        self,
        physical: Tensor,
        scalar: Tensor,
        steps: int,
        seed: int,
    ) -> Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        generator = torch.Generator(device=physical.device).manual_seed(seed)
        state = torch.randn(
            (physical.shape[0], 3, physical.shape[-2], physical.shape[-1]),
            device=physical.device,
            dtype=physical.dtype,
            generator=generator,
        ) * physical[:, -1:]
        step_size = 1.0 / steps
        for step in range(steps):
            flow_time = torch.full(
                (physical.shape[0],),
                (step + 0.5) / steps,
                device=physical.device,
                dtype=physical.dtype,
            )
            state = (state + step_size * self(state, physical, scalar, flow_time)) \
                * physical[:, -1:]
        return state


class AxialAttentionBlock(nn.Module):
    """Axial processor used by the deterministic NeuralD MPP implementation."""

    def __init__(self, channels: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_width = nn.LayerNorm(channels)
        self.width_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_height = nn.LayerNorm(channels)
        self.height_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(self, state: Tensor) -> Tensor:
        batch, channels, height, width = state.shape
        rows = state.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        normalized = self.norm_width(rows)
        attended, _ = self.width_attention(
            normalized, normalized, normalized, need_weights=False
        )
        rows = rows + attended
        state = rows.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

        columns = state.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        normalized = self.norm_height(columns)
        attended, _ = self.height_attention(
            normalized, normalized, normalized, need_weights=False
        )
        columns = columns + attended
        state = columns.reshape(batch, width, height, channels).permute(0, 3, 2, 1)

        tokens = state.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class MPPFlow(nn.Module):
    """Flow-Matching adaptation of NeuralD's deterministic MPP architecture."""

    def __init__(
        self,
        physical_channels: int = 12,
        scalar_dim: int = 15,
        width: int = 128,
        depth: int = 8,
        heads: int = 8,
        transport_skip: bool = False,
        endpoint_eos: bool = False,
        endpoint_eos_width: int = 64,
        endpoint_eos_depth: int = 6,
        endpoint_eos_direct: bool = False,
        endpoint_eos_warp: bool = False,
        endpoint_eos_kinematic_warp: bool = False,
        endpoint_vorticity: bool = False,
        endpoint_vorticity_residual: bool = False,
        endpoint_vorticity_dedicated: bool = False,
        endpoint_vorticity_phase_only: bool = False,
        endpoint_vorticity_scale: float = 1.0,
        endpoint_learned_projection: bool = False,
        endpoint_projection_attention: bool = False,
        endpoint_projection_spatial_attention: bool = False,
        endpoint_multilevel_projection: bool = False,
        endpoint_spectral_shaping: bool = False,
        endpoint_boundary_memory: bool = False,
        endpoint_boundary_memory_width: int = 32,
        endpoint_boundary_memory_depth: int = 3,
        endpoint_motion_transport_adapter: bool = False,
        endpoint_motion_transport_width: int = 32,
        endpoint_motion_transport_depth: int = 2,
        endpoint_geometry_latent: bool = False,
        endpoint_geometry_latent_width: int = 32,
        endpoint_geometry_latent_depth: int = 2,
        endpoint_prescribed_motion_warp: bool = False,
        endpoint_prescribed_motion_residual: bool = False,
        endpoint_prescribed_motion_band_cells: float = 8.0,
        endpoint_domain_boundary_adapter: bool = False,
        endpoint_domain_boundary_width: int = 32,
        endpoint_domain_boundary_depth: int = 2,
        endpoint_crop_ghost_feature: bool = False,
        endpoint_rollout_closure: bool = False,
        endpoint_rollout_closure_width: int = 32,
        endpoint_curl_memory: bool = False,
        endpoint_curl_memory_width: int = 32,
        endpoint_curl_memory_depth: int = 2,
        endpoint_feature_schwarz: bool = False,
        endpoint_tile_communication: bool = False,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("MPP width must be divisible by the attention head count")
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.width = width
        self.transport_skip = transport_skip
        # RevIN statistics contain mean and standard deviation for q_n=(u,v,p).
        condition_dim = scalar_dim + 6
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.embed = nn.Sequential(
            nn.Conv2d(3 + physical_channels, width, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            AxialAttentionBlock(width, heads) for _ in range(depth)
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(width, width // 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 3, kernel_size=3, padding=1),
        )
        self.output_affine = nn.Linear(width, 6)
        transport_width = max(width // 2, 16)
        self.transport_stem = nn.Conv2d(
            3 + physical_channels, transport_width, kernel_size=3, padding=1
        ) if transport_skip else None
        self.transport_condition = nn.Linear(
            width, 2 * transport_width
        ) if transport_skip else None
        self.transport_output = nn.Conv2d(
            transport_width, 3, kernel_size=3, padding=1
        ) if transport_skip else None
        self.endpoint_eos_width = endpoint_eos_width
        self.endpoint_eos_direct = endpoint_eos_direct
        self.endpoint_eos_warp = endpoint_eos_warp
        self.endpoint_eos_kinematic_warp = endpoint_eos_kinematic_warp
        self.endpoint_vorticity = endpoint_vorticity
        self.endpoint_vorticity_residual = endpoint_vorticity_residual
        self.endpoint_vorticity_dedicated = endpoint_vorticity_dedicated
        self.endpoint_vorticity_phase_only = endpoint_vorticity_phase_only
        self.endpoint_vorticity_scale = endpoint_vorticity_scale
        self.endpoint_learned_projection = endpoint_learned_projection
        self.endpoint_projection_attention = endpoint_projection_attention
        self.endpoint_projection_spatial_attention = endpoint_projection_spatial_attention
        self.endpoint_multilevel_projection = endpoint_multilevel_projection
        self.endpoint_spectral_shaping = endpoint_spectral_shaping
        self.endpoint_boundary_memory = endpoint_boundary_memory
        self.endpoint_boundary_memory_width = endpoint_boundary_memory_width
        self.endpoint_boundary_memory_depth = endpoint_boundary_memory_depth
        self.endpoint_motion_transport_adapter = endpoint_motion_transport_adapter
        self.endpoint_motion_transport_width = endpoint_motion_transport_width
        self.endpoint_motion_transport_depth = endpoint_motion_transport_depth
        self.endpoint_geometry_latent = endpoint_geometry_latent
        self.endpoint_geometry_latent_width = endpoint_geometry_latent_width
        self.endpoint_geometry_latent_depth = endpoint_geometry_latent_depth
        self.endpoint_prescribed_motion_warp = endpoint_prescribed_motion_warp
        self.endpoint_prescribed_motion_residual = (
            endpoint_prescribed_motion_residual
        )
        if (
            self.endpoint_prescribed_motion_residual
            and not self.endpoint_prescribed_motion_warp
        ):
            raise ValueError(
                "prescribed motion residual requires prescribed motion warp"
            )
        self.prescribed_motion_residual_gate = (
            nn.Parameter(torch.zeros(()))
            if self.endpoint_prescribed_motion_residual else None
        )
        if endpoint_prescribed_motion_band_cells <= 0.0:
            raise ValueError("prescribed motion band must be positive")
        self.endpoint_prescribed_motion_band_cells = (
            endpoint_prescribed_motion_band_cells
        )
        self.endpoint_domain_boundary_adapter = endpoint_domain_boundary_adapter
        self.endpoint_domain_boundary_width = endpoint_domain_boundary_width
        self.endpoint_domain_boundary_depth = endpoint_domain_boundary_depth
        self.endpoint_crop_ghost_feature = endpoint_crop_ghost_feature
        self.endpoint_rollout_closure = endpoint_rollout_closure
        self.endpoint_rollout_closure_width = endpoint_rollout_closure_width
        self.endpoint_curl_memory = endpoint_curl_memory
        self.endpoint_curl_memory_width = endpoint_curl_memory_width
        self.endpoint_curl_memory_depth = endpoint_curl_memory_depth
        if endpoint_curl_memory and min(
            endpoint_curl_memory_width, endpoint_curl_memory_depth,
        ) <= 0:
            raise ValueError("curl-memory dimensions must be positive")
        self.endpoint_feature_schwarz = endpoint_feature_schwarz
        self.feature_schwarz_gate = (
            nn.Parameter(torch.zeros(())) if endpoint_feature_schwarz else None
        )
        # A single pooled, geometry-relative communication after the endpoint
        # processor.  The communicator's final FiLM map is zero initialized,
        # so enabling it is an exact identity upgrade of an old checkpoint.
        self.endpoint_tile_communication = endpoint_tile_communication
        self.endpoint_tile_communicator = (
            TileGridCommunicator(
                endpoint_eos_width, scalar_dim, dimension=2,
                heads=(
                    8 if endpoint_eos_width % 8 == 0 else
                    4 if endpoint_eos_width % 4 == 0 else
                    2 if endpoint_eos_width % 2 == 0 else 1
                ),
            )
            if endpoint_tile_communication else None
        )
        self.gradient_checkpointing = False
        self.register_buffer(
            "endpoint_state_to_delta_ratio", torch.ones(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "endpoint_state_mean", torch.zeros(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "endpoint_state_std", torch.ones(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "endpoint_delta_mean", torch.zeros(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "endpoint_delta_std", torch.ones(1, 3, 1, 1), persistent=False
        )
        self.endpoint_stem: nn.Conv2d | None = None
        self.endpoint_condition: nn.Module | None = None
        self.endpoint_blocks: nn.ModuleList | None = None
        self.endpoint_output: nn.Conv2d | None = None
        self.endpoint_displacement: nn.Conv2d | None = None
        self.endpoint_warp_embed: nn.Conv2d | None = None
        self.endpoint_vorticity_output: nn.Conv2d | None = None
        self.vorticity_stem: nn.Conv2d | None = None
        self.vorticity_condition: nn.Module | None = None
        self.vorticity_blocks: nn.ModuleList | None = None
        self.vorticity_output: nn.Conv2d | None = None
        self.projection_stem: nn.Conv2d | None = None
        self.projection_condition: nn.Module | None = None
        self.projection_blocks: nn.ModuleList | None = None
        self.projection_output: nn.Conv2d | None = None
        self.projection_attention_norm: nn.LayerNorm | None = None
        self.projection_attention: nn.MultiheadAttention | None = None
        self.projection_attention_gate: nn.Parameter | None = None
        self.projection_attention_affine: nn.Linear | None = None
        self.projection_spatial_attention_norm: nn.LayerNorm | None = None
        self.projection_spatial_attention: nn.MultiheadAttention | None = None
        self.projection_spatial_attention_output: nn.Conv2d | None = None
        self.projection_spatial_position: nn.Parameter | None = None
        self.boundary_memory_stem: nn.Conv2d | None = None
        self.boundary_memory_condition: nn.Module | None = None
        self.boundary_memory_blocks: nn.ModuleList | None = None
        self.boundary_memory_candidate: nn.Conv2d | None = None
        self.boundary_memory_gate: nn.Conv2d | None = None
        self.boundary_memory_wall_mix: nn.Conv2d | None = None
        self.motion_transport_stem: nn.Conv2d | None = None
        self.motion_transport_condition: nn.Module | None = None
        self.motion_transport_blocks: nn.ModuleList | None = None
        self.motion_transport_output: nn.Conv2d | None = None
        self.motion_transport_warp_embed: nn.Conv2d | None = None
        self.geometry_latent_stem: nn.Conv2d | None = None
        self.geometry_latent_condition: nn.Module | None = None
        self.geometry_latent_blocks: nn.ModuleList | None = None
        self.geometry_latent_output: nn.Conv2d | None = None
        self.domain_boundary_stem: nn.Conv2d | None = None
        self.domain_boundary_condition: nn.Module | None = None
        self.domain_boundary_blocks: nn.ModuleList | None = None
        self.domain_boundary_output: nn.Conv2d | None = None
        self.domain_boundary_ghost_embed: nn.Conv2d | None = None
        self.rollout_closure_stem: nn.Conv2d | None = None
        self.rollout_closure_condition: nn.Module | None = None
        self.rollout_closure_block: nn.Module | None = None
        self.rollout_closure_output: nn.Conv2d | None = None
        self.curl_memory_stem: nn.Conv2d | None = None
        self.curl_memory_condition: nn.Module | None = None
        self.curl_memory_blocks: nn.ModuleList | None = None
        self.curl_memory_output: nn.Conv2d | None = None
        if endpoint_eos:
            self.enable_endpoint_eos(endpoint_eos_width, endpoint_eos_depth)
        with torch.no_grad():
            self.output_affine.weight.zero_()
            self.output_affine.bias.zero_()
            if self.transport_output is not None:
                self.transport_output.weight.zero_()
                self.transport_output.bias.zero_()

    def enable_endpoint_eos(self, width: int = 64, depth: int = 6) -> None:
        """Add a full-resolution clean-endpoint corrector to an existing Flow model."""
        if self.endpoint_stem is not None:
            return
        self.endpoint_eos_width = width
        self.endpoint_stem = nn.Conv2d(
            3 + self.physical_channels, width, kernel_size=5, padding=2
        )
        self.endpoint_condition = nn.Sequential(
            nn.Linear(self.scalar_dim + 6, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        dilation_cycle = (1, 2, 4, 1, 2, 4)
        self.endpoint_blocks = nn.ModuleList(
            FiLMResidualBlock(
                width,
                dilation_cycle[index % len(dilation_cycle)],
                spectral_shaping=self.endpoint_spectral_shaping,
            )
            for index in range(depth)
        )
        self.endpoint_output = nn.Conv2d(width, 3, kernel_size=3, padding=1)
        if self.endpoint_vorticity:
            if self.endpoint_vorticity_dedicated:
                self.vorticity_stem = nn.Conv2d(
                    self.physical_channels, width, kernel_size=5, padding=2
                )
                self.vorticity_condition = nn.Sequential(
                    nn.Linear(self.scalar_dim + 6, 2 * width),
                    nn.GELU(),
                    nn.Linear(2 * width, width),
                )
                self.vorticity_blocks = nn.ModuleList(
                    FiLMResidualBlock(
                        width,
                        (1, 2, 4)[index % 3],
                        spectral_shaping=self.endpoint_spectral_shaping,
                    )
                    for index in range(depth)
                )
                self.vorticity_output = nn.Conv2d(
                    width, 1, kernel_size=3, padding=1
                )
                nn.init.zeros_(self.vorticity_output.weight)
                nn.init.zeros_(self.vorticity_output.bias)
            else:
                self.endpoint_vorticity_output = nn.Conv2d(
                    width, 1, kernel_size=3, padding=1
                )
                nn.init.zeros_(self.endpoint_vorticity_output.weight)
                nn.init.zeros_(self.endpoint_vorticity_output.bias)
        if self.endpoint_eos_warp:
            self.endpoint_displacement = nn.Conv2d(width, 2, kernel_size=3, padding=1)
            self.endpoint_warp_embed = nn.Conv2d(3, width, kernel_size=3, padding=1)
            nn.init.zeros_(self.endpoint_displacement.weight)
            nn.init.zeros_(self.endpoint_displacement.bias)
        if self.endpoint_learned_projection:
            if not self.endpoint_vorticity:
                raise ValueError("learned projection requires endpoint vorticity")
            self.projection_stem = nn.Conv2d(
                self.physical_channels + 4, width, kernel_size=5, padding=2
            )
            self.projection_condition = nn.Sequential(
                nn.Linear(self.scalar_dim + 6, 2 * width),
                nn.GELU(),
                nn.Linear(2 * width, width),
            )
            self.projection_blocks = nn.ModuleList(
                FiLMResidualBlock(
                    width,
                    (1, 2, 4)[index % 3],
                    spectral_shaping=self.endpoint_spectral_shaping,
                )
                for index in range(depth)
            )
            self.projection_output = nn.Conv2d(width, 3, kernel_size=3, padding=1)
            nn.init.zeros_(self.projection_output.weight)
            nn.init.zeros_(self.projection_output.bias)
            if self.endpoint_projection_attention:
                attention_heads = 8 if width % 8 == 0 else 4
                self.projection_attention_norm = nn.LayerNorm(width)
                self.projection_attention = nn.MultiheadAttention(
                    width, attention_heads, batch_first=True
                )
                self.projection_attention_affine = nn.Linear(width, 2 * width)
                nn.init.zeros_(self.projection_attention_affine.weight)
                nn.init.zeros_(self.projection_attention_affine.bias)
                self.projection_attention_gate = nn.Parameter(torch.ones(()))
            if self.endpoint_projection_spatial_attention:
                attention_heads = 8 if width % 8 == 0 else 4
                self.projection_spatial_attention_norm = nn.LayerNorm(width)
                self.projection_spatial_attention = nn.MultiheadAttention(
                    width, attention_heads, batch_first=True
                )
                self.projection_spatial_attention_output = nn.Conv2d(
                    width, width, kernel_size=1
                )
                nn.init.zeros_(self.projection_spatial_attention_output.weight)
                nn.init.zeros_(self.projection_spatial_attention_output.bias)
                self.projection_spatial_position = nn.Parameter(
                    torch.zeros(1, 1, 16, width)
                )
                nn.init.normal_(self.projection_spatial_position, std=.02)
        if self.endpoint_boundary_memory:
            if self.physical_channels < 15:
                raise ValueError("boundary memory requires two-step physical history")
            memory_width = self.endpoint_boundary_memory_width
            self.boundary_memory_stem = nn.Conv2d(
                self.physical_channels + 3, memory_width, kernel_size=5, padding=2,
            )
            self.boundary_memory_condition = nn.Sequential(
                nn.Linear(self.scalar_dim + 6, 2 * memory_width),
                nn.GELU(),
                nn.Linear(2 * memory_width, memory_width),
            )
            self.boundary_memory_blocks = nn.ModuleList(
                FiLMResidualBlock(
                    memory_width, (1, 2, 4)[index % 3],
                    spectral_shaping=self.endpoint_spectral_shaping,
                )
                for index in range(self.endpoint_boundary_memory_depth)
            )
            self.boundary_memory_candidate = nn.Conv2d(
                memory_width, 2, kernel_size=3, padding=1,
            )
            self.boundary_memory_gate = nn.Conv2d(
                memory_width, 2, kernel_size=3, padding=1,
            )
            self.boundary_memory_wall_mix = nn.Conv2d(
                memory_width, 2, kernel_size=3, padding=1,
            )
            nn.init.zeros_(self.boundary_memory_candidate.weight)
            nn.init.zeros_(self.boundary_memory_candidate.bias)
            nn.init.zeros_(self.boundary_memory_gate.weight)
            nn.init.constant_(self.boundary_memory_gate.bias, -4.0)
            nn.init.zeros_(self.boundary_memory_wall_mix.weight)
            nn.init.zeros_(self.boundary_memory_wall_mix.bias)
        if self.endpoint_motion_transport_adapter:
            motion_width = self.endpoint_motion_transport_width
            self.motion_transport_stem = nn.Conv2d(
                self.physical_channels + 3, motion_width,
                kernel_size=5, padding=2,
            )
            self.motion_transport_condition = nn.Sequential(
                nn.Linear(self.scalar_dim + 6, 2 * motion_width),
                nn.GELU(),
                nn.Linear(2 * motion_width, motion_width),
            )
            self.motion_transport_blocks = nn.ModuleList(
                FiLMResidualBlock(
                    motion_width, (1, 2, 4)[index % 3],
                    spectral_shaping=self.endpoint_spectral_shaping,
                )
                for index in range(self.endpoint_motion_transport_depth)
            )
            self.motion_transport_output = nn.Conv2d(
                motion_width, 3, kernel_size=3, padding=1,
            )
            if self.endpoint_prescribed_motion_warp:
                self.motion_transport_warp_embed = nn.Conv2d(
                    3, motion_width, kernel_size=1, bias=False,
                )
            nn.init.zeros_(self.motion_transport_output.weight)
            nn.init.zeros_(self.motion_transport_output.bias)
        if self.endpoint_geometry_latent:
            geometry_width = self.endpoint_geometry_latent_width
            self.geometry_latent_stem = nn.Conv2d(
                7, geometry_width, kernel_size=5, padding=2,
            )
            self.geometry_latent_condition = nn.Sequential(
                nn.Linear(self.scalar_dim + 6, 2 * geometry_width),
                nn.GELU(),
                nn.Linear(2 * geometry_width, geometry_width),
            )
            self.geometry_latent_blocks = nn.ModuleList(
                FiLMResidualBlock(
                    geometry_width, (1, 2, 4)[index % 3],
                    spectral_shaping=self.endpoint_spectral_shaping,
                )
                for index in range(self.endpoint_geometry_latent_depth)
            )
            self.geometry_latent_output = nn.Conv2d(
                geometry_width, width, kernel_size=1,
            )
            nn.init.zeros_(self.geometry_latent_output.weight)
            nn.init.zeros_(self.geometry_latent_output.bias)
        if self.endpoint_domain_boundary_adapter:
            boundary_width = self.endpoint_domain_boundary_width
            self.domain_boundary_stem = nn.Conv2d(
                self.physical_channels + 3, boundary_width,
                kernel_size=5, padding=2,
            )
            self.domain_boundary_condition = nn.Sequential(
                nn.Linear(self.scalar_dim + 6, 2 * boundary_width),
                nn.GELU(),
                nn.Linear(2 * boundary_width, boundary_width),
            )
            self.domain_boundary_blocks = nn.ModuleList(
                FiLMResidualBlock(
                    boundary_width, (1, 2, 4)[index % 3],
                    spectral_shaping=self.endpoint_spectral_shaping,
                )
                for index in range(self.endpoint_domain_boundary_depth)
            )
            self.domain_boundary_output = nn.Conv2d(
                boundary_width, 3, kernel_size=3, padding=1,
            )
            if self.endpoint_crop_ghost_feature:
                self.domain_boundary_ghost_embed = nn.Conv2d(
                    3, boundary_width, kernel_size=1, bias=False,
                )
            nn.init.zeros_(self.domain_boundary_output.weight)
            nn.init.zeros_(self.domain_boundary_output.bias)
        if self.endpoint_rollout_closure:
            closure_width = self.endpoint_rollout_closure_width
            self.rollout_closure_stem = nn.Conv2d(
                self.physical_channels + 4, closure_width, kernel_size=3, padding=1,
            )
            self.rollout_closure_condition = nn.Sequential(
                nn.Linear(self.scalar_dim + 6, 2 * closure_width),
                nn.GELU(),
                nn.Linear(2 * closure_width, closure_width),
            )
            self.rollout_closure_block = FiLMResidualBlock(closure_width, 1)
            self.rollout_closure_output = nn.Conv2d(
                closure_width, 3, kernel_size=3, padding=1,
            )
            nn.init.zeros_(self.rollout_closure_output.weight)
            nn.init.zeros_(self.rollout_closure_output.bias)
        if self.endpoint_curl_memory:
            curl_width = self.endpoint_curl_memory_width
            self.curl_memory_stem = nn.Conv2d(
                12, curl_width, kernel_size=5, padding=2,
            )
            self.curl_memory_condition = nn.Sequential(
                nn.Linear(self.scalar_dim + 6, 2 * curl_width),
                nn.GELU(),
                nn.Linear(2 * curl_width, curl_width),
            )
            self.curl_memory_blocks = nn.ModuleList(
                FiLMResidualBlock(curl_width, (1, 2, 4)[index % 3])
                for index in range(self.endpoint_curl_memory_depth)
            )
            self.curl_memory_output = nn.Conv2d(
                curl_width, 3, kernel_size=3, padding=1,
            )
            nn.init.zeros_(self.curl_memory_output.weight)
            nn.init.zeros_(self.curl_memory_output.bias)
        nn.init.zeros_(self.endpoint_output.weight)
        nn.init.zeros_(self.endpoint_output.bias)

    def set_endpoint_statistics(
        self,
        state_mean: tuple[float, float, float],
        state_std: tuple[float, float, float],
        delta_std: tuple[float, float, float],
        delta_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        ratio = torch.tensor(state_std, dtype=torch.float32) / torch.tensor(
            delta_std, dtype=torch.float32
        ).clamp_min(1.0e-12)
        self.endpoint_state_to_delta_ratio.copy_(ratio[None, :, None, None])
        self.endpoint_state_std.copy_(
            torch.tensor(state_std, dtype=torch.float32)[None, :, None, None]
        )
        self.endpoint_state_mean.copy_(
            torch.tensor(state_mean, dtype=torch.float32)[None, :, None, None]
        )
        self.endpoint_delta_mean.copy_(
            torch.tensor(delta_mean, dtype=torch.float32)[None, :, None, None]
        )
        self.endpoint_delta_std.copy_(
            torch.tensor(delta_std, dtype=torch.float32)[None, :, None, None]
        )

    def correct_endpoint_fields(
        self,
        endpoint: Tensor,
        physical: Tensor,
        scalar: Tensor,
        tile_count: int | None = None,
        tile_exchange: Callable[[Tensor], Tensor] | None = None,
        motion_physical: Tensor | None = None,
        motion_scalar: Tensor | None = None,
        tile_grid_shape: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return clean normalized increment and optional physical vorticity."""
        if self.endpoint_stem is None:
            return endpoint, None
        assert self.endpoint_condition is not None
        assert self.endpoint_blocks is not None
        assert self.endpoint_output is not None
        physical_normalized, statistics = self.revin_previous_field(physical)
        condition = self.endpoint_condition(torch.cat((scalar, statistics), dim=1))
        endpoint_input = torch.zeros_like(endpoint) if self.endpoint_eos_direct else endpoint
        hidden = self.endpoint_stem(
            torch.cat((endpoint_input, physical_normalized), dim=1)
        )
        prescribed_motion_increment = None
        if self.geometry_latent_stem is not None:
            if motion_physical is None or motion_scalar is None:
                raise ValueError(
                    "geometry latent requires endpoint motion conditions"
                )
            assert self.geometry_latent_condition is not None
            assert self.geometry_latent_blocks is not None
            assert self.geometry_latent_output is not None
            prescribed_motion_increment = self._prescribed_motion_increment(
                motion_physical, motion_scalar,
            )
            _, motion_statistics = self.revin_previous_field(motion_physical)
            geometry_condition = self.geometry_latent_condition(torch.cat(
                (motion_scalar, motion_statistics), dim=1,
            ))
            geometry_features = torch.cat((
                prescribed_motion_increment,
                motion_physical[:, 3:5],
                motion_physical[:, 7:9] - motion_physical[:, 5:7],
            ), dim=1)
            geometry_hidden = self.geometry_latent_stem(geometry_features)
            for block in self.geometry_latent_blocks:
                geometry_hidden = self._residual_block(
                    block, geometry_hidden, geometry_condition,
                )
            distance_cells = 128.0 * torch.minimum(
                motion_physical[:, 3:4].abs(),
                motion_physical[:, 4:5].abs(),
            )
            swept = torch.exp(
                -distance_cells / self.endpoint_prescribed_motion_band_cells
            ) * motion_physical[:, -1:]
            geometry_cell = (
                swept.amax(dim=(-2, -1), keepdim=True) >= .367879
            ).to(hidden)
            hidden = hidden + geometry_cell * self.geometry_latent_output(
                F.silu(geometry_hidden)
            )
        for block in self.endpoint_blocks:
            hidden = self._residual_block(block, hidden, condition)
            hidden = self._feature_schwarz(hidden, tile_exchange)
        if self.endpoint_tile_communicator is not None:
            if tile_grid_shape is not None and (
                tile_count is None or math.prod(tile_grid_shape) != tile_count
            ):
                raise ValueError("tile_grid_shape is incompatible with tile_count")
            hidden = self.endpoint_tile_communicator(
                hidden, scalar, tile_grid_shape,
            )
        transported_increment = None
        if self.endpoint_displacement is not None:
            assert self.endpoint_warp_embed is not None
            batch, _, height, width = hidden.shape
            displacement = 4.0 * torch.tanh(self.endpoint_displacement(hidden))
            if self.endpoint_eos_kinematic_warp:
                previous_physical = (
                    physical[:, :3] * self.endpoint_state_std.to(physical)
                    + self.endpoint_state_mean.to(physical)
                )
                dt_over_dx = scalar[:, 11, None, None]
                displacement = displacement + torch.stack(
                    (
                        previous_physical[:, 0] * dt_over_dx,
                        previous_physical[:, 1] * dt_over_dx,
                    ),
                    dim=1,
                ).clamp(-4.0, 4.0)
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, height, device=hidden.device, dtype=hidden.dtype),
                torch.linspace(-1.0, 1.0, width, device=hidden.device, dtype=hidden.dtype),
                indexing="ij",
            )
            grid = torch.stack((xx, yy), dim=-1)[None].expand(batch, -1, -1, -1).clone()
            grid[..., 0] -= 2.0 * displacement[:, 0] / max(width - 1, 1)
            grid[..., 1] -= 2.0 * displacement[:, 1] / max(height - 1, 1)
            previous = physical[:, :3]
            warped = F.grid_sample(
                previous, grid, mode="bilinear", padding_mode="border",
                align_corners=True,
            )
            transported_state = warped - previous
            hidden = hidden + self.endpoint_warp_embed(transported_state)
            transported_increment = (
                transported_state
                * self.endpoint_state_to_delta_ratio.to(transported_state)
            )
        output = self.endpoint_output(F.silu(hidden))
        if transported_increment is not None:
            output = output + transported_increment
        result = output if self.endpoint_eos_direct else endpoint + output
        if self.endpoint_prescribed_motion_warp:
            if motion_physical is None or motion_scalar is None:
                raise ValueError(
                    "prescribed motion warp requires endpoint motion conditions"
                )
            if prescribed_motion_increment is None:
                prescribed_motion_increment = self._prescribed_motion_increment(
                    motion_physical, motion_scalar,
                )
        vorticity = None
        if self.vorticity_stem is not None:
            assert self.vorticity_condition is not None
            assert self.vorticity_blocks is not None
            assert self.vorticity_output is not None
            vorticity_condition = self.vorticity_condition(
                torch.cat((scalar, statistics), dim=1)
            )
            vorticity_physical = physical_normalized
            if self.endpoint_vorticity_phase_only:
                vorticity_physical = physical_normalized.clone()
                vorticity_physical[:, :3] = 0.0
            vorticity_hidden = self.vorticity_stem(vorticity_physical)
            for block in self.vorticity_blocks:
                vorticity_hidden = self._residual_block(
                    block, vorticity_hidden, vorticity_condition,
                )
                vorticity_hidden = self._feature_schwarz(
                    vorticity_hidden, tile_exchange,
                )
            vorticity = self.vorticity_output(F.silu(vorticity_hidden))[:, 0]
            vorticity = vorticity * self.endpoint_vorticity_scale
            vorticity = vorticity * physical[:, -1]
        elif self.endpoint_vorticity_output is not None:
            vorticity = self.endpoint_vorticity_output(F.silu(hidden))[:, 0]
            vorticity = vorticity * self.endpoint_vorticity_scale
            vorticity = vorticity * physical[:, -1]
        if self.projection_stem is not None and vorticity is not None:
            assert self.projection_condition is not None
            assert self.projection_blocks is not None
            assert self.projection_output is not None
            projection_condition = self.projection_condition(
                torch.cat((scalar, statistics), dim=1)
            )
            dimensionless_vorticity = (
                vorticity / max(self.endpoint_vorticity_scale, 1.0e-12)
            )[:, None]
            projection_hidden = self.projection_stem(torch.cat(
                (physical_normalized, result, dimensionless_vorticity), dim=1
            ))
            for block in self.projection_blocks:
                projection_hidden = self._residual_block(
                    block, projection_hidden, projection_condition,
                )
                projection_hidden = self._feature_schwarz(
                    projection_hidden, tile_exchange,
                )
            if self.projection_attention is not None and tile_count is not None:
                assert self.projection_attention_norm is not None
                assert self.projection_attention_gate is not None
                assert self.projection_attention_affine is not None
                if projection_hidden.shape[0] % tile_count:
                    raise ValueError("local tile batch is not divisible by tile_count")
                batch_size = projection_hidden.shape[0] // tile_count
                tokens = projection_hidden.mean(dim=(-1, -2)).reshape(
                    batch_size, tile_count, -1
                )
                normalized_tokens = self.projection_attention_norm(tokens)
                attended, _ = self.projection_attention(
                    normalized_tokens, normalized_tokens, normalized_tokens,
                    need_weights=False,
                )
                scale, shift = self.projection_attention_affine(attended).chunk(2, dim=-1)
                scale = scale.reshape(-1, scale.shape[-1])[:, :, None, None]
                shift = shift.reshape(-1, shift.shape[-1])[:, :, None, None]
                gate = self.projection_attention_gate
                projection_hidden = projection_hidden * (1.0 + gate * scale) \
                    + gate * shift
            if (
                self.projection_spatial_attention is not None
                and tile_count is not None
            ):
                assert self.projection_spatial_attention_norm is not None
                assert self.projection_spatial_attention_output is not None
                assert self.projection_spatial_position is not None
                if projection_hidden.shape[0] % tile_count:
                    raise ValueError("local tile batch is not divisible by tile_count")
                batch_size = projection_hidden.shape[0] // tile_count
                pooled = F.adaptive_avg_pool2d(projection_hidden, (4, 4))
                tokens = pooled.flatten(2).transpose(1, 2).reshape(
                    batch_size, tile_count * 16, -1
                )
                position = self.projection_spatial_position.expand(
                    batch_size, tile_count, -1, -1
                ).reshape(batch_size, tile_count * 16, -1)
                normalized_tokens = self.projection_spatial_attention_norm(
                    tokens + position
                )
                attended, _ = self.projection_spatial_attention(
                    normalized_tokens, normalized_tokens, normalized_tokens,
                    need_weights=False,
                )
                attended = attended.reshape(
                    batch_size * tile_count, 4, 4, -1
                ).permute(0, 3, 1, 2)
                attended = F.interpolate(
                    attended, size=projection_hidden.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                projection_hidden = projection_hidden \
                    + self.projection_spatial_attention_output(attended)
            result = result + self.projection_output(F.silu(projection_hidden))
        if self.motion_transport_stem is not None:
            if motion_physical is None or motion_scalar is None:
                raise ValueError(
                    "motion transport adapter requires endpoint motion conditions"
                )
            assert self.motion_transport_condition is not None
            assert self.motion_transport_blocks is not None
            assert self.motion_transport_output is not None
            motion_normalized, motion_statistics = self.revin_previous_field(
                motion_physical
            )
            motion_condition = self.motion_transport_condition(torch.cat(
                (motion_scalar, motion_statistics), dim=1,
            ))
            motion_hidden = self.motion_transport_stem(torch.cat(
                (motion_normalized, result), dim=1,
            ))
            if self.motion_transport_warp_embed is not None:
                assert prescribed_motion_increment is not None
                motion_hidden = motion_hidden + self.motion_transport_warp_embed(
                    prescribed_motion_increment
                )
            for block in self.motion_transport_blocks:
                motion_hidden = self._residual_block(
                    block, motion_hidden, motion_condition,
                )
            # Source and destination SDFs define the swept moving-wall band.
            swept_band = torch.exp(
                -16.0 * torch.minimum(
                    motion_physical[:, 3:4].abs(),
                    motion_physical[:, 4:5].abs(),
                )
            )
            motion_correction = 0.25 * torch.tanh(
                self.motion_transport_output(F.silu(motion_hidden))
            )
            result = result + (
                swept_band * motion_correction * motion_physical[:, -1:]
            )
        if self.endpoint_prescribed_motion_residual:
            assert prescribed_motion_increment is not None
            assert self.prescribed_motion_residual_gate is not None
            # The network now learns a residual around known local rigid-pose
            # transport instead of deleting and recreating the moving wall.
            result = result + torch.tanh(
                self.prescribed_motion_residual_gate
            ) * prescribed_motion_increment
        if self.domain_boundary_stem is not None:
            assert self.domain_boundary_condition is not None
            assert self.domain_boundary_blocks is not None
            assert self.domain_boundary_output is not None
            boundary_condition = self.domain_boundary_condition(
                torch.cat((scalar, statistics), dim=1)
            )
            boundary_hidden = self.domain_boundary_stem(torch.cat(
                (physical_normalized, result), dim=1,
            ))
            if self.domain_boundary_ghost_embed is not None:
                ghost = self._nearest_valid_ghost_feature(
                    physical_normalized[:, :3], physical[:, -1:],
                )
                boundary_hidden = boundary_hidden \
                    + self.domain_boundary_ghost_embed(ghost)
            for block in self.domain_boundary_blocks:
                boundary_hidden = self._residual_block(
                    block, boundary_hidden, boundary_condition,
                )
            # Only a global-domain halo contains invalid cells. Interior
            # overlap halos are populated from neighboring Cartesian cells.
            boundary_tile = (
                physical[:, -1:].amin(dim=(-2, -1), keepdim=True) < 0.5
            ).to(result.dtype)
            correction = 0.25 * torch.tanh(
                self.domain_boundary_output(F.silu(boundary_hidden))
            )
            result = result + boundary_tile * correction * physical[:, -1:]
        if self.boundary_memory_stem is not None:
            assert self.boundary_memory_condition is not None
            assert self.boundary_memory_blocks is not None
            assert self.boundary_memory_candidate is not None
            assert self.boundary_memory_gate is not None
            assert self.boundary_memory_wall_mix is not None
            memory_condition = self.boundary_memory_condition(
                torch.cat((scalar, statistics), dim=1)
            )
            memory_hidden = self.boundary_memory_stem(torch.cat(
                (physical_normalized, result), dim=1,
            ))
            for block in self.boundary_memory_blocks:
                memory_hidden = self._residual_block(
                    block, memory_hidden, memory_condition,
                )
            history_increment = physical[:, -4:-2]
            wall_increment = (
                physical[:, 7:9] - physical[:, 5:7]
                - self.endpoint_delta_mean[:, :2].to(physical)
            ) / self.endpoint_delta_std[:, :2].to(physical).clamp_min(1.0e-12)
            wall_mix = torch.sigmoid(
                self.boundary_memory_wall_mix(F.silu(memory_hidden))
            )
            candidate = (
                (1.0 - wall_mix) * history_increment
                + wall_mix * wall_increment
                + 0.5 * torch.tanh(
                self.boundary_memory_candidate(F.silu(memory_hidden))
                )
            )
            gate = torch.sigmoid(
                self.boundary_memory_gate(F.silu(memory_hidden))
            )
            # SDF is normalized by four patch widths. This envelope covers the
            # moving-body band while leaving the wake endpoint unchanged.
            boundary = torch.exp(-16.0 * physical[:, 4:5].abs())
            boundary = boundary * physical[:, -1:]
            velocity = result[:, :2] + boundary * gate * (
                candidate - result[:, :2]
            )
            result = torch.cat((velocity, result[:, 2:]), dim=1)
        if self.rollout_closure_stem is not None:
            assert self.rollout_closure_condition is not None
            assert self.rollout_closure_block is not None
            assert self.rollout_closure_output is not None
            closure_condition = self.rollout_closure_condition(
                torch.cat((scalar, statistics), dim=1)
            )
            closure_vorticity = torch.zeros_like(result[:, :1])
            if vorticity is not None:
                closure_vorticity = (
                    vorticity / max(self.endpoint_vorticity_scale, 1.0e-12)
                )[:, None]
            closure_hidden = self.rollout_closure_stem(torch.cat(
                (physical_normalized, result, closure_vorticity), dim=1,
            ))
            closure_hidden = self._residual_block(
                self.rollout_closure_block, closure_hidden, closure_condition,
            )
            correction = 0.25 * torch.tanh(
                self.rollout_closure_output(F.silu(closure_hidden))
            )
            result = result + correction * physical[:, -1:]
        if self.curl_memory_stem is not None:
            assert self.curl_memory_condition is not None
            assert self.curl_memory_blocks is not None
            assert self.curl_memory_output is not None
            if self.physical_channels < 15:
                raise ValueError("curl memory requires two-step physical history")
            current_curl = self._cell_index_curl(physical_normalized[:, :2])
            history_curl = self._cell_index_curl(physical[:, -4:-2])
            endpoint_vorticity = torch.zeros_like(current_curl)
            if vorticity is not None:
                endpoint_vorticity = (
                    vorticity / max(self.endpoint_vorticity_scale, 1.0e-12)
                )[:, None]
            curl_features = torch.cat((
                current_curl, history_curl, result, endpoint_vorticity,
                physical[:, 3:5], physical[:, 5:9],
            ), dim=1)
            curl_condition = self.curl_memory_condition(
                torch.cat((scalar, statistics), dim=1)
            )
            curl_hidden = self.curl_memory_stem(curl_features)
            for block in self.curl_memory_blocks:
                curl_hidden = self._residual_block(
                    block, curl_hidden, curl_condition,
                )
            curl_correction = 0.25 * torch.tanh(
                self.curl_memory_output(F.silu(curl_hidden))
            )
            result = result + curl_correction * physical[:, -1:]
        return result * physical[:, -1:], vorticity

    @staticmethod
    def _cell_index_curl(velocity: Tensor) -> Tensor:
        """Dimensionless centered curl on a local Cartesian tile."""
        if velocity.ndim != 4 or velocity.shape[1] != 2:
            raise ValueError("velocity must have shape B,2,H,W")
        padded = F.pad(velocity, (1, 1, 1, 1), mode="replicate")
        du_dy = 0.5 * (
            padded[:, 0:1, 2:, 1:-1] - padded[:, 0:1, :-2, 1:-1]
        )
        dv_dx = 0.5 * (
            padded[:, 1:2, 1:-1, 2:] - padded[:, 1:2, 1:-1, :-2]
        )
        return dv_dx - du_dy

    @staticmethod
    def _nearest_valid_ghost_feature(state: Tensor, valid: Tensor) -> Tensor:
        """Nearest-state extrapolation outside a rectangular crop tile."""
        if state.ndim != 4 or valid.shape != (
            state.shape[0], 1, state.shape[2], state.shape[3]
        ):
            raise ValueError("state and valid mask have incompatible shapes")
        present = valid[:, 0] > 0.5
        if not bool(present.flatten(1).any(dim=1).all()):
            raise ValueError("every Cartesian tile must contain a valid cell")
        rows = present.any(dim=-1)
        columns = present.any(dim=-2)
        first_y = rows.to(torch.int64).argmax(dim=1)
        last_y = state.shape[-2] - 1 \
            - rows.flip(1).to(torch.int64).argmax(dim=1)
        first_x = columns.to(torch.int64).argmax(dim=1)
        last_x = state.shape[-1] - 1 \
            - columns.flip(1).to(torch.int64).argmax(dim=1)
        y = torch.arange(state.shape[-2], device=state.device)[None]
        x = torch.arange(state.shape[-1], device=state.device)[None]
        y = torch.maximum(torch.minimum(y, last_y[:, None]), first_y[:, None])
        x = torch.maximum(torch.minimum(x, last_x[:, None]), first_x[:, None])
        linear = (
            y[:, :, None] * state.shape[-1] + x[:, None, :]
        ).reshape(state.shape[0], 1, -1).expand(-1, state.shape[1], -1)
        filled = state.flatten(2).gather(2, linear).reshape_as(state)
        return (filled - state) * (1.0 - valid)

    def _prescribed_motion_increment(
        self, physical: Tensor, scalar: Tensor,
    ) -> Tensor:
        """Transport near-body state exactly between prescribed rigid poses.

        The local network then predicts only the residual fluid update. The
        four-cell overlap is sufficient for the measured two-cell maximum L6
        body displacement, so the warp remains a local Cartesian-cell EoS.
        """
        if physical.shape[1] < 12 or scalar.shape[1] < 12:
            raise ValueError("prescribed motion warp requires airfoil conditions")
        previous = (
            physical[:, :3] * self.endpoint_state_std.to(physical)
            + self.endpoint_state_mean.to(physical)
        )
        x = 10.0 * physical[:, 9]
        y = 10.0 * physical[:, 10]
        delta_angle = math.pi * (scalar[:, 5] - scalar[:, 3])
        cosine = delta_angle.cos()[:, None, None]
        sine = delta_angle.sin()[:, None, None]
        source_x = cosine * x + sine * y
        source_y = -sine * x + cosine * y

        dt = 0.01 * scalar[:, 9]
        dx = (dt / scalar[:, 11].clamp_min(1.0e-12))[:, None, None]
        height, width = previous.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(
                -1.0, 1.0, height, device=physical.device,
                dtype=physical.dtype,
            ),
            torch.linspace(
                -1.0, 1.0, width, device=physical.device,
                dtype=physical.dtype,
            ),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1)[None].expand(
            previous.shape[0], -1, -1, -1,
        ).clone()
        grid[..., 0] += 2.0 * (source_x - x) / (
            dx * max(width - 1, 1)
        )
        grid[..., 1] += 2.0 * (source_y - y) / (
            dx * max(height - 1, 1)
        )
        warped = F.grid_sample(
            previous, grid, mode="bilinear", padding_mode="border",
            align_corners=True,
        )
        source_u, source_v = warped[:, 0].clone(), warped[:, 1].clone()
        warped[:, 0] = cosine * source_u - sine * source_v
        warped[:, 1] = sine * source_u + cosine * source_v

        # SDF channels are normalized by 4 * core * dx with core=32.
        distance_cells = 128.0 * torch.minimum(
            physical[:, 3].abs(), physical[:, 4].abs(),
        )
        gate = torch.exp(
            -distance_cells / self.endpoint_prescribed_motion_band_cells
        )[:, None]
        solid_union = ((physical[:, 3] < 0.0) | (physical[:, 4] < 0.0))[:, None]
        gate = torch.maximum(gate, solid_union.to(gate.dtype))
        physical_increment = gate * (warped - previous) * physical[:, -1:]
        return physical_increment / self.endpoint_delta_std.to(physical)

    def _residual_block(
        self,
        block: nn.Module,
        state: Tensor,
        condition: Tensor,
    ) -> Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(block, state, condition, use_reentrant=False)
        return block(state, condition)

    def _feature_schwarz(
        self,
        state: Tensor,
        tile_exchange: Callable[[Tensor], Tensor] | None,
    ) -> Tensor:
        if self.feature_schwarz_gate is None or tile_exchange is None:
            return state
        exchanged = tile_exchange(state)
        return state + torch.tanh(self.feature_schwarz_gate) * (exchanged - state)

    def correct_endpoint(
        self,
        endpoint: Tensor,
        physical: Tensor,
        scalar: Tensor,
    ) -> Tensor:
        """Apply the learned neural EoS at a projected clean Flow endpoint."""
        return self.correct_endpoint_fields(endpoint, physical, scalar)[0]

    @staticmethod
    def revin_previous_field(physical: Tensor) -> tuple[Tensor, Tensor]:
        valid = physical[:, -1:]
        previous = physical[:, :3]
        count = valid.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0)
        mean = (previous * valid).sum(dim=(-1, -2), keepdim=True) / count
        variance = ((previous - mean).square() * valid).sum(
            dim=(-1, -2), keepdim=True
        ) / count
        standard_deviation = variance.clamp_min(1.0e-6).sqrt()
        normalized = (previous - mean) / standard_deviation * valid
        transformed = torch.cat((normalized, physical[:, 3:]), dim=1)
        statistics = torch.cat((mean.flatten(1), standard_deviation.flatten(1)), dim=1)
        return transformed, statistics

    def forward(
        self,
        flow_state: Tensor,
        physical: Tensor,
        scalar: Tensor,
        flow_time: Tensor,
    ) -> Tensor:
        if physical.shape[1] != self.physical_channels:
            raise ValueError(
                f"expected {self.physical_channels} physical channels, got {physical.shape[1]}"
            )
        physical_normalized, statistics = self.revin_previous_field(physical)
        condition = self.time_mlp(timestep_embedding(flow_time, self.width))
        condition = condition + self.condition_mlp(torch.cat((scalar, statistics), dim=1))
        full_resolution_input = torch.cat((flow_state, physical_normalized), dim=1)
        latent = self.embed(full_resolution_input)
        latent = latent + condition[:, :, None, None]
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                latent = checkpoint(block, latent, use_reentrant=False)
            else:
                latent = block(latent)
        output = self.decode(latent)
        scale, shift = self.output_affine(condition).chunk(2, dim=1)
        output = output * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        if self.transport_stem is not None:
            assert self.transport_condition is not None
            assert self.transport_output is not None
            transport_scale, transport_shift = self.transport_condition(condition).chunk(2, dim=1)
            transport = self.transport_stem(full_resolution_input)
            transport = transport * (1.0 + transport_scale[:, :, None, None]) \
                + transport_shift[:, :, None, None]
            output = output + self.transport_output(F.gelu(transport))
        return output * physical[:, -1:]

    @torch.no_grad()
    def sample(
        self,
        physical: Tensor,
        scalar: Tensor,
        steps: int,
        seed: int,
    ) -> Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        generator = torch.Generator(device=physical.device).manual_seed(seed)
        state = torch.randn(
            (physical.shape[0], 3, physical.shape[-2], physical.shape[-1]),
            device=physical.device,
            dtype=physical.dtype,
            generator=generator,
        ) * physical[:, -1:]
        step_size = 1.0 / steps
        for step in range(steps):
            flow_time = torch.full(
                (physical.shape[0],),
                (step + 0.5) / steps,
                device=physical.device,
                dtype=physical.dtype,
            )
            state = (state + step_size * self(state, physical, scalar, flow_time)) \
                * physical[:, -1:]
        return state
