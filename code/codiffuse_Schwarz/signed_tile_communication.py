"""Signed one-pass residual integral kernels.

Kept separate from :mod:`tile_communication` so validation jobs can hash and
freeze the established softmax implementation independently of this research
ablation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .tile_communication import _grid_coordinates


Tensor = torch.Tensor


class SignedResidualKernelAttention(nn.Module):
    """Apply a signed, L1-normalized context kernel to a residual signal.

    With ``s`` determined only by context and relative layout, weights are
    ``tanh(s_ijh) / sum_j |tanh(s_ijh)|``. Value/output paths are bias-free, so
    the module is exactly linear in the residual, zero-preserving, permits
    directional cancellation, and remains one global application.
    """

    def __init__(
        self,
        context_width: int,
        residual_width: int,
        width: int,
        heads: int,
        dimension: int,
    ) -> None:
        super().__init__()
        if (
            min(context_width, residual_width, width, heads) <= 0
            or width % heads
            or dimension not in (2, 3)
        ):
            raise ValueError("invalid signed residual-kernel dimensions")
        self.context_width = context_width
        self.residual_width = residual_width
        self.width = width
        self.heads = heads
        self.dimension = dimension
        self.head_width = width // heads
        self.context_norm = nn.LayerNorm(context_width)
        self.qk = nn.Linear(context_width, 2 * width, bias=False)
        self.value = nn.Linear(residual_width, width, bias=False)
        edge_width = 2 * dimension + 1
        self.relative_score = nn.Sequential(
            nn.Linear(edge_width, max(16, 2 * heads)),
            nn.SiLU(),
            nn.Linear(max(16, 2 * heads), heads, bias=False),
        )
        self.output = nn.Linear(width, width, bias=False)

    def _relative_score(self, shape: tuple[int, ...], reference: Tensor) -> Tensor:
        coordinates = _grid_coordinates(shape, reference.device)
        scale = reference.new_tensor([max(size - 1, 1) for size in shape])
        delta = (coordinates[:, None] - coordinates[None, :]) / scale
        edge = torch.cat((
            delta, delta.abs(), delta.square().sum(-1, keepdim=True).sqrt(),
        ), dim=-1).to(dtype=reference.dtype)
        return self.relative_score(edge).permute(2, 0, 1)

    def kernel_weights(
        self,
        context: Tensor,
        grid_shape: tuple[int, ...],
        max_graph_radius: int | None = None,
    ) -> Tensor:
        """Return signed ``B,H,N,N`` weights for diagnostics/application."""
        if context.ndim != 3 or context.shape[-1] != self.context_width:
            raise ValueError("context must have shape B,N,context_width")
        if (
            len(grid_shape) != self.dimension
            or math.prod(grid_shape) != context.shape[1]
        ):
            raise ValueError("grid shape is incompatible with context tokens")
        if max_graph_radius is not None and max_graph_radius < 0:
            raise ValueError("max_graph_radius must be nonnegative or None")
        batch, count = context.shape[:2]
        qk = self.qk(self.context_norm(context)).reshape(
            batch, count, 2, self.heads, self.head_width,
        ).permute(2, 0, 3, 1, 4)
        query, key = qk.unbind(0)
        score = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(self.head_width)
        score = score + self._relative_score(grid_shape, context)[None]
        if max_graph_radius is not None:
            coordinates = _grid_coordinates(grid_shape, context.device)
            graph_distance = (
                coordinates[:, None] - coordinates[None, :]
            ).abs().amax(-1)
            allowed = graph_distance[None, None] <= max_graph_radius
        else:
            allowed = torch.ones(
                1, 1, count, count, dtype=torch.bool, device=context.device,
            )
        signed = torch.tanh(score.float()) * allowed
        denominator = signed.abs().sum(-1, keepdim=True).clamp_min(1.0e-6)
        return (signed / denominator).to(dtype=context.dtype)

    def forward(
        self,
        context: Tensor,
        residual: Tensor,
        grid_shape: tuple[int, ...],
        max_graph_radius: int | None = None,
    ) -> Tensor:
        if residual.ndim != 3 or residual.shape[-1] != self.residual_width:
            raise ValueError("residual must have shape B,N,residual_width")
        if context.shape[:2] != residual.shape[:2]:
            raise ValueError("context and residual token layouts must match")
        batch, count = context.shape[:2]
        value = self.value(residual).reshape(
            batch, count, self.heads, self.head_width,
        ).permute(0, 2, 1, 3)
        mixed = torch.matmul(
            self.kernel_weights(context, grid_shape, max_graph_radius), value,
        ).transpose(1, 2).reshape(batch, count, self.width)
        return self.output(mixed)


class SignedResidualKernelCodeMap(nn.Module):
    """One signed residual-kernel message plus bias-free local/code maps."""

    def __init__(
        self,
        context_width: int,
        residual_width: int,
        token_width: int,
        code_width: int,
        heads: int,
        dimension: int,
        *,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if min(token_width, code_width) <= 0:
            raise ValueError("token/code widths must be positive")
        self.residual_width = residual_width
        self.local_value = nn.Linear(residual_width, token_width, bias=False)
        self.global_kernel = SignedResidualKernelAttention(
            context_width, residual_width, token_width, heads, dimension,
        )
        self.code_head = nn.Linear(2 * token_width, code_width, bias=False)
        if zero_init:
            nn.init.zeros_(self.code_head.weight)

    def forward(
        self,
        context: Tensor,
        residual: Tensor,
        grid_shape: tuple[int, ...],
        max_graph_radius: int | None = None,
    ) -> Tensor:
        message = self.global_kernel(
            context, residual, grid_shape,
            max_graph_radius=max_graph_radius,
        )
        local = self.local_value(residual)
        return self.code_head(torch.cat((local, message), dim=-1))
