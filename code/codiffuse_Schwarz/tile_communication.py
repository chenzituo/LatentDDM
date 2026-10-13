"""One-pass relative communication between Cartesian operator tiles.

The communicator deliberately receives only pooled local features, the
Cartesian *layout shape*, and per-case physical controls.  It never receives
absolute world coordinates.  Pairwise attention biases depend on relative
integer tile offsets, while a small grid convolution supplies directed
nearest-neighbour transport.  A zero-initialized FiLM decoder makes the module
an exact identity when attached to a pretrained local endpoint model.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
PositionMode = Literal["cover_normalized", "tile_units"]


def _group_count(width: int) -> int:
    groups = min(16, width)
    while width % groups:
        groups -= 1
    return groups


def _grid_coordinates(shape: Sequence[int], device: torch.device) -> Tensor:
    axes = [torch.arange(size, device=device, dtype=torch.float32) for size in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(shape))


def _relative_edge_features(
    shape: Sequence[int], reference: Tensor, position_mode: PositionMode,
) -> Tensor:
    """Return signed/absolute relative offsets and distance.

    ``cover_normalized`` preserves the released checkpoints.  ``tile_units``
    is the domain-extension coordinate: an adjacent pair has the same feature
    regardless of how many same-size tiles are added elsewhere in the cover.
    """
    if position_mode not in ("cover_normalized", "tile_units"):
        raise ValueError("unknown relative-position coordinate mode")
    coordinates = _grid_coordinates(shape, reference.device)
    delta = coordinates[:, None] - coordinates[None, :]
    if position_mode == "cover_normalized":
        scale = reference.new_tensor([max(size - 1, 1) for size in shape])
        delta = delta / scale
    edge = torch.cat(
        (delta, delta.abs(), delta.square().sum(-1, keepdim=True).sqrt()),
        dim=-1,
    )
    return edge.to(dtype=reference.dtype)


class RelativeTileAttention(nn.Module):
    """Full tile attention with learned relative-offset bias.

    Full attention is intentional: pressure and other coarse/global modes must
    cross an arbitrarily long cover in one requested endpoint.  The expensive
    spatial fields remain local; only one pooled token per tile participates.
    """

    def __init__(
        self, width: int, heads: int, dimension: int, *,
        position_mode: PositionMode = "cover_normalized",
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or dimension not in (2, 3):
            raise ValueError("width/heads must be compatible and dimension must be 2 or 3")
        self.width = width
        self.heads = heads
        self.dimension = dimension
        self.head_width = width // heads
        if position_mode not in ("cover_normalized", "tile_units"):
            raise ValueError("unknown relative-position coordinate mode")
        self.position_mode = position_mode
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        # Signed offset, absolute offset, and Euclidean separation.
        edge_width = 2 * dimension + 1
        self.relative_bias = nn.Sequential(
            nn.Linear(edge_width, max(16, 2 * heads)),
            nn.SiLU(),
            nn.Linear(max(16, 2 * heads), heads, bias=False),
        )
        self.output = nn.Linear(width, width, bias=False)

    def _bias(self, shape: tuple[int, ...], reference: Tensor) -> Tensor:
        edge = _relative_edge_features(shape, reference, self.position_mode)
        return self.relative_bias(edge).permute(2, 0, 1)

    def forward(
        self,
        tokens: Tensor,
        grid_shape: tuple[int, ...],
        max_graph_radius: int | None = None,
    ) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError("tokens must have shape B,N,width")
        if len(grid_shape) != self.dimension or math.prod(grid_shape) != tokens.shape[1]:
            raise ValueError("grid shape is incompatible with tile tokens")
        if max_graph_radius is not None and max_graph_radius < 0:
            raise ValueError("max_graph_radius must be nonnegative or None")
        batch, count = tokens.shape[:2]
        qkv = self.qkv(self.norm(tokens)).reshape(
            batch, count, 3, self.heads, self.head_width,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores + self._bias(grid_shape, tokens)[None]
        if max_graph_radius is not None:
            coordinates = _grid_coordinates(grid_shape, tokens.device)
            graph_distance = (
                coordinates[:, None] - coordinates[None, :]
            ).abs().amax(-1)
            scores = scores.masked_fill(
                graph_distance[None, None] > max_graph_radius,
                torch.finfo(scores.dtype).min,
            )
        attention = F.softmax(scores.float(), dim=-1).to(dtype=tokens.dtype)
        mixed = torch.matmul(attention, value).transpose(1, 2).reshape(batch, count, self.width)
        return self.output(mixed)


class ResidualKernelAttention(nn.Module):
    """Apply one context-conditioned global kernel to a residual signal.

    Query/key weights and relative-layout bias depend only on ``context``;
    values and output are bias-free linear maps of ``residual``.  For fixed
    context this module is therefore exactly homogeneous and additive in the
    supplied compatibility right-hand side.  In particular, a zero residual
    produces a bitwise-zero message.  This is the attention analogue of one
    application of a geometry-conditioned reduced Green/Schur inverse.
    """

    def __init__(
        self,
        context_width: int,
        residual_width: int,
        width: int,
        heads: int,
        dimension: int,
        *,
        position_mode: PositionMode = "cover_normalized",
    ) -> None:
        super().__init__()
        if (
            min(context_width, residual_width, width, heads) <= 0
            or width % heads
            or dimension not in (2, 3)
        ):
            raise ValueError("invalid residual-kernel attention dimensions")
        self.context_width = context_width
        self.residual_width = residual_width
        self.width = width
        self.heads = heads
        self.dimension = dimension
        self.head_width = width // heads
        if position_mode not in ("cover_normalized", "tile_units"):
            raise ValueError("unknown relative-position coordinate mode")
        self.position_mode = position_mode
        self.context_norm = nn.LayerNorm(context_width)
        self.qk = nn.Linear(context_width, 2 * width, bias=False)
        self.value = nn.Linear(residual_width, width, bias=False)
        edge_width = 2 * dimension + 1
        self.relative_bias = nn.Sequential(
            nn.Linear(edge_width, max(16, 2 * heads)),
            nn.SiLU(),
            nn.Linear(max(16, 2 * heads), heads, bias=False),
        )
        self.output = nn.Linear(width, width, bias=False)

    def _bias(self, shape: tuple[int, ...], reference: Tensor) -> Tensor:
        edge = _relative_edge_features(shape, reference, self.position_mode)
        return self.relative_bias(edge).permute(2, 0, 1)

    def forward(
        self,
        context: Tensor,
        residual: Tensor,
        grid_shape: tuple[int, ...],
        max_graph_radius: int | None = None,
    ) -> Tensor:
        if context.ndim != 3 or context.shape[-1] != self.context_width:
            raise ValueError("context must have shape B,N,context_width")
        if residual.ndim != 3 or residual.shape[-1] != self.residual_width:
            raise ValueError("residual must have shape B,N,residual_width")
        if context.shape[:2] != residual.shape[:2]:
            raise ValueError("context and residual token layouts must match")
        if (
            len(grid_shape) != self.dimension
            or math.prod(grid_shape) != context.shape[1]
        ):
            raise ValueError("grid shape is incompatible with residual tokens")
        if max_graph_radius is not None and max_graph_radius < 0:
            raise ValueError("max_graph_radius must be nonnegative or None")

        weight = self._weights(context, grid_shape, max_graph_radius)
        return self._apply_weights(weight, residual)

    def _weights(
        self,
        context: Tensor,
        grid_shape: tuple[int, ...],
        max_graph_radius: int | None,
    ) -> Tensor:
        """Return the geometry-only kernel used by every residual channel."""
        batch, count = context.shape[:2]
        qk = self.qk(self.context_norm(context)).reshape(
            batch, count, 2, self.heads, self.head_width,
        ).permute(2, 0, 3, 1, 4)
        query, key = qk.unbind(0)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(self.head_width)
        scores = scores + self._bias(grid_shape, context)[None]
        if max_graph_radius is not None:
            coordinates = _grid_coordinates(grid_shape, context.device)
            graph_distance = (
                coordinates[:, None] - coordinates[None, :]
            ).abs().amax(-1)
            scores = scores.masked_fill(
                graph_distance[None, None] > max_graph_radius,
                torch.finfo(scores.dtype).min,
            )
        return F.softmax(scores.float(), dim=-1).to(dtype=context.dtype)

    def _apply_weights(self, weight: Tensor, residual: Tensor) -> Tensor:
        """Apply one already-built global kernel to residual value channels."""
        batch, count = residual.shape[:2]
        value = self.value(residual).reshape(
            batch, count, self.heads, self.head_width,
        ).permute(0, 2, 1, 3)
        mixed = torch.matmul(weight, value).transpose(1, 2).reshape(
            batch, count, self.width,
        )
        return self.output(mixed)

    def forward_decomposed(
        self,
        context: Tensor,
        residual: Tensor,
        grid_shape: tuple[int, ...],
        split_index: int,
        max_graph_radius: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return total and prefix-channel messages from one global kernel.

        Query/key scores and the global communication weights are constructed
        exactly once.  The second value stream only exposes the contribution
        of ``residual[..., :split_index]`` under that same kernel; it is not a
        second communication round.  This is useful for auditable warm-start
        residual corrections whose zero suffix exactly recovers a frozen
        prefix operator.
        """
        if context.ndim != 3 or context.shape[-1] != self.context_width:
            raise ValueError("context must have shape B,N,context_width")
        if residual.ndim != 3 or residual.shape[-1] != self.residual_width:
            raise ValueError("residual must have shape B,N,residual_width")
        if context.shape[:2] != residual.shape[:2]:
            raise ValueError("context and residual token layouts must match")
        if not 0 < split_index <= self.residual_width:
            raise ValueError("split index must select a nonempty residual prefix")
        if (
            len(grid_shape) != self.dimension
            or math.prod(grid_shape) != context.shape[1]
        ):
            raise ValueError("grid shape is incompatible with residual tokens")
        if max_graph_radius is not None and max_graph_radius < 0:
            raise ValueError("max_graph_radius must be nonnegative or None")
        prefix = torch.zeros_like(residual)
        prefix[..., :split_index] = residual[..., :split_index]
        weight = self._weights(context, grid_shape, max_graph_radius)
        return (
            self._apply_weights(weight, residual),
            self._apply_weights(weight, prefix),
        )


class StateGatedResidualKernelAttention(ResidualKernelAttention):
    r"""Let executable source state condition a residual-valued message.

    :class:`ResidualKernelAttention` already lets context choose where a
    residual is sent through its query/key weights.  This extension also lets
    the *source* context choose how the residual coordinates are embedded:

    ``V_j = W_0 q_j + sum_r a_r(c_j) W_r q_j``.

    The coefficient map is initialized to zero, so the module is an exact
    identity lift of an existing residual kernel.  The bases are initialized
    nonzero, which gives the coefficient map a live first gradient after a
    fitted base kernel is loaded.  For every fixed context the full operation
    remains bias-free linear in ``residual``; a zero residual therefore still
    produces a bitwise-zero message.  Context changes routing and the local
    residual chart, but is never transmitted as a value on its own.
    """

    def __init__(
        self,
        context_width: int,
        residual_width: int,
        width: int,
        heads: int,
        dimension: int,
        *,
        value_rank: int = 4,
        position_mode: PositionMode = "cover_normalized",
    ) -> None:
        super().__init__(
            context_width, residual_width, width, heads, dimension,
            position_mode=position_mode,
        )
        if value_rank <= 0:
            raise ValueError("state-gated value rank must be positive")
        self.value_rank = value_rank
        self.state_value_coefficients = nn.Linear(
            context_width, value_rank, bias=False,
        )
        self.state_value_basis = nn.Parameter(torch.empty(
            value_rank, width, residual_width,
        ))
        nn.init.normal_(
            self.state_value_basis,
            std=1.0 / math.sqrt(float(value_rank * residual_width)),
        )
        # Exact identity lift of ResidualKernelAttention.  Keeping the basis
        # live avoids the zero-times-zero gradient deadlock.
        nn.init.zeros_(self.state_value_coefficients.weight)

    def _state_projected_value(self, context: Tensor, residual: Tensor) -> Tensor:
        coefficients = self.state_value_coefficients(
            self.context_norm(context),
        )
        dynamic = torch.einsum(
            "bnr,rwd,bnd->bnw",
            coefficients, self.state_value_basis, residual,
        )
        return self.value(residual) + dynamic

    def _apply_state_weights(
        self, weight: Tensor, context: Tensor, residual: Tensor,
    ) -> Tensor:
        # In evaluation mode an exactly closed state gate must reproduce the
        # parent kernel through the identical floating-point program, not only
        # the same algebra.  Computing a zero dynamic GEMM and adding it can
        # change CUDA kernel selection/rounding by a few ulps on large covers.
        # Training deliberately takes the differentiable path below so the
        # zero-initialized coefficient map retains its live first gradient.
        if not self.training and not bool(torch.count_nonzero(
            self.state_value_coefficients.weight,
        )):
            return self._apply_weights(weight, residual)
        batch, count = residual.shape[:2]
        value = self._state_projected_value(context, residual).reshape(
            batch, count, self.heads, self.head_width,
        ).permute(0, 2, 1, 3)
        mixed = torch.matmul(weight, value).transpose(1, 2).reshape(
            batch, count, self.width,
        )
        return self.output(mixed)

    def forward(
        self,
        context: Tensor,
        residual: Tensor,
        grid_shape: tuple[int, ...],
        max_graph_radius: int | None = None,
    ) -> Tensor:
        # Reuse the base class's complete shape/radius validation before
        # applying the state-gated value stream.
        if context.ndim != 3 or context.shape[-1] != self.context_width:
            raise ValueError("context must have shape B,N,context_width")
        if residual.ndim != 3 or residual.shape[-1] != self.residual_width:
            raise ValueError("residual must have shape B,N,residual_width")
        if context.shape[:2] != residual.shape[:2]:
            raise ValueError("context and residual token layouts must match")
        if (
            len(grid_shape) != self.dimension
            or math.prod(grid_shape) != context.shape[1]
        ):
            raise ValueError("grid shape is incompatible with residual tokens")
        if max_graph_radius is not None and max_graph_radius < 0:
            raise ValueError("max_graph_radius must be nonnegative or None")
        weight = self._weights(context, grid_shape, max_graph_radius)
        return self._apply_state_weights(weight, context, residual)

    def forward_decomposed(
        self,
        context: Tensor,
        residual: Tensor,
        grid_shape: tuple[int, ...],
        split_index: int,
        max_graph_radius: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not 0 < split_index <= self.residual_width:
            raise ValueError("split index must select a nonempty residual prefix")
        # Calling forward validates every shared input.  We avoid using its
        # output so query/key weights are still constructed only once below.
        if context.ndim != 3 or context.shape[-1] != self.context_width:
            raise ValueError("context must have shape B,N,context_width")
        if residual.ndim != 3 or residual.shape[-1] != self.residual_width:
            raise ValueError("residual must have shape B,N,residual_width")
        if context.shape[:2] != residual.shape[:2]:
            raise ValueError("context and residual token layouts must match")
        if (
            len(grid_shape) != self.dimension
            or math.prod(grid_shape) != context.shape[1]
        ):
            raise ValueError("grid shape is incompatible with residual tokens")
        if max_graph_radius is not None and max_graph_radius < 0:
            raise ValueError("max_graph_radius must be nonnegative or None")
        prefix = torch.zeros_like(residual)
        prefix[..., :split_index] = residual[..., :split_index]
        weight = self._weights(context, grid_shape, max_graph_radius)
        return (
            self._apply_state_weights(weight, context, residual),
            self._apply_state_weights(weight, context, prefix),
        )


class ResidualKernelCodeMap(nn.Module):
    """One residual-kernel message plus a bias-free local/global code head."""

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
        position_mode: PositionMode = "cover_normalized",
    ) -> None:
        super().__init__()
        if min(token_width, code_width) <= 0:
            raise ValueError("token/code widths must be positive")
        self.residual_width = residual_width
        self.local_value = nn.Linear(residual_width, token_width, bias=False)
        self.global_kernel = ResidualKernelAttention(
            context_width, residual_width, token_width, heads, dimension,
            position_mode=position_mode,
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

    def forward_decomposed(
        self,
        context: Tensor,
        residual: Tensor,
        grid_shape: tuple[int, ...],
        split_index: int,
        max_graph_radius: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return total and frozen-prefix codes from one shared message."""
        message, prefix_message = self.global_kernel.forward_decomposed(
            context, residual, grid_shape, split_index,
            max_graph_radius=max_graph_radius,
        )
        prefix = torch.zeros_like(residual)
        prefix[..., :split_index] = residual[..., :split_index]
        local = self.local_value(residual)
        prefix_local = self.local_value(prefix)
        return (
            self.code_head(torch.cat((local, message), dim=-1)),
            self.code_head(torch.cat((prefix_local, prefix_message), dim=-1)),
        )


class StateGatedResidualKernelCodeMap(ResidualKernelCodeMap):
    """Residual code map with an identity-lifted state-gated value stream."""

    def __init__(
        self,
        context_width: int,
        residual_width: int,
        token_width: int,
        code_width: int,
        heads: int,
        dimension: int,
        *,
        value_rank: int = 4,
        zero_init: bool = True,
        position_mode: PositionMode = "cover_normalized",
    ) -> None:
        nn.Module.__init__(self)
        if min(token_width, code_width) <= 0:
            raise ValueError("token/code widths must be positive")
        self.residual_width = residual_width
        self.local_value = nn.Linear(residual_width, token_width, bias=False)
        self.global_kernel = StateGatedResidualKernelAttention(
            context_width, residual_width, token_width, heads, dimension,
            value_rank=value_rank, position_mode=position_mode,
        )
        self.code_head = nn.Linear(2 * token_width, code_width, bias=False)
        if zero_init:
            nn.init.zeros_(self.code_head.weight)


class CenteredResidualFieldEncoder(nn.Module):
    """Compress an explicit patchwise field defect into a centered token.

    Every affine map is bias free and every nonlinearity fixes zero.  Hence an
    identically zero physical defect produces a bitwise-zero token, unlike a
    generic hidden patch embedding.  The adaptive spatial bins retain more of
    the within-patch mismatch than face moments alone while keeping the global
    communication bandwidth independent of mesh resolution.
    """

    def __init__(
        self,
        field_channels: int,
        token_width: int,
        dimension: int,
        *,
        hidden_width: int = 16,
        bins_per_axis: int = 3,
    ) -> None:
        super().__init__()
        if min(
            field_channels, token_width, hidden_width, bins_per_axis,
        ) <= 0 or dimension not in (2, 3):
            raise ValueError("invalid centered residual-field encoder dimensions")
        self.field_channels = int(field_channels)
        self.token_width = int(token_width)
        self.dimension = int(dimension)
        self.hidden_width = int(hidden_width)
        self.bins_per_axis = int(bins_per_axis)
        convolution = nn.Conv2d if dimension == 2 else nn.Conv3d
        pooling = nn.AdaptiveAvgPool2d if dimension == 2 else nn.AdaptiveAvgPool3d
        self.features = nn.Sequential(
            convolution(field_channels, hidden_width, 3, padding=1, bias=False),
            nn.SiLU(),
            convolution(hidden_width, hidden_width, 3, padding=1, bias=False),
            nn.SiLU(),
        )
        pooled_shape = (bins_per_axis,) * dimension
        self.pool = pooling(pooled_shape)
        self.projection = nn.Linear(
            hidden_width * bins_per_axis**dimension, token_width, bias=False,
        )

    def forward(self, defect: Tensor) -> Tensor:
        expected_ndim = self.dimension + 2
        if defect.ndim != expected_ndim or defect.shape[1] != self.field_channels:
            raise ValueError("field defect has incompatible shape")
        features = self.pool(self.features(defect)).flatten(1)
        return self.projection(features)


class ResidualFieldCodeLift(nn.Module):
    """Add a rich residual-field value stream through an existing kernel.

    ``attention_weight`` is constructed once by the parent residual kernel.
    This module applies that same geometry-conditioned matrix to a centered
    field-defect token and returns an additive response-code correction.  Its
    final head is initialized to zero, making it an exact architectural lift
    of the parent LatentDDM.  It is still one message: semantic residuals and
    field-sketch residuals are two value streams under one attention matrix.
    """

    def __init__(
        self,
        field_channels: int,
        sketch_width: int,
        token_width: int,
        code_width: int,
        heads: int,
        dimension: int,
        *,
        hidden_width: int = 16,
        bins_per_axis: int = 3,
        auxiliary_residual_width: int = 0,
        global_only_suffix_width: int = 0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if (
            min(sketch_width, token_width, code_width, heads) <= 0
            or token_width % heads
            or auxiliary_residual_width < 0
            or not 0 <= global_only_suffix_width <= code_width
        ):
            raise ValueError("invalid residual-field code-lift dimensions")
        self.token_width = int(token_width)
        self.code_width = int(code_width)
        self.heads = int(heads)
        self.head_width = token_width // heads
        self.dimension = int(dimension)
        self.auxiliary_residual_width = int(auxiliary_residual_width)
        self.global_only_suffix_width = int(global_only_suffix_width)
        self.encoder = CenteredResidualFieldEncoder(
            field_channels, sketch_width, dimension,
            hidden_width=hidden_width, bins_per_axis=bins_per_axis,
        )
        value_width = sketch_width + self.auxiliary_residual_width
        self.local_value = nn.Linear(value_width, token_width, bias=False)
        self.global_value = nn.Linear(value_width, token_width, bias=False)
        self.global_output = nn.Linear(token_width, token_width, bias=False)
        self.code_head = nn.Linear(2 * token_width, code_width, bias=False)
        if zero_init:
            nn.init.zeros_(self.code_head.weight)

    def forward(
        self,
        attention_weight: Tensor,
        field_defect: Tensor,
        grid_shape: tuple[int, ...],
        auxiliary_residual: Tensor | None = None,
    ) -> Tensor:
        if len(grid_shape) != self.dimension:
            raise ValueError("grid shape dimension does not match field sketch")
        tile_count = math.prod(grid_shape)
        if field_defect.shape[0] % tile_count:
            raise ValueError("field-defect batch is not divisible by tile count")
        batch = field_defect.shape[0] // tile_count
        expected_weight = (batch, self.heads, tile_count, tile_count)
        if attention_weight.shape != expected_weight:
            raise ValueError(
                f"attention weight must have shape {expected_weight}"
            )
        sketch = self.encoder(field_defect).reshape(batch, tile_count, -1)
        if self.auxiliary_residual_width:
            expected = (batch, tile_count, self.auxiliary_residual_width)
            if auxiliary_residual is None or auxiliary_residual.shape != expected:
                raise ValueError(
                    f"auxiliary residual must have shape {expected}"
                )
            sketch = torch.cat((sketch, auxiliary_residual.to(sketch)), dim=-1)
        elif auxiliary_residual is not None:
            raise ValueError("this residual-field lift has no auxiliary channel")
        local = self.local_value(sketch)
        value = self.global_value(sketch).reshape(
            batch, tile_count, self.heads, self.head_width,
        ).permute(0, 2, 1, 3)
        mixed = torch.matmul(attention_weight, value).transpose(1, 2).reshape(
            batch, tile_count, self.token_width,
        )
        global_message = self.global_output(mixed)
        combined = torch.cat((local, global_message), dim=-1)
        if not self.global_only_suffix_width:
            return self.code_head(combined)
        split = self.code_width - self.global_only_suffix_width
        prefix = F.linear(combined, self.code_head.weight[:split])
        # The appended response modes are selected only by communicated
        # residual values.  This makes their causal role structural: local
        # residual capacity cannot impersonate the new global correction.
        suffix = F.linear(
            global_message, self.code_head.weight[split:, self.token_width:],
        )
        return torch.cat((prefix, suffix), dim=-1)


class TileGridCommunicator(nn.Module):
    """Modulate dense local features after one global tile communication.

    The output has the same shape as ``hidden``.  At initialization it is
    exactly equal to ``hidden``.  ``scalar`` is used only as a per-case physical
    control (for example freestream or motion), never as spatial position.
    """

    def __init__(
        self,
        width: int,
        scalar_dim: int,
        dimension: int,
        heads: int = 4,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if width <= 0 or scalar_dim < 0 or dimension not in (2, 3):
            raise ValueError("invalid communicator dimensions")
        if width % heads:
            raise ValueError("communicator width must be divisible by heads")
        self.width = width
        self.scalar_dim = scalar_dim
        self.dimension = dimension
        convolution = nn.Conv2d if dimension == 2 else nn.Conv3d
        self.local_norm = nn.GroupNorm(_group_count(width), width)
        self.local_mixer = convolution(width, width, 3, padding=1)
        self.attention = RelativeTileAttention(width, heads, dimension)
        self.scalar = (
            nn.Sequential(nn.Linear(scalar_dim, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
            if scalar_dim else None
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, 2 * width),
        )
        if zero_init:
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self,
        hidden: Tensor,
        scalar: Tensor,
        grid_shape: tuple[int, ...] | None,
        max_graph_radius: int | None = None,
    ) -> Tensor:
        expected_ndim = self.dimension + 2
        if hidden.ndim != expected_ndim or hidden.shape[1] != self.width:
            raise ValueError("hidden field has incompatible shape")
        if grid_shape is None:
            return hidden
        if len(grid_shape) != self.dimension:
            raise ValueError("grid shape dimension does not match communicator")
        tile_count = math.prod(grid_shape)
        if hidden.shape[0] % tile_count:
            raise ValueError("local batch is not divisible by the tile grid")
        batch = hidden.shape[0] // tile_count
        if scalar.ndim != 2 or scalar.shape != (hidden.shape[0], self.scalar_dim):
            raise ValueError("scalar controls have incompatible shape")

        tokens = hidden.flatten(2).mean(-1).reshape(batch, tile_count, self.width)
        grid = tokens.transpose(1, 2).reshape(batch, self.width, *grid_shape)
        local = self.local_mixer(F.silu(self.local_norm(grid)))
        local_tokens = local.reshape(batch, self.width, tile_count).transpose(1, 2)
        global_tokens = self.attention(
            tokens, grid_shape, max_graph_radius=max_graph_radius,
        )
        case_context = tokens.mean(1)
        if self.scalar is not None:
            case_scalar = scalar.reshape(batch, tile_count, self.scalar_dim).mean(1)
            case_context = case_context + self.scalar(case_scalar)
        context = case_context[:, None].expand(-1, tile_count, -1)
        scale, shift = self.fusion(torch.cat((local_tokens, global_tokens, context), dim=-1)).chunk(2, -1)
        broadcast = (batch * tile_count, self.width) + (1,) * self.dimension
        return hidden * (1.0 + scale.reshape(broadcast)) + shift.reshape(broadcast)
