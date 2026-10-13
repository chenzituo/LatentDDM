"""Geometry-aware latent composition of already executable local fields.

The local neural operator is deliberately outside this module.  A case adapter
first evaluates that frozen operator and encodes each completed local field as
one token ``z_i^0 = C(x_i, v_i)``.  This module then performs a configurable
number of lightweight, globally coupled Transformer layers.

The returned correction coordinates are *centered* against an otherwise
identical diagonal (self-only) execution of the stack.  Consequently disabling
cross-subdomain edges gives an exact zero correction for every parameter value;
the corresponding model is exactly the frozen local/core baseline.  This is a
stronger and cleaner identity than relying on a zero initialization.

Only relative node displacements enter attention.  Absolute world coordinates
are never added to a token, which is important for translating the same local
operator and composer to a new geometry cover.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def regular_grid_positions(
    grid_shape: Sequence[int], reference: Tensor,
) -> Tensor:
    """Return tile-center coordinates in tile units for a Cartesian cover."""
    if len(grid_shape) not in (2, 3) or any(int(size) <= 0 for size in grid_shape):
        raise ValueError("grid_shape must contain two or three positive sizes")
    axes = [
        torch.arange(int(size), device=reference.device, dtype=reference.dtype)
        for size in grid_shape
    ]
    return torch.stack(
        torch.meshgrid(*axes, indexing="ij"), dim=-1,
    ).reshape(-1, len(grid_shape))


def _batched_positions(positions: Tensor, batch: int, count: int) -> Tensor:
    if positions.ndim == 2:
        if positions.shape[0] != count:
            raise ValueError("positions and tokens contain different node counts")
        return positions[None].expand(batch, -1, -1)
    if positions.ndim == 3 and positions.shape[:2] == (batch, count):
        return positions
    raise ValueError("positions must have shape N,D or B,N,D")


def _edge_features(positions: Tensor) -> Tensor:
    # Receiver minus source.  Translation cancels exactly before the learned
    # relative-geometry map is evaluated.
    delta = positions[:, :, None] - positions[:, None, :]
    return torch.cat(
        (delta, delta.abs(), delta.square().sum(-1, keepdim=True).sqrt()),
        dim=-1,
    )


def _validated_graph_mask(
    graph_mask: Tensor | None,
    *,
    batch: int,
    count: int,
    device: torch.device,
) -> Tensor:
    if graph_mask is None:
        return torch.ones(batch, count, count, dtype=torch.bool, device=device)
    mask = graph_mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 2 and mask.shape == (count, count):
        mask = mask[None].expand(batch, -1, -1)
    elif mask.ndim != 3 or mask.shape != (batch, count, count):
        raise ValueError("graph_mask must have shape N,N or B,N,N")
    diagonal = torch.diagonal(mask, dim1=-2, dim2=-1)
    if not bool(diagonal.all()):
        raise ValueError("every geometry-graph node must retain its self edge")
    return mask


class RelativeGeometryAttention(nn.Module):
    """Multi-head self-attention whose only positional signal is relative."""

    def __init__(self, width: int, heads: int, dimension: int) -> None:
        super().__init__()
        if (
            width <= 0 or heads <= 0 or width % heads
            or dimension not in (2, 3)
        ):
            raise ValueError("invalid relative-attention dimensions")
        self.width = width
        self.heads = heads
        self.dimension = dimension
        self.head_width = width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        edge_width = 2 * dimension + 1
        bias_width = max(16, 2 * heads)
        self.relative_bias = nn.Sequential(
            nn.Linear(edge_width, bias_width),
            nn.SiLU(),
            nn.Linear(bias_width, heads, bias=False),
        )
        self.output = nn.Linear(width, width, bias=False)

    def forward(
        self,
        tokens: Tensor,
        positions: Tensor,
        graph_mask: Tensor | None = None,
    ) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError("tokens must have shape B,N,width")
        batch, count = tokens.shape[:2]
        positions = _batched_positions(
            positions.to(device=tokens.device, dtype=tokens.dtype), batch, count,
        )
        if positions.shape[-1] != self.dimension:
            raise ValueError("position dimension differs from attention dimension")
        mask = _validated_graph_mask(
            graph_mask, batch=batch, count=count, device=tokens.device,
        )

        qkv = self.qkv(self.norm(tokens)).reshape(
            batch, count, 3, self.heads, self.head_width,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_width)
        relative_bias = self.relative_bias(_edge_features(positions)).permute(
            0, 3, 1, 2,
        )
        scores = scores + relative_bias
        scores = scores.masked_fill(
            ~mask[:, None], torch.finfo(scores.dtype).min,
        )
        weights = F.softmax(scores.float(), dim=-1).to(dtype=tokens.dtype)
        mixed = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch, count, self.width,
        )
        return self.output(mixed)

    def forward_diagonal(self, tokens: Tensor) -> Tensor:
        """Evaluate the exact self-only attention map without an ``N x N`` map.

        With only the diagonal edge admitted, softmax has one entry and is
        identically one.  Queries, keys, and relative biases therefore cancel
        from the function; only the value projection and output map remain.
        """
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError("tokens must have shape B,N,width")
        normalized = self.norm(tokens)
        value_weight = self.qkv.weight[2 * self.width:]
        value = F.linear(normalized, value_weight)
        return self.output(value)


class FieldTokenTransformerLayer(nn.Module):
    """One pre-norm latent interaction layer on subdomain field tokens."""

    def __init__(
        self,
        width: int,
        heads: int,
        dimension: int,
        *,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if expansion <= 0:
            raise ValueError("feed-forward expansion must be positive")
        self.attention = RelativeGeometryAttention(width, heads, dimension)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Linear(expansion * width, width),
        )

    def forward(
        self,
        tokens: Tensor,
        positions: Tensor,
        graph_mask: Tensor,
    ) -> Tensor:
        tokens = tokens + self.attention(tokens, positions, graph_mask)
        return tokens + self.ffn(self.ffn_norm(tokens))

    def forward_diagonal(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention.forward_diagonal(tokens)
        return tokens + self.ffn(self.ffn_norm(tokens))


@dataclass(frozen=True)
class FieldTokenComposition:
    """Auditable outputs of the centered latent communication stack."""

    initial_tokens: Tensor
    global_tokens: Tensor
    diagonal_tokens: Tensor
    message: Tensor
    global_code: Tensor
    diagonal_code: Tensor
    correction_code: Tensor


class CenteredFieldTokenTransformer(nn.Module):
    r"""Stack latent interaction layers and isolate their non-local effect.

    Given locally encoded field tokens ``z^0``, the same stack is evaluated on
    the geometry graph ``G`` and its diagonal graph ``I``:

    ``z_G^L = T^L(z^0; G)``, ``z_I^L = T^L(z^0; I)``.

    A shared nonlinear head is applied to both executions and subtracted.  The
    resulting correction coordinate is therefore exactly zero when ``G=I``.
    A case-specific response chart can decode that coordinate into a spatial
    field correction ``delta_i``.
    """

    def __init__(
        self,
        width: int,
        code_width: int,
        heads: int,
        dimension: int,
        *,
        depth: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if min(width, code_width, heads, depth) <= 0:
            raise ValueError("latent stack dimensions and depth must be positive")
        self.width = width
        self.code_width = code_width
        self.heads = heads
        self.dimension = dimension
        self.depth = depth
        self.layers = nn.ModuleList([
            FieldTokenTransformerLayer(
                width, heads, dimension, expansion=expansion,
            )
            for _ in range(depth)
        ])
        self.code_head = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, code_width),
        )
        # Centering supplies the exact identity; a small live final map avoids
        # introducing a dead first gradient into the interaction stack.
        final = self.code_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, std=1.0e-3)
        nn.init.zeros_(final.bias)

    @property
    def communication_layers(self) -> int:
        return len(self.layers)

    def forward(
        self,
        initial_tokens: Tensor,
        *,
        grid_shape: Sequence[int] | None = None,
        positions: Tensor | None = None,
        graph_mask: Tensor | None = None,
        centered: bool = True,
    ) -> FieldTokenComposition:
        if initial_tokens.ndim != 3 or initial_tokens.shape[-1] != self.width:
            raise ValueError("initial_tokens must have shape B,N,width")
        batch, count = initial_tokens.shape[:2]
        if positions is None:
            if grid_shape is None or math.prod(grid_shape) != count:
                raise ValueError("grid_shape must identify every latent token")
            positions = regular_grid_positions(grid_shape, initial_tokens)
        elif grid_shape is not None and math.prod(grid_shape) != count:
            raise ValueError("grid_shape and token count disagree")
        positions = _batched_positions(
            positions.to(initial_tokens), batch, count,
        )
        if positions.shape[-1] != self.dimension:
            raise ValueError("geometry graph has the wrong spatial dimension")
        global_mask = _validated_graph_mask(
            graph_mask, batch=batch, count=count, device=initial_tokens.device,
        )
        diagonal_mask = torch.eye(
            count, dtype=torch.bool, device=initial_tokens.device,
        )[None].expand(batch, -1, -1)

        if not centered:
            # Ordinary whole-graph self-attention.  This path supports a clean
            # ablation of the G-I construction and deliberately avoids paying
            # for a diagonal companion execution that does not affect output.
            global_tokens = initial_tokens
            for layer in self.layers:
                global_tokens = layer(global_tokens, positions, global_mask)
            global_code = self.code_head(torch.cat(
                (initial_tokens, global_tokens), dim=-1,
            ))
            return FieldTokenComposition(
                initial_tokens=initial_tokens,
                global_tokens=global_tokens,
                diagonal_tokens=initial_tokens,
                message=global_tokens - initial_tokens,
                global_code=global_code,
                diagonal_code=torch.zeros_like(global_code),
                correction_code=global_code,
            )

        diagonal_tokens = initial_tokens
        if torch.equal(global_mask, diagonal_mask):
            for layer in self.layers:
                diagonal_tokens = layer.forward_diagonal(diagonal_tokens)
            # Preserve the centering invariant bitwise, including floating
            # evaluation order, when the geometry graph has no cross edges.
            global_tokens = diagonal_tokens
        else:
            global_tokens = initial_tokens
            for layer in self.layers:
                global_tokens = layer(global_tokens, positions, global_mask)
                diagonal_tokens = layer.forward_diagonal(diagonal_tokens)

        global_code = self.code_head(torch.cat(
            (initial_tokens, global_tokens), dim=-1,
        ))
        diagonal_code = self.code_head(torch.cat(
            (initial_tokens, diagonal_tokens), dim=-1,
        ))
        return FieldTokenComposition(
            initial_tokens=initial_tokens,
            global_tokens=global_tokens,
            diagonal_tokens=diagonal_tokens,
            message=global_tokens - diagonal_tokens,
            global_code=global_code,
            diagonal_code=diagonal_code,
            correction_code=global_code - diagonal_code,
        )


class CenteredPortFieldTokenExtension(nn.Module):
    r"""Communicate several local field summaries without dense ``(NP)^2`` attention.

    ``initial_tokens`` has shape ``B,N,P,W``: ``N`` executable local
    subdomains and ``P`` relative-coordinate field ports per subdomain.  The
    same geometry-aware stack communicates each port across the ``N``
    subdomains, so its attention work is ``O(PN^2)`` rather than
    ``O(P^2N^2)``.  A shared nonlinear head then fuses the port responses for
    each subdomain.

    The fusion is evaluated on both the full-graph and diagonal-graph states
    and subtracted.  Consequently the extension is exactly zero whenever
    cross-subdomain edges are absent, even after training.  Its final fusion
    map is initialized to zero so it can be added to an existing one-token
    composer as an exact functional child while retaining a live gradient in
    that final map.
    """

    def __init__(
        self,
        width: int,
        code_width: int,
        heads: int,
        dimension: int,
        *,
        ports: int,
        depth: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if ports <= 1:
            raise ValueError("a port extension requires at least two ports")
        self.width = width
        self.code_width = code_width
        self.ports = ports
        self.communication = CenteredFieldTokenTransformer(
            width, code_width, heads, dimension,
            depth=depth, expansion=expansion,
        )
        fused_width = ports * code_width
        hidden = max(2 * code_width, fused_width // 2)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_width),
            nn.Linear(fused_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, code_width, bias=False),
        )
        final = self.fusion[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)

    @property
    def communication_layers(self) -> int:
        return self.communication.communication_layers

    def _port_batch_positions(
        self, positions: Tensor | None, batch: int,
    ) -> Tensor | None:
        if positions is None or positions.ndim == 2:
            return positions
        if positions.ndim != 3 or positions.shape[0] != batch:
            raise ValueError("port positions must have shape N,D or B,N,D")
        return positions[:, None].expand(
            batch, self.ports, *positions.shape[1:],
        ).reshape(batch * self.ports, *positions.shape[1:])

    def _port_batch_mask(
        self, graph_mask: Tensor | None, batch: int,
    ) -> Tensor | None:
        if graph_mask is None or graph_mask.ndim == 2:
            return graph_mask
        if graph_mask.ndim != 3 or graph_mask.shape[0] != batch:
            raise ValueError("port graph mask must have shape N,N or B,N,N")
        return graph_mask[:, None].expand(
            batch, self.ports, *graph_mask.shape[1:],
        ).reshape(batch * self.ports, *graph_mask.shape[1:])

    def _unflatten_token(self, value: Tensor, batch: int, count: int) -> Tensor:
        return value.reshape(
            batch, self.ports, count, self.width,
        ).permute(0, 2, 1, 3).contiguous()

    def _unflatten_code(self, value: Tensor, batch: int, count: int) -> Tensor:
        return value.reshape(
            batch, self.ports, count, self.code_width,
        ).permute(0, 2, 1, 3).reshape(
            batch, count, self.ports * self.code_width,
        )

    def forward(
        self,
        initial_tokens: Tensor,
        *,
        grid_shape: Sequence[int] | None = None,
        positions: Tensor | None = None,
        graph_mask: Tensor | None = None,
    ) -> FieldTokenComposition:
        if (
            initial_tokens.ndim != 4
            or initial_tokens.shape[2:] != (self.ports, self.width)
        ):
            raise ValueError("port tokens must have shape B,N,P,width")
        batch, count = initial_tokens.shape[:2]
        flattened = initial_tokens.permute(0, 2, 1, 3).reshape(
            batch * self.ports, count, self.width,
        )
        port_result = self.communication(
            flattened,
            grid_shape=grid_shape,
            positions=self._port_batch_positions(positions, batch),
            graph_mask=self._port_batch_mask(graph_mask, batch),
        )
        global_code = self.fusion(self._unflatten_code(
            port_result.global_code, batch, count,
        ))
        diagonal_code = self.fusion(self._unflatten_code(
            port_result.diagonal_code, batch, count,
        ))
        initial = self._unflatten_token(
            port_result.initial_tokens, batch, count,
        )
        global_tokens = self._unflatten_token(
            port_result.global_tokens, batch, count,
        )
        diagonal_tokens = self._unflatten_token(
            port_result.diagonal_tokens, batch, count,
        )
        return FieldTokenComposition(
            initial_tokens=initial,
            global_tokens=global_tokens,
            diagonal_tokens=diagonal_tokens,
            message=global_tokens - diagonal_tokens,
            global_code=global_code,
            diagonal_code=diagonal_code,
            correction_code=global_code - diagonal_code,
        )


class CenteredDensePortTokenExtension(nn.Module):
    r"""Communicate oriented ports as distinct relative-position tokens.

    Unlike :class:`CenteredPortFieldTokenExtension`, which communicates each
    port index independently, this extension flattens ``N`` subdomains and
    ``P`` oriented ports into one small ``NP`` graph. A receiver port may
    therefore attend to the opposite face of another subdomain. Edges within
    one subdomain are omitted: local port fusion happens only in the shared
    nonlinear head, while every nonzero centered message remains attributable
    to cross-subdomain communication.

    The graph admits each token's self edge and all allowed cross-subdomain
    edges. For one subdomain it is exactly diagonal, so the correction is
    identically zero for arbitrary trained weights.
    """

    def __init__(
        self,
        width: int,
        code_width: int,
        heads: int,
        dimension: int,
        *,
        ports: int,
        depth: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if ports <= 1:
            raise ValueError("a dense port extension requires at least two ports")
        self.width = int(width)
        self.code_width = int(code_width)
        self.ports = int(ports)
        self.dimension = int(dimension)
        self.communication = CenteredFieldTokenTransformer(
            width, code_width, heads, dimension,
            depth=depth, expansion=expansion,
        )
        fused_width = ports * code_width
        hidden = max(2 * code_width, fused_width // 2)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_width),
            nn.Linear(fused_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, code_width, bias=False),
        )
        final = self.fusion[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)

    @property
    def communication_layers(self) -> int:
        return self.communication.communication_layers

    def _positions(
        self,
        initial_tokens: Tensor,
        grid_shape: Sequence[int],
        port_offsets: Tensor,
    ) -> Tensor:
        batch, count = initial_tokens.shape[:2]
        base = regular_grid_positions(grid_shape, initial_tokens)
        offsets = port_offsets.to(initial_tokens)
        if offsets.shape != (self.ports, self.dimension):
            raise ValueError("port offsets must have shape P,D")
        positions = base[:, None] + offsets[None]
        return positions.reshape(count * self.ports, self.dimension)[None].expand(
            batch, -1, -1,
        )

    def _graph_mask(
        self,
        count: int,
        reference: Tensor,
        tile_graph_mask: Tensor | None,
    ) -> Tensor:
        tile_index = torch.arange(
            count, device=reference.device,
        ).repeat_interleave(self.ports)
        cross_tile = tile_index[:, None] != tile_index[None, :]
        token_count = count * self.ports
        mask = cross_tile | torch.eye(
            token_count, dtype=torch.bool, device=reference.device,
        )
        if tile_graph_mask is None:
            return mask
        tile_mask = tile_graph_mask.to(device=reference.device, dtype=torch.bool)
        if tile_mask.shape != (count, count):
            raise ValueError("tile graph mask must have shape N,N")
        expanded = tile_mask.repeat_interleave(
            self.ports, dim=0,
        ).repeat_interleave(self.ports, dim=1)
        return mask & expanded

    @staticmethod
    def _token_field(
        value: Tensor, batch: int, count: int, ports: int, width: int,
    ) -> Tensor:
        return value.reshape(batch, count, ports, width)

    def _fused_code(self, value: Tensor, batch: int, count: int) -> Tensor:
        return self.fusion(value.reshape(
            batch, count, self.ports * self.code_width,
        ))

    def forward(
        self,
        initial_tokens: Tensor,
        *,
        grid_shape: Sequence[int],
        port_offsets: Tensor,
        tile_graph_mask: Tensor | None = None,
    ) -> FieldTokenComposition:
        if (
            initial_tokens.ndim != 4
            or initial_tokens.shape[2:] != (self.ports, self.width)
            or math.prod(grid_shape) != initial_tokens.shape[1]
        ):
            raise ValueError("dense port tokens must have shape B,N,P,width")
        batch, count = initial_tokens.shape[:2]
        flattened = initial_tokens.reshape(
            batch, count * self.ports, self.width,
        )
        port_result = self.communication(
            flattened,
            positions=self._positions(initial_tokens, grid_shape, port_offsets),
            graph_mask=self._graph_mask(count, initial_tokens, tile_graph_mask),
        )
        global_codes = port_result.global_code.reshape(
            batch, count, self.ports, self.code_width,
        )
        diagonal_codes = port_result.diagonal_code.reshape_as(global_codes)
        global_code = self._fused_code(global_codes, batch, count)
        diagonal_code = self._fused_code(diagonal_codes, batch, count)
        global_tokens = self._token_field(
            port_result.global_tokens, batch, count, self.ports, self.width,
        )
        diagonal_tokens = self._token_field(
            port_result.diagonal_tokens, batch, count, self.ports, self.width,
        )
        return FieldTokenComposition(
            initial_tokens=initial_tokens,
            global_tokens=global_tokens,
            diagonal_tokens=diagonal_tokens,
            message=global_tokens - diagonal_tokens,
            global_code=global_code,
            diagonal_code=diagonal_code,
            correction_code=global_code - diagonal_code,
        )
