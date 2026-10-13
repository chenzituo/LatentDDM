"""The only interface that a new physical system must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn.functional as F

from .cover import CartesianCover


Tensor = torch.Tensor


@dataclass
class PhysicalTransition:
    """System-independent description of one physical time transition.

    `payload` holds geometry, coefficients, forcing, controls, or masks. Values
    that live on the grid should be [B,C,H,W]; global values should be [B,K].
    The adapter, rather than the solver, assigns meaning to those values.
    """

    previous: Tensor
    current: Tensor
    target: Tensor | None
    dt: Tensor
    payload: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.previous.shape != self.current.shape or self.current.ndim != 4:
            raise ValueError("previous/current states must share shape [B,C,H,W]")
        if self.target is not None and self.target.shape != self.current.shape:
            raise ValueError("target and current state shapes differ")
        if self.dt.shape != (self.current.shape[0],):
            raise ValueError("dt must have one value per batch item")

    @property
    def batch_size(self) -> int:
        return self.current.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return tuple(self.current.shape[-2:])


@dataclass(frozen=True)
class ConditionBatch:
    """Dense spatial channels and global scalars consumed by local operators."""

    spatial: Tensor
    scalars: Tensor
    valid: Tensor | None = None

    def tiled(self, cover: CartesianCover) -> tuple[Tensor, Tensor]:
        batch = self.spatial.shape[0]
        valid = self.valid
        if valid is None:
            valid = torch.ones(
                batch, 1, cover.height, cover.width,
                device=self.spatial.device, dtype=self.spatial.dtype,
            )
        physical = cover.tile(torch.cat((self.spatial, valid), dim=1))
        physical[:, -1:] = cover.valid_tiles(
            batch, physical.device, physical.dtype
        )
        scalar = self.scalars[:, None].expand(-1, cover.tile_count, -1).reshape(
            batch * cover.tile_count, -1
        )
        return physical, scalar


class PhysicsAdapter(ABC):
    """Maps a physical problem to the normalized local-endpoint representation.

    A new system should keep all PDE-specific decisions here: normalization,
    geometry/control encoding, target semantics, AMR indicators, conservative
    transfer, and conversion of a generated endpoint back to physical units.
    """

    state_channels: int
    spatial_condition_channels: int
    scalar_condition_dim: int

    @abstractmethod
    def conditions(
        self, transition: PhysicalTransition, flow_time: Tensor, level_scale: float,
    ) -> ConditionBatch:
        """Build conditions at endpoint phase 1 (or experimental Flow time)."""

    @abstractmethod
    def normalized_target(self, transition: PhysicalTransition) -> Tensor:
        """Return the clean supervised endpoint, usually normalized delta-q."""

    def normalize_endpoint(
        self, transition: PhysicalTransition, next_state: Tensor,
    ) -> Tensor:
        """Normalize an arbitrary candidate next state using target semantics.

        Adaptive direct inference uses this operation to express a prolonged
        coarse prediction in the fine-level endpoint coordinates.  The default
        implementation preserves compatibility with adapters that implement
        normalization through ``normalized_target``.
        """
        return self.normalized_target(replace(transition, target=next_state))

    @abstractmethod
    def apply_endpoint(
        self, transition: PhysicalTransition, normalized_endpoint: Tensor,
    ) -> Tensor:
        """Convert a predicted normalized endpoint to the next physical state."""

    @abstractmethod
    def refinement_indicator(self, transition: PhysicalTransition) -> Tensor:
        """Return a causal nonnegative [B,1,H,W] refinement score."""

    def coarsen(
        self, transition: PhysicalTransition, factor: int,
    ) -> PhysicalTransition:
        """Default cell-average transfer; override for non-extensive variables."""
        if factor <= 1:
            raise ValueError("coarsening factor must exceed one")

        def restrict(value: Tensor) -> Tensor:
            if value.ndim == 4 and value.shape[-2:] == transition.spatial_shape:
                return F.avg_pool2d(value, factor, factor)
            return value

        return PhysicalTransition(
            previous=restrict(transition.previous),
            current=restrict(transition.current),
            target=None if transition.target is None else restrict(transition.target),
            dt=transition.dt,
            payload={key: restrict(value) for key, value in transition.payload.items()},
            metadata=dict(transition.metadata),
        )
