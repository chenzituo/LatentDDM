"""Low-rank adaptation utilities for matched global Darcy controls.

The adapters are deliberately training-only.  A checkpoint stores the merged
weights under the ordinary global-model contract, so the existing evaluator
sees exactly the same architecture as the source Transolver or Patch
Transformer.  The unmerged state is retained separately for resumable
training and for auditing the number of updated parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    """Frozen linear map plus a centered trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(
            rank,
            base.in_features,
            device=base.weight.device,
            dtype=base.weight.dtype,
        ))
        self.lora_b = nn.Parameter(torch.zeros(
            base.out_features,
            rank,
            device=base.weight.device,
            dtype=base.weight.dtype,
        ))
        nn.init.kaiming_uniform_(self.lora_a, a=5.0**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = F.linear(inputs, self.base.weight, self.base.bias)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return frozen + self.scale * update

    def merged_linear(self) -> nn.Linear:
        """Return an ordinary Linear containing the frozen map and update."""

        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = self.scale * torch.matmul(self.lora_b, self.lora_a)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + delta.to(
                device=self.base.weight.device, dtype=self.base.weight.dtype,
            ))
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


class LowRankMultiheadAttention(nn.Module):
    """Frozen MHA with centered low-rank updates to QKV and output maps."""

    def __init__(
        self,
        base: nn.MultiheadAttention,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not base._qkv_same_embed_dim:
            raise ValueError("low-rank MHA requires equal query/key/value width")
        if base.in_proj_weight is None:
            raise ValueError("low-rank MHA requires a packed input projection")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if dropout:
            raise ValueError("packed low-rank MHA currently requires zero dropout")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scale = self.alpha / self.rank
        width = base.embed_dim
        device = base.in_proj_weight.device
        dtype = base.in_proj_weight.dtype
        self.lora_in_a = nn.Parameter(torch.empty(
            rank, width, device=device, dtype=dtype,
        ))
        self.lora_in_b = nn.Parameter(torch.zeros(
            3 * width, rank, device=device, dtype=dtype,
        ))
        self.lora_out_a = nn.Parameter(torch.empty(
            rank, width, device=device, dtype=dtype,
        ))
        self.lora_out_b = nn.Parameter(torch.zeros(
            width, rank, device=device, dtype=dtype,
        ))
        nn.init.kaiming_uniform_(self.lora_in_a, a=5.0**0.5)
        nn.init.kaiming_uniform_(self.lora_out_a, a=5.0**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def batch_first(self) -> bool:
        """Mirror the public layout flag inspected by TransformerEncoder."""

        return self.base.batch_first

    @property
    def in_proj_bias(self) -> Tensor | None:
        """Expose the frozen bias for TransformerEncoder compatibility."""

        return self.base.in_proj_bias

    @property
    def _qkv_same_embed_dim(self) -> bool:
        """Disable fused encoder paths that would bypass low-rank deltas.

        The wrapped base does use a common Q/K/V width, but PyTorch's fused
        TransformerEncoder implementation consumes raw projection tensors and
        would therefore omit the dynamically composed low-rank update.
        Reporting ``False`` here safely selects the ordinary module forward.
        """

        return False

    def _weights(self) -> tuple[Tensor, Tensor]:
        in_delta = self.scale * torch.matmul(self.lora_in_b, self.lora_in_a)
        out_delta = self.scale * torch.matmul(self.lora_out_b, self.lora_out_a)
        return (
            self.base.in_proj_weight + in_delta.to(
                device=self.base.in_proj_weight.device,
                dtype=self.base.in_proj_weight.dtype,
            ),
            self.base.out_proj.weight + out_delta.to(
                device=self.base.out_proj.weight.device,
                dtype=self.base.out_proj.weight.dtype,
            ),
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        batched = query.ndim == 3
        if self.base.batch_first and batched:
            query, key, value = (item.transpose(0, 1) for item in (query, key, value))
        in_weight, out_weight = self._weights()
        output, weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            embed_dim_to_check=self.base.embed_dim,
            num_heads=self.base.num_heads,
            in_proj_weight=in_weight,
            in_proj_bias=self.base.in_proj_bias,
            bias_k=self.base.bias_k,
            bias_v=self.base.bias_v,
            add_zero_attn=self.base.add_zero_attn,
            dropout_p=self.base.dropout,
            out_proj_weight=out_weight,
            out_proj_bias=self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.base.batch_first and batched:
            output = output.transpose(0, 1)
        return output, weights

    def merged_state_dict(self) -> dict[str, Tensor]:
        in_weight, out_weight = self._weights()
        state = {
            "in_proj_weight": in_weight.detach().clone(),
            "out_proj.weight": out_weight.detach().clone(),
        }
        if self.base.in_proj_bias is not None:
            state["in_proj_bias"] = self.base.in_proj_bias.detach().clone()
        if self.base.bias_k is not None:
            state["bias_k"] = self.base.bias_k.detach().clone()
        if self.base.bias_v is not None:
            state["bias_v"] = self.base.bias_v.detach().clone()
        if self.base.out_proj.bias is not None:
            state["out_proj.bias"] = self.base.out_proj.bias.detach().clone()
        return state


@dataclass(frozen=True)
class AdaptationSummary:
    rank: int
    placement: str
    adapted_linear_maps: int
    adapted_module_paths: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int


def install_low_rank_adapters(
    model: nn.Module,
    rank: int,
    alpha: float | None = None,
    dropout: float = 0.0,
    placement: str = "all_linear",
) -> AdaptationSummary:
    """Freeze ``model`` and replace eligible Linear maps with low-rank deltas.

    Packed ``nn.MultiheadAttention`` modules receive low-rank QKV and output
    updates as a unit.  Transolver's explicit attention projections are
    ordinary Linear modules and are adapted by the same centered rule.
    """

    if rank <= 0:
        raise ValueError("rank must be positive")
    if placement not in {"all_linear", "attention_ffn"}:
        raise ValueError(f"unsupported low-rank placement {placement!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapted = 0
    adapted_paths: list[str] = []

    def eligible_linear(path: str) -> bool:
        if placement == "all_linear":
            return True
        segments = path.split(".")
        return (
            "attention" in segments
            or "mlp" in segments
            or segments[-1] in {"linear1", "linear2"}
        )

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal adapted
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.MultiheadAttention):
                setattr(
                    parent,
                    name,
                    LowRankMultiheadAttention(
                        child, rank=rank, alpha=alpha, dropout=dropout,
                    ),
                )
                adapted += 2
                adapted_paths.append(path)
                continue
            if isinstance(child, nn.Linear) and eligible_linear(path):
                setattr(
                    parent,
                    name,
                    LowRankLinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                adapted += 1
                adapted_paths.append(path)
            else:
                visit(child, path)

    visit(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if not adapted or not trainable:
        raise RuntimeError("no eligible linear maps were adapted")
    return AdaptationSummary(
        rank=int(rank),
        placement=placement,
        adapted_linear_maps=adapted,
        adapted_module_paths=tuple(adapted_paths),
        trainable_parameters=trainable,
        frozen_parameters=frozen,
    )


def merged_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return a standard-model state dict without mutating ``model``."""

    state: dict[str, Tensor] = {}

    def visit(module: nn.Module, prefix: str = "") -> None:
        for name, parameter in module._parameters.items():
            if parameter is not None:
                state[prefix + name] = parameter.detach().clone()
        for name, buffer in module._buffers.items():
            if buffer is not None and name not in module._non_persistent_buffers_set:
                state[prefix + name] = buffer.detach().clone()
        for name, child in module.named_children():
            child_prefix = prefix + name + "."
            if isinstance(child, LowRankLinear):
                merged = child.merged_linear()
                for key, value in merged.state_dict().items():
                    state[child_prefix + key] = value.detach().clone()
            elif isinstance(child, LowRankMultiheadAttention):
                for key, value in child.merged_state_dict().items():
                    state[child_prefix + key] = value.detach().clone()
            else:
                visit(child, child_prefix)

    visit(model)
    return state


def adaptation_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return only the auditable trainable low-rank state."""

    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
