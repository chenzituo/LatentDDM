"""Deterministically lift a fitted relative Green map to a dynamic kernel.

No data, labels, or PDE operator are consulted.  The accepted value
communicator and fitted signed 5x5 relative kernel are copied exactly.  The
dynamic low-rank coefficient head is zero, so the migrated checkpoint has the
same input/output map while retaining a live first-step gradient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .residual_port_composer import (
    DynamicRelativeGreenFluxResidualPortCommunicator,
    RelativeGreenFluxResidualPortCommunicator,
)
from .train_residual_port_chart import sha256
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT,
    CONTRACT,
)


RUN_CONTRACT = "darcy_dynamic_relative_green_identity_lift_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--residual-rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def _dimensions(checkpoint: dict[str, Any]) -> dict[str, int]:
    cfg = checkpoint["communicator_spec"]
    kernel = checkpoint["communicator"].get("flux_green.kernel")
    if kernel is None or kernel.ndim != 4:
        raise ValueError("checkpoint lacks a signed relative-Green kernel")
    return {
        "physical_channels": int(cfg["physical_channels"]),
        "scalar_dim": int(cfg["scalar_dim"]),
        "code_width": int(kernel.shape[2]),
        "prefix_width": int(cfg["value_prefix_width"]),
        "flux_width": int(kernel.shape[3]),
        "token_width": int(cfg["token_width"]),
        "heads": int(cfg["heads"]),
        "max_offset": int(cfg.get("relative_green_max_offset", 2)),
    }


def _construct_source(
    dimensions: dict[str, int],
) -> RelativeGreenFluxResidualPortCommunicator:
    return RelativeGreenFluxResidualPortCommunicator(**dimensions)


def _construct_lifted(
    dimensions: dict[str, int], hidden_width: int, residual_rank: int,
) -> DynamicRelativeGreenFluxResidualPortCommunicator:
    return DynamicRelativeGreenFluxResidualPortCommunicator(
        **dimensions, hidden_width=hidden_width, residual_rank=residual_rank,
    )


@torch.inference_mode()
def _synthetic_equivalence(
    source: RelativeGreenFluxResidualPortCommunicator,
    lifted: DynamicRelativeGreenFluxResidualPortCommunicator,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, tiles, size = 2, 9, 6
    local = torch.randn(tiles * batch, 1, size, size, generator=generator)
    physical = torch.randn(
        tiles * batch, 29, size, size, generator=generator,
    )
    scalar = torch.randn(tiles * batch, 4, generator=generator)
    valid = torch.ones(tiles * batch, 1, size, size)
    aligned = torch.randn(
        batch, tiles, source.aligned_width, generator=generator,
    )
    source_output = source.forward_decomposed(
        local, physical, scalar, valid, (3, 3), source.prefix_width,
        aligned_features=aligned,
    )
    lifted_output = lifted.forward_decomposed(
        local, physical, scalar, valid, (3, 3), lifted.prefix_width,
        aligned_features=aligned,
    )
    maximum = max(
        float((left - right).abs().max())
        for left, right in zip(source_output, lifted_output, strict=True)
    )
    if maximum != 0.0:
        raise RuntimeError(
            "zero-residual dynamic lift changed the source map: "
            f"maximum absolute difference {maximum}"
        )
    return maximum


def lift(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("dynamic lift output must not overwrite its input")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("input is not a residual-port communicator checkpoint")
    cfg = checkpoint.get("communicator_spec", {})
    if cfg.get("analyzer_mode") != "value_flux_relative_green":
        raise ValueError("input must use value_flux_relative_green analysis")
    if cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
        raise ValueError("input lacks the homogeneous compatibility contract")
    if cfg.get("absolute_tile_identifier", False):
        raise ValueError("dynamic lift requires a translation-relative source")
    if min(args.hidden_width, args.residual_rank) <= 0:
        raise ValueError("hidden width and residual rank must be positive")

    dimensions = _dimensions(checkpoint)
    source = _construct_source(dimensions).eval()
    source.load_state_dict(checkpoint["communicator"], strict=True)
    torch.manual_seed(args.seed)
    lifted = _construct_lifted(
        dimensions, args.hidden_width, args.residual_rank,
    ).eval()
    lifted.base.load_state_dict(source.base.state_dict(), strict=True)
    with torch.no_grad():
        lifted.flux_green.kernel.copy_(source.flux_green.kernel)
    if torch.count_nonzero(lifted.flux_green.coefficient_head.weight):
        raise RuntimeError("dynamic residual is not identity-initialized")
    maximum = _synthetic_equivalence(source, lifted, args.seed)

    dynamic_parameters = (
        sum(parameter.numel() for parameter in lifted.flux_green.parameters())
        - lifted.flux_green.kernel.numel()
    )
    output_cfg = dict(cfg)
    output_cfg.update({
        "analyzer_mode": "value_flux_dynamic_relative_green",
        "kernel_type": "accepted_relative_green_plus_low_rank_dynamic_residual",
        "dynamic_kernel_formula": "K_delta+sum_r a_ijr(h_i,h_j,delta)B_r",
        "dynamic_kernel_context": "geometry_only",
        "dynamic_hidden_width": args.hidden_width,
        "dynamic_residual_rank": args.residual_rank,
        "dynamic_residual_initialization": "exact_zero_live_final_gradient",
        "dynamic_parameter_count_excluding_fixed_kernel": dynamic_parameters,
        "fixed_context_q_linearity": True,
        "zero_signal_zero_increment_bitwise": True,
        "identity_lift_synthetic_max_abs": maximum,
        "one_global_communication_stage": True,
        "base_attention_plus_dynamic_relative_convolution_disclosed": True,
    })
    payload = dict(checkpoint)
    payload.pop("optimizer", None)
    payload.pop("scheduler", None)
    payload.update({
        "contract": CONTRACT,
        "phase": RUN_CONTRACT,
        "communicator": lifted.state_dict(),
        "communicator_spec": output_cfg,
        "source_relative_green_checkpoint": str(input_path),
        "source_relative_green_sha256": sha256(input_path),
        "identity_lift_synthetic_max_abs": maximum,
        "lift_used_data": False,
        "target_or_pde_labels_used_during_lift": False,
        "exact_global_solution_used_at_inference": False,
        "test_subsets_opened": False,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    result = {
        "contract": RUN_CONTRACT,
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "identity_lift_synthetic_max_abs": maximum,
        "dynamic_parameter_count_excluding_fixed_kernel": dynamic_parameters,
        "test_subsets_opened": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(lift(parse_args()), indent=2))
