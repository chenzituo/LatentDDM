"""Identity-lift the accepted Darcy kernel to executable-state value gating.

The lift copies every accepted parameter.  Two multiplicative paths are added:
one gates provisional-field context into the routing token, and one gates a
low-rank residual-value basis.  Both path coefficients are exactly zero, so no
data, labels, or PDE operator are needed and the source map is reproduced bit
for bit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .residual_port_composer import (
    DynamicRelativeGreenFluxResidualPortCommunicator,
    StateGatedDynamicRelativeGreenFluxResidualPortCommunicator,
)
from .train_residual_port_chart import sha256
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT,
)


RUN_CONTRACT = "darcy_state_gated_dynamic_identity_lift_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def _dimensions(checkpoint: dict[str, Any]) -> dict[str, int]:
    cfg = checkpoint["communicator_spec"]
    kernel = checkpoint["communicator"].get("flux_green.kernel")
    if kernel is None or kernel.ndim != 4:
        raise ValueError("checkpoint lacks the dynamic relative-Green kernel")
    return {
        "physical_channels": int(cfg["physical_channels"]),
        "scalar_dim": int(cfg["scalar_dim"]),
        "code_width": int(kernel.shape[2]),
        "prefix_width": int(cfg["value_prefix_width"]),
        "flux_width": int(kernel.shape[3]),
        "token_width": int(cfg["token_width"]),
        "heads": int(cfg["heads"]),
        "max_offset": int(cfg.get("relative_green_max_offset", 2)),
        "hidden_width": int(cfg["dynamic_hidden_width"]),
        "residual_rank": int(cfg["dynamic_residual_rank"]),
    }


@torch.inference_mode()
def _synthetic_equivalence(
    source: DynamicRelativeGreenFluxResidualPortCommunicator,
    lifted: StateGatedDynamicRelativeGreenFluxResidualPortCommunicator,
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
            f"state-gated identity lift changed the source map: {maximum}"
        )
    zero = torch.zeros_like(aligned)
    zero_code = lifted.forward_decomposed(
        local, physical, scalar, valid, (3, 3), lifted.prefix_width,
        aligned_features=zero,
    )[0]
    if torch.count_nonzero(zero_code):
        raise RuntimeError("zero compatibility signal produced a nonzero code")
    return maximum


def lift(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve(); output_path = args.output.resolve()
    if input_path == output_path or output_path.exists():
        raise FileExistsError("state-gated lift output must be a new path")
    if args.value_rank <= 0:
        raise ValueError("value rank must be positive")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("communicator_spec", {})
    if checkpoint.get("contract") != CONTRACT or cfg.get(
        "analyzer_mode"
    ) != "value_flux_dynamic_relative_green":
        raise ValueError("input is not the accepted dynamic Darcy communicator")
    if cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
        raise ValueError("input lacks the homogeneous compatibility contract")

    dimensions = _dimensions(checkpoint)
    source = DynamicRelativeGreenFluxResidualPortCommunicator(**dimensions).eval()
    source.load_state_dict(checkpoint["communicator"], strict=True)
    torch.manual_seed(args.seed)
    lifted = StateGatedDynamicRelativeGreenFluxResidualPortCommunicator(
        **dimensions, value_rank=args.value_rank,
    ).eval()
    lifted.base.geometry_analyzer.load_state_dict(
        source.base.geometry_analyzer.state_dict(), strict=True,
    )
    incompatible = lifted.base.code_map.load_state_dict(
        source.base.code_map.state_dict(), strict=False,
    )
    expected_missing = {
        "global_kernel.state_value_coefficients.weight",
        "global_kernel.state_value_basis",
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "unexpected base-map lift mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    lifted.flux_green.load_state_dict(source.flux_green.state_dict(), strict=True)
    if torch.count_nonzero(lifted.base.state_context_gate) or torch.count_nonzero(
        lifted.base.code_map.global_kernel.state_value_coefficients.weight
    ):
        raise RuntimeError("new state gates are not exactly zero")
    maximum = _synthetic_equivalence(source, lifted, args.seed)

    output_cfg = dict(cfg)
    output_cfg.update({
        "analyzer_mode": "value_flux_state_gated_dynamic_relative_green",
        "kernel_type": "state_gated_dynamic_relative_green",
        "state_value_rank": args.value_rank,
        "dynamic_kernel_context": "geometry_plus_zero_gated_provisional_state",
        "state_context_summary": "local mean/rms and four face mean/rms pairs",
        "state_context_is_never_value": True,
        "fixed_context_q_linearity": True,
        "zero_signal_zero_increment_bitwise": True,
        "identity_lift_synthetic_max_abs": maximum,
        "one_global_communication_stage": True,
    })
    payload = dict(checkpoint)
    payload.pop("optimizer", None); payload.pop("scheduler", None)
    payload.update({
        "contract": CONTRACT,
        "phase": RUN_CONTRACT,
        "communicator": lifted.state_dict(),
        "communicator_spec": output_cfg,
        "source_accepted_checkpoint": str(input_path),
        "source_accepted_sha256": sha256(input_path),
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
        "input": str(input_path), "input_sha256": sha256(input_path),
        "output": str(output_path), "output_sha256": sha256(output_path),
        "value_rank": args.value_rank,
        "identity_lift_synthetic_max_abs": maximum,
        "target_or_pde_labels_used_during_lift": False,
        "test_subsets_opened": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(lift(parse_args()), indent=2))
