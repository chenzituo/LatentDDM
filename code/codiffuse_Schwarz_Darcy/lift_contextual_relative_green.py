"""Lift a fitted relative-Green communicator to identity-gated context.

This is a deterministic architecture migration, not a training program.  The
accepted value communicator and signed relative kernel are copied exactly,
while every geometry-gate parameter is initialized to zero.  Since the gate is
``exp(tanh(0)) == 1``, the lifted checkpoint has exactly the same input/output
map before any contextual fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .residual_port_composer import (
    ContextualRelativeGreenFluxResidualPortCommunicator,
    RelativeGreenFluxResidualPortCommunicator,
)
from .train_residual_port_chart import sha256
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT,
    CONTRACT,
)


RUN_CONTRACT = "darcy_contextual_relative_green_identity_lift_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def _construct(
    checkpoint: dict[str, Any], *, contextual: bool,
) -> RelativeGreenFluxResidualPortCommunicator:
    cfg = checkpoint["communicator_spec"]
    state = checkpoint["communicator"]
    kernel = state.get("flux_green.kernel")
    if kernel is None or kernel.ndim != 4:
        raise ValueError("checkpoint lacks a signed relative-Green kernel")
    code_width, flux_width = int(kernel.shape[2]), int(kernel.shape[3])
    prefix_width = int(cfg["value_prefix_width"])
    communicator_type = (
        ContextualRelativeGreenFluxResidualPortCommunicator
        if contextual else RelativeGreenFluxResidualPortCommunicator
    )
    return communicator_type(
        int(cfg["physical_channels"]), int(cfg["scalar_dim"]),
        code_width, prefix_width, flux_width=flux_width,
        token_width=int(cfg["token_width"]), heads=int(cfg["heads"]),
        max_offset=int(cfg.get("relative_green_max_offset", 2)),
    )


@torch.inference_mode()
def _synthetic_equivalence(
    source: RelativeGreenFluxResidualPortCommunicator,
    lifted: ContextualRelativeGreenFluxResidualPortCommunicator,
    seed: int,
) -> float:
    """Return the maximum absolute output difference on fixed synthetic data."""
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
            "identity-gated contextual lift changed the source map: "
            f"maximum absolute difference {maximum}"
        )
    return maximum


def lift(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("contextual lift output must not overwrite its input")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("input is not a residual-port communicator checkpoint")
    cfg = checkpoint.get("communicator_spec", {})
    if cfg.get("analyzer_mode") != "value_flux_relative_green":
        raise ValueError("input must use value_flux_relative_green analysis")
    if cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
        raise ValueError("input lacks the homogeneous compatibility contract")
    if cfg.get("absolute_tile_identifier", False):
        raise ValueError("contextual lift requires a translation-relative source")

    source = _construct(checkpoint, contextual=False).eval()
    source.load_state_dict(checkpoint["communicator"], strict=True)
    lifted = _construct(checkpoint, contextual=True).eval()
    lifted.base.load_state_dict(source.base.state_dict(), strict=True)
    with torch.no_grad():
        lifted.flux_green.kernel.copy_(source.flux_green.kernel)
    gate_parameters = (
        lifted.flux_green.receiver_gate.weight,
        lifted.flux_green.source_gate.weight,
        lifted.flux_green.relative_gate,
    )
    if any(torch.count_nonzero(parameter) for parameter in gate_parameters):
        raise RuntimeError("contextual gates are not identity-initialized")
    maximum = _synthetic_equivalence(source, lifted, args.seed)

    output_cfg = dict(cfg)
    output_cfg.update({
        "analyzer_mode": "value_flux_contextual_relative_green",
        "kernel_type": "accepted_softmax_plus_contextual_relative_green_5x5",
        "contextual_gate": "exp_tanh_receiver_plus_source_plus_relative",
        "contextual_gate_context": "geometry_only",
        "contextual_gate_initialization": "exact_identity",
        "contextual_gate_parameters": sum(
            parameter.numel() for parameter in gate_parameters
        ),
        "fixed_context_q_linearity": True,
        "zero_signal_zero_increment_bitwise": True,
        "identity_lift_synthetic_max_abs": maximum,
        "one_global_communication_stage": True,
        "base_attention_plus_relative_convolution_disclosed": True,
    })
    payload = dict(checkpoint)
    # Stale optimizer state cannot safely be resumed after changing topology.
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
        "contextual_gate_parameters": output_cfg["contextual_gate_parameters"],
        "test_subsets_opened": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(lift(parse_args()), indent=2))
