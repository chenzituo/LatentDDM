"""Matched validation comparison of nonlinear and tied-linear response charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonlinear-summary", type=Path, required=True)
    parser.add_argument("--linear-summary", type=Path, required=True)
    parser.add_argument("--nonlinear-checkpoint", type=Path, required=True)
    parser.add_argument("--linear-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compare(args: argparse.Namespace) -> dict[str, object]:
    nonlinear = json.loads(args.nonlinear_summary.read_text())
    linear = json.loads(args.linear_summary.read_text())
    if nonlinear.get("selection_split") != "validation" \
            or linear.get("selection_split") != "validation":
        raise ValueError("response-chart comparison is validation-only")
    nonlinear_checkpoint = torch.load(
        args.nonlinear_checkpoint, map_location="cpu", weights_only=False,
    )
    linear_checkpoint = torch.load(
        args.linear_checkpoint, map_location="cpu", weights_only=False,
    )
    if nonlinear_checkpoint.get("source_local_sha256") \
            != linear_checkpoint.get("source_local_sha256"):
        raise ValueError("response-chart controls use different frozen locals")
    nonlinear_communicator = nonlinear_checkpoint["communicator_spec"]
    linear_communicator = linear_checkpoint["communicator_spec"]
    controlled_keys = (
        "token_width", "heads", "analyzer_mode", "aligned_width",
        "relative_attention_count", "absolute_tile_identifier",
    )
    if any(
        nonlinear_communicator.get(key) != linear_communicator.get(key)
        for key in controlled_keys
    ):
        raise ValueError("response-chart comparison changes communicator capacity")
    nonlinear_chart_parameters = int(
        nonlinear_checkpoint["chart_spec"]["parameter_count"]
    )
    linear_chart_parameters = int(
        linear_checkpoint["chart_spec"]["parameter_count"]
    )
    relative_parameter_gap = abs(
        linear_chart_parameters - nonlinear_chart_parameters
    ) / nonlinear_chart_parameters
    if relative_parameter_gap >= 0.02:
        raise ValueError("response charts are not parameter matched within 2 percent")
    metrics = (
        "relative_l2", "energy_relative_square", "residual_relative_square",
        "interface_value_square", "interface_flux_square", "wall_seconds",
    )
    deltas = {}
    for key, nonlinear_values in nonlinear["summary"].items():
        if key not in linear["summary"]:
            continue
        linear_values = linear["summary"][key]
        deltas[key] = {
            metric: (
                None if nonlinear_values[metric] is None
                or linear_values[metric] is None
                else float(linear_values[metric]) - float(nonlinear_values[metric])
            )
            for metric in metrics
        }
    result = {
        "contract": "darcy_residual_port_chart_ablation_v1",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "same_frozen_local": True,
        "same_communicator_architecture": True,
        "nonlinear_chart_parameters": nonlinear_chart_parameters,
        "linear_chart_parameters": linear_chart_parameters,
        "relative_chart_parameter_gap": relative_parameter_gap,
        "delta_tied_linear_minus_nonlinear": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(compare(parse_args()), indent=2))
