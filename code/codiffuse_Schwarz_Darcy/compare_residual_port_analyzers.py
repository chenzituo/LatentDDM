"""Compare generic and defect-aligned analyzers on the same validation ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-summary", type=Path, required=True)
    parser.add_argument("--defect-summary", type=Path, required=True)
    parser.add_argument("--generic-checkpoint", type=Path, required=True)
    parser.add_argument("--defect-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def parameter_count(checkpoint: dict) -> int:
    return sum(tensor.numel() for tensor in checkpoint["communicator"].values())


def compare(args: argparse.Namespace) -> dict:
    generic = json.loads(args.generic_summary.read_text())
    defect = json.loads(args.defect_summary.read_text())
    if generic.get("selection_split") != "validation" \
            or defect.get("selection_split") != "validation":
        raise ValueError("analyzer comparison is validation-only")
    if generic.get("local_checkpoint") != defect.get("local_checkpoint"):
        raise ValueError("analyzer controls use different frozen local checkpoints")
    generic_checkpoint = load_checkpoint(args.generic_checkpoint)
    defect_checkpoint = load_checkpoint(args.defect_checkpoint)
    if generic_checkpoint.get("source_chart_sha256") \
            != defect_checkpoint.get("source_chart_sha256"):
        raise ValueError("analyzer controls use different frozen response charts")
    generic_parameters = parameter_count(generic_checkpoint)
    defect_parameters = parameter_count(defect_checkpoint)
    if generic_parameters != defect_parameters:
        raise ValueError("analyzer ablation does not have matched parameter capacity")
    metrics = (
        "relative_l2", "energy_relative_square", "residual_relative_square",
        "interface_value_square", "interface_flux_square", "wall_seconds",
    )
    deltas = {}
    for key, generic_values in generic["summary"].items():
        if not key.startswith("predicted/") or key not in defect["summary"]:
            continue
        defect_values = defect["summary"][key]
        deltas[key] = {
            metric: (
                None if generic_values[metric] is None or defect_values[metric] is None
                else float(defect_values[metric]) - float(generic_values[metric])
            )
            for metric in metrics
        }
    result = {
        "contract": "darcy_residual_port_analyzer_ablation_v1",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "same_frozen_local": True,
        "same_frozen_chart_required_by_job_contract": True,
        "matched_communicator_parameters": generic_parameters,
        "delta_defect_minus_generic": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(compare(parse_args()), indent=2))
