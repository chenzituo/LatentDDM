"""Select the smallest response chart that clears validation-only gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = (
    "relative_l2", "energy_relative_square", "residual_relative_square",
    "interface_value_square", "interface_flux_square",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-oracle", type=Path, required=True)
    parser.add_argument("--linear-spectrum", type=Path, required=True)
    parser.add_argument("--nonlinear-oracle", type=Path, required=True)
    parser.add_argument("--nonlinear-spectrum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def chart_gate(oracle: dict, spectrum: dict) -> dict[str, object]:
    if oracle.get("selection_split") != "validation" \
            or spectrum.get("selection_split") != "validation":
        raise ValueError("chart gate accepts validation artifacts only")
    local, predicted = oracle["summary"]["local"], oracle["summary"]["oracle"]
    finite = all(
        math.isfinite(float(local[name])) and math.isfinite(float(predicted[name]))
        for name in METRICS
    )
    improvements = {
        "relative_l2": float(predicted["relative_l2"]) < float(local["relative_l2"]),
        **{
            name: float(predicted[name]) <= 1.01 * float(local[name])
            for name in METRICS if name != "relative_l2"
        },
    }
    conditioning = spectrum.get(
        "supported_artificial_coordinates_standardized",
        spectrum["standardized_output_coordinates"],
    )
    supported = float(conditioning.get(
        "mean_supported_coordinates", spectrum.get("port_width", 1),
    ))
    required_effective_rank = max(1.0, 0.5 * supported)
    condition_pass = (
        float(conditioning["minimum_top_singular_value"]) > 1.0e-6
        and float(conditioning["mean_effective_rank_at_1e-2"])
        >= required_effective_rank
    )
    return {
        "finite": finite,
        "strict_improvement": improvements,
        "conditioning_pass": condition_pass,
        "required_effective_rank": required_effective_rank,
        "clears": finite and all(improvements.values()) and condition_pass,
        "chart_parameters": int(oracle["chart_parameters"]),
        "port_width": int(oracle.get("port_width", 0)),
        "chart_checkpoint": oracle.get("chart_checkpoint"),
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    linear_oracle = json.loads(args.linear_oracle.read_text())
    nonlinear_oracle = json.loads(args.nonlinear_oracle.read_text())
    linear_spectrum = json.loads(args.linear_spectrum.read_text())
    nonlinear_spectrum = json.loads(args.nonlinear_spectrum.read_text())
    hashes = {
        linear_oracle["source_local_sha256"],
        nonlinear_oracle["source_local_sha256"],
    }
    if len(hashes) != 1:
        raise ValueError("chart gate candidates use different frozen locals")
    gates = {
        "tied_linear_green": chart_gate(linear_oracle, linear_spectrum),
        "nonlinear_12mode": chart_gate(nonlinear_oracle, nonlinear_spectrum),
    }
    eligible = [name for name, gate in gates.items() if gate["clears"]]
    selected = min(
        eligible, key=lambda name: gates[name]["chart_parameters"],
        default=None,
    )
    result = {
        "contract": "darcy_response_chart_gate_audit_v1",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "criteria": {
            "all_oracle_metrics_strictly_improve_over_frozen_local": list(METRICS),
            "minimum_top_singular_value": 1.0e-6,
            "minimum_mean_effective_rank_at_1e-2": 3.0,
            "selection_among_passing": "smallest chart parameter count",
        },
        "gates": gates,
        "selected_chart": selected,
        "communicator_training_authorized": selected is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if selected is None:
        raise RuntimeError("no response chart cleared every oracle/conditioning gate")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2))
