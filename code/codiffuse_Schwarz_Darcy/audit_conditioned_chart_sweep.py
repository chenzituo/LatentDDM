"""Select the smallest feasible geometry-only Darcy port chart on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_response_chart_gate import chart_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for degree in range(3):
        parser.add_argument(f"--degree{degree}-oracle", type=Path, required=True)
        parser.add_argument(f"--degree{degree}-spectrum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit(args: argparse.Namespace) -> dict[str, object]:
    gates = {}
    local_hashes = set()
    for degree in range(3):
        oracle = json.loads(getattr(args, f"degree{degree}_oracle").read_text())
        spectrum = json.loads(getattr(args, f"degree{degree}_spectrum").read_text())
        local_hashes.add(oracle["source_local_sha256"])
        gate = chart_gate(oracle, spectrum)
        expected_width = 4 * (degree + 1)
        if gate["port_width"] != expected_width:
            raise ValueError(
                f"degree {degree} expected {expected_width} port coordinates"
            )
        gates[f"degree_{degree}"] = gate
    if len(local_hashes) != 1:
        raise ValueError("conditioned charts do not share one frozen local")
    eligible = [name for name, gate in gates.items() if gate["clears"]]
    selected = min(
        eligible,
        key=lambda name: (
            gates[name]["port_width"], gates[name]["chart_parameters"],
        ),
        default=None,
    )
    result = {
        "contract": "darcy_conditioned_chart_sweep_gate_v1",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "chart_type": "geometry_tied_linear_green",
        "criteria": {
            "field": "strictly improves frozen local",
            "energy_residual_value_flux": "candidate <= 1.01 * frozen local",
            "conditioning": "supported artificial-face coordinates only",
            "selection": "smallest port width, then parameter count",
        },
        "gates": gates,
        "selected_candidate": selected,
        "selected_chart_checkpoint": (
            gates[selected]["chart_checkpoint"] if selected is not None else None
        ),
        "communicator_training_authorized": selected is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if selected is None:
        raise RuntimeError("no conditioned chart clears every gate")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2))
