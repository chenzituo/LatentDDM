"""Summarize the predeclared depth-1/2/4 field-token study across cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEPTHS = (1, 2, 4)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def optional(path: Path) -> dict[str, Any] | None:
    return read(path) if path.exists() else None


def collect() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    definitions = {
        "darcy": (
            ROOT / "codiffuse_Schwarz_Darcy/outputs",
            "best_validation_relative_l2",
        ),
        "car": (
            ROOT / "codiffuse_Schwarz_Car/outputs",
            "best_validation_relative_l2",
        ),
        "airfoil": (
            ROOT / "codiffuse_Schwarz_Airfoil/outputs",
            "best_validation_selection_score",
        ),
    }
    for case, (base, criterion) in definitions.items():
        rows = []
        for depth in DEPTHS:
            run = base / f"field_token_core_depth{depth}_seed20260810"
            summary = optional(run / "summary.json")
            if summary is None:
                rows.append({"depth": depth, "status": "pending"})
                continue
            row: dict[str, Any] = {
                "depth": depth,
                "status": "complete",
                "selection_criterion": criterion,
                "selection_value": float(summary[criterion]),
                "local_core_validation": float(
                    summary[
                        "local_core_validation_relative_l2"
                        if case != "airfoil"
                        else "local_core_validation_field_relative_l2"
                    ]
                ),
                "trainable_parameter_count": int(summary["trainable_parameter_count"]),
                "summary": str((run / "summary.json").resolve()),
            }
            common_query = optional(run / "common_query_validation.json")
            rollout = optional(run / "rollout_h20_validation.json")
            if common_query is not None:
                row["common_query"] = common_query["summary"]
            if rollout is not None:
                row["rollout_h20"] = rollout["aggregate"]
            rows.append(row)
        completed = [row for row in rows if row["status"] == "complete"]
        selected = min(
            completed, key=lambda row: float(row["selection_value"]),
        )["depth"] if completed else None
        cases[case] = {
            "selection_rule": f"minimum {criterion} on the declared validation split",
            "selected_depth": selected,
            "rows": rows,
        }
    return {
        "contract": "three_case_field_token_depth_study_v1",
        "depths": list(DEPTHS),
        "local_endpoint_recalled_between_layers": False,
        "assembly": "disjoint_owned_core",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "codiffuse_Schwarz/outputs/field_token_depth_study_v1.json",
    )
    args = parser.parse_args()
    payload = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
