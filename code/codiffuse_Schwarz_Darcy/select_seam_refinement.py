"""Select a validation-safe Darcy seam refinement without viewing a figure.

The primary model remains a supervised, no-discrete-operator LatentDDM.  A
candidate is admissible only when its complete 128-case field error is within
one percent of the frozen parent and its owned-core seam diagnostic improves.
Among admissible continuations we select the smallest seam diagnostic, using
field error only as a tie break.  The rendered case is never read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_evaluation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("contract") != "darcy_field_token_validation_v1"
        or payload.get("assembly") != "disjoint_owned_core"
        or payload.get("test_subsets_opened") is not False
        or int(payload.get("cases", 0)) != 128
    ):
        raise ValueError(f"inadmissible Darcy validation artifact: {path}")
    checkpoint = Path(payload["composer_checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != payload[
        "composer_checkpoint_sha256"
    ]:
        raise ValueError(f"checkpoint lineage mismatch: {path}")
    metrics = payload["metrics"]
    return {
        "name": path.parent.name,
        "evaluation": str(path.resolve()),
        "evaluation_sha256": sha256(path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "relative_l2": float(metrics["relative_l2"]),
        "seam_error": float(metrics["seam_error"]),
        "trace_consistency": float(metrics["trace_consistency"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field-tolerance", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.field_tolerance <= 0.1:
        raise ValueError("field tolerance must lie in [0, 0.1]")
    baseline = read_evaluation(args.baseline)
    candidates = [
        read_evaluation(path)
        for path in sorted(args.candidate_root.glob("*/validation_128.json"))
    ]
    if not candidates:
        raise ValueError("no seam-refinement evaluations were found")
    field_limit = baseline["relative_l2"] * (1.0 + args.field_tolerance)
    for row in candidates:
        row["field_gate_passed"] = row["relative_l2"] <= field_limit
        row["seam_gate_passed"] = row["seam_error"] < baseline["seam_error"]
        row["admissible"] = row["field_gate_passed"] and row[
            "seam_gate_passed"
        ]
    admissible = [row for row in candidates if row["admissible"]]
    selected = min(
        admissible, key=lambda row: (row["seam_error"], row["relative_l2"]),
    ) if admissible else baseline
    result = {
        "contract": "darcy_validation_safe_seam_selection_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "visual_appearance_used_for_selection": False,
        "selection_rule": {
            "admissibility": (
                "128-case relative L2 no more than 1% above the frozen parent "
                "and strictly smaller owned-core seam error"
            ),
            "ranking": "smallest seam error; relative L2 tie break",
        },
        "baseline": baseline,
        "field_error_limit": field_limit,
        "candidates": candidates,
        "selected": selected,
        "relative_l2_change_fraction": (
            selected["relative_l2"] / baseline["relative_l2"] - 1.0
        ),
        "seam_error_change_fraction": (
            selected["seam_error"] / baseline["seam_error"] - 1.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
