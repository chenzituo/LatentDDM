"""Publish the owned-core Darcy SNI validation frontier with one initialization.

The original validation run evaluated two starting states in one artifact.
The paper now defines SNI uniquely as the recurrence initialized by the frozen
Local/Core endpoint.  This utility extracts that exact recorded arm without
re-evaluating, renaming, or altering any metric.  Its output contains only the
paper method and records the hash of the immutable source run for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/core_sni_initializations_v1.json"
)
DEFAULT_OUTPUT = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/core_sni_local_initialization_v2.json"
)
CONTRACT = "darcy_core_ras_sni_local_initialization_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build(source: dict[str, Any], source_hash: str) -> dict[str, Any]:
    require(
        source.get("contract") == "darcy_core_ras_sni_two_initializations_v1",
        "source is not the frozen owned-core validation run",
    )
    require(
        source.get("split") == "validation"
        and source.get("protected_test_cases_opened") == 0
        and source.get("assembly") == "core_ras",
        "source split or assembly differs from the paper protocol",
    )
    require(
        source.get("local_initialization_equals_sni_sweep0_max_abs") == 0.0,
        "recorded SNI sweep zero is not exactly Local/Core",
    )
    prefix = "init_local_core/"
    summary = {
        key: value for key, value in source["summary"].items()
        if key.startswith(prefix)
    }
    rows = [
        row for row in source["rows"]
        if row.get("initialization") == "local_core"
    ]
    require(summary and rows, "source has no Local/Core-initialized SNI rows")
    require(
        all(row.get("initialization") == "local_core" for row in rows),
        "filtered SNI evidence contains another initialization",
    )
    candidates = [
        (key, value) for key, value in summary.items()
        if not key.endswith("sweep_0000")
    ]
    best_key, best_value = min(
        candidates, key=lambda item: item[1]["relative_l2_mean"],
    )
    result = {
        "contract": CONTRACT,
        "split": "validation",
        "protected_test_cases_opened": 0,
        "cases": int(source["cases"]),
        "data": source["data"],
        "local_checkpoint": source["local_checkpoint"],
        "local_checkpoint_sha256": source["local_checkpoint_sha256"],
        "assembly": "core_ras",
        "initialization": "local_core",
        "taus": source["taus"],
        "checkpoints": source["checkpoints"],
        "max_sweeps": int(source["max_sweeps"]),
        "local_initialization_equals_sni_sweep0_max_abs": 0.0,
        "cost_definition": (
            "SNI-k includes one Local/Core endpoint call and k sequential "
            "owned-core response sweeps"
        ),
        "best_validation_field": {
            "configuration": best_key,
            **best_value,
        },
        "summary": summary,
        "rows": rows,
        "provenance": {
            "kind": "exact method-arm extraction; no metric recomputation",
            "source_artifact_sha256": source_hash,
            "selection": "initialization == local_core",
        },
    }
    serialized = json.dumps(result, sort_keys=True)
    require("init_zero" not in serialized, "non-paper SNI arm leaked into output")
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    result = build(source, sha256(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "contract": result["contract"],
        "cases": result["cases"],
        "best_validation_field": result["best_validation_field"],
    }, indent=2))


if __name__ == "__main__":
    main()
