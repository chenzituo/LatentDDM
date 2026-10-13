"""Freeze one 128^2 Darcy composer using validation data only.

Architectures are ranked by the mean complete-field relative L2 across three
optimization seeds at the shared 16-label budget.  The copied checkpoint is
the seed closest to its architecture mean, which avoids choosing an unusually
lucky seed for the qualitative figure or timing audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch


EXPECTED_DATA = Path(__file__).resolve().parent / "data/darcy_domain_extension_128_v1.h5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(run: Path) -> dict[str, Any]:
    seed_dirs = sorted(path for path in run.glob("seed*") if path.is_dir())
    if len(seed_dirs) != 3:
        raise ValueError(f"{run} must contain exactly three seed directories")
    rows: list[dict[str, Any]] = []
    local_reference: float | None = None
    for seed_dir in seed_dirs:
        validation_path = seed_dir / "validation_128.json"
        checkpoint_path = seed_dir / "checkpoint.pt"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        local_path = Path(checkpoint["source_local_checkpoint"])
        chart_path = Path(checkpoint["source_chart_checkpoint"])
        if (
            validation.get("contract") != "darcy_field_token_validation_v1"
            or Path(validation.get("data", "")).resolve() != EXPECTED_DATA.resolve()
            or int(validation.get("cases", -1)) != 128
            or validation.get("composer_checkpoint") != str(checkpoint_path.resolve())
            or checkpoint.get("composer_spec", {}).get("assembly")
            != "disjoint_owned_core"
            or not local_path.is_file()
            or not chart_path.is_file()
            or sha256(local_path) != checkpoint.get("source_local_sha256")
            or sha256(chart_path) != checkpoint.get("source_chart_sha256")
        ):
            raise ValueError(f"invalid Darcy target-domain artifact: {validation_path}")
        metrics = validation["metrics"]
        local = float(metrics["local_core_relative_l2"])
        if local_reference is None:
            local_reference = local
        elif not np.isclose(local, local_reference, rtol=0.0, atol=1.0e-9):
            raise ValueError("Darcy candidates do not share one Local/Core field")
        rows.append({
            "seed": int(seed_dir.name.removeprefix("seed")),
            "relative_l2": float(metrics["relative_l2"]),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "validation": str(validation_path.resolve()),
            "validation_sha256": sha256(validation_path),
            "communication_depth": int(checkpoint["composer_spec"]["communication_depth"]),
            "code_width": int(checkpoint["composer_spec"]["code_width"]),
            "local_checkpoint": str(local_path.resolve()),
            "local_checkpoint_sha256": checkpoint["source_local_sha256"],
            "chart_checkpoint": str(chart_path.resolve()),
            "chart_checkpoint_sha256": checkpoint["source_chart_sha256"],
        })
    mean = float(np.mean([row["relative_l2"] for row in rows]))
    std = float(np.std([row["relative_l2"] for row in rows], ddof=1))
    representative = min(
        rows,
        key=lambda row: (abs(float(row["relative_l2"]) - mean), int(row["seed"])),
    )
    return {
        "run": str(run.resolve()),
        "architecture": run.parent.parent.name + "/" + run.parent.name,
        "mean_relative_l2": mean,
        "sample_std_relative_l2": std,
        "local_core_relative_l2": local_reference,
        "representative": representative,
        "seeds": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    candidates = [summarize(run) for run in args.runs]
    local = float(candidates[0]["local_core_relative_l2"])
    if any(
        not np.isclose(float(row["local_core_relative_l2"]), local, rtol=0.0, atol=1.0e-9)
        for row in candidates[1:]
    ):
        raise ValueError("Darcy architectures do not share one frozen local operator")
    selected = min(candidates, key=lambda row: (
        float(row["mean_relative_l2"]),
        float(row["sample_std_relative_l2"]),
        int(row["representative"]["communication_depth"]),
        int(row["representative"]["code_width"]),
    ))
    representative = selected["representative"]
    args.output_dir.mkdir(parents=True)
    sources = {
        "checkpoint.pt": Path(representative["checkpoint"]),
        "local_checkpoint.pt": Path(representative["local_checkpoint"]),
        "chart_checkpoint.pt": Path(representative["chart_checkpoint"]),
    }
    copied: dict[str, dict[str, str]] = {}
    for name, source in sources.items():
        destination = args.output_dir / name
        shutil.copy2(source, destination)
        if sha256(destination) != sha256(source):
            raise RuntimeError(f"copied Darcy artifact changed: {name}")
        copied[name] = {"path": str(destination.resolve()), "sha256": sha256(destination)}
    result = {
        "contract": "darcy_128_composer_validation_selection_v1",
        "selection_split": "validation",
        "target_training_labels": 16,
        "protected_test_cases_opened": 0,
        "selection_rule": (
            "minimize three-seed mean complete-field relative L2; copy the "
            "seed closest to the selected architecture mean"
        ),
        "selected": selected,
        "candidates": candidates,
        "copied": copied,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
