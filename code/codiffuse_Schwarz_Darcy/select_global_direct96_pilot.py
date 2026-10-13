"""Select one source-trained Point, Slice, and Patch model on validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def method_name(path: Path) -> str:
    name = path.name
    for method in ("point", "slice", "patch"):
        if name.startswith(f"{method}_"):
            return method
    raise ValueError(f"cannot infer method from {path}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    candidates: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("*_seed*/validation_250/summary.json")):
        run = summary_path.parent.parent
        checkpoint_path = run / "checkpoint.pt"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        score = float(
            summary["summary"]["validation/sweep_001"]["relative_l2"]
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        if checkpoint.get("current_policy") != "zero" \
                or checkpoint.get("use_discrete_residual") is not False:
            raise ValueError(f"{checkpoint_path} is not a direct PDE-agnostic model")
        spec = checkpoint["model_spec"]
        method = method_name(run)
        if method == "patch" and tuple(spec.get("patch_size", ())) != (32, 32):
            raise ValueError(
                f"{checkpoint_path} violates the fixed 32-cell Patch contract"
            )
        candidates.append({
            "method": method,
            "run": str(run),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "validation_summary": str(summary_path),
            "validation_relative_l2": score,
            "parameter_count": int(sum(
                value.numel() for value in checkpoint["model"].values()
            )),
            "model_spec": spec,
        })
    if len(candidates) != 6:
        raise RuntimeError(f"expected six completed pilot arms, found {len(candidates)}")

    selected: dict[str, dict[str, object]] = {}
    selection_root = root / "selected"
    for method in ("point", "slice", "patch"):
        eligible = [row for row in candidates if row["method"] == method]
        if len(eligible) != 2:
            raise RuntimeError(f"expected two {method} pilot arms")
        winner = min(eligible, key=lambda row: float(row["validation_relative_l2"]))
        selected[method] = winner
        destination = selection_root / method
        destination.mkdir(parents=True, exist_ok=True)
        link = destination / "checkpoint.pt"
        target = Path(str(winner["checkpoint"]))
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(target, destination))

    result = {
        "contract": "darcy_global_direct96_validation_selection_v1",
        "selection_split": "source_validation_250",
        "selection_rule": "minimum one-pass mean relative L2 within method",
        "current_policy": "zero",
        "use_discrete_residual": False,
        "global_models_trained_end_to_end": True,
        "patch_size_constraint": 32,
        "protected_test_cases_opened": 0,
        "selected": selected,
        "candidates": candidates,
    }
    selection_root.mkdir(parents=True, exist_ok=True)
    temporary = selection_root / "selection.json.partial"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(selection_root / "selection.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
