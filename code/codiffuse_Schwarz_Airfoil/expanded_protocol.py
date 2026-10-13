"""Frozen, incrementally discoverable split for the 256-case airfoil corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


FORMAT_VERSION = 1


def trajectory_name(index: int, level: int = 5) -> str:
    return f"cyclic_{index:03d}_level{level}_window.h5"


def make_frozen_assignment(
    base_split: dict,
    *,
    expansion_start: int = 32,
    expansion_stop: int = 256,
    fold_count: int = 10,
    test_fold: int = 0,
    validation_fold: int = 1,
) -> dict[str, list[str]]:
    """Keep legacy cases fixed and fold new randomized LHS rows by ID.

    The expansion plan independently permutes every Latin-hypercube parameter
    column before assigning IDs.  An ID fold is therefore stable before data
    generation and remains approximately stratified in every parameter.
    """
    if expansion_start < 0 or expansion_stop <= expansion_start:
        raise ValueError("invalid expansion interval")
    if fold_count < 3 or not 0 <= test_fold < fold_count:
        raise ValueError("invalid test fold")
    if not 0 <= validation_fold < fold_count or validation_fold == test_fold:
        raise ValueError("invalid validation fold")
    assignment = {
        key: list(base_split[key]) for key in ("train", "validation", "test")
    }
    for index in range(expansion_start, expansion_stop):
        fold = (index - expansion_start) % fold_count
        key = (
            "test" if fold == test_fold
            else "validation" if fold == validation_fold
            else "train"
        )
        assignment[key].append(trajectory_name(index))
    return assignment


def file_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return bool(handle.attrs.get("complete", False))
    except OSError:
        return False


def available_paired_names(root: Path, names: list[str]) -> list[str]:
    available = []
    for name in names:
        coarse = root / name
        fine = root / name.replace("_level5_", "_level6_")
        if file_complete(coarse) and file_complete(fine):
            available.append(name)
    return available


def build_manifest(
    root: Path,
    base_split: dict,
    *,
    expansion_start: int = 32,
    expansion_stop: int = 256,
) -> dict:
    expected = make_frozen_assignment(
        base_split, expansion_start=expansion_start,
        expansion_stop=expansion_stop,
    )
    available = {
        key: available_paired_names(root, expected[key])
        for key in ("train", "validation", "test")
    }
    return {
        "format_version": FORMAT_VERSION,
        "policy": "legacy_fixed_plus_lhs_id_folds",
        "root": str(root.resolve()),
        "expansion_start": expansion_start,
        "expansion_stop": expansion_stop,
        "fold_count": 10,
        "test_fold": 0,
        "validation_fold": 1,
        "expected": expected,
        "available": available,
        "expected_counts": {key: len(value) for key, value in expected.items()},
        "available_counts": {key: len(value) for key, value in available.items()},
        "paired_complete_count": sum(len(value) for value in available.values()),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def split_from_source(path: Path) -> dict[str, list[str]]:
    """Read either an expanded protocol or an exact trajectory split."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload["available"] if "available" in payload else payload
    split = {key: list(source[key]) for key in ("train", "validation", "test")}
    flattened = [name for key in split for name in split[key]]
    if not all(split.values()):
        raise ValueError("all split partitions must be nonempty")
    if len(flattened) != len(set(flattened)):
        raise ValueError("trajectory split partitions overlap")
    return split


def write_or_validate_snapshot(source: Path, output: Path) -> dict[str, list[str]]:
    """Freeze current availability so a running experiment cannot drift."""
    proposed = split_from_source(source)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != proposed:
            raise RuntimeError(
                f"existing snapshot does not match split source: {output}"
            )
        return existing
    output.write_text(json.dumps(proposed, indent=2) + "\n", encoding="utf-8")
    return proposed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expansion-start", type=int, default=32)
    parser.add_argument("--expansion-stop", type=int, default=256)
    args = parser.parse_args()
    base_split = json.loads(args.base_split.read_text(encoding="utf-8"))
    manifest = build_manifest(
        args.data_root, base_split,
        expansion_start=args.expansion_start,
        expansion_stop=args.expansion_stop,
    )
    write_manifest(args.output, manifest)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "expected_counts": manifest["expected_counts"],
        "available_counts": manifest["available_counts"],
        "paired_complete_count": manifest["paired_complete_count"],
    }))


if __name__ == "__main__":
    main()
