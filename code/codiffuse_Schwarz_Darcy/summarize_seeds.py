"""Aggregate mean/std metrics across the three fixed Darcy training seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=("20260805", "20260806", "20260807"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    combined: dict[str, object] = {"seeds": list(args.seeds), "domains": {}}
    for domain in ("A", "B", "C", "D"):
        summaries = []
        for seed in args.seeds:
            path = args.root / f"direct_endpoint_seed{seed}" / f"domain_{domain}_composition" / "summary.json"
            summaries.append(json.loads(path.read_text())["summary"])
        keys = sorted(set.intersection(*(set(summary) for summary in summaries)))
        domain_output: dict[str, object] = {}
        for key in keys:
            metrics = sorted(set.intersection(*(set(summary[key]) for summary in summaries)))
            entry: dict[str, object] = {}
            for metric in metrics:
                if metric == "cases":
                    continue
                values = np.asarray([float(summary[key][metric]) for summary in summaries])
                entry[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
                rows.append({
                    "domain": domain, "method_sweep": key, "metric": metric,
                    "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                })
            domain_output[key] = entry
        combined["domains"][domain] = domain_output  # type: ignore[index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("domain", "method_sweep", "metric", "mean", "std"))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

