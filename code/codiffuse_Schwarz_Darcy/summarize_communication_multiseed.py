"""Aggregate model-seed uncertainty for local and communicated Darcy POU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


DOMAINS = tuple("ABCD")
METRICS = (
    "relative_l2", "energy_relative_error", "residual_rms",
    "value_jump", "flux_jump", "wall_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("codiffuse_Schwarz_Darcy/outputs"))
    parser.add_argument("--local-pattern", default="full_domain_local_pou_seed*")
    parser.add_argument("--communication-pattern", default="full_domain_relative_comm_seed*")
    parser.add_argument("--communication-stage", choices=("proposal", "both"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def complete_runs(root: Path, pattern: str) -> list[tuple[str, Path]]:
    result = []
    for run in sorted(path for path in root.glob(pattern) if path.is_dir()):
        match = re.search(r"seed(\d+)$", run.name)
        if match is None:
            continue
        required = [
            run / f"domain_{domain}_composition" / "iteration_metrics.csv"
            for domain in DOMAINS
        ]
        if all(path.exists() for path in required):
            result.append((match.group(1), run))
    if not result:
        raise FileNotFoundError(f"no complete runs matched {pattern!r}")
    return result


def sweep_metric(run: Path, domain: str, sweep: int, metric: str) -> float:
    path = run / f"domain_{domain}_composition" / "iteration_metrics.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["sweep"]) == sweep]
    if not rows:
        raise ValueError(f"no sweep {sweep} rows in {path}")
    return float(np.mean([float(row[metric]) for row in rows]))


def aggregate(runs: list[tuple[str, Path]], domain: str) -> dict:
    result = {"seed_count": len(runs), "seeds": [seed for seed, _ in runs]}
    for metric in METRICS:
        values = np.asarray(
            [sweep_metric(run, domain, 1, metric) for _, run in runs],
            dtype=np.float64,
        )
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std())
    return result


def build_summary(args: argparse.Namespace) -> dict:
    local_runs = complete_runs(args.root, args.local_pattern)
    communication_runs = complete_runs(args.root, args.communication_pattern)
    domains = {
        domain: {
            "local_pou": aggregate(local_runs, domain),
            "communication": aggregate(communication_runs, domain),
        }
        for domain in DOMAINS
    }
    local_equal = np.asarray([
        domains[domain]["local_pou"]["relative_l2_mean"] for domain in DOMAINS
    ])
    communication_equal = np.asarray([
        domains[domain]["communication"]["relative_l2_mean"] for domain in DOMAINS
    ])
    shared_seeds = sorted(
        set(seed for seed, _ in local_runs) & set(seed for seed, _ in communication_runs)
    )
    paired_changes = []
    local_map, communication_map = dict(local_runs), dict(communication_runs)
    for seed in shared_seeds:
        local_value = np.mean([
            sweep_metric(local_map[seed], domain, 1, "relative_l2")
            for domain in DOMAINS
        ])
        communication_value = np.mean([
            sweep_metric(communication_map[seed], domain, 1, "relative_l2")
            for domain in DOMAINS
        ])
        paired_changes.append(communication_value / local_value - 1.0)
    return {
        "contract": "darcy_communication_model_seed_aggregate_v1",
        "communication_stage": args.communication_stage,
        "local_seeds": [seed for seed, _ in local_runs],
        "communication_seeds": [seed for seed, _ in communication_runs],
        "domains": domains,
        "equal_domain_one_pass": {
            "local_relative_l2_mean": float(local_equal.mean()),
            "communication_relative_l2_mean": float(communication_equal.mean()),
            "relative_change_of_means": float(
                communication_equal.mean() / local_equal.mean() - 1.0
            ),
            "paired_seed_relative_changes": paired_changes,
        },
    }


def plot(summary: dict, path: Path) -> None:
    x = np.arange(len(DOMAINS))
    width = 0.36
    local = [summary["domains"][d]["local_pou"]["relative_l2_mean"] for d in DOMAINS]
    local_std = [summary["domains"][d]["local_pou"]["relative_l2_std"] for d in DOMAINS]
    communicated = [summary["domains"][d]["communication"]["relative_l2_mean"] for d in DOMAINS]
    communicated_std = [summary["domains"][d]["communication"]["relative_l2_std"] for d in DOMAINS]
    figure, axis = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    axis.bar(x - width / 2, local, width, yerr=local_std, capsize=3, label="local POU")
    axis.bar(
        x + width / 2, communicated, width, yerr=communicated_std,
        capsize=3, label=f"{summary['communication_stage']} communication",
    )
    axis.set_xticks(x, DOMAINS)
    axis.set_ylabel(r"one-pass relative $L_2$")
    axis.set_title("Darcy model-seed uncertainty")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(summary, args.output)
    print(json.dumps(summary["equal_domain_one_pass"], indent=2))


if __name__ == "__main__":
    main()
