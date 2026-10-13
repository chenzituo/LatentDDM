"""Aggregate global Transolver against local and two-stage Darcy communication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DOMAINS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--communication",
        type=Path,
        default=Path(
            "codiffuse_Schwarz_Darcy/outputs/"
            "darcy_communication_comparison.json"
        ),
    )
    parser.add_argument(
        "--global-root", type=Path,
        default=Path("codiffuse_Schwarz_Darcy/outputs"),
    )
    parser.add_argument(
        "--global-pattern",
        default="global_transolver_seed*/matched_evaluation/summary.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "codiffuse_Schwarz_Darcy/outputs/"
            "darcy_local_global_comparison.json"
        ),
    )
    return parser.parse_args()


def available_global_summaries(
    root: Path, pattern: str,
) -> list[tuple[str, dict[str, Any]]]:
    records = []
    for path in sorted(root.glob(pattern)):
        match = re.search(r"global_transolver(?:_matched_pretraining)?_seed(\d+)", str(path))
        if match is None:
            raise ValueError(f"cannot recover global seed from {path}")
        seed = match.group(1)
        records.append((seed, json.loads(path.read_text(encoding="utf-8"))))
    if not records:
        raise FileNotFoundError("no completed matched global evaluations")
    return records


def global_values(
    runs: list[tuple[str, dict[str, Any]]], domain: str, sweep: int, metric: str,
) -> np.ndarray:
    key = f"{domain}/sweep_{sweep:03d}"
    return np.asarray(
        [float(payload["summary"][key][metric]) for _, payload in runs],
        dtype=np.float64,
    )


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    communication = json.loads(args.communication.read_text(encoding="utf-8"))
    global_runs = available_global_summaries(args.global_root, args.global_pattern)
    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        local = communication["domains"][domain]["local_pou"]
        message = communication["domains"][domain]["one_message"]
        global_l2 = global_values(global_runs, domain, 1, "relative_l2")
        global_energy = global_values(
            global_runs, domain, 1, "energy_relative_error",
        )
        global_residual = global_values(global_runs, domain, 1, "residual_rms")
        domains[domain] = {
            "local_pou": {
                key: float(local[key]) for key in (
                    "relative_l2", "energy_relative_error", "residual_rms",
                )
            },
            "one_message": {
                key: float(message[key]) for key in (
                    "relative_l2", "energy_relative_error", "residual_rms",
                )
            },
            "global_transolver": {
                "relative_l2_mean": float(global_l2.mean()),
                "relative_l2_std": float(global_l2.std()),
                "energy_relative_error_mean": float(global_energy.mean()),
                "energy_relative_error_std": float(global_energy.std()),
                "residual_rms_mean": float(global_residual.mean()),
                "residual_rms_std": float(global_residual.std()),
            },
        }

    sweeps: dict[str, Any] = {}
    for sweep in range(1, 17):
        global_per_seed = np.stack([
            global_values(global_runs, domain, sweep, "relative_l2")
            for domain in DOMAINS
        ]).mean(0)
        sweeps[str(sweep)] = {
            "local_pou": float(np.mean([
                communication["domains"][domain]["local_pou"]
                ["relative_l2_by_sweep"][str(sweep)] for domain in DOMAINS
            ])),
            "one_message": float(np.mean([
                communication["domains"][domain]["one_message"]
                ["relative_l2_by_sweep"][str(sweep)] for domain in DOMAINS
            ])),
            "global_transolver_mean": float(global_per_seed.mean()),
            "global_transolver_std": float(global_per_seed.std()),
        }

    result = {
        "contract": "darcy_matched_local_two_stage_global_v2",
        "communication_call_sites": ["proposal", "projector"],
        "schema_note": "one_message keys are retained for artifact compatibility",
        "global_seeds": [seed for seed, _ in global_runs],
        "global_seed_count": len(global_runs),
        "domains": domains,
        "equal_domain_one_pass": {
            "local_pou_relative_l2": float(np.mean([
                domains[d]["local_pou"]["relative_l2"] for d in DOMAINS
            ])),
            "one_message_relative_l2": float(np.mean([
                domains[d]["one_message"]["relative_l2"] for d in DOMAINS
            ])),
            "global_transolver_relative_l2_mean": float(np.mean([
                domains[d]["global_transolver"]["relative_l2_mean"]
                for d in DOMAINS
            ])),
        },
        "relative_l2_by_sweep": sweeps,
        "interpretation_guard": (
            "Global seed count is explicit; do not call this a three-seed "
            "comparison until global_seed_count equals 3. Recurrent sweeps "
            "are diagnostic and not test-selected inference."
        ),
    }
    return result


def plot_summary(result: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    x = np.arange(len(DOMAINS))
    width = 0.25
    local = [result["domains"][d]["local_pou"]["relative_l2"] for d in DOMAINS]
    message = [result["domains"][d]["one_message"]["relative_l2"] for d in DOMAINS]
    global_mean = [
        result["domains"][d]["global_transolver"]["relative_l2_mean"]
        for d in DOMAINS
    ]
    global_std = [
        result["domains"][d]["global_transolver"]["relative_l2_std"]
        for d in DOMAINS
    ]
    axes[0].bar(x - width, local, width, label="local POU")
    axes[0].bar(x, message, width, label="two-stage comm. + POU")
    axes[0].bar(
        x + width, global_mean, width, yerr=global_std,
        label=f"global Transolver ({result['global_seed_count']} seed)",
        capsize=3,
    )
    axes[0].set_xticks(x, DOMAINS)
    axes[0].set_ylabel(r"one-pass relative $L_2$")
    axes[0].set_title("Held-out geometry families")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    sweep_numbers = np.arange(1, 17)
    rows = result["relative_l2_by_sweep"]
    axes[1].plot(
        sweep_numbers, [rows[str(s)]["local_pou"] for s in sweep_numbers],
        "--", label="local POU",
    )
    axes[1].plot(
        sweep_numbers, [rows[str(s)]["one_message"] for s in sweep_numbers],
        label="two-stage comm. + POU",
    )
    global_curve = np.asarray([
        rows[str(s)]["global_transolver_mean"] for s in sweep_numbers
    ])
    global_curve_std = np.asarray([
        rows[str(s)]["global_transolver_std"] for s in sweep_numbers
    ])
    axes[1].plot(sweep_numbers, global_curve, label="global Transolver")
    axes[1].fill_between(
        sweep_numbers, global_curve - global_curve_std,
        global_curve + global_curve_std, alpha=0.18,
    )
    axes[1].set_xlabel("diagnostic reapplication sweep")
    axes[1].set_ylabel(r"equal-domain relative $L_2$")
    axes[1].set_title("Inference-depth behavior")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    result = build_summary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_summary(result, args.output.with_suffix(".png"))
    print(json.dumps(result["equal_domain_one_pass"], indent=2))


if __name__ == "__main__":
    main()
