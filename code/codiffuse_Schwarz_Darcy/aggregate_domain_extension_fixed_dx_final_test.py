"""Aggregate the one-time fixed-dx Darcy protected-test report.

All choices come from the completed validation summary.  In particular, this
script never selects a global family or an SNI sweep from protected-test error.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .summarize_domain_extension_fixed_dx import (
    SEEDS,
    align,
    hierarchical_interval,
    indexed_error,
    read_csv,
    relative_reduction_of_means,
)


METHODS = ("latentddm", "point", "slice", "patch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def latent_rows(path: Path, method: str = "latentddm") -> dict[int, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("contract") != "darcy_input_correction_matched_test_v1":
        raise ValueError(f"not a protected-test LatentDDM artifact: {path}")
    if document.get("split") != "test" or int(
        document.get("protected_test_cases_opened", -1)
    ) != 64:
        raise ValueError(f"incomplete protected-test artifact: {path}")
    result = indexed_error(
        [row for row in document["rows"] if row["method"] == method],
    )
    if len(result) != 64:
        raise ValueError(f"protected-test method has {len(result)} cases: {path}")
    return result


def latent_timing(path: Path) -> dict[int, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return indexed_error(
        [row for row in document["rows"] if row["method"] == "latentddm"],
        "seconds",
    )


def global_rows(path: Path, key: str = "relative_l2") -> dict[int, float]:
    summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    if summary.get("groups") != ["test"] or int(
        summary.get("protected_test_cases_opened", -1)
    ) != 64:
        raise ValueError(f"incomplete protected-test global artifact: {path}")
    rows = [
        row for row in read_csv(path)
        if row["group"] == "test" and int(row["sweep"]) == 1
    ]
    result = indexed_error(rows, key)
    if len(result) != 64:
        raise ValueError(f"protected-test method has {len(result)} cases: {path}")
    return result


def sni_rows(path: Path, sweep: int, key: str = "relative_l2") -> dict[int, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("split") != "test" or int(
        document.get("protected_test_cases_opened", -1)
    ) != 64:
        raise ValueError(f"incomplete protected-test SNI artifact: {path}")
    rows = [
        row for row in document["rows"]
        if abs(float(row["tau"]) - 1.0) <= 1.0e-12
        and int(row["sweep"]) == sweep
    ]
    result = indexed_error(rows, key)
    if len(result) != 64:
        raise ValueError(f"protected-test SNI has {len(result)} cases: {path}")
    return result


def stack(paths: list[Path], reader) -> tuple[list[int], np.ndarray]:
    cases, values = align(*(reader(path) for path in paths))
    return cases, np.stack(values)


def main() -> None:
    args = parse_args()
    validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    if validation.get("contract") != "darcy_fixed_dx_extension_paired_validation_v2":
        raise ValueError("unexpected validation selection artifact")
    if int(validation.get("protected_test_cases_opened", -1)) != 0:
        raise ValueError("validation selection opened protected cases")
    rng = np.random.default_rng(args.seed)
    report: dict[str, object] = {
        "contract": "darcy_fixed_dx_extension_protected_test_v1",
        "selection_source": str(args.validation_summary.resolve()),
        "selection_uses_protected_test": False,
        "protected_test_cases_per_scale": 64,
        "scales": {},
    }
    table: list[dict[str, object]] = []
    for blocks in (4, 8, 16):
        key = f"{blocks}x{blocks}"
        root = args.root / key / "final_test_v1"
        zero_paths = {
            "latentddm": root / "latent_zero_shot.json",
            **{
                method: root / f"{method}_zero_shot/iteration_metrics.csv"
                for method in ("point", "slice", "patch")
            },
        }
        adapted_paths = {
            "latentddm": [root / f"latent_seed{seed}.json" for seed in SEEDS],
            **{
                method: [
                    root / f"{method}_seed{seed}/iteration_metrics.csv"
                    for seed in SEEDS
                ]
                for method in ("point", "slice", "patch")
            },
        }
        readers = {
            "latentddm": latent_rows,
            "point": global_rows,
            "slice": global_rows,
            "patch": global_rows,
        }
        payload: dict[str, object] = {"methods": {}}
        matrices: dict[str, tuple[list[int], np.ndarray]] = {}
        for method in METHODS:
            reader = readers[method]
            zero = reader(zero_paths[method])
            cases, matrix = stack(adapted_paths[method], reader)
            zero_aligned = np.asarray([zero[index] for index in cases])
            zero_matrix = np.broadcast_to(zero_aligned[None], matrix.shape)
            gain = (zero_matrix - matrix) / np.maximum(zero_matrix, 1.0e-12)
            matrices[method] = cases, matrix
            entry = {
                "zero_shot_error": hierarchical_interval(
                    zero_aligned, args.bootstrap, rng,
                ),
                "adapted_error": hierarchical_interval(
                    matrix, args.bootstrap, rng,
                ),
                "paired_adaptation_gain": hierarchical_interval(
                    gain, args.bootstrap, rng,
                ),
                "aggregate_mean_error_reduction": relative_reduction_of_means(
                    zero_matrix, matrix,
                ),
            }
            payload["methods"][method] = entry
            table.append({
                "blocks": blocks,
                "method": method,
                "zero_shot_relative_l2": entry["zero_shot_error"]["mean"],
                "adapted_relative_l2": entry["adapted_error"]["mean"],
                "aggregate_mean_error_reduction": entry[
                    "aggregate_mean_error_reduction"
                ],
                "paired_casewise_adaptation_gain": entry[
                    "paired_adaptation_gain"
                ]["mean"],
            })

        selected_global = str(validation["scales"][key]["best_global"])
        if selected_global not in ("point", "slice", "patch"):
            raise ValueError(f"invalid validation-selected global family: {key}")
        latent_cases, latent = matrices["latentddm"]
        global_cases, global_values = matrices[selected_global]
        if latent_cases != global_cases:
            raise ValueError("protected-test case order differs across methods")
        gain_over_global = (
            global_values - latent
        ) / np.maximum(global_values, 1.0e-12)
        payload["validation_selected_best_global"] = selected_global
        payload["latent_gain_over_validation_selected_global"] = (
            hierarchical_interval(gain_over_global, args.bootstrap, rng)
        )
        payload[
            "latent_aggregate_mean_error_reduction_over_validation_selected_global"
        ] = relative_reduction_of_means(global_values, latent)

        sni_path = root / "sni_tau1.json"
        sni16 = sni_rows(sni_path, 16)
        validation_match = validation["scales"][key]["sni_matched_accuracy"]
        matched_sweep = int(validation_match["sweep"])
        matched = sni_rows(sni_path, matched_sweep)
        matched_seconds = sni_rows(sni_path, matched_sweep, "seconds")
        latent_seconds = stack(adapted_paths["latentddm"], latent_timing)[1]
        payload["sni"] = {
            "policy": "tau=1 fixed; sweep choices copied from validation",
            "sweep16_error": hierarchical_interval(
                np.asarray(list(sni16.values())), args.bootstrap, rng,
            ),
            "validation_matched_sweep": matched_sweep,
            "matched_sweep_error": hierarchical_interval(
                np.asarray(list(matched.values())), args.bootstrap, rng,
            ),
            "matched_sweep_seconds": hierarchical_interval(
                np.asarray(list(matched_seconds.values())), args.bootstrap, rng,
            ),
            "latent_seconds": hierarchical_interval(
                latent_seconds, args.bootstrap, rng,
            ),
            "latent_speedup_at_validation_matched_sweep": float(
                np.mean(list(matched_seconds.values()))
                / max(float(latent_seconds.mean()), 1.0e-12)
            ),
        }
        report["scales"][key] = payload

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "method_summary.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
