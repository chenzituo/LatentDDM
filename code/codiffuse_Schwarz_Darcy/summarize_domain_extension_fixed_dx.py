"""Aggregate paired validation gains for the fixed-dx extension benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SEEDS = (20260810, 20260811, 20260812)
METHODS = ("latentddm", "point", "slice", "patch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-global-selection", type=Path, required=True)
    parser.add_argument("--source-latent-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def indexed_error(
    rows: Iterable[dict[str, str]], key: str = "relative_l2",
) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in rows:
        index = int(row["case_index"])
        if index in result:
            raise ValueError(f"duplicate case {index}")
        result[index] = float(row[key])
    return result


def align(*values: dict[int, float]) -> tuple[list[int], list[np.ndarray]]:
    common = sorted(set.intersection(*(set(value) for value in values)))
    if not common:
        raise ValueError("paired metric inputs have no common cases")
    return common, [
        np.asarray([value[index] for index in common], dtype=np.float64)
        for value in values
    ]


def hierarchical_interval(
    values: np.ndarray, draws: int, rng: np.random.Generator,
) -> dict[str, float]:
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("bootstrap values must have seed x case shape")
    seeds, cases = values.shape
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        seed_index = rng.integers(0, seeds, size=seeds)
        case_index = rng.integers(0, cases, size=cases)
        samples[draw] = values[np.ix_(seed_index, case_index)].mean()
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "seed_count": int(seeds),
        "case_count": int(cases),
    }


def relative_reduction_of_means(
    baseline: np.ndarray, candidate: np.ndarray,
) -> float:
    """Return the fractional reduction of the aggregate mean error.

    This is deliberately distinct from the mean of paired per-case relative
    reductions.  Both are useful, but they answer different questions and
    must not share one label in the paper artifacts.
    """

    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.size == 0:
        raise ValueError("relative-reduction arrays must be nonempty and aligned")
    return float(
        (baseline.mean() - candidate.mean())
        / max(float(baseline.mean()), 1.0e-12)
    )


def adaptation_metadata(paths: list[Path]) -> dict[str, object]:
    """Summarize the measured optimization cost across adaptation seeds."""
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    wall_seconds = np.asarray([
        float(document["wall_seconds"]) for document in documents
    ], dtype=np.float64)
    trainable = {
        int(document["trainable_parameter_count"]) for document in documents
    }
    exposures = {
        int(document.get(
            "sample_exposures",
            len(document["history"]) * int(document["samples_per_epoch"]),
        ))
        for document in documents
    }
    train_cases = {int(document["train_cases"]) for document in documents}
    if len(trainable) != 1 or len(exposures) != 1 or len(train_cases) != 1:
        raise ValueError("adaptation seeds do not share one training contract")
    return {
        "wall_seconds_mean": float(wall_seconds.mean()),
        "wall_seconds_min": float(wall_seconds.min()),
        "wall_seconds_max": float(wall_seconds.max()),
        "trainable_parameter_count": trainable.pop(),
        "target_train_cases": train_cases.pop(),
        "sample_exposures": exposures.pop(),
        "seed_count": int(wall_seconds.size),
    }


def global_rows(
    path: Path, key: str = "relative_l2",
) -> dict[int, float]:
    rows = [
        row for row in read_csv(path)
        if int(row["sweep"]) == 1 and row["group"] in ("validation", "val")
    ]
    return indexed_error(rows, key)


def latent_rows(path: Path) -> dict[int, float]:
    return indexed_error(read_csv(path), "latentddm_relative_l2")


def local_rows(path: Path) -> dict[int, float]:
    return indexed_error(read_csv(path), "local_core_relative_l2")


def oracle_rows(path: Path) -> dict[int, float]:
    return indexed_error(read_csv(path), "oracle_family_relative_l2")


def input_correction_rows(
    path: Path, method: str, key: str = "relative_l2",
) -> dict[int, float]:
    """Read one per-case method from the on-manifold LatentDDM evaluator."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("contract") != "darcy_input_correction_matched_validation_v1":
        raise ValueError(f"unexpected input-correction contract in {path}")
    if int(document.get("protected_test_cases_opened", -1)) != 0:
        raise ValueError(f"validation artifact opened protected cases: {path}")
    selected = [row for row in document["rows"] if row["method"] == method]
    return indexed_error(selected, key)


def input_chart_oracle_rows(path: Path, modes: int = 4) -> dict[int, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("contract") != "darcy_boundary_chart_capacity_oracle_v1":
        raise ValueError(f"unexpected boundary-chart contract in {path}")
    if int(document.get("protected_test_cases_opened", -1)) != 0:
        raise ValueError(f"oracle diagnostic opened protected cases: {path}")
    return indexed_error(
        document["rows"], f"projected_trace_modes_{modes}_relative_l2",
    )


def adapted_matrix(
    paths: list[Path], reader,
) -> tuple[list[int], np.ndarray]:
    maps = [reader(path) for path in paths]
    cases, aligned = align(*maps)
    return cases, np.stack(aligned)


def sni_selection(
    path: Path, *, tau: float = 1.0, sweep: int = 16,
) -> dict[str, object]:
    """Read one predeclared SNI operating point without target-side tuning."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    selected_rows = [
        row for row in rows
        if abs(float(row["tau"]) - tau) <= 1.0e-12
        and int(row["sweep"]) == sweep
    ]
    if not selected_rows:
        raise ValueError(
            f"SNI report lacks fixed operating point tau={tau}, sweep={sweep}: {path}"
        )
    selected = indexed_error(selected_rows)
    seconds = indexed_error(selected_rows, "seconds")
    baseline = indexed_error([
        row for row in rows
        if abs(float(row["tau"]) - tau) <= 1.0e-12
        and int(row["sweep"]) == 0
    ])
    cases, (local, error) = align(baseline, selected)
    return {
        "tau": tau,
        "sweep": sweep,
        "cases": cases,
        "error": error,
        "seconds": np.asarray([seconds[index] for index in cases]),
        "local": local,
        "gain": (local - error) / np.maximum(local, 1.0e-12),
    }


def sni_operating_points(
    path: Path, *, tau: float = 1.0,
) -> list[dict[str, object]]:
    """Return the fixed-relaxation SNI trajectory with aligned latency."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    grouped: dict[tuple[float, int], list[dict]] = {}
    for row in rows:
        sweep = int(row["sweep"])
        if sweep > 0 and abs(float(row["tau"]) - tau) <= 1.0e-12:
            grouped.setdefault((float(row["tau"]), sweep), []).append(row)
    points = []
    for (tau, sweep), selected_rows in sorted(grouped.items()):
        selected = indexed_error(selected_rows)
        seconds = indexed_error(selected_rows, "seconds")
        baseline = indexed_error([
            row for row in rows
            if abs(float(row["tau"]) - tau) <= 1.0e-12
            and int(row["sweep"]) == 0
        ])
        cases, (local, error) = align(baseline, selected)
        points.append({
            "tau": tau,
            "sweep": sweep,
            "cases": cases,
            "error": error,
            "seconds": np.asarray([seconds[index] for index in cases]),
            "gain": (local - error) / np.maximum(local, 1.0e-12),
        })
    if not points:
        raise ValueError(f"SNI report has no positive-sweep point: {path}")
    return points


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap draw count must be positive")
    rng = np.random.default_rng(args.seed)
    source_selection = json.loads(
        args.source_global_selection.read_text(encoding="utf-8")
    )
    source_latent = json.loads(
        args.source_latent_summary.read_text(encoding="utf-8")
    )
    if source_latent.get("contract") == "darcy_input_correction_matched_validation_v1":
        source_latent_error = float(
            source_latent["summary"]["latentddm"]["relative_l2_mean"]
        )
    else:
        source_latent_error = float(source_latent["best_validation_relative_l2"])
    source_error = {
        "latentddm": source_latent_error,
        **{
            method: float(
                source_selection["selected"][method]["validation_relative_l2"]
            )
            for method in ("point", "slice", "patch")
        },
    }
    source_best_global = min(
        ("point", "slice", "patch"), key=lambda method: source_error[method],
    )

    result: dict[str, object] = {
        "contract": "darcy_fixed_dx_extension_paired_validation_v2",
        "latentddm_contract": "F(x,b0) -> attention -> F(x,b0+Delta_b)",
        "overlap_assembly": "cosine_normalized_partition_of_unity",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "bootstrap_draws": args.bootstrap,
        "source_error": source_error,
        "source_best_global": source_best_global,
        "source_latent_relative_gain_over_best_global": float(
            (source_error[source_best_global] - source_error["latentddm"])
            / max(source_error[source_best_global], 1.0e-12)
        ),
        "sni_primary_policy": {
            "initialization": "weighted local-core assembly",
            "relaxation": 1.0,
            "reported_sweeps": 16,
            "selection": "fixed before target validation aggregation",
            "other_relaxations": "robustness diagnostics only",
        },
        "scales": {},
    }
    flat_rows: list[dict[str, object]] = []
    for blocks in (4, 8, 16):
        scale_root = args.root / f"{blocks}x{blocks}"
        latent_zero_path = scale_root / "zero_shot_input_latent/validation.json"
        local = input_correction_rows(latent_zero_path, "local_core")
        oracle = input_chart_oracle_rows(
            scale_root / (
                "zero_shot_input_latent/chart_capacity_validation64.json"
            ),
            modes=4,
        )
        zero_maps = {
            "latentddm": input_correction_rows(latent_zero_path, "latentddm")
        }
        for method in ("point", "slice", "patch"):
            zero_maps[method] = global_rows(
                scale_root / f"global_zero_shot/{method}/iteration_metrics.csv"
            )

        adapted: dict[str, tuple[list[int], np.ndarray]] = {}
        adapted_timing: dict[str, tuple[list[int], np.ndarray]] = {}
        adaptation_summaries: dict[str, list[Path]] = {}
        latent_paths = [
            scale_root / f"input_latent_budget16/seed{seed}/validation_64.json"
            for seed in SEEDS
        ]
        adapted["latentddm"] = adapted_matrix(
            latent_paths,
            lambda path: input_correction_rows(path, "latentddm"),
        )
        adapted_timing["latentddm"] = adapted_matrix(
            latent_paths,
            lambda path: input_correction_rows(path, "latentddm", "seconds"),
        )
        adaptation_summaries["latentddm"] = [
            path.parent / "summary.json" for path in latent_paths
        ]
        for method in ("point", "slice", "patch"):
            method_paths = [
                scale_root / (
                    f"global_budget16/{method}_seed{seed}/"
                    "validation_64/iteration_metrics.csv"
                )
                for seed in SEEDS
            ]
            adapted[method] = adapted_matrix(method_paths, global_rows)
            adapted_timing[method] = adapted_matrix(
                method_paths,
                lambda path: global_rows(path, "wall_seconds"),
            )
            adaptation_summaries[method] = [
                path.parents[1] / "summary.json" for path in method_paths
            ]

        scale_payload: dict[str, object] = {"methods": {}}
        method_means = {
            method: float(matrix.mean())
            for method, (_, matrix) in adapted.items()
            if method != "latentddm"
        }
        best_global = min(method_means, key=method_means.get)
        for method in METHODS:
            cases, matrix = adapted[method]
            zero = zero_maps[method]
            _, (zero_aligned,) = align({
                index: zero[index] for index in cases if index in zero
            })
            if len(zero_aligned) != len(cases):
                raise ValueError(f"{method} zero-shot and adapted cases differ")
            zero_matrix = np.broadcast_to(zero_aligned[None], matrix.shape)
            adaptation_gain = (
                zero_matrix - matrix
            ) / np.maximum(zero_matrix, 1.0e-12)
            _, (local_aligned,) = align({
                index: local[index] for index in cases if index in local
            })
            local_matrix = np.broadcast_to(local_aligned[None], matrix.shape)
            communication_gain = (
                local_matrix - matrix
            ) / np.maximum(local_matrix, 1.0e-12)
            method_payload = {
                "adapted_error": hierarchical_interval(
                    matrix, args.bootstrap, rng,
                ),
                "zero_shot_error": hierarchical_interval(
                    zero_matrix, args.bootstrap, rng,
                ),
                "paired_adaptation_gain": hierarchical_interval(
                    adaptation_gain, args.bootstrap, rng,
                ),
                "aggregate_mean_error_reduction": relative_reduction_of_means(
                    zero_matrix, matrix,
                ),
                "paired_gain_over_local": hierarchical_interval(
                    communication_gain, args.bootstrap, rng,
                ),
                "extension_degradation_vs_source": float(
                    matrix.mean() / max(source_error[method], 1.0e-12)
                ),
                "adaptation_cost": adaptation_metadata(
                    adaptation_summaries[method]
                ),
                "inference_seconds_per_case": hierarchical_interval(
                    adapted_timing[method][1], args.bootstrap, rng,
                ),
            }
            scale_payload["methods"][method] = method_payload
            flat_rows.append({
                "blocks": blocks,
                "method": method,
                "adapted_relative_l2": method_payload["adapted_error"]["mean"],
                "aggregate_mean_error_reduction": method_payload[
                    "aggregate_mean_error_reduction"
                ],
                "paired_casewise_adaptation_gain": method_payload[
                    "paired_adaptation_gain"
                ]["mean"],
                "gain_over_local": method_payload["paired_gain_over_local"]["mean"],
                "extension_degradation": method_payload[
                    "extension_degradation_vs_source"
                ],
            })

        latent_cases, latent_matrix = adapted["latentddm"]
        global_cases, global_matrix = adapted[best_global]
        if latent_cases != global_cases:
            raise ValueError("LatentDDM and best global cases differ")
        relative_to_global = (
            global_matrix - latent_matrix
        ) / np.maximum(global_matrix, 1.0e-12)
        _, (oracle_aligned,) = align({
            index: oracle[index] for index in latent_cases if index in oracle
        })
        oracle_matrix = np.broadcast_to(oracle_aligned[None], latent_matrix.shape)
        local_aligned = np.asarray([local[index] for index in latent_cases])
        local_matrix = np.broadcast_to(local_aligned[None], latent_matrix.shape)
        recovered = 1.0 - np.maximum(
            latent_matrix - oracle_matrix, 0.0,
        ) / np.maximum(local_matrix - oracle_matrix, 1.0e-12)
        scale_payload["best_global"] = best_global
        scale_payload["latent_paired_gain_over_best_global"] = hierarchical_interval(
            relative_to_global, args.bootstrap, rng,
        )
        scale_payload["latent_aggregate_mean_error_reduction_over_best_global"] = (
            relative_reduction_of_means(global_matrix, latent_matrix)
        )
        latent_cost = scale_payload["methods"]["latentddm"]["adaptation_cost"]
        global_cost = scale_payload["methods"][best_global]["adaptation_cost"]
        scale_payload["latent_adaptation_efficiency_vs_best_global"] = {
            "trainable_parameter_fraction": float(
                latent_cost["trainable_parameter_count"]
                / max(global_cost["trainable_parameter_count"], 1)
            ),
            "wall_time_fraction": float(
                latent_cost["wall_seconds_mean"]
                / max(global_cost["wall_seconds_mean"], 1.0e-12)
            ),
            "sample_exposure_ratio": float(
                latent_cost["sample_exposures"]
                / max(global_cost["sample_exposures"], 1)
            ),
        }
        scale_payload["latent_fraction_available_family_gain_recovered"] = (
            hierarchical_interval(recovered, args.bootstrap, rng)
        )
        sni_path = scale_root / "sni_dense_validation64.json"
        sni = sni_selection(sni_path, tau=1.0, sweep=16)
        scale_payload["sni_fixed_operating_point"] = {
            "tau": sni["tau"],
            "sweep": sni["sweep"],
            "error": hierarchical_interval(sni["error"], args.bootstrap, rng),
            "paired_gain_over_local": hierarchical_interval(
                sni["gain"], args.bootstrap, rng,
            ),
            "inference_seconds_per_case": hierarchical_interval(
                sni["seconds"], args.bootstrap, rng,
            ),
        }
        flat_rows.append({
            "blocks": blocks,
            "method": "sni",
            "adapted_relative_l2": float(sni["error"].mean()),
            "aggregate_mean_error_reduction": float("nan"),
            "paired_casewise_adaptation_gain": float("nan"),
            "gain_over_local": float(sni["gain"].mean()),
            "extension_degradation": float("nan"),
        })
        latent_seconds = float(
            scale_payload["methods"]["latentddm"]
            ["inference_seconds_per_case"]["mean"]
        )
        sni_seconds = float(
            scale_payload["sni_fixed_operating_point"]
            ["inference_seconds_per_case"]["mean"]
        )
        scale_payload["latent_speedup_over_fixed_sni16"] = (
            sni_seconds / max(latent_seconds, 1.0e-12)
        )
        latent_error = float(
            scale_payload["methods"]["latentddm"]["adapted_error"]["mean"]
        )
        operating_points = sni_operating_points(sni_path, tau=1.0)
        scale_payload["sni_accuracy_time_curve"] = [{
            "tau": point["tau"],
            "sweep": point["sweep"],
            "relative_l2_mean": float(point["error"].mean()),
            "error_ratio_to_latentddm": float(
                point["error"].mean() / max(latent_error, 1.0e-12)
            ),
            "seconds_per_case_mean": float(point["seconds"].mean()),
        } for point in operating_points]
        eligible = [
            point for point in operating_points
            if float(point["error"].mean()) <= latent_error
        ]
        if eligible:
            matched = min(
                eligible, key=lambda point: float(point["seconds"].mean()),
            )
            matched_seconds = float(matched["seconds"].mean())
            scale_payload["sni_matched_accuracy"] = {
                "tau": matched["tau"],
                "sweep": matched["sweep"],
                "error": hierarchical_interval(
                    matched["error"], args.bootstrap, rng,
                ),
                "inference_seconds_per_case": hierarchical_interval(
                    matched["seconds"], args.bootstrap, rng,
                ),
                "latent_speedup": matched_seconds / max(
                    latent_seconds, 1.0e-12,
                ),
            }
        else:
            scale_payload["sni_matched_accuracy"] = None
        result["scales"][f"{blocks}x{blocks}"] = scale_payload

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "method_summary.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
