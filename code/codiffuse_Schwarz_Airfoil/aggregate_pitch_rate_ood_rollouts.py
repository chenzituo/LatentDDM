"""Aggregate the matched ID/OOD Airfoil H20 evaluation matrix.

The aggregation unit is trajectory, not frame.  For adapted models, seed means
receive equal weight and uncertainty resamples both optimization/subset seeds
and trajectories.  Force error is one relative norm over each complete rollout
window, which remains well conditioned when instantaneous lift or drag crosses
zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


METHODS = ("latent", "point", "slice", "patch")
SEEDS = (20260814, 20260815, 20260816)
BUDGETS = (0, 1, 2, 4, 8, 16)
REGIMES = ("id", "ood")
HORIZON = 20
TARGET_ADAPTATION_SAMPLES_PER_EPOCH = 256
STATE_LATENT_CONTRACT = "airfoil_absolute_robin_state_latentddm_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-part", choices=("validation", "test"), default="validation",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--zero-shot-only", action="store_true",
        help="require and aggregate only the frozen budget-zero comparison",
    )
    parser.add_argument(
        "--zero-shot-freeze", type=Path,
        help="validation-frozen checkpoint manifest required for zero-shot test",
    )
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def expected_path(
    root: Path, method: str, budget: int, seed: int, regime: str,
    split_part: str,
) -> Path:
    return (
        root / method / f"budget{budget:02d}" / f"seed{seed}"
        / f"{regime}_{split_part}.json"
    )


def normalize_rows(
    document: dict[str, object], *, method: str, seed: int,
    split_part: str,
) -> list[dict[str, object]]:
    expected_split = f"manifest_{split_part}"
    if document.get("split") != expected_split:
        raise ValueError(
            f"{method} artifact declares {document.get('split')!r}, "
            f"expected {expected_split!r}"
        )
    protected = int(document.get("protected_test_trajectories_opened", -1))
    if split_part == "validation" and protected != 0:
        raise ValueError(f"{method} validation opened protected trajectories")
    if split_part == "test" and protected <= 0:
        raise ValueError(f"{method} test report did not declare protected access")
    if int(document.get("horizon", -1)) != HORIZON:
        raise ValueError(f"{method} evaluation is not H{HORIZON}")
    if int(document.get("starts_per_trajectory", -1)) != 3:
        raise ValueError(f"{method} evaluation does not use three starts")
    if document.get("model_owned_rollout") is not True:
        raise ValueError(f"{method} evaluation is not model-owned")
    if document.get("reference_used_by_inference") is not False:
        raise ValueError(f"{method} inference used a reference field")
    contract = str(document.get("contract"))
    if contract == "airfoil_fixed_cadence_global_rollout_v1":
        source = document["rows"]["rollout"]
        field_time = "seconds"
        drag = "drag"
        lift = "lift"
    elif contract in (
        "airfoil_absolute_robin_latentddm_rollout_v1",
        "airfoil_absolute_robin_rollout_v1",
        "airfoil_latentddm_rollout_v2",
        "airfoil_local_sni_rollout_v2",
    ):
        source = document["records"]
        field_time = "inference_seconds"
        drag = "c_drag"
        lift = "c_lift"
    else:
        raise ValueError(f"unknown rollout contract {contract}")
    rows = []
    for row in source:
        rows.append({
            "seed": int(seed),
            "trajectory": str(row.get("trajectory_name", row["trajectory"])),
            "start": int(row["start"]),
            "step": int(row["step"]),
            "field": float(row["field_relative_l2"]),
            "vorticity": float(row["vorticity_relative_l2"]),
            "seconds": float(row[field_time]),
            "drag_prediction": float(row[f"{drag}_prediction"]),
            "drag_reference": float(row[f"{drag}_reference"]),
            "lift_prediction": float(row[f"{lift}_prediction"]),
            "lift_reference": float(row[f"{lift}_reference"]),
        })
    return rows


def rollout_identity(
    rows: Iterable[dict[str, object]], *, seed: int,
) -> tuple[tuple[str, int, int], ...]:
    identity = tuple(sorted(
        (
            str(row["trajectory"]), int(row["start"]), int(row["step"]),
        )
        for row in rows if int(row["seed"]) == seed
    ))
    if not identity or len(set(identity)) != len(identity):
        raise ValueError(f"invalid rollout identity for seed {seed}")
    windows: dict[tuple[str, int], list[int]] = {}
    for trajectory, start, step in identity:
        windows.setdefault((trajectory, start), []).append(step)
    if any(steps != list(range(1, HORIZON + 1)) for steps in windows.values()):
        raise ValueError(f"incomplete H{HORIZON} rollout identity for seed {seed}")
    return identity


def validate_matched_units(
    matrix: dict[str, dict[str, dict[int, list[dict[str, object]]]]],
) -> None:
    """Require identical trajectory/start/step units across every comparator."""
    for regime in REGIMES:
        for budget in BUDGETS:
            seeds = SEEDS[:1] if budget == 0 else SEEDS
            for seed in seeds:
                identities: dict[str, tuple[tuple[str, int, int], ...]] = {}
                for method in METHODS:
                    rows = matrix[regime][method].get(budget, [])
                    if any(int(row["seed"]) == seed for row in rows):
                        identities[method] = rollout_identity(rows, seed=seed)
                if budget == 0:
                    sni_rows = matrix[regime]["sni"].get(0, [])
                    if any(int(row["seed"]) == seed for row in sni_rows):
                        identities["sni"] = rollout_identity(
                            sni_rows, seed=seed,
                        )
                if len(identities) < 2:
                    continue
                reference_name = sorted(identities)[0]
                reference = identities[reference_name]
                for method, identity in identities.items():
                    if identity != reference:
                        raise ValueError(
                            f"unmatched {regime} budget{budget:02d} seed{seed} "
                            f"rollout units: {method} versus {reference_name}"
                        )


def scalar_clusters(
    rows: Iterable[dict[str, object]], metric: str, *, step: int | None,
) -> dict[int, dict[str, float]]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if step is not None and int(row["step"]) != step:
            continue
        grouped.setdefault(
            (int(row["seed"]), str(row["trajectory"])), [],
        ).append(float(row[metric]))
    result: dict[int, dict[str, float]] = {}
    for (seed, trajectory), values in grouped.items():
        result.setdefault(seed, {})[trajectory] = float(np.mean(values))
    return result


def force_clusters(
    rows: Iterable[dict[str, object]], *, horizon: int,
) -> dict[int, dict[str, float]]:
    windows = force_window_errors(rows, horizon=horizon)
    trajectory_values: dict[tuple[int, str], list[float]] = {}
    for (seed, trajectory, _), value in windows.items():
        trajectory_values.setdefault((seed, trajectory), []).append(value)
    result: dict[int, dict[str, float]] = {}
    for (seed, trajectory), values in trajectory_values.items():
        result.setdefault(seed, {})[trajectory] = float(np.mean(values))
    return result


def force_window_errors(
    rows: Iterable[dict[str, object]], *, horizon: int,
) -> dict[tuple[int, str, int], float]:
    """Return one force-vector relative norm for each complete rollout."""
    windows: dict[tuple[int, str, int], list[dict[str, object]]] = {}
    for row in rows:
        if int(row["step"]) <= horizon:
            windows.setdefault((
                int(row["seed"]), str(row["trajectory"]), int(row["start"]),
            ), []).append(row)
    result: dict[tuple[int, str, int], float] = {}
    for (seed, trajectory, start), window in windows.items():
        window.sort(key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in window]
        if steps != list(range(1, horizon + 1)):
            raise ValueError(
                f"incomplete H{horizon} force window: "
                f"seed={seed}, trajectory={trajectory}, start={start}"
            )
        prediction = np.asarray([
            [row["drag_prediction"], row["lift_prediction"]]
            for row in window
        ], dtype=np.float64)
        reference = np.asarray([
            [row["drag_reference"], row["lift_reference"]]
            for row in window
        ], dtype=np.float64)
        result[(seed, trajectory, start)] = float(
            np.linalg.norm(prediction - reference)
            / max(np.linalg.norm(reference), 1.0e-20)
        )
    return result


def scalar_relative_gain_clusters(
    baseline: Iterable[dict[str, object]],
    adapted: Iterable[dict[str, object]],
    metric: str,
    *,
    step: int,
) -> dict[int, dict[str, float]]:
    """Pair endpoint errors by trajectory, rollout start, and horizon."""
    baseline_index = {
        (str(row["trajectory"]), int(row["start"]), int(row["step"])):
        float(row[metric])
        for row in baseline if int(row["step"]) == step
    }
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in adapted:
        if int(row["step"]) != step:
            continue
        key = (str(row["trajectory"]), int(row["start"]), step)
        if key not in baseline_index:
            raise ValueError(f"adapted {metric} row has no zero-shot pair: {key}")
        zero = baseline_index[key]
        gain = (zero - float(row[metric])) / max(zero, 1.0e-20)
        grouped.setdefault(
            (int(row["seed"]), str(row["trajectory"])), [],
        ).append(gain)
    if not grouped:
        raise ValueError(f"paired {metric} gain has no rows")
    result: dict[int, dict[str, float]] = {}
    for (seed, trajectory), values in grouped.items():
        result.setdefault(seed, {})[trajectory] = float(np.mean(values))
    return result


def force_relative_gain_clusters(
    baseline: Iterable[dict[str, object]],
    adapted: Iterable[dict[str, object]],
    *,
    horizon: int,
) -> dict[int, dict[str, float]]:
    baseline_windows = force_window_errors(baseline, horizon=horizon)
    baseline_index: dict[tuple[str, int], float] = {}
    for (_, trajectory, start), value in baseline_windows.items():
        key = trajectory, start
        if key in baseline_index and not np.isclose(baseline_index[key], value):
            raise ValueError("zero-shot force windows disagree across duplicate seeds")
        baseline_index[key] = value
    adapted_windows = force_window_errors(adapted, horizon=horizon)
    grouped: dict[tuple[int, str], list[float]] = {}
    for (seed, trajectory, start), value in adapted_windows.items():
        key = trajectory, start
        if key not in baseline_index:
            raise ValueError(f"adapted force window has no zero-shot pair: {key}")
        zero = baseline_index[key]
        grouped.setdefault((seed, trajectory), []).append(
            (zero - value) / max(zero, 1.0e-20)
        )
    if not grouped:
        raise ValueError("paired force gain has no windows")
    result: dict[int, dict[str, float]] = {}
    for (seed, trajectory), values in grouped.items():
        result.setdefault(seed, {})[trajectory] = float(np.mean(values))
    return result


def hierarchical_summary(
    clusters: dict[int, dict[str, float]], *, samples: int, seed: int,
) -> dict[str, float | int]:
    if not clusters or any(not values for values in clusters.values()):
        raise ValueError("cannot summarize empty trajectory clusters")
    seed_keys = sorted(clusters)
    point = float(np.mean([
        np.mean(list(clusters[key].values())) for key in seed_keys
    ]))
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = generator.choice(
            seed_keys, size=len(seed_keys), replace=True,
        )
        seed_means = []
        for sampled in sampled_seeds:
            values = np.asarray(
                list(clusters[int(sampled)].values()), dtype=np.float64,
            )
            seed_means.append(float(np.mean(generator.choice(
                values, size=values.size, replace=True,
            ))))
        draws[index] = np.mean(seed_means)
    return {
        "mean": point,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "seed_count": len(seed_keys),
        "trajectory_clusters": int(sum(map(len, clusters.values()))),
    }


def summarize_rows(
    rows: list[dict[str, object]], *, samples: int, identity: tuple[object, ...],
    step: int = HORIZON,
) -> dict[str, object]:
    return {
        metric: hierarchical_summary(
            scalar_clusters(rows, metric, step=step), samples=samples,
            seed=stable_seed(*identity, metric, step),
        )
        for metric in ("field", "vorticity")
    } | {
        "force": hierarchical_summary(
            force_clusters(rows, horizon=step), samples=samples,
            seed=stable_seed(*identity, "force", step),
        ),
        "seconds_per_step": hierarchical_summary(
            scalar_clusters(rows, "seconds", step=None), samples=samples,
            seed=stable_seed(*identity, "seconds"),
        ),
    }


def summarize_scalar_step(
    rows: list[dict[str, object]], *, samples: int,
    identity: tuple[object, ...], step: int,
) -> dict[str, object]:
    """Summarize field diagnostics at one declared rollout step.

    The adaptation headline remains terminal H20.  This companion summary
    makes H1 negative transfer auditable without silently changing the
    deployment-oriented checkpoint or publication gate.
    """
    return {
        metric: hierarchical_summary(
            scalar_clusters(rows, metric, step=step), samples=samples,
            seed=stable_seed(*identity, metric, step),
        )
        for metric in ("field", "vorticity")
    }


def summarize_scalar_relative_gains(
    baseline: list[dict[str, object]],
    adapted: list[dict[str, object]], *, samples: int,
    identity: tuple[object, ...], step: int,
) -> dict[str, object]:
    return {
        metric: hierarchical_summary(
            scalar_relative_gain_clusters(
                baseline, adapted, metric, step=step,
            ),
            samples=samples,
            seed=stable_seed(
                *identity, "paired_relative_gain", metric, step,
            ),
        )
        for metric in ("field", "vorticity")
    }


def summarize_relative_gains(
    baseline: list[dict[str, object]],
    adapted: list[dict[str, object]],
    *,
    samples: int,
    identity: tuple[object, ...],
    step: int = HORIZON,
) -> dict[str, object]:
    result = {}
    for metric in ("field", "vorticity"):
        result[metric] = hierarchical_summary(
            scalar_relative_gain_clusters(
                baseline, adapted, metric, step=step,
            ),
            samples=samples,
            seed=stable_seed(*identity, "paired_relative_gain", metric, step),
        )
    result["force"] = hierarchical_summary(
        force_relative_gain_clusters(
            baseline, adapted, horizon=step,
        ),
        samples=samples,
        seed=stable_seed(*identity, "paired_relative_gain", "force", step),
    )
    return result


def load_matrix(
    root: Path, *, split_part: str, allow_incomplete: bool,
    budgets: tuple[int, ...] = BUDGETS,
) -> tuple[dict[str, dict[str, dict[int, list[dict[str, object]]]]], list[str]]:
    matrix = {
        regime: {method: {} for method in (*METHODS, "sni")}
        for regime in REGIMES
    }
    missing = []
    seen_checkpoints: dict[tuple[str, str, int], set[str]] = {}
    for method in METHODS:
        for budget in budgets:
            seeds = SEEDS[:1] if budget == 0 else SEEDS
            for seed in seeds:
                for regime in REGIMES:
                    path = expected_path(
                        root, method, budget, seed, regime, split_part,
                    )
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    document = json.loads(path.read_text(encoding="utf-8"))
                    checkpoint = str(document.get("checkpoint_sha256", ""))
                    identity = (regime, method, budget)
                    seen = seen_checkpoints.setdefault(identity, set())
                    if checkpoint and checkpoint in seen:
                        continue
                    if checkpoint:
                        seen.add(checkpoint)
                    matrix[regime][method].setdefault(budget, []).extend(
                        normalize_rows(
                            document, method=method, seed=seed,
                            split_part=split_part,
                        )
                    )
    for regime in REGIMES:
        path = root / "sni" / f"{regime}_{split_part}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        matrix[regime]["sni"][0] = normalize_rows(
            document, method="sni", seed=SEEDS[0], split_part=split_part,
        )
    if missing and not allow_incomplete:
        raise FileNotFoundError(
            f"evaluation matrix is missing {len(missing)} files; first: {missing[0]}"
        )
    validate_matched_units(matrix)
    return matrix, missing


def adaptation_seconds(
    experiment_root: Path, method: str, budget: int, seed: int,
) -> float | None:
    """Return measured target-training wall time, including validation."""
    if budget == 0 or method == "sni":
        return 0.0
    metadata = adaptation_run_metadata(
        experiment_root, method, budget, seed,
    )
    return None if metadata is None else float(metadata["seconds"])


def adaptation_run_metadata(
    experiment_root: Path, method: str, budget: int, seed: int,
) -> dict[str, float | int] | None:
    """Read one seed's measured optimization and trainable-size contract."""
    if budget <= 0 or method == "sni":
        return None
    if method == "latent":
        selection_path = experiment_root / "source_latent_family_selected_v1.json"
        family_contract = None
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            family_contract = selection.get("selected", {}).get(
                "composer_contract"
            )
        family_directory = (
            "target_state_latentddm_adapt_v1"
            if family_contract == STATE_LATENT_CONTRACT else
            "target_latentddm_adapt_v2"
        )
        directory = (
            experiment_root / family_directory
            / f"budget{budget:02d}" / f"seed{seed}"
        )
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            return None
        document = json.loads(summary_path.read_text(encoding="utf-8"))
        if document.get("contract") not in (
            "airfoil_latentddm_training_summary_v1",
            "airfoil_state_latentddm_training_summary_v1",
        ):
            raise ValueError("LatentDDM adaptation summary has the wrong contract")
        return {
            "seconds": float(document.get(
                "adaptation_seconds_total",
                document["training_seconds_total"],
            )),
            "trainable_parameter_count": int(
                document["trainable_parameter_count"]
            ),
            "epochs_completed": int(document["epochs_completed"]),
            "samples_per_epoch": int(document["samples_per_epoch"]),
            "sample_exposures": int(document["sample_exposures"]),
        }
    path = (
        experiment_root / "target_global_adapt_v1" / method
        / f"budget{budget:02d}" / f"seed{seed}" / "summary.json"
    )
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    value = document.get("training_seconds_total")
    if value is None:
        return None
    epochs = int(document["epochs_completed"])
    return {
        "seconds": float(value),
        "trainable_parameter_count": int(document["parameter_count"]),
        "epochs_completed": epochs,
        "samples_per_epoch": TARGET_ADAPTATION_SAMPLES_PER_EPOCH,
        "sample_exposures": epochs * TARGET_ADAPTATION_SAMPLES_PER_EPOCH,
    }


def timing_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "seed_count": int(array.size),
    }


def adaptation_contract_summary(
    records: list[dict[str, float | int]],
) -> dict[str, object]:
    if not records:
        raise ValueError("adaptation contract has no seed records")
    parameters = {int(record["trainable_parameter_count"]) for record in records}
    per_epoch = {int(record["samples_per_epoch"]) for record in records}
    if len(parameters) != 1 or len(per_epoch) != 1:
        raise ValueError("adaptation seeds disagree on model or exposure contract")
    return {
        "trainable_parameter_count": parameters.pop(),
        "samples_per_epoch": per_epoch.pop(),
        "seconds": timing_summary([
            float(record["seconds"]) for record in records
        ]),
        "epochs_completed": timing_summary([
            float(record["epochs_completed"]) for record in records
        ]),
        "sample_exposures": timing_summary([
            float(record["sample_exposures"]) for record in records
        ]),
    }


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    if args.bootstrap_samples < 100:
        raise ValueError("use at least 100 bootstrap samples")
    matrix, missing = load_matrix(
        args.evaluation_root, split_part=args.split_part,
        allow_incomplete=args.allow_incomplete,
        budgets=(0,) if args.zero_shot_only else BUDGETS,
    )
    curves: dict[str, object] = {}
    rollout: dict[str, object] = {}
    experiment_root = args.evaluation_root.parent
    deployment_selection = None
    if args.split_part == "test":
        if args.zero_shot_only:
            if args.zero_shot_freeze is None or not args.zero_shot_freeze.is_file():
                raise FileNotFoundError(
                    "zero-shot test aggregate requires its validation freeze"
                )
            deployment_selection = json.loads(
                args.zero_shot_freeze.read_text(encoding="utf-8")
            )
            if (
                deployment_selection.get("contract")
                != "airfoil_pitch_rate_ood_zero_shot_freeze_v1"
                or int(deployment_selection.get(
                    "protected_test_trajectories_opened", -1,
                )) != 0
            ):
                raise ValueError("zero-shot checkpoint map is not validation-only")
        else:
            selection_path = (
                args.evaluation_root / "adaptation_selection_validation.json"
            )
            if not selection_path.is_file():
                raise FileNotFoundError(
                    "protected aggregate requires validation-frozen adaptation selection"
                )
            deployment_selection = json.loads(
                selection_path.read_text(encoding="utf-8")
            )
            if (
                deployment_selection.get("contract")
                != "airfoil_pitch_rate_ood_adaptation_selection_v1"
                or int(deployment_selection.get(
                    "protected_test_trajectories_opened", -1,
                )) != 0
            ):
                raise ValueError("adaptation checkpoint map is not validation-only")
    for regime in REGIMES:
        curves[regime] = {}
        rollout[regime] = {}
        for method in (*METHODS, "sni"):
            curves[regime][method] = {}
            baseline = matrix[regime][method].get(0)
            for budget, rows in sorted(matrix[regime][method].items()):
                entry = summarize_rows(
                    rows, samples=args.bootstrap_samples,
                    identity=(regime, method, budget, "endpoint"),
                )
                entry["h1_diagnostic"] = summarize_scalar_step(
                    rows, samples=args.bootstrap_samples,
                    identity=(regime, method, budget, "h1_diagnostic"),
                    step=1,
                )
                timing_seeds = SEEDS[:1] if budget == 0 else SEEDS
                timing = [
                    adaptation_seconds(experiment_root, method, budget, seed)
                    for seed in timing_seeds
                ]
                if all(value is not None for value in timing):
                    entry["adaptation_seconds"] = timing_summary([
                        float(value) for value in timing if value is not None
                    ])
                if budget > 0:
                    metadata = [
                        adaptation_run_metadata(
                            experiment_root, method, budget, seed,
                        )
                        for seed in timing_seeds
                    ]
                    if all(value is not None for value in metadata):
                        entry["adaptation_contract"] = adaptation_contract_summary([
                            value for value in metadata if value is not None
                        ])
                    if baseline is None:
                        raise ValueError(
                            f"{regime}/{method} has adaptation without zero-shot rows"
                        )
                    entry["paired_relative_gain"] = summarize_relative_gains(
                        baseline, rows, samples=args.bootstrap_samples,
                        identity=(regime, method, budget),
                    )
                    entry["paired_relative_gain_h1"] = (
                        summarize_scalar_relative_gains(
                            baseline, rows, samples=args.bootstrap_samples,
                            identity=(regime, method, budget, "h1_diagnostic"),
                            step=1,
                        )
                    )
                curves[regime][method][str(budget)] = entry
            endpoint_budget = (
                0 if args.zero_shot_only or method == "sni" else 16
            )
            rows = matrix[regime][method].get(endpoint_budget)
            if rows:
                rollout[regime][method] = {
                    f"step_{step:02d}": summarize_rows(
                        rows, samples=args.bootstrap_samples,
                        identity=(regime, method, endpoint_budget, "rollout"),
                        step=step,
                    )
                    for step in range(1, HORIZON + 1)
                }
    result = {
        "contract": "airfoil_pitch_rate_ood_rollout_aggregate_v1",
        "evaluation_root": str(args.evaluation_root.resolve()),
        "complete": not missing,
        "missing": missing,
        "report_split": args.split_part,
        "protected_test_evaluation": args.split_part == "test",
        "horizon": HORIZON,
        "budgets": (0,) if args.zero_shot_only else BUDGETS,
        "zero_shot_only": bool(args.zero_shot_only),
        "methods": (*METHODS, "sni"),
        "uncertainty": (
            "95% hierarchical bootstrap over subset/optimization seed and "
            "trajectory; starts remain within their trajectory cluster"
        ),
        "field_metric": "fluid-mask relative L2, identical for all methods",
        "force_metric": "relative L2 over each model-owned rollout window",
        "adaptation_selection": deployment_selection,
        "curves": curves,
        "rollout": rollout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = aggregate(parse_args())
    print(json.dumps({
        "complete": result["complete"],
        "missing": len(result["missing"]),
        "output": result["contract"],
    }), flush=True)


if __name__ == "__main__":
    main()
