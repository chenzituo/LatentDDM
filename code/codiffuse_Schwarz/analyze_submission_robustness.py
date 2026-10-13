"""Audit evaluation uncertainty and experimental units for the submission.

The audit never opens a protected test split.  It aggregates correlated
Airfoil frames by trajectory and Car queries by vehicle before bootstrapping;
individual cells or points are not treated as independent observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "field_relative_l2",
    "vorticity_relative_l2",
    "interface_relative_l2",
    "divergence_rms",
    "force_relative_l2",
    "drag_relative_abs",
    "lift_relative_abs",
)


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def bootstrap_mean(
    values: np.ndarray, *, seed: int, draws: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or draws <= 0:
        raise ValueError("bootstrap requires a nonempty vector and draws > 0")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, .025)),
        "ci95_high": float(np.quantile(samples, .975)),
        "probability_mean_below_zero": float(np.mean(samples < 0.0)),
        "experimental_units": int(values.size),
        "bootstrap_draws": int(draws),
    }


def means_by_unit(
    rows: list[dict], unit: str, metric: str,
) -> dict[str, float]:
    units = sorted({str(row[unit]) for row in rows})
    return {
        value: float(np.mean([
            float(row[metric]) for row in rows if str(row[unit]) == value
        ]))
        for value in units
    }


def airfoil_rows(local: dict, global_: dict, regime: str) -> dict[str, list[dict]]:
    names = [Path(name).stem for name in local["evaluation_trajectories"]]
    local_rows = {}
    for method in ("local", "latentddm"):
        local_rows[method] = [
            {**row, "trajectory_name": names[int(row["trajectory"])]}
            for row in local["records"][regime][method]
        ]
    if regime == "one_step":
        rows = global_["rows"]["one_step"]
    else:
        rows = global_["rows"]["rollout"]
        if regime == "step20":
            rows = [row for row in rows if int(row["step"]) == 20]
    for method in ("transolver", "patch_transformer"):
        local_rows[method] = [
            {**row, "trajectory_name": str(row["trajectory"])}
            for row in rows if row["method"] == method
        ]
    return local_rows


def airfoil_audit(
    local: dict, global_: dict, *, seed: int, draws: int,
) -> dict:
    if local["protected_test_trajectories_opened"] != 0 \
            or global_["protected_test_trajectories_opened"] != 0:
        raise ValueError("Airfoil audit source opened protected test data")
    expected = {Path(name).stem for name in local["evaluation_trajectories"]}
    global_names = {Path(name).stem for name in global_["evaluation_trajectories"]}
    if expected != global_names:
        raise ValueError("local and global Airfoil trajectory sets differ")
    regimes = {}
    for regime_index, regime in enumerate(("one_step", "all_step", "step20")):
        rows = airfoil_rows(local, global_, regime)
        result = {}
        for metric_index, metric in enumerate(METRICS):
            by_method = {
                method: means_by_unit(values, "trajectory_name", metric)
                for method, values in rows.items()
            }
            if any(set(values) != expected for values in by_method.values()):
                raise ValueError(f"{regime}/{metric}: incomplete trajectory pairing")
            reference = by_method["latentddm"]
            result[metric] = {
                method: bootstrap_mean(
                    np.asarray([
                        values[name] - reference[name] for name in sorted(expected)
                    ]),
                    seed=seed + 1009 * regime_index + 31 * metric_index
                    + 7919 * method_index,
                    draws=draws,
                )
                for method_index, (method, values) in enumerate(by_method.items())
                if method != "latentddm"
            }
        regimes[regime] = result
    return {
        "evaluation_split": global_["evaluation_split"],
        "experimental_unit": "trajectory",
        "trajectory_count": len(expected),
        "method_minus_latentddm": regimes,
    }


def car_audit(*, seed: int, draws: int) -> dict:
    local = load(
        "codiffuse_Schwarz_Car/outputs/volume_state_gated_weight_retraction_v1/"
        "full_volume_validation_lineage/validation_metrics.json"
    )
    patch = load(
        "codiffuse_Schwarz_Car/outputs/global_vit_fullvolume_stablepos_v1/"
        "validation_metrics.json"
    )
    transolver = load(
        "codiffuse_Schwarz_Car/outputs/global_point_v1/validation_metrics_v1.json"
    )
    if local["protected_test_accessed"] or patch["protected_test_accessed"] \
            or transolver["protected_test_accessed"]:
        raise ValueError("Car audit source opened protected test data")
    local_rows = {row["design_id"]: row for row in local["per_design"]}
    patch_rows = {row["design_id"]: row for row in patch["records"]}
    trans_rows = {
        row["design_id"]: row for row in transolver["per_design"]
        if row["query_space"] == "native_cfd"
    }
    designs = sorted(local_rows)
    if set(designs) != set(patch_rows) or set(designs) != set(trans_rows):
        raise ValueError("Car global/local vehicle sets differ")
    native = {
        "local": np.asarray([
            local_rows[name]["local_native_magnitude_rel_l2"] for name in designs
        ]),
        "latentddm": np.asarray([
            local_rows[name]["damped_native_magnitude_rel_l2"] for name in designs
        ]),
        "transolver": np.asarray([
            trans_rows[name]["magnitude_rel_l2"] for name in designs
        ]),
        "patch_transformer": np.asarray([
            patch_rows[name]["native_speed_relative_l2"] for name in designs
        ]),
    }
    paired = {
        method: bootstrap_mean(
            values - native["latentddm"],
            seed=seed + 10007 + 101 * index, draws=draws,
        )
        for index, (method, values) in enumerate(native.items())
        if method != "latentddm"
    }
    local_minus_latent = {}
    for index, (label, local_key, latent_key) in enumerate((
        ("wake", "local_wake_magnitude_rel_l2", "damped_wake_magnitude_rel_l2"),
        ("overlap", "local_overlap_rms_normalized", "damped_overlap_rms_normalized"),
        ("divergence", "local_divergence_rms", "damped_divergence_rms"),
    )):
        differences = np.asarray([
            local_rows[name][latent_key] - local_rows[name][local_key]
            for name in designs
        ])
        local_minus_latent[label] = bootstrap_mean(
            differences, seed=seed + 20011 + index, draws=draws,
        )
    return {
        "evaluation_split": "validation",
        "experimental_unit": "vehicle design",
        "design_count": len(designs),
        "native_method_minus_latentddm": paired,
        "latentddm_minus_local_secondary": local_minus_latent,
    }


def darcy_audit(*, seed: int, draws: int) -> dict:
    sni = load(
        "codiffuse_Schwarz_Darcy/outputs/darcy_weighted_pou_sni_two_init_v1.json"
    )
    if sni["protected_test_cases_opened"] != 0:
        raise ValueError("Darcy SNI opened protected test data")
    patch_path = ROOT / (
        "codiffuse_Schwarz_Darcy/outputs/"
        "global_vit_face_v2_stablepos_seed20260805/validation_only_250_gpu/"
        "iteration_metrics.csv"
    )
    with patch_path.open(newline="", encoding="utf-8") as stream:
        patch_rows = [row for row in csv.DictReader(stream) if int(row["sweep"]) == 1]
    if len(patch_rows) != 250:
        raise ValueError("Darcy Patch audit is not the full 250-case validation")
    patch_values = np.asarray([float(row["relative_l2"]) for row in patch_rows])
    return {
        "evaluation_split": "validation",
        "patch_evaluation_cases": 250,
        "patch_relative_l2": bootstrap_mean(
            patch_values, seed=seed + 30011, draws=draws,
        ),
        "transolver_training_seeds": 3,
        "patch_training_seeds": 1,
        "init_sni_zero_identity_max_abs": float(
            sni["local_pou_equals_pou_sni_sweep0_max_abs"]
        ),
        "zero_sni_one_sweep_identity_max_abs": float(
            sni["tau1_zero_sweep1_equals_local_pou_max_abs"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz/workshop/submission/"
            "experimental_robustness_audit.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()
    exact_local = load(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "global_response_rank16_onpolicy_rollin3_lr5e5_v1/"
        "local_latent_main_validation_clean.json"
    )
    exact_global = load(
        "codiffuse_Schwarz_Airfoil/outputs/global100_v1_finest/"
        "global_exact_main_validation_v1.json"
    )
    prospective_local = load(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "global_response_rank16_onpolicy_rollin3_lr5e5_v1/"
        "local_latent_expanded21_clean.json"
    )
    prospective_global = load(
        "codiffuse_Schwarz_Airfoil/outputs/global100_v1_finest/"
        "global_prospective_geometry_validation_v1.json"
    )
    result = {
        "contract": "latentddm_submission_experimental_robustness_audit_v1",
        "protected_test_sets_opened": 0,
        "bootstrap_seed": args.seed,
        "bootstrap_draws": args.bootstrap_draws,
        "airfoil_checkpoint_validation": airfoil_audit(
            exact_local, exact_global, seed=args.seed, draws=args.bootstrap_draws,
        ),
        "airfoil_prospective_geometries": airfoil_audit(
            prospective_local, prospective_global,
            seed=args.seed + 50021, draws=args.bootstrap_draws,
        ),
        "car": car_audit(seed=args.seed, draws=args.bootstrap_draws),
        "darcy": darcy_audit(seed=args.seed, draws=args.bootstrap_draws),
        "optimization_seed_scope": {
            "darcy_transolver": 3,
            "darcy_patch_transformer": 1,
            "car_global_and_latent_models": 1,
            "airfoil_global_and_latent_models": 1,
            "interpretation": (
                "paired intervals quantify evaluation-unit uncertainty, not "
                "optimization-seed uncertainty"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "protected_test_sets_opened": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
