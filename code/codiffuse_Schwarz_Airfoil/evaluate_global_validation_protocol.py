"""Evaluate global Airfoil models on the exact paper validation protocol.

The paper's Local/Core, LatentDDM, and Flux-SNI numbers use five validation
trajectories, eight fixed one-step transitions per trajectory, and two fixed
20-step rollouts per trajectory.  This evaluator applies the same split,
sample locations, finest-grid field metric, and model-owned rollout contract
to independently trained point/anchor (AB-UPT), slice (Transolver), and
full-domain Patch Transformer models.  No designated test trajectory is
opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .compare_head_to_head_v3 import (
    abupt_step,
    global_method_name,
    load_abupt,
    load_global,
    model_input_batch,
    transolver_step,
)
from .dataset import FieldStatistics
from .evaluate_airfoil_sni import metrics
from .final_holdout_contract import (
    FINAL_HOLDOUT_CONTRACT,
    FINAL_HOLDOUT_CONTRACTS,
    is_final_holdout,
    validate_unopened_manifest,
)
from .train_multiresolution_projection import batched
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


MATCHED_DATA_OBJECTIVE_KEYS = (
    "epochs",
    "samples_per_epoch",
    "batch_size",
    "min_epochs",
    "state_weight",
    "vorticity_weight",
    "near_body_weight",
    "quiet_weight",
    "force_weight",
    "validation_samples_per_trajectory",
    "fine_probability",
    "seed",
)

OPTIMIZER_KEYS = ("lr", "min_lr", "lr_patience", "patience")


def fixed_indices(length: int, count: int) -> list[int]:
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    return np.linspace(0, length - 1, min(length, count), dtype=int).tolist()


def rollout_starts(length: int, horizon: int, count: int) -> list[int]:
    maximum = length - horizon
    if maximum < 0:
        raise ValueError("rollout horizon exceeds the paired trajectory")
    if count < 1:
        raise ValueError("rollout count must be positive")
    if count == 1:
        return [maximum // 2]
    return sorted(set(np.linspace(0, maximum, count, dtype=int).tolist()))


def select_evaluation_names(
    checkpoint_split: dict,
    manifest: dict | None = None,
) -> tuple[list[str], str, dict]:
    if manifest is None:
        return list(checkpoint_split["validation"]), "validation", {
            "source": "checkpoint validation split",
            "checkpoint_validation_overlap": len(checkpoint_split["validation"]),
            "checkpoint_test_overlap": 0,
        }
    expected = manifest.get("expected", {})
    if manifest.get("contract") in FINAL_HOLDOUT_CONTRACTS:
        names = list(manifest.get("selected", ()))
        source = "prospectively hashed manifest selected"
        label = "boundary_final_holdout_v3" if (
            manifest.get("contract") == FINAL_HOLDOUT_CONTRACT
        ) else "boundary_final_holdout_v2"
        strict_disjoint = True
    elif "validation" in expected:
        names = list(expected["validation"])
        source = "manifest expected.validation"
        label = "prospective_expanded_validation"
        strict_disjoint = False
    else:
        raise ValueError(
            "evaluation manifest has neither a recognized selected list nor "
            "expected.validation"
        )
    checkpoint_validation = set(checkpoint_split["validation"])
    checkpoint_test = set(checkpoint_split["test"])
    checkpoint_source = {
        name for values in checkpoint_split.values() for name in values
    }
    overlap = set(names) & checkpoint_source
    if overlap and strict_disjoint:
        raise ValueError(
            "prospective evaluation trajectories overlap the global-model "
            f"source split: {sorted(overlap)}"
        )
    if not strict_disjoint:
        names = [name for name in names if name not in checkpoint_source]
    if not names:
        raise ValueError("prospective manifest supplied no disjoint trajectories")
    return names, label, {
        "source": source,
        "checkpoint_validation_overlap": len(set(names) & checkpoint_validation),
        "checkpoint_test_overlap": len(set(names) & checkpoint_test),
        "manifest_candidate_count": len(names),
        "disjoint_evaluation_count": len(names),
    }


def validate_reporting_selection(
    manifest: dict | None, selection_path: Path | None,
) -> dict[str, str] | None:
    """Require a frozen validation choice before opening the final holdout."""
    is_final = is_final_holdout(manifest)
    if not is_final:
        if selection_path is not None:
            raise ValueError("--selection is only valid with the final manifest")
        return None
    if selection_path is None:
        raise ValueError("final-holdout evaluation requires --selection")
    payload = selection_path.read_bytes()
    selection = json.loads(payload)
    selected_sha = selection.get("selected", {}).get("checkpoint_sha256")
    copied_sha = selection.get("copied_checkpoint_sha256")
    if (
        selection.get("contract")
        != "airfoil_boundary_composer_validation_selection_v1"
        or not (selected_sha or copied_sha)
    ):
        raise ValueError("selection does not freeze a reporting checkpoint")
    accepted_sha = {
        selected_sha,
        copied_sha,
    }
    validate_unopened_manifest(
        manifest, selection_path=selection_path,
        selected_checkpoint_hashes=accepted_sha,
    )
    return {
        "path": str(selection_path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def mean_metrics(rows: list[dict]) -> dict[str, float | int]:
    excluded = {"method", "trajectory", "item", "start", "step"}
    keys = sorted({
        key for row in rows for key, value in row.items()
        if key not in excluded and isinstance(value, (int, float))
    })
    result: dict[str, float | int] = {"examples": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_p90"] = float(np.quantile(values, .9))
    return result


def rollout_force_summary(rows: list[dict], horizon: int) -> dict[str, float]:
    """Aggregate force with the same per-window norm as local methods."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["trajectory"]), int(row["start"])), [],
        ).append(row)
    relative = []
    for key, window in groups.items():
        window.sort(key=lambda row: int(row["step"]))
        if [int(row["step"]) for row in window] != list(range(1, horizon + 1)):
            raise ValueError(f"incomplete rollout force series for {key}")
        prediction = np.asarray([
            [row["drag_prediction"], row["lift_prediction"]]
            for row in window
        ], dtype=np.float64)
        reference = np.asarray([
            [row["drag_reference"], row["lift_reference"]]
            for row in window
        ], dtype=np.float64)
        relative.append(float(
            np.linalg.norm(prediction - reference)
            / max(np.linalg.norm(reference), 1.0e-20)
        ))
    values = np.asarray(relative, dtype=np.float64)
    return {
        "force_relative_l2_mean": float(values.mean()),
        "force_relative_l2_p90": float(np.quantile(values, .90)),
        "force_relative_l2_windows": int(values.size),
    }


def validate_training_lineage(
    name: str, checkpoint: dict, checkpoint_path: Path,
) -> dict[str, object]:
    """Verify random initialization, including interrupted continuations.

    The 100-epoch Transolver campaign was continued in its original directory
    after an interrupted allocation.  Its final CLI therefore points
    ``pretrained`` back to the same checkpoint.  A contiguous epoch-1 history
    distinguishes that continuation from external pretraining.  AB-UPT used
    the same continuation protocol but moved its output to a second directory;
    the source checkpoint and combined epoch-1 history are both verified.
    """

    args = checkpoint.get("args", {})
    summary_path = checkpoint_path.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = summary.get("history", [])
    epochs = [int(row["epoch"]) for row in history]
    expected_epochs = list(range(1, int(args["epochs"]) + 1))
    if epochs != expected_epochs:
        raise ValueError(f"{name} training history is not a complete epoch-1 run")
    if int(summary.get("final_epoch", -1)) != int(args["epochs"]):
        raise ValueError(f"{name} did not complete its declared epochs")

    source = args.get("pretrained")
    resumed = bool(args.get("resume_in_place", False))
    if name == "patch_transformer":
        if source is not None or resumed:
            raise ValueError("paper Patch Transformer must start from scratch")
        continuation = "none"
    elif name == "transolver":
        if source is None and not resumed:
            continuation = "none"
        elif resumed and Path(source).resolve() == checkpoint_path.resolve():
            continuation = "same-directory"
        else:
            raise ValueError("Transolver has external pretraining provenance")
    elif name == "point_anchor":
        if source is None:
            continuation = "none"
        else:
            source_path = Path(source)
            if source_path.exists():
                initial = torch.load(
                    source_path, map_location="cpu", weights_only=False,
                )
                if initial.get("args", {}).get("pretrained") is not None:
                    raise ValueError("point/anchor source did not start from scratch")
                for key in (
                    "model_name", "training_contract", "split", "statistics",
                ):
                    if initial.get(key) != checkpoint.get(key):
                        raise ValueError(
                            f"point/anchor continuation changed checkpoint {key}"
                        )
                if int(summary.get("start_epoch", -1)) != int(initial["epoch"]):
                    raise ValueError(
                        "point/anchor continuation epoch is inconsistent"
                    )
                continuation = "moved-directory"
            elif int(summary.get("start_epoch", -1)) == 49:
                # The artifact cleanup retained the selected checkpoint and
                # complete epoch-1..100 history but removed the epoch-49
                # continuation source.  Preserve that limitation explicitly
                # instead of silently treating this as external pretraining.
                continuation = "moved-directory-source-cleaned"
            else:
                raise ValueError(
                    "point/anchor continuation checkpoint is missing and its "
                    "retained history does not identify the expected boundary"
                )
    else:
        raise ValueError(f"unsupported global baseline {name!r}")
    return {
        "trained_from_scratch": True,
        "trained_from_random_initialization": True,
        "continuation_of_same_random_start_run": continuation != "none",
        "continuation_layout": continuation,
        "continuation_source_available": (
            source is None or Path(source).exists()
        ),
        "history_starts_at_epoch": epochs[0],
        "history_ends_at_epoch": epochs[-1],
        "history_epochs": len(epochs),
        "summary": str(summary_path.resolve()),
    }


def timed_prediction(
    method: str,
    model,
    sample: dict,
    previous: torch.Tensor,
    current: torch.Tensor,
    dataset,
    statistics: FieldStatistics,
    device: torch.device,
    context_stride: int | None = None,
) -> tuple[torch.Tensor, float, dict]:
    batch = batched(sample, "fine")
    batch["q_nm1"] = previous
    batch["q_n"] = current
    inference_batch = model_input_batch(batch)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if method == "point_anchor":
        if context_stride is None:
            raise ValueError("point/anchor inference requires a context stride")
        prediction = abupt_step(
            model, inference_batch, dataset, statistics, context_stride, device,
        )
    else:
        prediction = transolver_step(
            model, inference_batch, dataset, statistics, device,
        )
    torch.cuda.synchronize(device)
    return prediction, time.perf_counter() - started, batch


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda")
    checkpoints = {
        "point_anchor": torch.load(
            args.point_anchor, map_location="cpu", weights_only=False,
        ),
        "transolver": torch.load(
            args.transolver, map_location="cpu", weights_only=False,
        ),
        "patch_transformer": torch.load(
            args.patch_transformer, map_location="cpu", weights_only=False,
        ),
    }
    if checkpoints["point_anchor"].get("model_name") not in {
        "dynamic_abupt", "abupt",
    }:
        raise ValueError("--point-anchor is not an AB-UPT checkpoint")
    if global_method_name(checkpoints["transolver"]) != "transolver":
        raise ValueError("--transolver is not a Transolver checkpoint")
    if global_method_name(checkpoints["patch_transformer"]) != "patch_transformer":
        raise ValueError("--patch-transformer is not a global ViT checkpoint")
    if tuple(
        checkpoints["patch_transformer"].get("model_args", {}).get(
            "patch_size", (),
        )
    ) != (32, 32):
        raise ValueError("Airfoil Patch Transformer is not owned-core matched")
    training_lineage = {
        name: validate_training_lineage(name, checkpoints[name], path)
        for name, path in (
            ("point_anchor", args.point_anchor),
            ("transolver", args.transolver),
            ("patch_transformer", args.patch_transformer),
        )
    }
    reference = checkpoints["transolver"]
    for name, checkpoint in checkpoints.items():
        if checkpoint["split"] != reference["split"]:
            raise ValueError(f"{name} uses a different trajectory split")
        if checkpoint["statistics"] != reference["statistics"]:
            raise ValueError(f"{name} uses different normalization statistics")
        if name != "point_anchor" and checkpoint.get("level_policy") != \
                "finest_available":
            raise ValueError(f"{name} must use the finest-available policy")
    matched_training = {
        key: reference["args"].get(key) for key in MATCHED_DATA_OBJECTIVE_KEYS
    }
    for name, checkpoint in checkpoints.items():
        for key, expected in matched_training.items():
            actual = checkpoint["args"].get(key)
            if actual != expected:
                raise ValueError(
                    f"global data/objective mismatch for {key}: "
                    f"Transolver={expected!r}, {name}={actual!r}"
                )

    statistics = FieldStatistics.from_json(reference["statistics"])
    point_model, point_context_stride = load_abupt(
        checkpoints["point_anchor"], device,
    )
    models = {
        "point_anchor": point_model.requires_grad_(False),
        "transolver": load_global(
            checkpoints["transolver"], device,
        ).requires_grad_(False),
        "patch_transformer": load_global(
            checkpoints["patch_transformer"], device,
        ).requires_grad_(False),
    }
    context_strides = {
        "point_anchor": point_context_stride,
        "transolver": None,
        "patch_transformer": None,
    }
    if args.evaluation_manifest is None:
        manifest = None
        manifest_metadata = None
    else:
        manifest_bytes = args.evaluation_manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        manifest_metadata = {
            "path": str(args.evaluation_manifest.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    reporting_selection_metadata = validate_reporting_selection(
        manifest, args.selection,
    )
    evaluation_names, evaluation_split, split_metadata = (
        select_evaluation_names(reference["split"], manifest)
    )
    for name in evaluation_names:
        for level in ("level5", "level6"):
            candidate = args.data_root / name.replace("level5", level)
            if not candidate.exists():
                raise FileNotFoundError(candidate)
    corpus = PairedTrajectoryCorpus(args.data_root, evaluation_names, "all")
    one_step_rows: list[dict] = []
    rollout_rows: list[dict] = []
    try:
        # One untimed prediction per architecture removes first-call setup from
        # the reported resident-model latency.
        first_sample = corpus.paired[0][0]
        first_previous = first_sample["fine_q_nm1"][None].to(device)
        first_current = first_sample["fine_q_n"][None].to(device)
        for name, model in models.items():
            timed_prediction(
                name, model, first_sample, first_previous, first_current,
                corpus.fine[0], statistics, device,
                context_strides[name],
            )

        for trajectory, paired in enumerate(corpus.paired):
            trajectory_name = paired.coarse_path.stem
            dataset = corpus.fine[trajectory]
            for item in fixed_indices(
                len(paired), args.one_step_samples_per_trajectory,
            ):
                sample = paired[item]
                previous = sample["fine_q_nm1"][None].to(device)
                current = sample["fine_q_n"][None].to(device)
                target = sample["fine_q_np1"][None].to(device)
                for name, model in models.items():
                    prediction, seconds, batch = timed_prediction(
                        name, model, sample, previous, current, dataset,
                        statistics, device, context_strides[name],
                    )
                    one_step_rows.append({
                        "method": name,
                        "trajectory": trajectory_name,
                        "item": int(item),
                        **metrics(prediction, target, batch, dataset),
                        "seconds": float(seconds),
                    })

            for start in rollout_starts(
                len(paired), args.horizon, args.rollouts_per_trajectory,
            ):
                initial = paired[start]
                histories = {
                    name: (
                        initial["fine_q_nm1"][None].to(device),
                        initial["fine_q_n"][None].to(device),
                    )
                    for name in models
                }
                for step in range(args.horizon):
                    sample = paired[start + step]
                    target = sample["fine_q_np1"][None].to(device)
                    for name, model in models.items():
                        previous, current = histories[name]
                        prediction, seconds, batch = timed_prediction(
                            name, model, sample, previous, current, dataset,
                            statistics, device, context_strides[name],
                        )
                        rollout_rows.append({
                            "method": name,
                            "trajectory": trajectory_name,
                            "start": int(start),
                            "step": step + 1,
                            **metrics(prediction, target, batch, dataset),
                            "seconds": float(seconds),
                        })
                        histories[name] = (current, prediction)
    finally:
        corpus.close()

    summary = {}
    for name in models:
        one = [row for row in one_step_rows if row["method"] == name]
        rollout = [row for row in rollout_rows if row["method"] == name]
        final = [row for row in rollout if row["step"] == args.horizon]
        rollout_all = mean_metrics(rollout)
        rollout_all.update(rollout_force_summary(rollout, args.horizon))
        summary[name] = {
            "one_step": mean_metrics(one),
            "rollout_all_steps": rollout_all,
            "rollout_final_step": mean_metrics(final),
        }

    return {
        "contract": (
            "airfoil_global_exact_main_validation_v2"
            if manifest is None
            else "airfoil_global_prospective_geometry_validation_v2"
        ),
        "evaluation_split": evaluation_split,
        "evaluation_trajectories": evaluation_names,
        "evaluation_selection": split_metadata,
        "evaluation_manifest": manifest_metadata,
        "validation_selection": reporting_selection_metadata,
        "protected_test_trajectories_opened": 0,
        "final_holdout_trajectories_opened": (
            len(evaluation_names)
            if evaluation_split in {
                "boundary_final_holdout_v2", "boundary_final_holdout_v3",
            } else 0
        ),
        "one_step_selection": (
            f"{args.one_step_samples_per_trajectory} linspace transitions "
            "per trajectory"
        ),
        "rollout_selection": (
            f"{args.rollouts_per_trajectory} linspace starts per trajectory"
        ),
        "one_step_samples_per_trajectory": args.one_step_samples_per_trajectory,
        "rollouts_per_trajectory": args.rollouts_per_trajectory,
        "horizon": args.horizon,
        "field_metric": "full finest L6 physical field relative L2",
        "patch_size": [32, 32],
        "model_owned_rollout": True,
        "rollout_force_metric": (
            "per-window relative L2 over the complete signed drag/lift series"
        ),
        "future_reference_is_model_input": False,
        "one_untimed_warmup_per_model": True,
        "matched_data_and_objective_hyperparameters": matched_training,
        "architecture_specific_optimizer_hyperparameters": {
            name: {
                key: checkpoint["args"].get(key) for key in OPTIMIZER_KEYS
            }
            for name, checkpoint in checkpoints.items()
        },
        "checkpoints": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "contract": checkpoints[name].get("training_contract"),
                "epoch": int(checkpoints[name]["epoch"]),
                "parameters": sum(
                    parameter.numel() for parameter in models[name].parameters()
                ),
                **training_lineage[name],
                "uses_pou": False,
            }
            for name, path in (
                ("point_anchor", args.point_anchor),
                ("transolver", args.transolver),
                ("patch_transformer", args.patch_transformer),
            )
        },
        "summary": summary,
        "rows": {"one_step": one_step_rows, "rollout": rollout_rows},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--point-anchor", type=Path, required=True)
    parser.add_argument("--transolver", type=Path, required=True)
    parser.add_argument("--patch-transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-manifest", type=Path,
        help=(
            "optional expanded design manifest; expected.validation entries "
            "overlapping checkpoint validation/test are excluded"
        ),
    )
    parser.add_argument(
        "--selection", type=Path,
        help="validation-only LatentDDM selection required for final holdout",
    )
    parser.add_argument("--one-step-samples-per-trajectory", type=int, default=8)
    parser.add_argument("--rollouts-per-trajectory", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    payload = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
