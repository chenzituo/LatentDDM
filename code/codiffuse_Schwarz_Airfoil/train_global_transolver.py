"""Train a full-field global baseline on two-history airfoil dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics
from codiffuse_Schwarz_Airfoil.expanded_protocol import write_or_validate_snapshot
from codiffuse_Schwarz_Airfoil.force_diagnostics import (
    force_coefficient_relative_square,
)
from codiffuse_Schwarz_Airfoil.train_global_abupt import (
    INPUT_CHANNELS,
    make_inputs,
    masked_relative_square,
    per_sample_relative,
    physical_state,
    physical_loss_terms as loss_terms,
    relative_square,
    sha256,
    trajectory_batch,
)
from codiffuse_Schwarz_Airfoil.trajectory_corpus import (
    TrajectoryCorpus,
    write_or_validate_split,
)
from codiffuse_Schwarz_Airfoil.training_resume import (
    atomic_torch_save,
    atomic_write_json,
    merged_training_history,
    require_complete_training_prefix,
)
from codiffuse_Schwarz_Airfoil.vorticity_projection import periodic_vorticity
from src.neural_eos.transolver import StructuredTransolverOperator
from codiffuse_Schwarz.global_vit import GlobalPatchTransformer


Tensor = torch.Tensor
TRANSOLVER_INPUT_CHANNELS = INPUT_CHANNELS + (
    "log2_dx_relative", "normalized_dt_over_dx", "viscous_step_scale",
)
BALANCED_LEVEL_CONTRACT = "global_two_state_structured_transolver_v3"
FINEST_AVAILABLE_CONTRACT = "global_two_state_transolver_finest_available_v1"
COARSE_ONLY_CONTRACT = "global_two_state_transolver_l5_fixed_cadence_v2"
VIT_BALANCED_LEVEL_CONTRACT = "global_two_state_patch_transformer_v1"
VIT_FINEST_AVAILABLE_CONTRACT = "global_two_state_patch_transformer_finest_available_v1"
VIT_COARSE_ONLY_CONTRACT = "global_two_state_patch_transformer_l5_fixed_cadence_v2"


def make_resolution_inputs(
    batch: dict[str, Tensor], dataset, statistics: FieldStatistics,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    inputs, target = make_inputs(batch, dataset, statistics, device)
    dt = (batch["time_np1"] - batch["time_n"]).to(device)
    reference_dx = 20.0 / (64 * 2**5)
    reynolds = float(dataset.physical_parameters["reynolds"])
    values = torch.stack((
        torch.full_like(dt, np.log2(dataset.dx / reference_dx)),
        dt / dataset.dx / 2.56,
        100.0 * dt / (reynolds * dataset.dx**2),
    ), dim=1)
    values = values[:, :, None, None].expand(
        -1, -1, inputs.shape[-2], inputs.shape[-1],
    )
    return torch.cat((inputs, values), dim=1), target


@torch.no_grad()
def evaluate(
    model: StructuredTransolverOperator | GlobalPatchTransformer,
    corpora: dict[str, TrajectoryCorpus],
    statistics: FieldStatistics,
    device: torch.device,
    samples_per_trajectory: int,
) -> dict[str, float]:
    model.eval()
    level_metrics = {}
    for level, corpus in corpora.items():
        state_values: list[float] = []
        vorticity_values: list[float] = []
        near_values: list[float] = []
        force_values: list[float] = []
        for dataset in corpus.datasets:
            count = min(samples_per_trajectory, len(dataset))
            for item in np.linspace(0, len(dataset) - 1, count, dtype=int):
                sample = dataset[int(item)]
                batch = {key: value[None] for key, value in sample.items()}
                inputs, target_delta = make_resolution_inputs(
                    batch, dataset, statistics, device,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction_delta = model(inputs).float()
                prediction = physical_state(
                    prediction_delta, batch, statistics, device,
                )
                target = batch["q_np1"].to(device)
                state_values.extend(
                    per_sample_relative(prediction, target).cpu().tolist()
                )
                vorticity_values.extend(per_sample_relative(
                    periodic_vorticity(prediction, dataset.dx),
                    periodic_vorticity(target, dataset.dx),
                ).cpu().tolist())
                near = batch["sdf_np1"].to(device).abs() <= 8.0 * dataset.dx
                near_values.append(float(torch.sqrt(masked_relative_square(
                    prediction_delta[:, :2], target_delta[:, :2], near,
                ))))
                force_values.append(float(torch.sqrt(
                    force_coefficient_relative_square(
                        prediction, target, batch["sdf_np1"].to(device),
                        batch["motion_np1"].to(device), dx=dataset.dx,
                        window=dataset.window, penalty_time=dataset.penalty_time,
                        rotation_sign=dataset.rotation_sign,
                    ).clamp_min(0.0)
                )))
        level_metrics[level] = {
            "state_l2_mean": float(np.mean(state_values)),
            "state_l2_p90": float(np.quantile(state_values, 0.9)),
            "vorticity_l2_mean": float(np.mean(vorticity_values)),
            "vorticity_l2_p90": float(np.quantile(vorticity_values, 0.9)),
            "near_body_delta_l2_mean": float(np.mean(near_values)),
            "force_l2_mean": float(np.mean(force_values)),
        }
    result = {
        f"{level}_{key}": value
        for level, metrics in level_metrics.items()
        for key, value in metrics.items()
    }
    for key in level_metrics[next(iter(level_metrics))]:
        result[f"balanced_{key}"] = float(np.mean([
            metrics[key] for metrics in level_metrics.values()
        ]))
    return result


def selection_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    """Use one checkpoint-selection rule for Point, Slice, and Patch."""
    return float(
        metrics["balanced_state_l2_mean"]
        + args.selection_vorticity_weight
        * metrics["balanced_vorticity_l2_mean"]
        + args.selection_force_weight
        * np.log1p(metrics["balanced_force_l2_mean"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument(
        "--initialize", type=Path,
        help=(
            "source Slice/Patch checkpoint for full-model target adaptation; "
            "unlike --pretrained, this permits a new split and optimizer"
        ),
    )
    parser.add_argument("--history-log", type=Path)
    parser.add_argument("--resume-in-place", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--trajectory-count", type=int, default=32)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples-per-epoch", type=int, default=1024)
    parser.add_argument("--fine-probability", type=float, default=0.5)
    parser.add_argument(
        "--level-policy", choices=("balanced", "finest_available", "coarse_only"),
        default="balanced",
        help=(
            "balanced reproduces the archived independent L5/L6 campaign; "
            "finest_available trains only the full L6 AMReX covering grid; "
            "coarse_only trains only the synchronized L5 field"
        ),
    )
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--min-lr", type=float, default=5.0e-6)
    parser.add_argument("--width", type=int, default=368)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--local-width", type=int, default=96)
    parser.add_argument("--local-depth", type=int, default=4)
    parser.add_argument(
        "--architecture", choices=("transolver", "global_vit"),
        default="transolver",
        help="global architecture trained jointly from random initialization",
    )
    parser.add_argument(
        "--patch-size", type=int, default=16,
        help="non-overlapping square patch width for --architecture global_vit",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--vorticity-weight", type=float, default=1.0)
    parser.add_argument("--near-body-weight", type=float, default=0.5)
    parser.add_argument("--quiet-weight", type=float, default=0.25)
    parser.add_argument("--force-weight", type=float, default=.05)
    parser.add_argument("--selection-vorticity-weight", type=float, default=.10)
    parser.add_argument("--selection-force-weight", type=float, default=.005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if args.resume_in_place and args.pretrained is None:
        raise ValueError("--resume-in-place requires --pretrained")
    if args.pretrained is not None and args.initialize is not None:
        raise ValueError("--pretrained and --initialize are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume_in_place)
    if not 0.0 <= args.fine_probability <= 1.0:
        raise ValueError("fine-probability must lie in [0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    split = (
        write_or_validate_snapshot(
            args.split_manifest, args.output_dir / "trajectory_split.json",
        )
        if args.split_manifest is not None
        else write_or_validate_split(
            args.data_root, args.output_dir / "trajectory_split.json",
            trajectory_count=args.trajectory_count,
        )
    )
    fine_train_names = [name.replace("_level5_", "_level6_") for name in split["train"]]
    fine_validation_names = [
        name.replace("_level5_", "_level6_") for name in split["validation"]
    ]
    if args.level_policy == "finest_available":
        train = {
            "l6": TrajectoryCorpus(
                args.data_root, fine_train_names, history_steps=2,
            ),
        }
        validation = {
            "l6": TrajectoryCorpus(
                args.data_root, fine_validation_names, history_steps=2,
            ),
        }
        training_contract = (
            VIT_FINEST_AVAILABLE_CONTRACT
            if args.architecture == "global_vit"
            else FINEST_AVAILABLE_CONTRACT
        )
    elif args.level_policy == "coarse_only":
        train = {
            "l5": TrajectoryCorpus(
                args.data_root, split["train"], history_steps=2,
            ),
        }
        validation = {
            "l5": TrajectoryCorpus(
                args.data_root, split["validation"], history_steps=2,
            ),
        }
        training_contract = (
            VIT_COARSE_ONLY_CONTRACT
            if args.architecture == "global_vit"
            else COARSE_ONLY_CONTRACT
        )
    else:
        train = {
            "l5": TrajectoryCorpus(
                args.data_root, split["train"], history_steps=2,
            ),
            "l6": TrajectoryCorpus(
                args.data_root, fine_train_names, history_steps=2,
            ),
        }
        validation = {
            "l5": TrajectoryCorpus(
                args.data_root, split["validation"], history_steps=2,
            ),
            "l6": TrajectoryCorpus(
                args.data_root, fine_validation_names, history_steps=2,
            ),
        }
        training_contract = (
            VIT_BALANCED_LEVEL_CONTRACT
            if args.architecture == "global_vit"
            else BALANCED_LEVEL_CONTRACT
        )
    source = torch.load(args.statistics, map_location="cpu", weights_only=False)
    statistics = FieldStatistics.from_json(source["statistics"])
    if args.architecture == "global_vit":
        model = GlobalPatchTransformer(
            input_channels=len(TRANSOLVER_INPUT_CHANNELS),
            output_channels=3,
            dimensions=2,
            patch_size=args.patch_size,
            width=args.width,
            depth=args.depth,
            heads=args.heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
        ).to(device)
        model_args = model.model_args
        model_name = "global_patch_transformer"
    else:
        model_args = {
            "input_channels": len(TRANSOLVER_INPUT_CHANNELS), "width": args.width,
            "depth": args.depth, "heads": args.heads, "slices": args.slices,
            "mlp_ratio": args.mlp_ratio, "local_width": args.local_width,
            "local_depth": args.local_depth,
        }
        model = StructuredTransolverOperator(**model_args).to(device)
        model_name = "structured_transolver"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    resume = None
    initialization = None
    start_epoch = 0
    if args.pretrained is not None:
        resume = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        if resume.get("training_contract") != training_contract:
            raise ValueError("pretrained Transolver uses a different level policy")
        if resume["model_args"] != model_args:
            raise ValueError("pretrained Transolver architecture does not match")
        if resume["split"] != split:
            raise ValueError("pretrained Transolver split does not match")
        model.load_state_dict(resume["model"])
        start_epoch = int(resume["epoch"])
        if args.epochs <= start_epoch:
            raise ValueError("epochs must exceed the pretrained checkpoint epoch")
        if not args.resume_in_place:
            atomic_torch_save(args.output_dir / "checkpoint.pt", resume)
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        if "scheduler" in resume:
            scheduler.load_state_dict(resume["scheduler"])
    elif args.initialize is not None:
        initialization = torch.load(
            args.initialize, map_location="cpu", weights_only=False,
        )
        if initialization.get("training_contract") != training_contract:
            raise ValueError("initialization uses a different level policy")
        if initialization.get("model_args") != model_args:
            raise ValueError("initialization architecture does not match")
        model.load_state_dict(initialization["model"])
    best = float("inf")
    summary_path = args.output_dir / "summary.json"
    if resume is not None:
        best = float(resume.get(
            "selection_score", selection_score(resume["validation"], args),
        ))
        if args.resume_in_place and summary_path.exists():
            prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best = float(prior_summary.get(
                "best_validation_selection_score",
                prior_summary.get("best_validation_vorticity_l2", best),
            ))
    history = merged_training_history(
        start_epoch,
        summary_path=summary_path if args.resume_in_place else None,
        history_log=args.history_log,
    )
    if resume is not None:
        require_complete_training_prefix(history, start_epoch)
    stale = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        generator = random.Random(args.seed + epoch * 1_000_003)
        records = []
        steps = args.max_train_batches or args.samples_per_epoch
        for _ in range(steps):
            level = (
                "l6" if args.level_policy == "finest_available"
                else "l5" if args.level_policy == "coarse_only"
                else "l6" if generator.random() < args.fine_probability else "l5"
            )
            dataset, batch = train[level].random_batch(args.batch_size, generator)
            inputs, target_delta = make_resolution_inputs(
                batch, dataset, statistics, device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction_delta = model(inputs).float()
                prediction = physical_state(
                    prediction_delta, batch, statistics, device,
                )
                target = batch["q_np1"].to(device)
                terms = loss_terms(
                    prediction_delta, target_delta, prediction, target,
                    batch["sdf_np1"].to(device), dataset.dx,
                )
                terms["force"] = force_coefficient_relative_square(
                    prediction, target, batch["sdf_np1"].to(device),
                    batch["motion_np1"].to(device), dx=dataset.dx,
                    window=dataset.window, penalty_time=dataset.penalty_time,
                    rotation_sign=dataset.rotation_sign,
                )
                loss = (
                    terms["delta"] + args.state_weight * terms["state"]
                    + args.vorticity_weight * terms["vorticity"]
                    + args.near_body_weight * terms["near_body_delta"]
                    + args.quiet_weight * terms["quiet_vorticity"]
                    + args.force_weight * terms["force"]
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            records.append({
                **{key: float(value.detach()) for key, value in terms.items()},
                "objective": float(loss.detach()),
            })
        metrics = evaluate(
            model, validation, statistics, device,
            args.max_validation_batches
            or args.validation_samples_per_trajectory,
        )
        selection = selection_score(metrics, args)
        record = {
            "epoch": epoch,
            "selection_score": selection,
            "train_loss": float(np.mean([row["objective"] for row in records])),
            "train_delta_mse": float(np.mean([row["delta"] for row in records])),
            "train_state_l2": float(np.sqrt(np.mean([row["state"] for row in records]))),
            "train_vorticity_l2": float(np.sqrt(np.mean([
                row["vorticity"] for row in records
            ]))),
            "train_near_body_delta_l2": float(np.sqrt(np.mean([
                row["near_body_delta"] for row in records
            ]))),
            "train_quiet_vorticity_l2": float(np.sqrt(np.mean([
                row["quiet_vorticity"] for row in records
            ]))),
            "train_force_l2": float(np.sqrt(np.mean([
                row["force"] for row in records
            ]))),
            **{f"validation_{key}": value for key, value in metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
        }
        improved = selection < best
        if improved:
            best, stale = selection, 0
        else:
            stale += 1
        scheduler.step(selection)
        checkpoint = {
                "model_name": model_name,
                "model": model.state_dict(), "model_args": model_args,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args), "statistics": statistics.to_json(),
                "input_channels": TRANSOLVER_INPUT_CHANNELS, "split": split,
                "epoch": epoch, "validation": metrics,
                "selection_score": selection,
                "training_contract": training_contract,
                "level_policy": args.level_policy,
                "initialization": (
                    str(args.initialize.resolve())
                    if args.initialize is not None else None
                ),
                "initialization_sha256": (
                    sha256(args.initialize) if args.initialize is not None else None
                ),
                "source_split": (
                    initialization.get("split")
                    if initialization is not None else None
                ),
                "full_model_adaptation": args.initialize is not None,
            }
        if improved:
            atomic_torch_save(args.output_dir / "checkpoint.pt", checkpoint)
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            atomic_torch_save(args.output_dir / "last_checkpoint.pt", checkpoint)
        history.append(record)
        print(json.dumps(record), flush=True)
        atomic_write_json(args.output_dir / "summary.json", {
            "model": (
                "Global Patch Transformer"
                if args.architecture == "global_vit"
                else "Structured Transolver"
            ),
            "training_contract": training_contract,
            "level_policy": args.level_policy,
            "parameter_count": parameter_count,
            "training_transitions_per_level": {
                level: corpus.transition_count for level, corpus in train.items()
            },
            "start_epoch": start_epoch,
            "initialization": (
                str(args.initialize.resolve())
                if args.initialize is not None else None
            ),
            "initialization_sha256": (
                sha256(args.initialize) if args.initialize is not None else None
            ),
            "best_validation_selection_score": best,
            "epochs_completed": len(history),
            "training_seconds_total": float(sum(
                row.get("seconds", 0.0) for row in history
            )),
            "final_epoch": history[-1]["epoch"], "history": history,
        })
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    for corpus in (*train.values(), *validation.values()):
        corpus.close()


if __name__ == "__main__":
    main()
