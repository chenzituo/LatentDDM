"""Train a full-domain geometry-aware AB-UPT trajectory baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.neural_eos.abupt import ABUPTScalarOperator
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics
from codiffuse_Schwarz_Airfoil.expanded_protocol import write_or_validate_snapshot
from codiffuse_Schwarz_Airfoil.force_diagnostics import (
    force_coefficient_relative_square,
)
from codiffuse_Schwarz_Airfoil.trajectory_corpus import (
    TrajectoryCorpus,
    write_or_validate_split,
)
from codiffuse_Schwarz_Airfoil.vorticity_projection import periodic_vorticity


INPUT_CHANNELS = (
    "q_n_u", "q_n_v", "q_n_p", "history_du", "history_dv", "history_dp",
    "sdf_n", "sdf_np1", "angle_n", "omega_n", "accel_n", "phase_n",
    "angle_np1", "omega_np1", "accel_np1", "phase_np1", "x", "y",
    "reynolds", "thickness", "pivot", "dt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_inputs(
    batch: dict[str, torch.Tensor], dataset, statistics: FieldStatistics,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_nm1 = batch["q_nm1"].to(device)
    q_n = batch["q_n"].to(device)
    q_np1 = batch["q_np1"].to(device)
    state_mean = torch.tensor(statistics.state_mean, device=device)[None, :, None, None]
    state_std = torch.tensor(statistics.state_std, device=device)[None, :, None, None]
    delta_mean = torch.tensor(statistics.delta_mean, device=device)[None, :, None, None]
    delta_std = torch.tensor(statistics.delta_std, device=device)[None, :, None, None]
    current = (q_n - state_mean) / state_std
    history = (q_n - q_nm1) / delta_std
    sdf = torch.cat((batch["sdf_n"], batch["sdf_np1"]), dim=1).to(device)
    sdf = (sdf / (8.0 * dataset.dx)).clamp(-4.0, 4.0) / 4.0
    motion = torch.cat((batch["motion_n"], batch["motion_np1"]), dim=1).to(device)
    motion[:, (0, 4)] /= torch.pi
    motion[:, (2, 6)] /= 3.0
    batch_size, _, height, width = q_n.shape
    motion = motion[:, :, None, None].expand(-1, -1, height, width)
    x0, _, y0, _ = dataset.window
    x = x0 + (torch.arange(width, device=device) + 0.5) * dataset.dx
    y = y0 + (torch.arange(height, device=device) + 0.5) * dataset.dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((xx / 10.0, yy / 10.0))[None].expand(batch_size, -1, -1, -1)
    params = dataset.physical_parameters
    dt = (batch["time_np1"] - batch["time_n"]).to(device)
    scalars = torch.stack((
        torch.full_like(dt, np.log(float(params["reynolds"]) / 10_000.0)),
        torch.full_like(dt, (float(params["thickness"]) - 0.15) / 0.05),
        torch.full_like(dt, (float(params["pivot_fraction"]) - 0.25) / 0.10),
        dt / 0.025,
    ), dim=1)[:, :, None, None].expand(-1, -1, height, width)
    inputs = torch.cat((current, history, sdf, motion, coordinates, scalars), dim=1)
    target = (q_np1 - q_n - delta_mean) / delta_std
    return inputs, target


def predict(
    model: ABUPTScalarOperator, inputs: torch.Tensor, context_stride: int,
) -> torch.Tensor:
    context = inputs[:, :, ::context_stride, ::context_stride]
    context_tokens = context.flatten(2).transpose(1, 2)
    query_tokens = inputs.flatten(2).transpose(1, 2)
    output = model(context_tokens, query_tokens)
    return output.transpose(1, 2).reshape(inputs.shape[0], 3, *inputs.shape[-2:])


def physical_state(
    normalized_delta: torch.Tensor, batch: dict[str, torch.Tensor],
    statistics: FieldStatistics, device: torch.device,
) -> torch.Tensor:
    mean = torch.tensor(statistics.delta_mean, device=device)[None, :, None, None]
    std = torch.tensor(statistics.delta_std, device=device)[None, :, None, None]
    return batch["q_n"].to(device) + normalized_delta * std + mean


def relative_square(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().sum() / target.square().sum().clamp_min(1.0e-20)


def per_sample_relative(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction - target).square().flatten(1).sum(1)
    denominator = target.square().flatten(1).sum(1).clamp_min(1.0e-20)
    return torch.sqrt(numerator / denominator)


def masked_relative_square(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(prediction.dtype)
    while weight.ndim < prediction.ndim:
        weight = weight.unsqueeze(1)
    numerator = ((prediction - target).square() * weight).sum()
    denominator = (target.square() * weight).sum().clamp_min(1.0e-20)
    return numerator / denominator


def physical_loss_terms(
    prediction_delta: torch.Tensor,
    target_delta: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    sdf: torch.Tensor,
    dx: float,
) -> dict[str, torch.Tensor]:
    """Shared fixed-cadence objective for Point, Slice, and Patch."""
    predicted_vorticity = periodic_vorticity(prediction, dx)
    target_vorticity = periodic_vorticity(target, dx)
    near_body = sdf.abs() <= 8.0 * dx
    amplitude = target_vorticity.abs().flatten(1).amax(1)[:, None, None]
    quiet = (
        (sdf[:, 0].abs() > 8.0 * dx)
        & (target_vorticity.abs() < 0.02 * amplitude)
    )
    velocity_scale = target_delta[:, :2].square().sum().clamp_min(1.0e-20)
    vorticity_scale = target_vorticity.square().sum().clamp_min(1.0e-20)
    return {
        "delta": (prediction_delta - target_delta).square().mean(),
        "state": relative_square(prediction, target),
        "vorticity": relative_square(predicted_vorticity, target_vorticity),
        "near_body_delta": (
            (prediction_delta[:, :2] - target_delta[:, :2]).square()
            * near_body.to(prediction_delta.dtype)
        ).sum() / velocity_scale,
        "quiet_vorticity": (
            (predicted_vorticity - target_vorticity).square()
            * quiet.to(predicted_vorticity.dtype)
        ).sum() / vorticity_scale,
    }


def trajectory_batch(corpus, batch):
    indices = batch.pop("trajectory_index")
    if not torch.all(indices == indices[0]):
        raise RuntimeError("mixed-trajectory batch violates physics metadata contract")
    return corpus.datasets[int(indices[0])], batch


@torch.no_grad()
def evaluate(
    model, corpus, loader, statistics, context_stride, device, max_batches=0,
) -> dict[str, float]:
    model.eval()
    state_values, vort_values = [], []
    for batch_index, loaded in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        dataset, batch = trajectory_batch(corpus, loaded)
        inputs, _ = make_inputs(batch, dataset, statistics, device)
        prediction = physical_state(predict(model, inputs, context_stride), batch, statistics, device)
        target = batch["q_np1"].to(device)
        state_values.extend(per_sample_relative(prediction, target).cpu().tolist())
        vort_values.extend(per_sample_relative(
            periodic_vorticity(prediction, dataset.dx),
            periodic_vorticity(target, dataset.dx),
        ).cpu().tolist())
    return {
        "state_l2_mean": float(np.mean(state_values)),
        "vorticity_l2_mean": float(np.mean(vort_values)),
        "vorticity_l2_p90": float(np.quantile(vort_values, 0.9)),
    }


@torch.no_grad()
def evaluate_trajectory_sampled(
    model,
    corpus,
    statistics,
    context_stride,
    device,
    samples_per_trajectory,
) -> dict[str, float]:
    """Evaluate every trajectory with the same deterministic sample count."""
    model.eval()
    state_values, vort_values, force_values = [], [], []
    for dataset in corpus.datasets:
        count = min(samples_per_trajectory, len(dataset))
        for item in np.linspace(0, len(dataset) - 1, count, dtype=int):
            sample = dataset[int(item)]
            batch = {key: value[None] for key, value in sample.items()}
            inputs, _ = make_inputs(batch, dataset, statistics, device)
            prediction = physical_state(
                predict(model, inputs, context_stride), batch, statistics, device,
            )
            target = batch["q_np1"].to(device)
            state_values.extend(per_sample_relative(prediction, target).cpu().tolist())
            vort_values.extend(per_sample_relative(
                periodic_vorticity(prediction, dataset.dx),
                periodic_vorticity(target, dataset.dx),
            ).cpu().tolist())
            force_values.append(float(torch.sqrt(
                force_coefficient_relative_square(
                    prediction, target, batch["sdf_np1"].to(device),
                    batch["motion_np1"].to(device), dx=dataset.dx,
                    window=dataset.window, penalty_time=dataset.penalty_time,
                    rotation_sign=dataset.rotation_sign,
                ).clamp_min(0.0)
            )))
    return {
        "state_l2_mean": float(np.mean(state_values)),
        "state_l2_p90": float(np.quantile(state_values, 0.9)),
        "vorticity_l2_mean": float(np.mean(vort_values)),
        "vorticity_l2_p90": float(np.quantile(vort_values, 0.9)),
        "force_l2_mean": float(np.mean(force_values)),
        "examples": len(state_values),
        "trajectories": len(corpus.datasets),
    }


def selection_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    return float(
        metrics["state_l2_mean"]
        + args.selection_vorticity_weight * metrics["vorticity_l2_mean"]
        + args.selection_force_weight * np.log1p(metrics["force_l2_mean"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--initialize", type=Path,
        help=(
            "source AB-UPT checkpoint for full-model target adaptation; the "
            "target optimizer is initialized from scratch"
        ),
    )
    parser.add_argument("--trajectory-count", type=int, default=32)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--d-model", type=int, default=352)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--context-stride", type=int, default=2)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--vorticity-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=.05)
    parser.add_argument("--near-body-weight", type=float, default=.5)
    parser.add_argument("--quiet-weight", type=float, default=.25)
    parser.add_argument("--selection-vorticity-weight", type=float, default=.10)
    parser.add_argument("--selection-force-weight", type=float, default=.005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--target-parameters", type=int, default=13_123_093)
    parser.add_argument("--parameter-tolerance", type=float, default=.05)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
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
    train = TrajectoryCorpus(args.data_root, split["train"], history_steps=2)
    validation = TrajectoryCorpus(args.data_root, split["validation"], history_steps=2)
    train_loader = train.loader(args.batch_size, True, args.seed, args.num_workers)
    source = torch.load(args.statistics, map_location="cpu", weights_only=False)
    statistics = FieldStatistics.from_json(source["statistics"])
    model = ABUPTScalarOperator(
        len(INPUT_CHANNELS), args.d_model, args.anchors, args.depth, args.heads,
        out_channels=3,
    ).to(device)
    model_args = {
        "in_channels": len(INPUT_CHANNELS),
        "d_model": args.d_model,
        "n_anchors": args.anchors,
        "depth": args.depth,
        "heads": args.heads,
        "out_channels": 3,
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    relative_parameter_gap = abs(parameter_count - args.target_parameters) / args.target_parameters
    if relative_parameter_gap > args.parameter_tolerance:
        raise ValueError(
            f"AB-UPT parameter count {parameter_count} differs from target "
            f"{args.target_parameters} by {relative_parameter_gap:.1%}"
        )
    initialization = None
    if args.initialize is not None:
        initialization = torch.load(
            args.initialize, map_location="cpu", weights_only=False,
        )
        if initialization.get("training_contract") \
                != "global_two_state_abupt_fixed_cadence_v4":
            raise ValueError("initialization is not a fixed-cadence AB-UPT")
        source_args = dict(initialization["args"])
        for name in ("d_model", "anchors", "depth", "heads", "context_stride"):
            if source_args.get(name) != getattr(args, name):
                raise ValueError(f"initialization and adaptation disagree on {name}")
        model.load_state_dict(initialization["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    best, history, stale = float("inf"), [], 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        losses = []
        components = []
        for batch_index, loaded in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            dataset, batch = trajectory_batch(train, loaded)
            inputs, target_delta = make_inputs(batch, dataset, statistics, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction_delta = predict(model, inputs, args.context_stride).float()
                prediction = physical_state(prediction_delta, batch, statistics, device)
                target = batch["q_np1"].to(device)
                terms = physical_loss_terms(
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
            losses.append(float(loss.detach()))
            components.append({
                **{key: float(value.detach()) for key, value in terms.items()},
                "objective": float(loss.detach()),
            })
        metrics = evaluate_trajectory_sampled(
            model, validation, statistics, args.context_stride, device,
            args.max_validation_batches
            or args.validation_samples_per_trajectory,
        )
        score = selection_score(metrics, args)
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "selection_score": score,
            "train_delta_mse": float(np.mean([value["delta"] for value in components])),
            "train_state_l2": float(np.sqrt(np.mean([
                value["state"] for value in components
            ]))),
            "train_vorticity_l2": float(np.sqrt(np.mean([
                value["vorticity"] for value in components
            ]))),
            "train_near_body_delta_l2": float(np.sqrt(np.mean([
                value["near_body_delta"] for value in components
            ]))),
            "train_quiet_vorticity_l2": float(np.sqrt(np.mean([
                value["quiet_vorticity"] for value in components
            ]))),
            "train_force_l2": float(np.sqrt(np.mean([
                value["force"] for value in components
            ]))),
            "training_samples": train.transition_count,
            **{
            f"validation_{key}": value for key, value in metrics.items()
        }, "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started}
        if score < best:
            best = score
            stale = 0
            torch.save({
                "model_name": "abupt", "model": model.state_dict(),
                "model_args": model_args,
                "args": vars(args), "statistics": statistics.to_json(),
                "input_channels": INPUT_CHANNELS, "split": split, "epoch": epoch,
                "validation": metrics, "selection_score": score,
                "training_contract": "global_two_state_abupt_fixed_cadence_v4",
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
            }, args.output_dir / "checkpoint.pt")
        else:
            stale += 1
        scheduler.step(score)
        history.append(record)
        print(json.dumps(record), flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    (args.output_dir / "summary.json").write_text(json.dumps({
        "model": "AB-UPT", "training_contract": "global_two_state_abupt_fixed_cadence_v4",
        "parameter_count": parameter_count,
        "relative_parameter_gap": relative_parameter_gap,
        "training_transitions": train.transition_count,
        "best_validation_selection_score": best,
        "epochs_completed": len(history),
        "early_stopped": len(history) < args.epochs,
        "training_seconds_total": float(sum(
            row.get("seconds", 0.0) for row in history
        )),
        "initialization": (
            str(args.initialize.resolve()) if args.initialize is not None else None
        ),
        "initialization_sha256": (
            sha256(args.initialize) if args.initialize is not None else None
        ),
        "history": history,
    }, indent=2), encoding="utf-8")
    train.close()
    validation.close()


if __name__ == "__main__":
    main()
