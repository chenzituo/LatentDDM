"""Robust one-step refinement of the multilevel residual-vorticity projection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from codiffuse_Schwarz_Airfoil.coupled_trajectory import CyclicWindowDataset
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics, denormalize_delta
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import (
    apply_endpoint_eos_fields,
    evaluate_multi_offset_rollouts,
    load_pretrained,
    make_transition,
)
from codiffuse_Schwarz_Airfoil.vorticity_projection import (
    periodic_vorticity,
    project_velocity_curl,
)


def stack_samples(
    dataset: CyclicWindowDataset,
    records: list[tuple[int, torch.Tensor | None]],
) -> dict[str, torch.Tensor]:
    samples = []
    for item, state in records:
        sample = dataset[item]
        if state is not None:
            sample["q_n"] = state
        samples.append(sample)
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


def smooth_field(field: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Local low-pass without introducing zero-valued domain boundaries."""
    radius = kernel_size // 2
    padded = F.pad(field[:, None], (radius,) * 4, mode="replicate")
    return F.avg_pool2d(padded, kernel_size, stride=1)[:, 0]


def spectral_error_losses(
    error: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separate recurrently dangerous low/mid bands from pixel-scale noise."""
    low = smooth_field(error, 25)
    high = error - smooth_field(error, 5)
    mid = error - low - high
    return tuple(component.square().sum() / denominator for component in (low, mid, high))


def projected_next(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    dataset: CyclicWindowDataset,
    statistics: FieldStatistics,
    fine_fraction: float,
    halo: int,
    seed: int,
    device: torch.device,
    placement_mode: str = "random",
) -> tuple[torch.Tensor, torch.Tensor]:
    transition = make_transition(
        "multilevel", batch, statistics, dataset, device, seed,
        fine_fraction, halo, placement_mode,
    )
    initial = torch.zeros_like(transition.fine.target)
    endpoint, vorticity = apply_endpoint_eos_fields(model, initial, transition, 1)
    if vorticity is None:
        raise RuntimeError("robust projection training requires a vorticity endpoint")
    baseline = transition.fine.q_n + denormalize_delta(endpoint, statistics)
    if bool(getattr(model, "endpoint_learned_projection", False)):
        return baseline, periodic_vorticity(baseline, dataset.dx)
    return project_velocity_curl(baseline, vorticity, dataset.dx), vorticity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--gate-lr", type=float, default=2.0e-3)
    parser.add_argument("--min-lr", type=float, default=2.0e-6)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--start-stride", type=int, default=20)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--fine-fraction", type=float, default=0.35)
    parser.add_argument("--state-weight", type=float, default=2.0)
    parser.add_argument("--quiet-weight", type=float, default=5.0)
    parser.add_argument("--spectral-shaping", action="store_true")
    parser.add_argument("--low-band-weight", type=float, default=2.0)
    parser.add_argument("--mid-band-weight", type=float, default=1.0)
    parser.add_argument("--high-band-weight", type=float, default=0.25)
    parser.add_argument("--unfreeze-endpoint", action="store_true")
    parser.add_argument("--max-replay-scale", type=float, default=1.0)
    parser.add_argument(
        "--replay-warmup-epochs", type=int, default=-1,
        help="Epochs used to reach max replay; zero starts at full replay.",
    )
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    statistics = FieldStatistics.from_json(checkpoint["statistics"])
    model, spec = load_pretrained(
        checkpoint, device, endpoint_spectral_shaping=args.spectral_shaping
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = []
    learned_projection = bool(getattr(model, "endpoint_learned_projection", False))
    for name, parameter in model.named_parameters():
        selected = name.startswith(
            "projection_" if learned_projection else "vorticity_"
        )
        if learned_projection and args.unfreeze_endpoint:
            selected = selected or name.startswith("endpoint_")
        if args.spectral_shaping and name.endswith("spectral_shaping_gate"):
            selected = True
        if selected:
            parameter.requires_grad = True
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("checkpoint has no trainable projection parameters")
    gates = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.endswith("spectral_shaping_gate")
    ]
    gate_ids = {id(parameter) for parameter in gates}
    ordinary = [parameter for parameter in trainable if id(parameter) not in gate_ids]
    groups = []
    if ordinary:
        groups.append({"params": ordinary, "lr": args.lr})
    if gates:
        groups.append({"params": gates, "lr": args.gate_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr
    )
    dataset = CyclicWindowDataset(args.data, "cycle2")
    starts = list(range(0, len(dataset) - args.horizon + 1, args.start_stride))
    if starts[-1] != len(dataset) - args.horizon:
        starts.append(len(dataset) - args.horizon)
    eval_args = argparse.Namespace(
        mode="multilevel", sampler="pnp", fine_fraction=args.fine_fraction,
        halo=args.halo, ensemble_size=2, placement_ensemble_size=2,
        eval_flow_steps=4, schwarz_iterations=1, schwarz_relaxation=0.7,
        endpoint_eos_iterations=1,
    )

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.eval()
        replay: list[tuple[int, torch.Tensor]] = []
        with torch.no_grad():
            for start in starts:
                current = dataset[start]["q_n"][None].to(device)
                for step in range(args.horizon):
                    item = start + step
                    # The recovery input belongs to time n, before advancing to n+1.
                    replay.append((item, current[0].cpu()))
                    sample = dataset[item]
                    batch = {key: value[None] for key, value in sample.items()}
                    batch["q_n"] = current
                    current, _ = projected_next(
                        model, batch, dataset, statistics, args.fine_fraction,
                        args.halo, args.seed + epoch * 1_000_003 + item,
                        device,
                    )

        warmup = (
            max(args.epochs // 2, 1)
            if args.replay_warmup_epochs < 0 else args.replay_warmup_epochs
        )
        scale = args.max_replay_scale * (
            1.0 if warmup == 0 else min(1.0, epoch / warmup)
        )
        records: list[tuple[int, torch.Tensor | None]] = [
            (item, None) for item in range(len(dataset))
        ]
        records.extend(
            (item, dataset[item]["q_n"] + scale * (state - dataset[item]["q_n"]))
            for item, state in replay
        )
        random.shuffle(records)
        model.train()
        losses = []
        for batch_index in range(0, len(records), args.batch_size):
            selected = records[batch_index : batch_index + args.batch_size]
            batch = stack_samples(dataset, selected)
            optimizer.zero_grad(set_to_none=True)
            prediction, predicted_vorticity = projected_next(
                model, batch, dataset, statistics, args.fine_fraction, args.halo,
                args.seed + epoch * 10_000_019 + batch_index, device,
            )
            target = batch["q_np1"].to(device)
            target_vorticity = periodic_vorticity(target, dataset.dx)
            error = predicted_vorticity - target_vorticity
            denominator = target_vorticity.square().sum().clamp_min(1.0e-20)
            vorticity_loss = error.square().sum() / denominator
            amplitude = target_vorticity.abs().flatten(1).amax(dim=1)[:, None, None]
            far = batch["sdf_np1"].to(device)[:, 0].abs() > 8.0 * dataset.dx
            quiet = far & (target_vorticity.abs() < 0.02 * amplitude)
            quiet_loss = (error.square() * quiet).sum() / denominator
            low_loss, mid_loss, high_loss = spectral_error_losses(
                error, denominator
            )
            state_loss = (
                (prediction - target).square().sum()
                / target.square().sum().clamp_min(1.0e-20)
            )
            loss = (
                vorticity_loss + args.quiet_weight * quiet_loss
                + args.state_weight * state_loss
                + args.low_band_weight * low_loss
                + args.mid_band_weight * mid_loss
                + args.high_band_weight * high_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append((
                float(vorticity_loss.detach()), float(quiet_loss.detach()),
                float(state_loss.detach()), float(low_loss.detach()),
                float(mid_loss.detach()), float(high_loss.detach()),
            ))
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_vorticity_l2": float(np.sqrt(np.mean([x[0] for x in losses]))),
            "train_quiet_l2": float(np.sqrt(np.mean([x[1] for x in losses]))),
            "train_state_l2": float(np.sqrt(np.mean([x[2] for x in losses]))),
            "train_low_band_l2": float(np.sqrt(np.mean([x[3] for x in losses]))),
            "train_mid_band_l2": float(np.sqrt(np.mean([x[4] for x in losses]))),
            "train_high_band_l2": float(np.sqrt(np.mean([x[5] for x in losses]))),
            "replay_scale": scale,
            "lr": optimizer.param_groups[0]["lr"],
            "gate_lr": optimizer.param_groups[-1]["lr"] if gates else None,
            "spectral_gates": {
                name: float(
                    0.75 * torch.tanh(parameter.detach().clamp_min(0.0)).cpu()
                )
                for name, parameter in model.named_parameters()
                if name.endswith("spectral_shaping_gate")
            },
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            metrics, rows = evaluate_multi_offset_rollouts(
                model, dataset, args.horizon, args.start_stride, statistics,
                eval_args, args.seed + 9_000_000 + epoch, device,
            )
            record.update(metrics)
            selection = metrics["worst_step20_vorticity_l2"]
            if selection < best:
                best = selection
                torch.save({
                    "model": model.state_dict(), "statistics": statistics.to_json(),
                    "model_spec": spec,
                    "args": {
                        **checkpoint.get("args", {}),
                        "endpoint_spectral_shaping": args.spectral_shaping,
                    },
                    "epoch": epoch, "validation": metrics,
                    "training_contract": (
                        "detached_pushforward_one_step_spectral_projection"
                    ),
                }, args.output_dir / "checkpoint.pt")
                with (args.output_dir / "multi_offset.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        history.append(record)
        print(json.dumps(record), flush=True)

    (args.output_dir / "summary.json").write_text(json.dumps({
        "training_contract": "detached_pushforward_one_step_spectral_projection",
        "best_worst_offset_step20_vorticity_l2": best,
        "gate_passed": best < 0.1,
        "history": history,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
