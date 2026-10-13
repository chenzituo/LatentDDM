"""Fine-tune one adaptive Cartesian operator on paired L5/L6 trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from codiffuse_Schwarz_Airfoil.coupled_trajectory import CyclicWindowDataset
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics
from codiffuse_Schwarz_Airfoil.expanded_protocol import write_or_validate_snapshot
from codiffuse_Schwarz_Airfoil.force_diagnostics import (
    force_coefficient_relative_square,
)
from codiffuse_Schwarz_Airfoil.multiresolution import PairedCartesianLevelDataset
from codiffuse_Schwarz_Airfoil.multiresolution_inference import (
    predict_multiresolution_step,
)
from codiffuse_Schwarz_Airfoil.train_multiresolution_projection import (
    batched,
    multiresolution_losses,
)
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import load_pretrained
from codiffuse_Schwarz_Airfoil.trajectory_corpus import write_or_validate_split


class PairedTrajectoryCorpus:
    """Keep paired fields and transition-condition views trajectory aligned."""

    def __init__(
        self,
        root: Path,
        names: list[str],
        scope: str,
        history_steps: int = 2,
    ) -> None:
        if history_steps not in (1, 2):
            raise ValueError("history_steps must be one or two")
        self.names = names
        self.paired: list[PairedCartesianLevelDataset] = []
        self.coarse: list[CyclicWindowDataset] = []
        self.fine: list[CyclicWindowDataset] = []
        for name in names:
            coarse = root / name
            match = re.search(r"_level(\d+)_", name)
            if match is None:
                raise ValueError(f"paired trajectory name has no level token: {name}")
            coarse_level = int(match.group(1))
            fine_name = (
                name[: match.start(1)] + str(coarse_level + 1)
                + name[match.end(1) :]
            )
            fine = root / fine_name
            if not fine.is_file():
                raise FileNotFoundError(f"missing paired fine trajectory: {fine}")
            self.paired.append(PairedCartesianLevelDataset(
                coarse, fine, scope, history_steps=history_steps,
            ))
            self.coarse.append(
                CyclicWindowDataset(coarse, scope, history_steps=history_steps)
            )
            self.fine.append(
                CyclicWindowDataset(fine, scope, history_steps=history_steps)
            )

    @property
    def transition_count(self) -> int:
        return sum(len(dataset) for dataset in self.paired)

    def random_sample(
        self, generator: random.Random,
    ) -> tuple[PairedCartesianLevelDataset, CyclicWindowDataset, CyclicWindowDataset, dict]:
        trajectory = generator.randrange(len(self.paired))
        paired = self.paired[trajectory]
        return (
            paired, self.coarse[trajectory], self.fine[trajectory],
            paired[generator.randrange(len(paired))],
        )

    def close(self) -> None:
        for dataset in (*self.paired, *self.coarse, *self.fine):
            dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-count", type=int, default=32)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=1024)
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3.0e-6)
    parser.add_argument("--min-lr", type=float, default=3.0e-7)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--fine-fraction", type=float, default=0.35)
    parser.add_argument("--fine-trust", type=float, default=.5)
    parser.add_argument("--vorticity-amplitude-fraction", type=float, default=.15)
    parser.add_argument("--coarse-state-weight", type=float, default=1.0)
    parser.add_argument("--fine-state-weight", type=float, default=1.0)
    parser.add_argument("--composite-vorticity-weight", type=float, default=4.0)
    parser.add_argument("--near-body-weight", type=float, default=1.0)
    parser.add_argument("--far-field-weight", type=float, default=2.0)
    parser.add_argument(
        "--domain-boundary-vorticity-weight", type=float, default=0.0,
    )
    parser.add_argument("--force-weight", type=float, default=.05)
    parser.add_argument("--selection-force-weight", type=float, default=.025)
    parser.add_argument(
        "--selection-domain-boundary-weight", type=float, default=0.0,
    )
    parser.add_argument("--enable-feature-schwarz", action="store_true")
    parser.add_argument("--enable-domain-boundary-adapter", action="store_true")
    parser.add_argument("--enable-motion-transport-adapter", action="store_true")
    parser.add_argument("--motion-transport-width", type=int, default=32)
    parser.add_argument("--motion-transport-depth", type=int, default=2)
    parser.add_argument("--motion-transport-lr", type=float, default=3.0e-5)
    parser.add_argument("--domain-boundary-width", type=int, default=32)
    parser.add_argument("--domain-boundary-depth", type=int, default=2)
    parser.add_argument("--domain-boundary-lr", type=float, default=3.0e-5)
    parser.add_argument("--domain-boundary-only", action="store_true")
    parser.add_argument("--feature-schwarz-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--unfreeze-endpoint", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def prediction_and_losses(
    model: torch.nn.Module,
    sample: dict,
    coarse: CyclicWindowDataset,
    fine: CyclicWindowDataset,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
):
    coarse_batch = batched(sample, "coarse")
    fine_batch = batched(sample, "fine")
    prediction = predict_multiresolution_step(
        model, coarse_batch, fine_batch, coarse, fine, statistics,
        fine_fraction=args.fine_fraction, halo=args.halo, seed=seed,
        device=device, fine_trust=args.fine_trust,
        vorticity_amplitude_fraction=args.vorticity_amplitude_fraction,
    )
    native_parent = None
    if "native_refinement_mask_np1" in sample:
        native_fine = sample["native_refinement_mask_np1"][None].to(
            device=device, dtype=torch.float32,
        )
        native_parent = F.max_pool2d(native_fine, kernel_size=2, stride=2) > 0.0
    losses = multiresolution_losses(
        prediction, coarse_batch["q_np1"].to(device),
        fine_batch["q_np1"].to(device), coarse_batch["sdf_np1"].to(device),
        fine_batch["sdf_np1"].to(device), coarse.dx, fine.dx,
        composite_mask=native_parent,
    )
    losses["force"] = force_coefficient_relative_square(
        prediction.composite_fine_view, fine_batch["q_np1"].to(device),
        fine_batch["sdf_np1"].to(device), fine_batch["motion_np1"].to(device),
        dx=fine.dx, window=fine.window, penalty_time=fine.penalty_time,
        rotation_sign=fine.rotation_sign,
    )
    return prediction, losses, native_parent


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    records = []
    for trajectory, paired in enumerate(corpus.paired):
        count = min(args.validation_samples_per_trajectory, len(paired))
        items = np.linspace(0, len(paired) - 1, count, dtype=int)
        for local, item in enumerate(items):
            prediction, losses, native_parent = prediction_and_losses(
                model, paired[int(item)], corpus.coarse[trajectory],
                corpus.fine[trajectory], statistics, args,
                args.seed + trajectory * 1_000_003 + local, device,
            )
            diagnostics = {}
            if native_parent is not None:
                selected = prediction.refinement_mask.bool()
                intersection = (selected & native_parent).float().sum()
                diagnostics = {
                    "native_refinement_recall": float(
                        intersection / native_parent.float().sum().clamp_min(1.0)
                    ),
                    "native_refinement_precision": float(
                        intersection / selected.float().sum().clamp_min(1.0)
                    ),
                }
            records.append({
                key: float(torch.sqrt(value.clamp_min(0.0)))
                for key, value in losses.items()
            } | {
                "executed_fine_tile_fraction": float(
                    prediction.executed_fine_tiles.float().mean()
                )
            } | diagnostics)
    return {
        f"{key}_mean": float(np.mean([record[key] for record in records]))
        for key in records[0]
    } | {
        "composite_vorticity_p90": float(np.quantile(
            [record["composite_vorticity"] for record in records], 0.9,
        )),
        "samples": len(records),
    }


def selection_score(
    metrics: dict[str, float], force_weight: float,
    domain_boundary_weight: float = 0.0,
) -> float:
    """Prefer vorticity gains only when they do not sacrifice surface loads."""
    if force_weight < 0.0 or domain_boundary_weight < 0.0:
        raise ValueError("selection weights must be nonnegative")
    return (
        metrics["composite_vorticity_mean"]
        + force_weight * math.log1p(metrics["force_mean"])
        + domain_boundary_weight
        * metrics.get("domain_boundary_vorticity_mean", 0.0)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
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
    train = PairedTrajectoryCorpus(args.data_root, split["train"], "all")
    validation = PairedTrajectoryCorpus(
        args.data_root, split["validation"], "all",
    )
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    model, spec = load_pretrained(
        checkpoint, device, history_steps=2,
        endpoint_feature_schwarz=args.enable_feature_schwarz,
        endpoint_motion_transport_adapter=args.enable_motion_transport_adapter,
        endpoint_motion_transport_width=args.motion_transport_width,
        endpoint_motion_transport_depth=args.motion_transport_depth,
        endpoint_domain_boundary_adapter=args.enable_domain_boundary_adapter,
        endpoint_domain_boundary_width=args.domain_boundary_width,
        endpoint_domain_boundary_depth=args.domain_boundary_depth,
    )
    if args.statistics.suffix == ".pt":
        payload = torch.load(
            args.statistics, map_location="cpu", weights_only=False,
        )
    else:
        payload = json.loads(args.statistics.read_text(encoding="utf-8"))
    statistics = FieldStatistics.from_json(payload["statistics"])
    model.set_endpoint_statistics(
        statistics.state_mean, statistics.state_std, statistics.delta_std,
        statistics.delta_mean,
    )
    model.gradient_checkpointing = args.gradient_checkpointing
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = []
    for name, parameter in model.named_parameters():
        selected = (
            name.startswith("projection_")
            or name.startswith("motion_transport_")
            or name.startswith("domain_boundary_")
            or name == "feature_schwarz_gate"
        )
        if args.unfreeze_endpoint:
            selected = selected or name.startswith(("endpoint_", "vorticity_"))
        if args.domain_boundary_only:
            selected = name.startswith("domain_boundary_")
        parameter.requires_grad = selected
        if selected:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("checkpoint has no endpoint/projection parameters")
    gate = [
        parameter for name, parameter in model.named_parameters()
        if name == "feature_schwarz_gate" and parameter.requires_grad
    ]
    gate_ids = {id(parameter) for parameter in gate}
    boundary = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("domain_boundary_") and parameter.requires_grad
    ]
    motion = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("motion_transport_") and parameter.requires_grad
    ]
    special_ids = gate_ids | {
        id(parameter) for parameter in (*boundary, *motion)
    }
    ordinary = [
        parameter for parameter in trainable if id(parameter) not in special_ids
    ]
    groups = []
    if ordinary:
        groups.append({"params": ordinary, "lr": args.lr})
    if gate:
        groups.append({
            "params": gate, "lr": args.feature_schwarz_lr,
            "weight_decay": 0.0,
        })
    if boundary:
        groups.append({"params": boundary, "lr": args.domain_boundary_lr})
    if motion:
        groups.append({"params": motion, "lr": args.motion_transport_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr,
    )
    baseline = evaluate(model, validation, statistics, args, device)
    best, stale = selection_score(
        baseline, args.selection_force_weight,
        args.selection_domain_boundary_weight,
    ), 0
    selected_validation = baseline
    history = [{
        "epoch": 0, "selection_score": best, **baseline, "lr": args.lr,
    }]
    torch.save({
        "model": model.state_dict(), "model_spec": spec,
        "statistics": statistics.to_json(), "args": vars(args),
        "split": split, "epoch": 0, "validation": baseline,
        "selection_score": best,
        "training_contract": "multitrajectory_paired_l5_l6_adaptive_v3",
    }, args.output_dir / "checkpoint.pt")
    print(json.dumps(history[-1]), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = random.Random(args.seed + epoch * 1_000_003)
        records = []
        for local in range(args.samples_per_epoch):
            _, coarse, fine, sample = train.random_sample(generator)
            optimizer.zero_grad(set_to_none=True)
            _, losses, _ = prediction_and_losses(
                model, sample, coarse, fine, statistics, args,
                args.seed + epoch * 10_000_019 + local, device,
            )
            loss = (
                args.coarse_state_weight * losses["coarse_state"]
                + args.fine_state_weight * losses["fine_state"]
                + args.composite_vorticity_weight * losses["composite_vorticity"]
                + args.near_body_weight * losses["near_body_velocity"]
                + args.far_field_weight * losses["far_field_vorticity"]
                + args.domain_boundary_vorticity_weight
                * losses["domain_boundary_vorticity"]
                + args.force_weight * losses["force"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            records.append({
                key: float(torch.sqrt(value.detach().clamp_min(0.0)))
                for key, value in losses.items()
            })
        metrics = evaluate(model, validation, statistics, args, device)
        selection = selection_score(
            metrics, args.selection_force_weight,
            args.selection_domain_boundary_weight,
        )
        record = {
            "epoch": epoch,
            "train_composite_vorticity_l2": float(np.mean([
                item["composite_vorticity"] for item in records
            ])),
            "selection_score": selection,
            **metrics, "lr": optimizer.param_groups[0]["lr"],
            "feature_schwarz_gate": (
                float(torch.tanh(model.feature_schwarz_gate.detach()))
                if model.feature_schwarz_gate is not None else None
            ),
        }
        if selection < best:
            best, stale = selection, 0
            selected_validation = metrics
            torch.save({
                "model": model.state_dict(), "model_spec": spec,
                "statistics": statistics.to_json(), "args": vars(args),
                "split": split, "epoch": epoch, "validation": metrics,
                "selection_score": selection,
                "training_contract": "multitrajectory_paired_l5_l6_adaptive_v3",
            }, args.output_dir / "checkpoint.pt")
        else:
            stale += 1
        scheduler.step()
        history.append(record)
        print(json.dumps(record), flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    (args.output_dir / "summary.json").write_text(json.dumps({
        "training_contract": "multitrajectory_paired_l5_l6_adaptive_v3",
        "trajectory_counts": {
            key: len(split[key]) for key in ("train", "validation", "test")
        },
        "training_transitions": train.transition_count,
        "best_selection_score": best,
        "best_composite_vorticity_l2": selected_validation[
            "composite_vorticity_mean"
        ],
        "best_force_l2": selected_validation["force_mean"],
        "epochs_completed": len(history) - 1, "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    train.close()
    validation.close()


if __name__ == "__main__":
    main()
