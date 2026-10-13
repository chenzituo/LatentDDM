"""Distill train-only value+flux Schur codes into one homogeneous message."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from codiffuse_Schwarz.tile_communication import ResidualKernelAttention

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .export_value_flux_teacher import CONTRACT as TEACHER_CONTRACT
from .latent_schur_teacher import inference_only_context
from .residual_port_composer import (
    SignedSourceLinearResidualPortCommunicator,
    SourceLinearResidualPortCommunicator,
)
from .train import seed_everything
from .train_residual_port_chart import (
    frozen_context, interface_losses, load_frozen_local, masked_mse,
    physics_losses, sha256,
)
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT, forward_composer,
    load_frozen_chart, source_linear_signal_width,
)
from .validate_latent_schur import measured_q, q_from_corrected


Tensor = torch.Tensor
ORDINARY_NONREGRESSION_FACTOR = 1.02
TEACHER_IMPROVEMENT_FACTOR = 0.95


class ValueFluxTeacherDataset(Dataset[dict[str, Any]]):
    """Join train PDE inputs to cached target-free codes by train case index."""

    def __init__(
        self, data: Path, teacher: Path, local_checkpoint: Path,
        chart_checkpoint: Path, *, holdout: bool,
    ) -> None:
        self.base = H5DarcyDataset(data, "train")
        self.teacher_path = teacher.resolve()
        with h5py.File(self.teacher_path, "r") as handle:
            if handle.attrs.get("contract") != TEACHER_CONTRACT:
                raise ValueError("value+flux teacher contract differs")
            if handle.attrs.get("source_split") != "train" or bool(
                handle.attrs.get("validation_or_test_opened", True)
            ):
                raise ValueError("teacher corpus is not train-only")
            if bool(handle.attrs.get("target_or_pde_labels_used", True)):
                raise ValueError("teacher corpus used prohibited labels/operators")
            if handle.attrs.get("source_data_sha256") != sha256(data):
                raise ValueError("teacher source-data identity differs")
            if handle.attrs.get("local_checkpoint_sha256") != sha256(local_checkpoint):
                raise ValueError("teacher local-checkpoint identity differs")
            if handle.attrs.get("chart_checkpoint_sha256") != sha256(chart_checkpoint):
                raise ValueError("teacher chart-checkpoint identity differs")
            indices = np.asarray(handle["train/case_index"], dtype=np.int64)
        selector = indices % 10 == 0
        self.rows = np.flatnonzero(selector if holdout else ~selector)
        self.case_indices = indices
        if not len(self.rows):
            raise ValueError("empty teacher train/holdout partition")
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy(); state["_handle"] = None
        return state

    def close(self) -> None:
        self.base.close()
        if self._handle is not None:
            self._handle.close(); self._handle = None

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = h5py.File(self.teacher_path, "r")
        row = int(self.rows[index])
        item = self.base[int(self.case_indices[row])]
        item["teacher_code"] = torch.from_numpy(np.asarray(
            self._handle["train/teacher_code"][row], dtype=np.float32,
        ))
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ordinary-validation-cases", type=int, default=60)
    parser.add_argument("--teacher-validation-cases", type=int, default=128)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--token-width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--field-weight", type=float, default=0.1)
    parser.add_argument("--interface-weight", type=float, default=0.02)
    parser.add_argument("--cycle-weight", type=float, default=0.1)
    parser.add_argument(
        "--kernel-type", choices=("softmax", "signed_l1_tanh"),
        default="softmax",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _loader(dataset, batch_size, workers, device, *, samples=None) -> DataLoader:
    sampler = None if samples is None else RandomSampler(
        dataset, replacement=samples > len(dataset), num_samples=samples,
    )
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=False, num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0, drop_last=sampler is not None,
    )


def _load_expanded_initial(
    communicator, path: Path, local_path: Path, chart_path: Path,
) -> dict[str, Any]:
    initial = torch.load(path, map_location="cpu", weights_only=False)
    if initial.get("contract") != CONTRACT:
        raise ValueError("initial checkpoint is not a one-message communicator")
    if initial.get("source_local_sha256") != sha256(local_path) \
            or initial.get("source_chart_sha256") != sha256(chart_path):
        raise ValueError("initial communicator lineage differs")
    if initial.get("communicator_spec", {}).get("analyzer_mode") != "source_linear":
        raise ValueError("initial communicator is not the canonical value-only q kernel")
    source = initial["communicator"]
    expanded = communicator.state_dict()
    if isinstance(communicator, SignedSourceLinearResidualPortCommunicator):
        # Signed normalization changes the meaning of the value/output path.
        # Transfer geometry-dependent context scores only; retain the signed
        # architecture's fresh bias-free values and exact-zero code head.
        for target_name in tuple(expanded):
            source_name = None
            if target_name.startswith("geometry_analyzer.") or \
                    ".global_kernel.context_norm." in target_name or \
                    ".global_kernel.qk." in target_name:
                source_name = target_name
            elif ".global_kernel.relative_score." in target_name:
                source_name = target_name.replace(
                    ".global_kernel.relative_score.",
                    ".global_kernel.relative_bias.",
                )
            if source_name is not None and source_name in source \
                    and source[source_name].shape == expanded[target_name].shape:
                expanded[target_name].copy_(source[source_name])
        communicator.load_state_dict(expanded, strict=True)
        return initial
    expandable = {
        "code_map.local_value.weight", "code_map.global_kernel.value.weight",
    }
    for name, target in expanded.items():
        if name not in source:
            raise ValueError(f"initial communicator lacks {name}")
        old = source[name]
        if old.shape == target.shape:
            target.copy_(old)
        elif name in expandable and old.ndim == 2 \
                and old.shape[0] == target.shape[0] and old.shape[1] < target.shape[1]:
            target.zero_(); target[:, :old.shape[1]].copy_(old)
        else:
            raise ValueError(f"cannot expand {name}: {tuple(old.shape)} -> {tuple(target.shape)}")
    communicator.load_state_dict(expanded, strict=True)
    return initial


def _complete_cycle_square(
    corrected: Tensor, context, batch: dict[str, Tensor], spec, scale,
    baseline_q: Tensor,
) -> Tensor:
    inference = inference_only_context(context)
    _, support = measured_q(inference, batch, spec, scale, "value_flux")
    corrected_q = q_from_corrected(
        corrected, inference, batch, spec, scale, "value_flux",
    )
    axes = tuple(range(1, baseline_q.ndim))
    numerator = (corrected_q.square() * support).sum(axes)
    denominator = (baseline_q.square() * support).sum(axes).clamp_min(1.0e-12)
    return (numerator / denominator).mean()


def _terms(
    communicator, chart, context, batch, teacher_code, spec, scale,
    solution_std: float,
) -> dict[str, Tensor]:
    inference = inference_only_context(context)
    q, _ = measured_q(inference, batch, spec, scale, "value_flux")
    assembled, predicted, _, detail = forward_composer(
        communicator, chart, context, scale, batch, aligned_features=q,
    )
    consensus = context.cover.tile(assembled)
    energy, residual = physics_losses(assembled, batch, solution_std)
    value, flux = interface_losses(detail["corrected"], assembled, context, batch)
    result = {
        "teacher_code": F.smooth_l1_loss(predicted, teacher_code),
        "assembled": masked_mse(assembled, context.target_normalized, batch["mask"]),
        "local": masked_mse(detail["corrected"], context.target_tiles, context.valid),
        "consensus": masked_mse(detail["corrected"], consensus, context.valid),
        "energy": energy, "residual": residual,
        "value_interface": value, "flux_interface": flux,
        "complete_q_cycle_square": _complete_cycle_square(
            detail["corrected"], context, batch, spec, scale, q,
        ),
    }
    return result


def _objective(terms: dict[str, Tensor], args) -> Tensor:
    # No discrete PDE operator contributes to the optimized objective. Energy
    # and FV residual are diagnostic selection guards only.
    return (
        terms["teacher_code"]
        + args.field_weight * (terms["assembled"] + 0.2 * terms["local"])
        + args.interface_weight * (
            terms["consensus"] + terms["value_interface"]
            + 0.1 * terms["flux_interface"]
        )
        + args.cycle_weight * terms["complete_q_cycle_square"]
    )


@torch.no_grad()
def _validate(
    communicator, chart, local, adapter, loader, device, spec, scale,
    args, maximum_cases: int,
) -> dict[str, float]:
    communicator.eval(); totals: dict[str, float] = {}; count = 0
    for host in loader:
        batch = move_batch(host, device)
        teacher = batch.pop("teacher_code", None)
        original = batch["solution"].shape[0]
        keep = min(original, maximum_cases - count)
        if keep <= 0: break
        for name, value in list(batch.items()):
            if isinstance(value, Tensor) and value.shape[0] == original:
                batch[name] = value[:keep]
        if teacher is None:
            teacher = torch.zeros(
                keep, 9, spec.code_width, device=device,
            )
        else:
            teacher = teacher[:keep]
        context = frozen_context(local, adapter, batch, spec, args.core, args.halo)
        terms = _terms(
            communicator, chart, context, batch, teacher, spec, scale,
            adapter.statistics.solution_std,
        )
        for name, value in terms.items():
            if name == "teacher_code" and "teacher_code" not in host:
                continue
            totals[name] = totals.get(name, 0.0) + float(value) * keep
        count += keep
        if count >= maximum_cases: break
    if count == 0: raise RuntimeError("empty distillation validation")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def _identity_audit(communicator, context, signal: Tensor) -> dict[str, float | bool]:
    zeros = torch.zeros_like(signal)
    zero_code, _ = communicator(
        context.local, context.physical, context.scalar, context.valid,
        context.cover.grid_shape, aligned_features=zeros,
    )
    left = 0.37 * signal; right = -0.21 * signal.roll(1, 1)
    combined, _ = communicator(
        context.local, context.physical, context.scalar, context.valid,
        context.cover.grid_shape, aligned_features=left + right,
    )
    first, _ = communicator(
        context.local, context.physical, context.scalar, context.valid,
        context.cover.grid_shape, aligned_features=left,
    )
    second, _ = communicator(
        context.local, context.physical, context.scalar, context.valid,
        context.cover.grid_shape, aligned_features=right,
    )
    error = float((combined - first - second).abs().max())
    scale = max(float(combined.abs().max()), 1.0)
    return {
        "zero_signal_max_abs_code": float(zero_code.abs().max()),
        "zero_signal_zero_code_bitwise": bool(torch.count_nonzero(zero_code) == 0),
        "fixed_geometry_linearity_max_abs_error": error,
        "fixed_geometry_linearity_relative_error": error / scale,
    }


def _payload(
    communicator, chart, adapter, spec, scale, chart_checkpoint, initial,
    optimizer, scheduler, args, epoch, history, initial_metrics, identity,
) -> dict[str, Any]:
    signal_width = source_linear_signal_width(spec) + 8
    return {
        "contract": CONTRACT,
        "phase": "train_only_value_flux_teacher_distillation",
        "epoch": epoch, "communicator": communicator.state_dict(),
        "chart": chart.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "statistics": adapter.statistics.to_json(), "port_spec": spec.to_json(),
        "port_scale": scale.detach().cpu(),
        "source_local_checkpoint": str(args.local_checkpoint.resolve()),
        "source_local_sha256": sha256(args.local_checkpoint),
        "source_chart_checkpoint": str(args.chart_checkpoint.resolve()),
        "source_chart_sha256": sha256(args.chart_checkpoint),
        "source_teacher_corpus": str(args.teacher.resolve()),
        "source_teacher_sha256": sha256(args.teacher),
        "initial_communicator_checkpoint": str(args.initial_checkpoint.resolve()),
        "initial_communicator_sha256": sha256(args.initial_checkpoint),
        "chart_spec": chart_checkpoint["chart_spec"],
        "communicator_spec": {
            "physical_channels": 29, "scalar_dim": 4,
            "token_width": args.token_width, "heads": args.heads,
            "relative_attention_count": 1, "absolute_tile_identifier": False,
            "analyzer_mode": "value_flux",
            "canonical_analyzer_contract": CANONICAL_ANALYZER_CONTRACT,
            "kernel_type": args.kernel_type,
            "compatibility_residual_sign": "q=d=P_chi(v)-v",
            "q_d_coordinates": [], "end_to_end_source_linearity_claimed": False,
            "aligned_width": signal_width,
            "source_linear_signal_width": signal_width,
            "qk_geometry_only": True, "bias_free_linear_value_and_output": True,
            "shared_residual_kernel_code_map": True,
            "zero_signal_zero_code_bitwise": True,
            "signal_definition": (
                "homogeneous POU value moments plus eight permeability-weighted "
                "normal-flux defect moments on artificial faces"
            ),
            "raw_source_or_boundary_coordinates": False,
            "compatible_endpoint_signal_zero_up_to_roundoff": True,
            "operator_free_interface_observations": True,
            "coordinate_roles": ["compatibility"] * spec.code_width,
            "qoi_coordinate_count": 0,
        },
        "teacher_selection": {
            "source_split": "train", "heldout_rule": "case_index modulo 10 equals 0",
            "target_or_pde_labels_used_by_teacher": False,
            "ordinary_nonregression_factor": ORDINARY_NONREGRESSION_FACTOR,
            "teacher_improvement_factor": TEACHER_IMPROVEMENT_FACTOR,
        },
        "initial_validation": initial_metrics, "identity_audit": identity,
        "training_splits": ["train", "validation"],
        "test_subsets_opened": False,
        "local_calls_per_case": "one batched call; local_nfe=tile_count",
        "args": vars(args), "history": history,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(args.chart_checkpoint, device)
    if chart_checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("chart/local lineage differs")
    signal_width = source_linear_signal_width(spec) + 8
    cls = (
        SignedSourceLinearResidualPortCommunicator
        if args.kernel_type == "signed_l1_tanh"
        else SourceLinearResidualPortCommunicator
    )
    communicator = cls(
        29, 4, spec.code_width, signal_width,
        token_width=args.token_width, heads=args.heads,
    ).to(device)
    communicator.analyzer_mode = "value_flux"
    initial = _load_expanded_initial(
        communicator, args.initial_checkpoint, args.local_checkpoint,
        args.chart_checkpoint,
    )
    attention_count = sum(
        isinstance(module, ResidualKernelAttention)
        for module in communicator.modules()
    )
    # SignedResidualKernelAttention is intentionally a different class; its
    # architecture still contains exactly one global kernel at code_map.
    if args.kernel_type == "softmax" and attention_count != 1:
        raise RuntimeError("distilled communicator must have exactly one attention")

    train_set = ValueFluxTeacherDataset(
        args.data, args.teacher, args.local_checkpoint, args.chart_checkpoint,
        holdout=False,
    )
    teacher_val_set = ValueFluxTeacherDataset(
        args.data, args.teacher, args.local_checkpoint, args.chart_checkpoint,
        holdout=True,
    )
    ordinary_set = H5DarcyDataset(args.data, "validation")
    if not ordinary_set.has_face_dirichlet:
        raise ValueError("ordinary validation must be face-resolved")
    train_loader = _loader(
        train_set, args.batch_size, args.workers, device,
        samples=args.samples_per_epoch,
    )
    teacher_val_loader = _loader(
        teacher_val_set, args.batch_size, min(args.workers, 2), device,
    )
    ordinary_loader = _loader(
        ordinary_set, args.batch_size, min(args.workers, 2), device,
    )
    optimizer = torch.optim.AdamW(
        communicator.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1.0e-6,
    )
    initial_teacher = _validate(
        communicator, chart, local, adapter, teacher_val_loader, device,
        spec, scale, args, args.teacher_validation_cases,
    )
    initial_ordinary = _validate(
        communicator, chart, local, adapter, ordinary_loader, device,
        spec, scale, args, args.ordinary_validation_cases,
    )
    first = next(iter(teacher_val_loader)); first_batch = move_batch(first, device)
    first_batch.pop("teacher_code")
    first_context = frozen_context(
        local, adapter, first_batch, spec, args.core, args.halo,
    )
    first_q = measured_q(
        inference_only_context(first_context), first_batch, spec, scale,
        "value_flux",
    )[0]
    identity = _identity_audit(communicator, first_context, first_q)
    initial_metrics = {"teacher": initial_teacher, "ordinary": initial_ordinary}
    history: list[dict[str, Any]] = []; best = math.inf
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_payload = _payload(
        communicator, chart, adapter, spec, scale, chart_checkpoint, initial,
        optimizer, scheduler, args, 0, history, initial_metrics, identity,
    )
    torch.save(initial_payload, args.output_dir / "initial_checkpoint.pt")

    guarded = (
        "assembled", "energy", "residual", "value_interface",
        "flux_interface", "complete_q_cycle_square",
    )
    for epoch in range(1, args.epochs + 1):
        communicator.train(); totals: dict[str, float] = {}; count = 0
        for host in train_loader:
            batch = move_batch(host, device); teacher_code = batch.pop("teacher_code")
            context = frozen_context(local, adapter, batch, spec, args.core, args.halo)
            optimizer.zero_grad(set_to_none=True)
            terms = _terms(
                communicator, chart, context, batch, teacher_code, spec, scale,
                adapter.statistics.solution_std,
            )
            loss = _objective(terms, args)
            loss.backward(); torch.nn.utils.clip_grad_norm_(communicator.parameters(), 1.0)
            optimizer.step(); batch_count = batch["solution"].shape[0]
            for name, value in {"objective": loss, **terms}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_count
            count += batch_count
        scheduler.step()
        teacher_validation = _validate(
            communicator, chart, local, adapter, teacher_val_loader, device,
            spec, scale, args, args.teacher_validation_cases,
        )
        ordinary_validation = _validate(
            communicator, chart, local, adapter, ordinary_loader, device,
            spec, scale, args, args.ordinary_validation_cases,
        )
        gates = {
            f"ordinary_{name}_nonregression": ordinary_validation[name]
            <= ORDINARY_NONREGRESSION_FACTOR * initial_ordinary[name]
            for name in guarded
        }
        gates.update({
            "heldout_train_teacher_code_improves":
                teacher_validation["teacher_code"]
                <= TEACHER_IMPROVEMENT_FACTOR * initial_teacher["teacher_code"],
            "heldout_train_complete_q_improves":
                teacher_validation["complete_q_cycle_square"]
                <= TEACHER_IMPROVEMENT_FACTOR
                * initial_teacher["complete_q_cycle_square"],
        })
        clears = all(gates.values())
        score = (
            teacher_validation["teacher_code"]
            / max(initial_teacher["teacher_code"], 1.0e-12)
            + teacher_validation["complete_q_cycle_square"]
            / max(initial_teacher["complete_q_cycle_square"], 1.0e-12)
        )
        row = {
            "epoch": epoch, "selection_score": score,
            "predeclared_gates_clear": clears,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"gate_{name}": value for name, value in gates.items()},
            **{f"train_{name}": value / max(count, 1) for name, value in totals.items()},
            **{f"teacher_validation_{name}": value for name, value in teacher_validation.items()},
            **{f"ordinary_validation_{name}": value for name, value in ordinary_validation.items()},
        }
        history.append(row)
        payload = _payload(
            communicator, chart, adapter, spec, scale, chart_checkpoint, initial,
            optimizer, scheduler, args, epoch, history, initial_metrics, identity,
        )
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        if clears and score < best:
            best = score; torch.save(payload, args.output_dir / "checkpoint.pt")
        print(json.dumps(row, sort_keys=True), flush=True)
    for dataset in (train_set, teacher_val_set, ordinary_set): dataset.close()
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    result = {
        "contract": CONTRACT, "best_selection_score": best,
        "selected_checkpoint_exists": (args.output_dir / "checkpoint.pt").exists(),
        "test_subsets_opened": False, "initial_validation": initial_metrics,
        "identity_audit": identity, "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    if not result["selected_checkpoint_exists"]:
        raise RuntimeError(
            "no value+flux communicator cleared the predeclared train/validation gates"
        )
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
