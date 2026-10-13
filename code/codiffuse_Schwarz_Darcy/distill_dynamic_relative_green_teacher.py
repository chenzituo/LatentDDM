"""Distill the train-only Schur teacher into the dynamic relative Green map.

This stage consumes only the 4,000-case ``train`` teacher export.  The frozen
local endpoint, geometry-tied linear chart, and accepted value communicator are
never optimized.  For fixed geometry, the only learned action remains

    z_flux,i = sum_j K_ij(h_i, h_j) q_flux,j,

so it is exactly linear and zero preserving in the compatibility signal.  A
deterministic train-corpus partition (case index modulo ten) selects epoch and
trust scale.  The selection gate requires both improved teacher-code recovery
and non-regression of the safeguarded ordinary solution field.  No validation
or test split is opened.

Solution fields are used only as *train-time* non-regression labels.  They,
the teacher code, and a discrete PDE operator are never inference inputs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .export_value_flux_teacher import CONTRACT as TEACHER_CONTRACT
from .latent_schur_teacher import inference_only_context
from .residual_port_composer import (
    DynamicRelativeGreenFluxResidualPortCommunicator,
)
from .train import seed_everything
from .train_incremental_flux_remote_schur import (
    _decomposed_fields, _load_communicator,
)
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT, load_frozen_chart,
)


Tensor = torch.Tensor
RUN_CONTRACT = "darcy_dynamic_relative_green_teacher_distillation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--min-lr", type=float, default=2.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--field-weight", type=float, default=2.0)
    parser.add_argument("--l2sp-weight", type=float, default=1.0e-4)
    parser.add_argument("--field-nonregression-factor", type=float, default=1.02)
    parser.add_argument("--teacher-improvement-factor", type=float, default=0.95)
    parser.add_argument("--selection-modulus", type=int, default=10)
    parser.add_argument("--selection-remainder", type=int, default=0)
    parser.add_argument(
        "--trust-scales", type=float, nargs="+",
        default=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _load_teacher(
    path: Path, data_path: Path, local_path: Path, chart_path: Path,
    maximum_cases: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and audit an explicitly train-only, target-free teacher export."""
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
        if attrs.get("contract") != TEACHER_CONTRACT:
            raise ValueError("teacher corpus has the wrong contract")
        if attrs.get("source_split") != "train":
            raise ValueError("teacher corpus is not train-only")
        if bool(attrs.get("validation_or_test_opened", True)):
            raise ValueError("teacher export opened validation or test data")
        if bool(attrs.get("target_or_pde_labels_used", True)):
            raise ValueError("teacher export used forbidden targets/PDE labels")
        expected = {
            "source_data_sha256": sha256(data_path),
            "local_checkpoint_sha256": sha256(local_path),
            "chart_checkpoint_sha256": sha256(chart_path),
        }
        for name, value in expected.items():
            if attrs.get(name) != value:
                raise ValueError(f"teacher lineage mismatch for {name}")
        indices = np.asarray(handle["train/case_index"], dtype=np.int64)
        codes = np.asarray(handle["train/teacher_code"], dtype=np.float32)
    if maximum_cases <= 1 or len(indices) < maximum_cases:
        raise ValueError(
            f"requested {maximum_cases} cases from a {len(indices)}-case teacher"
        )
    indices, codes = indices[:maximum_cases], codes[:maximum_cases]
    if codes.shape != (maximum_cases, 9, 4):
        raise ValueError(f"unexpected teacher-code shape {codes.shape}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("teacher case indices are not unique")
    return indices, codes, attrs


def _case_square(error: Tensor, weight: Tensor) -> Tensor:
    axes = tuple(range(1, error.ndim))
    return (error.square() * weight).sum(axes)


def _teacher_relative_square(
    prediction: Tensor, teacher: Tensor, accepted: Tensor,
) -> Tensor:
    numerator = (prediction - teacher).square().sum((1, 2))
    denominator = (accepted - teacher).square().sum((1, 2)).clamp_min(1.0e-12)
    return numerator / denominator


def _batch_terms(
    candidate: DynamicRelativeGreenFluxResidualPortCommunicator,
    chart: torch.nn.Module,
    local: torch.nn.Module,
    adapter,
    batch: dict[str, Tensor],
    teacher: Tensor,
    spec,
    scale: Tensor,
    args: argparse.Namespace,
) -> dict[str, Tensor]:
    supervised = frozen_context(
        local, adapter, batch, spec, args.core, args.halo,
    )
    context = inference_only_context(supervised)
    fields = _decomposed_fields(
        candidate, chart, context, batch, spec, scale,
    )
    code_ratio = _teacher_relative_square(
        fields["total_code"], teacher, fields["base_code"],
    )
    target = supervised.target_normalized
    weight = batch["mask"]
    base_assembled = context.cover.assemble_weighted(
        fields["base"], target.shape[0],
    ) * weight
    base_square = _case_square(base_assembled - target, weight)
    raw_square = _case_square(fields["raw_assembled"] - target, weight)
    deployed_square = _case_square(fields["assembled"] - target, weight)
    denominator = base_square.clamp_min(1.0e-12)
    raw_ratio = raw_square / denominator
    deployed_ratio = deployed_square / denominator
    # The teacher term supplies the direction; this hinge prevents the code
    # fit from purchasing it by degrading ordinary train fields.
    deployed_hinge = F.smooth_l1_loss(
        torch.relu(deployed_ratio - args.field_nonregression_factor),
        torch.zeros_like(deployed_ratio), beta=0.05,
    )
    # Train the actual raw direction as well as its target-free safeguarded
    # realization.  Otherwise an always-zero affine step could satisfy the
    # deployed field gate without learning a useful correction.
    raw_hinge = F.smooth_l1_loss(
        torch.relu(raw_ratio - args.field_nonregression_factor),
        torch.zeros_like(raw_ratio), beta=0.05,
    )
    return {
        "teacher_relative_square": code_ratio.mean(),
        "teacher_balanced_log": torch.log1p(code_ratio).mean(),
        "raw_field_ratio": raw_ratio.mean(),
        "deployed_field_ratio": deployed_ratio.mean(),
        "raw_field_nonregression_hinge": raw_hinge,
        "deployed_field_nonregression_hinge": deployed_hinge,
        "affine_alpha_mean": fields["alpha"].mean(),
        "increment_code_rms": (
            fields["total_code"] - fields["base_code"]
        ).square().mean().sqrt(),
    }


def _trust_parameters(candidate) -> tuple[torch.nn.Parameter, ...]:
    parameters = tuple(candidate.flux_green.trust_path_parameters())
    if len(parameters) != 2:
        raise RuntimeError("dynamic trust path must contain K and coefficient head")
    return parameters


def _set_trust_scale(
    parameters: tuple[torch.nn.Parameter, ...], terminal: tuple[Tensor, ...],
    trust: float,
) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, terminal, strict=True):
            parameter.copy_(trust * value)


@torch.no_grad()
def _evaluate(
    loader: DataLoader, lookup: dict[int, int], teacher_codes: np.ndarray,
    candidate, chart, local, adapter, spec, scale, args, device,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    cases = 0
    candidate.eval()
    for host in loader:
        batch = move_batch(host, device)
        rows = [lookup[int(value)] for value in batch["case_index"].cpu().numpy()]
        teacher = torch.from_numpy(teacher_codes[rows]).to(device)
        terms = _batch_terms(
            candidate, chart, local, adapter, batch, teacher, spec, scale, args,
        )
        count = int(batch["solution"].shape[0])
        cases += count
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + count * float(value)
    return {name: value / cases for name, value in totals.items()}


def _assert_trust_zero_base(
    candidate, chart, local, adapter, host, spec, scale, args, device,
) -> float:
    """Prove the learned branch has an exact accepted-base endpoint."""
    batch = move_batch(host, device)
    supervised = frozen_context(local, adapter, batch, spec, args.core, args.halo)
    context = inference_only_context(supervised)
    from .validate_latent_schur import measured_q
    from .train_residual_port_communicator import source_linear_signal_width

    aligned = measured_q(context, batch, spec, scale, "value_flux")[0]
    prefix = source_linear_signal_width(spec)
    trust_parameters = _trust_parameters(candidate)
    saved = tuple(parameter.detach().clone() for parameter in trust_parameters)
    _set_trust_scale(trust_parameters, saved, 0.0)
    total, accepted, _ = candidate.forward_decomposed(
        context.local, context.physical, context.scalar, context.valid,
        context.cover.grid_shape, prefix, aligned_features=aligned,
    )
    difference = float((total - accepted).abs().max())
    _set_trust_scale(trust_parameters, saved, 1.0)
    if difference != 0.0:
        raise RuntimeError(f"trust-zero failed exact base recovery: {difference}")
    return difference


@torch.no_grad()
def _assert_inference_target_invariance(
    candidate, local, adapter, host, spec, scale, args, device,
) -> float:
    """Replace train labels and prove that the deployed code is unchanged."""
    from .train_residual_port_communicator import source_linear_signal_width
    from .validate_latent_schur import measured_q

    first_batch = move_batch(host, device)
    second_batch = dict(first_batch)
    second_batch["solution"] = torch.full_like(first_batch["solution"], 37.0)
    outputs = []
    for batch in (first_batch, second_batch):
        context = inference_only_context(frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        ))
        aligned = measured_q(context, batch, spec, scale, "value_flux")[0]
        code = candidate.forward_decomposed(
            context.local, context.physical, context.scalar, context.valid,
            context.cover.grid_shape, source_linear_signal_width(spec),
            aligned_features=aligned,
        )[0]
        outputs.append((aligned, code))
    difference = max(
        float((left - right).abs().max())
        for left, right in zip(outputs[0], outputs[1], strict=True)
    )
    if difference > 1.0e-6:
        raise RuntimeError(
            f"dynamic inference depends on replaced solution labels: {difference}"
        )
    return difference


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("training extent must be positive")
    if args.selection_modulus <= 1 or not (
        0 <= args.selection_remainder < args.selection_modulus
    ):
        raise ValueError("invalid train-only selection partition")
    if any(not 0.0 <= value <= 1.0 for value in args.trust_scales):
        raise ValueError("trust scales must lie in [0,1]")
    if 0.0 not in args.trust_scales:
        raise ValueError("trust candidates must include the exact accepted base")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    indices, teacher_codes, teacher_attrs = _load_teacher(
        args.teacher, args.data, args.local_checkpoint, args.chart_checkpoint,
        args.maximum_cases,
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    candidate, initial = _load_communicator(
        args.initial_checkpoint, spec, device,
    )
    if not isinstance(candidate, DynamicRelativeGreenFluxResidualPortCommunicator):
        raise ValueError("initial checkpoint is not value_flux_dynamic_relative_green")
    cfg = initial["communicator_spec"]
    if cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
        raise ValueError("initial checkpoint lacks the homogeneous q contract")
    if initial.get("source_local_sha256") != sha256(args.local_checkpoint) or \
            initial.get("source_chart_sha256") != sha256(args.chart_checkpoint):
        raise ValueError("initial communicator lineage differs")
    if cfg.get("fixed_context_q_linearity") is not True or \
            cfg.get("dynamic_kernel_context") != "geometry_only":
        raise ValueError("dynamic checkpoint lacks fixed-context q-linearity")

    for module in (local, chart, candidate):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    parameters = tuple(candidate.flux_green.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    initial_parameters = tuple(parameter.detach().clone() for parameter in parameters)
    frozen_base = {
        name: value.detach().clone()
        for name, value in candidate.base.state_dict().items()
    }

    lookup = {int(index): row for row, index in enumerate(indices)}
    selection_mask = indices % args.selection_modulus == args.selection_remainder
    fit_indices = indices[~selection_mask]
    selection_indices = indices[selection_mask]
    if len(fit_indices) == 0 or len(selection_indices) == 0:
        raise ValueError("empty fit or train-selection partition")
    dataset = H5DarcyDataset(args.data, "train")
    if max(indices) >= len(dataset):
        raise ValueError("teacher case index exceeds the train dataset")
    generator = torch.Generator().manual_seed(args.seed)
    fit_loader = DataLoader(
        Subset(dataset, [int(index) for index in fit_indices]),
        batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    selection_loader = DataLoader(
        Subset(dataset, [int(index) for index in selection_indices]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers, 2), pin_memory=device.type == "cuda",
    )
    first_host = next(iter(selection_loader))
    trust_zero_error = _assert_trust_zero_base(
        candidate, chart, local, adapter, first_host, spec, scale, args, device,
    )
    target_invariance_error = _assert_inference_target_invariance(
        candidate, local, adapter, first_host, spec, scale, args, device,
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    best_score = math.inf
    for epoch in range(1, args.epochs + 1):
        # Only the dynamic kernel enters training mode.  Keep the accepted
        # base explicitly frozen in its deployment/evaluation mode.
        candidate.flux_green.train()
        candidate.base.eval()
        totals: dict[str, float] = {}
        cases = 0
        for host in fit_loader:
            batch = move_batch(host, device)
            rows = [lookup[int(value)] for value in batch["case_index"].cpu().numpy()]
            teacher = torch.from_numpy(teacher_codes[rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            terms = _batch_terms(
                candidate, chart, local, adapter, batch, teacher, spec, scale, args,
            )
            l2sp = sum(
                (parameter - initial_value).square().mean()
                for parameter, initial_value in zip(
                    parameters, initial_parameters, strict=True,
                )
            )
            loss = (
                terms["teacher_balanced_log"]
                + args.field_weight * (
                    terms["raw_field_nonregression_hinge"]
                    + terms["deployed_field_nonregression_hinge"]
                )
                + args.l2sp_weight * l2sp
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite dynamic distillation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            count = int(batch["solution"].shape[0])
            cases += count
            for name, value in {"objective": loss, "l2sp": l2sp, **terms}.items():
                totals[name] = totals.get(name, 0.0) + count * float(value.detach())
        scheduler.step()
        fit_metrics = {name: value / cases for name, value in totals.items()}
        trust_parameters = _trust_parameters(candidate)
        terminal = tuple(parameter.detach().clone() for parameter in trust_parameters)
        candidate.eval()
        for trust in sorted(set(args.trust_scales)):
            _set_trust_scale(trust_parameters, terminal, trust)
            metrics = _evaluate(
                selection_loader, lookup, teacher_codes, candidate, chart,
                local, adapter, spec, scale, args, device,
            )
            gates = {
                "teacher_improves_accepted": metrics["teacher_relative_square"]
                <= args.teacher_improvement_factor,
                "ordinary_field_nonregression": metrics["deployed_field_ratio"]
                <= args.field_nonregression_factor,
                "raw_direction_nonregression": metrics["raw_field_ratio"]
                <= args.field_nonregression_factor,
            }
            clears = all(gates.values())
            score = metrics["teacher_relative_square"] + max(
                metrics["deployed_field_ratio"] - 1.0, 0.0,
            )
            row = {
                "epoch": epoch, "trust_scale": trust, "clears": clears,
                "score": score, "gates": gates,
                **{f"fit_{name}": value for name, value in fit_metrics.items()},
                **{f"selection_{name}": value for name, value in metrics.items()},
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if clears and score < best_score:
                best_score = score
                selected = row
                payload = dict(initial)
                output_cfg = dict(cfg)
                output_cfg.update({
                    "analyzer_mode": "value_flux_dynamic_relative_green",
                    "teacher_distilled": True,
                    "deployment_selected_trust_scale": trust,
                    "base_value_operator_frozen": True,
                    "local_endpoint_frozen": True,
                    "response_chart_frozen": True,
                    "fixed_context_q_linearity": True,
                    "exact_trust_zero_base": True,
                    "one_global_communication_stage": True,
                })
                payload.update({
                    "contract": CONTRACT, "phase": RUN_CONTRACT,
                    "epoch": epoch, "communicator": copy.deepcopy(candidate.state_dict()),
                    "communicator_spec": output_cfg,
                    "initial_checkpoint": str(args.initial_checkpoint.resolve()),
                    "initial_checkpoint_sha256": sha256(args.initial_checkpoint),
                    "teacher_corpus": str(args.teacher.resolve()),
                    "teacher_corpus_sha256": sha256(args.teacher),
                    "source_data": str(args.data.resolve()),
                    "source_data_sha256": sha256(args.data),
                    "training_splits": ["train"],
                    "selection_partition": (
                        f"train case_index mod {args.selection_modulus} == "
                        f"{args.selection_remainder}"
                    ),
                    "fit_cases": int(len(fit_indices)),
                    "selection_cases": int(len(selection_indices)),
                    "selected_metrics": row,
                    "teacher_export_target_or_pde_labels_used": bool(
                        teacher_attrs["target_or_pde_labels_used"]
                    ),
                    "solution_targets_used_during_training": True,
                    "solution_targets_used_at_inference": False,
                    "teacher_code_used_at_inference": False,
                    "discrete_pde_operator_used_during_training": False,
                    "discrete_pde_operator_used_at_inference": False,
                    "exact_global_solution_used_at_inference": False,
                    "validation_or_test_opened": False,
                    "test_subsets_opened": False,
                    "trust_zero_base_max_abs": trust_zero_error,
                    "solution_replacement_invariance_max_abs": (
                        target_invariance_error
                    ),
                    "args": vars(args),
                })
                torch.save(payload, args.output_dir / "checkpoint.pt")
        _set_trust_scale(trust_parameters, terminal, 1.0)
        last = dict(initial)
        last.update({
            "contract": CONTRACT, "phase": RUN_CONTRACT, "epoch": epoch,
            "communicator": candidate.state_dict(), "history": history,
            "validation_or_test_opened": False, "test_subsets_opened": False,
            "args": vars(args),
        })
        torch.save(last, args.output_dir / "last_checkpoint.pt")
    frozen_base_error = max(
        float((value - frozen_base[name]).abs().max())
        for name, value in candidate.base.state_dict().items()
    )
    if frozen_base_error != 0.0:
        raise RuntimeError(f"accepted base changed during distillation: {frozen_base_error}")
    dataset.close()
    result = {
        "contract": RUN_CONTRACT,
        "selected": selected,
        "fit_cases": int(len(fit_indices)),
        "selection_cases": int(len(selection_indices)),
        "trust_zero_base_max_abs": trust_zero_error,
        "solution_replacement_invariance_max_abs": target_invariance_error,
        "frozen_base_max_abs_parameter_change": frozen_base_error,
        "fixed_context_q_linearity": True,
        "validation_or_test_opened": False,
        "test_subsets_opened": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        flat = [{
            key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            for key, value in row.items()
        } for row in history]
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    if selected is None:
        raise RuntimeError("dynamic teacher distillation did not clear train-only gates")
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
