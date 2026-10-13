"""Test executable-state residual values around the accepted Darcy map.

This is a train-only, identity-lifted ablation.  The accepted local endpoint,
response chart, geometry-conditioned residual kernel, and dynamic Green map are
frozen.  Only the two new zero-gated paths and their latent bases train.  Model
selection compares against the *full accepted LatentDDM*, not the weaker value
prefix, so a new checkpoint is accepted only when it improves teacher-code fit
without degrading the train-only held-out physical field.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .adapter import H5DarcyDataset
from .distill_dynamic_relative_green_teacher import (
    _case_square, _load_teacher, _teacher_relative_square,
)
from .evaluate import move_batch
from .latent_schur_teacher import inference_only_context
from .residual_port_composer import (
    DynamicRelativeGreenFluxResidualPortCommunicator,
    StateGatedDynamicRelativeGreenFluxResidualPortCommunicator,
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
RUN_CONTRACT = "darcy_state_gated_dynamic_teacher_distillation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--accepted-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr", type=float, default=2.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--field-weight", type=float, default=4.0)
    parser.add_argument("--l2sp-weight", type=float, default=1.0e-4)
    parser.add_argument("--field-nonregression-factor", type=float, default=1.01)
    parser.add_argument("--teacher-improvement-factor", type=float, default=0.99)
    parser.add_argument("--selection-modulus", type=int, default=10)
    parser.add_argument("--selection-remainder", type=int, default=0)
    parser.add_argument(
        "--trust-scales", type=float, nargs="+",
        default=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _batch_terms(
    candidate: StateGatedDynamicRelativeGreenFluxResidualPortCommunicator,
    reference: DynamicRelativeGreenFluxResidualPortCommunicator,
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
    candidate_fields = _decomposed_fields(
        candidate, chart, context, batch, spec, scale,
    )
    with torch.no_grad():
        reference_fields = _decomposed_fields(
            reference, chart, context, batch, spec, scale,
        )
    code_ratio = _teacher_relative_square(
        candidate_fields["total_code"], teacher,
        reference_fields["total_code"],
    )
    target = supervised.target_normalized
    weight = batch["mask"]
    reference_raw_square = _case_square(
        reference_fields["raw_assembled"] - target, weight,
    ).clamp_min(1.0e-12)
    reference_deployed_square = _case_square(
        reference_fields["assembled"] - target, weight,
    ).clamp_min(1.0e-12)
    raw_square = _case_square(
        candidate_fields["raw_assembled"] - target, weight,
    )
    deployed_square = _case_square(
        candidate_fields["assembled"] - target, weight,
    )
    raw_ratio = raw_square / reference_raw_square
    deployed_ratio = deployed_square / reference_deployed_square
    raw_hinge = F.smooth_l1_loss(
        torch.relu(raw_ratio - args.field_nonregression_factor),
        torch.zeros_like(raw_ratio), beta=0.02,
    )
    deployed_hinge = F.smooth_l1_loss(
        torch.relu(deployed_ratio - args.field_nonregression_factor),
        torch.zeros_like(deployed_ratio), beta=0.02,
    )
    return {
        "teacher_relative_square": code_ratio.mean(),
        "teacher_balanced_log": torch.log1p(code_ratio).mean(),
        "raw_field_ratio": raw_ratio.mean(),
        "deployed_field_ratio": deployed_ratio.mean(),
        "raw_field_nonregression_hinge": raw_hinge,
        "deployed_field_nonregression_hinge": deployed_hinge,
        "affine_alpha_mean": candidate_fields["alpha"].mean(),
        "state_code_delta_rms": (
            candidate_fields["total_code"] - reference_fields["total_code"]
        ).square().mean().sqrt(),
    }


@torch.no_grad()
def _evaluate(
    loader: DataLoader,
    lookup: dict[int, int],
    teacher_codes: np.ndarray,
    candidate,
    reference,
    chart,
    local,
    adapter,
    spec,
    scale,
    args,
    device,
) -> dict[str, float]:
    totals: dict[str, float] = {}; cases = 0
    candidate.eval(); reference.eval()
    for host in loader:
        batch = move_batch(host, device)
        rows = [lookup[int(value)] for value in batch["case_index"].cpu().numpy()]
        teacher = torch.from_numpy(teacher_codes[rows]).to(device)
        terms = _batch_terms(
            candidate, reference, chart, local, adapter, batch, teacher,
            spec, scale, args,
        )
        count = int(batch["solution"].shape[0]); cases += count
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + count * float(value)
    return {name: value / cases for name, value in totals.items()}


def _trust_parameters(
    candidate: StateGatedDynamicRelativeGreenFluxResidualPortCommunicator,
) -> tuple[torch.nn.Parameter, ...]:
    return tuple(candidate.trust_path_parameters())


def _set_trust_scale(
    parameters: tuple[torch.nn.Parameter, ...],
    terminal: tuple[Tensor, ...],
    trust: float,
) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, terminal, strict=True):
            parameter.copy_(trust * value)


@torch.no_grad()
def _real_identity_audit(
    candidate, reference, chart, local, adapter, host, spec, scale, args, device,
) -> dict[str, float]:
    batch = move_batch(host, device)
    supervised = frozen_context(local, adapter, batch, spec, args.core, args.halo)
    context = inference_only_context(supervised)
    candidate_fields = _decomposed_fields(
        candidate, chart, context, batch, spec, scale,
    )
    reference_fields = _decomposed_fields(
        reference, chart, context, batch, spec, scale,
    )
    result = {
        name: float((candidate_fields[name] - reference_fields[name]).abs().max())
        for name in ("total_code", "raw", "assembled")
    }
    if max(result.values()) != 0.0:
        raise RuntimeError(f"real Darcy identity lift failed: {result}")
    zero = torch.zeros_like(candidate_fields["aligned"])
    prefix = candidate.prefix_width
    zero_code = candidate.forward_decomposed(
        context.local, context.physical, context.scalar, context.valid,
        context.cover.grid_shape, prefix, aligned_features=zero,
    )[0]
    if torch.count_nonzero(zero_code):
        raise RuntimeError("zero residual produced a nonzero state-gated code")
    return result


def _trainable_parameters(candidate) -> tuple[torch.nn.Parameter, ...]:
    selected = [
        candidate.base.state_context_gate,
        candidate.base.code_map.global_kernel.state_value_coefficients.weight,
        candidate.base.code_map.global_kernel.state_value_basis,
        *candidate.base.state_analyzer.parameters(),
    ]
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in selected:
        if id(parameter) not in seen:
            unique.append(parameter); seen.add(id(parameter))
    return tuple(unique)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("training extent must be positive")
    if args.selection_modulus <= 1 or not (
        0 <= args.selection_remainder < args.selection_modulus
    ):
        raise ValueError("invalid train-only selection partition")
    if any(not 0.0 <= value <= 1.0 for value in args.trust_scales) or (
        0.0 not in args.trust_scales
    ):
        raise ValueError("trust scales must lie in [0,1] and include zero")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    indices, teacher_codes, teacher_attrs = _load_teacher(
        args.teacher, args.data, args.local_checkpoint, args.chart_checkpoint,
        args.maximum_cases,
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    candidate, initial = _load_communicator(args.initial_checkpoint, spec, device)
    reference, reference_payload = _load_communicator(
        args.accepted_reference, spec, device,
    )
    if not isinstance(
        candidate, StateGatedDynamicRelativeGreenFluxResidualPortCommunicator,
    ):
        raise ValueError("initial checkpoint is not the state-gated identity lift")
    if not isinstance(reference, DynamicRelativeGreenFluxResidualPortCommunicator) or isinstance(
        reference, StateGatedDynamicRelativeGreenFluxResidualPortCommunicator,
    ):
        raise ValueError("reference must be the accepted pre-extension dynamic map")
    cfg = initial["communicator_spec"]
    if cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
        raise ValueError("initial checkpoint lacks the homogeneous q contract")
    for payload in (initial, reference_payload):
        if payload.get("source_local_sha256") != sha256(args.local_checkpoint) or payload.get(
            "source_chart_sha256"
        ) != sha256(args.chart_checkpoint):
            raise ValueError("communicator lineage differs")
    if initial.get("source_accepted_sha256") != sha256(args.accepted_reference):
        raise ValueError("identity lift does not descend from the declared reference")

    for module in (local, chart, candidate, reference):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    parameters = _trainable_parameters(candidate)
    for parameter in parameters:
        parameter.requires_grad_(True)
    initial_parameters = tuple(parameter.detach().clone() for parameter in parameters)
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in candidate.named_parameters()
        if id(parameter) not in {id(value) for value in parameters}
    }

    lookup = {int(index): row for row, index in enumerate(indices)}
    selection_mask = indices % args.selection_modulus == args.selection_remainder
    fit_indices = indices[~selection_mask]; selection_indices = indices[selection_mask]
    if len(fit_indices) == 0 or len(selection_indices) == 0:
        raise ValueError("empty fit or train-selection partition")
    dataset = H5DarcyDataset(args.data, "train")
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
    identity_audit = _real_identity_audit(
        candidate, reference, chart, local, adapter, first_host,
        spec, scale, args, device,
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
    best_score = math.inf; best_safe_score = math.inf
    for epoch in range(1, args.epochs + 1):
        candidate.train(); reference.eval(); chart.eval(); local.eval()
        totals: dict[str, float] = {}; cases = 0
        for host in fit_loader:
            batch = move_batch(host, device)
            rows = [lookup[int(value)] for value in batch["case_index"].cpu().numpy()]
            teacher = torch.from_numpy(teacher_codes[rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            terms = _batch_terms(
                candidate, reference, chart, local, adapter, batch, teacher,
                spec, scale, args,
            )
            l2sp = sum(
                (parameter - origin).square().mean()
                for parameter, origin in zip(
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
                raise FloatingPointError("non-finite state-gated Darcy loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            count = int(batch["solution"].shape[0]); cases += count
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
                selection_loader, lookup, teacher_codes, candidate, reference,
                chart, local, adapter, spec, scale, args, device,
            )
            gates = {
                "teacher_improves_full_accepted": metrics[
                    "teacher_relative_square"
                ] <= args.teacher_improvement_factor,
                "deployed_field_nonregression": metrics[
                    "deployed_field_ratio"
                ] <= args.field_nonregression_factor,
                "raw_direction_nonregression": metrics[
                    "raw_field_ratio"
                ] <= args.field_nonregression_factor,
            }
            clears = all(gates.values())
            safe = gates["deployed_field_nonregression"] and gates[
                "raw_direction_nonregression"
            ]
            score = metrics["teacher_relative_square"] + 2.0 * max(
                metrics["deployed_field_ratio"] - 1.0, 0.0,
            )
            row = {
                "epoch": epoch, "trust_scale": trust, "clears": clears,
                "safe": safe, "score": score, "gates": gates,
                **{f"fit_{name}": value for name, value in fit_metrics.items()},
                **{f"selection_{name}": value for name, value in metrics.items()},
            }
            history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
            payload = dict(initial)
            output_cfg = dict(cfg)
            output_cfg.update({
                "teacher_distilled": True,
                "deployment_selected_trust_scale": trust,
                "accepted_reference_checkpoint": str(args.accepted_reference.resolve()),
                "accepted_reference_sha256": sha256(args.accepted_reference),
                "accepted_pre_extension_parameters_frozen": True,
                "fixed_context_q_linearity": True,
                "exact_trust_zero_accepted_map": True,
                "one_global_communication_stage": True,
            })
            payload.update({
                "contract": CONTRACT, "phase": RUN_CONTRACT,
                "epoch": epoch, "communicator": copy.deepcopy(candidate.state_dict()),
                "communicator_spec": output_cfg,
                "initial_checkpoint": str(args.initial_checkpoint.resolve()),
                "initial_checkpoint_sha256": sha256(args.initial_checkpoint),
                "accepted_reference": str(args.accepted_reference.resolve()),
                "accepted_reference_sha256": sha256(args.accepted_reference),
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
                "identity_lift_real_audit": identity_audit,
                "teacher_export_target_or_pde_labels_used": bool(
                    teacher_attrs["target_or_pde_labels_used"]
                ),
                "solution_targets_used_during_training": True,
                "solution_targets_used_at_inference": False,
                "teacher_code_used_at_inference": False,
                "discrete_pde_operator_used_during_training": False,
                "discrete_pde_operator_used_at_inference": False,
                "validation_or_test_opened": False,
                "test_subsets_opened": False,
                "args": vars(args),
            })
            if safe and score < best_safe_score:
                best_safe_score = score
                torch.save(payload, args.output_dir / "checkpoint_best_safe.pt")
            if clears and score < best_score:
                best_score = score; selected = row
                torch.save(payload, args.output_dir / "checkpoint.pt")
        _set_trust_scale(trust_parameters, terminal, 1.0)
        torch.save({
            **initial, "contract": CONTRACT, "phase": RUN_CONTRACT,
            "epoch": epoch, "communicator": candidate.state_dict(),
            "history": history, "args": vars(args),
            "validation_or_test_opened": False, "test_subsets_opened": False,
        }, args.output_dir / "last_checkpoint.pt")

    frozen_error = max(
        float((parameter - frozen[name]).abs().max())
        for name, parameter in candidate.named_parameters() if name in frozen
    )
    if frozen_error != 0.0:
        raise RuntimeError(f"accepted Darcy parameters changed: {frozen_error}")
    dataset.close()
    result = {
        "contract": RUN_CONTRACT,
        "selected": selected,
        "best_safe_score": None if not math.isfinite(best_safe_score) else best_safe_score,
        "fit_cases": int(len(fit_indices)),
        "selection_cases": int(len(selection_indices)),
        "identity_lift_real_audit": identity_audit,
        "frozen_accepted_max_abs_parameter_change": frozen_error,
        "trainable_parameter_count": sum(value.numel() for value in parameters),
        "fixed_context_q_linearity": True,
        "validation_or_test_opened": False,
        "test_subsets_opened": False,
        "history": history,
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
        writer.writeheader(); writer.writerows(flat)
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
