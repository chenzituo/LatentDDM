"""Fine-tune a signed relative-Green branch on train-only remote pairs.

The plain route optimizes the 800 translation-equivariant kernel entries.  A
contextual checkpoint additionally optimizes geometry-only scalar pair gates.
For either route, validation trust scaling follows a path whose zero endpoint
is the already accepted value-only communicator, so the ordinary-case floor is
an exact available fallback.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from .adapter import H5DarcyDataset
from .finetune_remote_pair_response import RemotePairDataset
from .train import seed_everything
from .train_incremental_flux_remote_schur import (
    ORDINARY_NONREGRESSION_FACTOR, PAIR_IMPROVEMENT_FACTOR, _load_communicator,
    objective, ordinary_metrics, pair_terms, validate_pairs,
)
from .train_residual_port_chart import load_frozen_local, sha256
from .train_residual_port_communicator import CONTRACT, load_frozen_chart


RUN_CONTRACT = "darcy_relative_green_flux_remote_finetune_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--ordinary-reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--raw-warmup-epochs", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ordinary-validation-cases", type=int, default=60)
    parser.add_argument("--pair-validation-batches", type=int, default=32)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--l2sp-weight", type=float, default=1e-3)
    parser.add_argument("--full-weight", type=float, default=200.0)
    parser.add_argument("--receiver-weight", type=float, default=3.0)
    parser.add_argument("--balanced-weight", type=float, default=2.0)
    parser.add_argument("--tail-weight", type=float, default=3.0)
    parser.add_argument("--gain-weight", type=float, default=0.01)
    parser.add_argument("--field-weight", type=float, default=0.0)
    parser.add_argument("--reference-weight", type=float, default=20.0)
    parser.add_argument("--schur-weight", type=float, default=0.5)
    parser.add_argument(
        "--trust-scales", type=float, nargs="+",
        default=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _set_scale(parameters, trained, scale: float) -> None:
    with torch.no_grad():
        for parameter, terminal in zip(parameters, trained, strict=True):
            parameter.copy_(scale * terminal)


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs <= 0 or not 0 <= args.raw_warmup_epochs <= args.epochs:
        raise ValueError("invalid training extent")
    if any(not 0 < value <= 1 for value in args.trust_scales):
        raise ValueError("trust scales must lie in (0,1]")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    candidate, initial = _load_communicator(args.initial_checkpoint, spec, device)
    reference, reference_payload = _load_communicator(
        args.ordinary_reference_checkpoint, spec, device,
    )
    analyzer_mode = getattr(candidate, "analyzer_mode", None)
    if analyzer_mode not in (
        "value_flux_relative_green",
        "value_flux_contextual_relative_green",
        "value_flux_dynamic_relative_green",
    ):
        raise ValueError("initial checkpoint is not a relative Green route")
    contextual = analyzer_mode == "value_flux_contextual_relative_green"
    dynamic = analyzer_mode == "value_flux_dynamic_relative_green"
    for payload in (initial, reference_payload):
        if payload["source_local_sha256"] != sha256(args.local_checkpoint) or \
                payload["source_chart_sha256"] != sha256(args.chart_checkpoint):
            raise ValueError("communicator/local/chart lineage differs")
    base = copy.deepcopy(candidate)
    with torch.no_grad():
        base.flux_green.kernel.zero_()
        if contextual:
            base.flux_green.receiver_gate.weight.zero_()
            base.flux_green.source_gate.weight.zero_()
            base.flux_green.relative_gate.zero_()
        if dynamic:
            for parameter in base.flux_green.trust_path_parameters():
                parameter.zero_()
    for model in (candidate, reference, base, chart, local):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
    if dynamic:
        parameters = list(candidate.flux_green.parameters())
        trust_parameters = list(candidate.flux_green.trust_path_parameters())
    else:
        parameters = [candidate.flux_green.kernel]
        trust_parameters = parameters
    if contextual:
        parameters.extend((
            candidate.flux_green.receiver_gate.weight,
            candidate.flux_green.source_gate.weight,
            candidate.flux_green.relative_gate,
        ))
        trust_parameters = parameters
    for parameter in parameters:
        parameter.requires_grad_(True)
    analytic_initial = [parameter.detach().clone() for parameter in parameters]
    train_set = RemotePairDataset(args.data, args.pairs, holdout=False)
    pair_set = RemotePairDataset(args.data, args.pairs, holdout=True)
    ordinary_set = H5DarcyDataset(args.data, "validation")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=RandomSampler(
            train_set, replacement=True, num_samples=args.samples_per_epoch,
            generator=generator,
        ),
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    pair_loader = DataLoader(
        pair_set, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers, 2), pin_memory=device.type == "cuda",
    )
    ordinary_loader = DataLoader(
        ordinary_set, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers, 2), pin_memory=device.type == "cuda",
    )
    model_tail = (reference, chart, local, adapter, device, spec, scale, args)
    baseline_pair = validate_pairs(
        pair_loader, base, *model_tail,
        maximum_batches=args.pair_validation_batches,
    )
    immutable_ordinary = ordinary_metrics(
        reference, chart, local, adapter, ordinary_loader, device, spec, scale,
        args, incremental=False,
    )
    base_ordinary = ordinary_metrics(
        base, chart, local, adapter, ordinary_loader, device, spec, scale,
        args, incremental=True,
    )
    warm_delta = max(
        abs(base_ordinary[name] - immutable_ordinary[name])
        for name in immutable_ordinary
    )
    if warm_delta > 1e-6:
        raise RuntimeError("zero relative Green does not recover immutable base")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []; selected = None; best = math.inf
    best_ordinary_safe_score = math.inf
    for epoch in range(1, args.epochs + 1):
        candidate.train(); totals = {}; batches = 0
        deploy = epoch > args.raw_warmup_epochs
        for host in train_loader:
            optimizer.zero_grad(set_to_none=True)
            terms = pair_terms(
                candidate, *model_tail, host=host, deploy=deploy,
            )
            penalty = sum(
                (parameter - initial).square().mean()
                for parameter, initial in zip(
                    parameters, analytic_initial, strict=True,
                )
            )
            loss = objective(terms, args, penalty)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite relative Green loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step(); batches += 1
            for name, value in {"objective": loss, **terms}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
        scheduler.step()
        trained = [parameter.detach().clone() for parameter in parameters]
        trust_trained = [
            parameter.detach().clone() for parameter in trust_parameters
        ]
        candidate.eval()
        for trust in sorted(set(args.trust_scales)):
            _set_scale(trust_parameters, trust_trained, trust)
            pair = validate_pairs(
                pair_loader, candidate, *model_tail,
                maximum_batches=args.pair_validation_batches,
            )
            ordinary = ordinary_metrics(
                candidate, chart, local, adapter, ordinary_loader, device,
                spec, scale, args, incremental=True,
            )
            ordinary_gates = {
                name: ordinary[name]
                <= ORDINARY_NONREGRESSION_FACTOR * immutable_ordinary[name]
                for name in immutable_ordinary
            }
            response_gates = {
                name: pair[name] <= PAIR_IMPROVEMENT_FACTOR * baseline_pair[name]
                for name in (
                    "full_response_relative_square",
                    "receiver_response_relative_square",
                    "receiver_balanced_log_relative_square",
                    "receiver_tail_log_relative_square",
                )
            }
            response_gates.update({
                "receiver_gain_physical": 0.05 <= pair["receiver_gain_mean"] <= 1.95,
                "schur_nonexpansive": pair["schur_cycle_relative_square"] <= 1.0,
            })
            clears = all(ordinary_gates.values()) and all(response_gates.values())
            score = sum(
                pair[name] / baseline_pair[name]
                for name in (
                    "full_response_relative_square",
                    "receiver_response_relative_square",
                    "receiver_balanced_log_relative_square",
                    "receiver_tail_log_relative_square",
                )
            )
            row = {
                "epoch": epoch, "trust_scale": trust, "score": score,
                "clears": clears,
                "training_field": "affine_retracted" if deploy else "raw_warmup",
                "ordinary_gates": ordinary_gates, "response_gates": response_gates,
                **{f"train_{k}": v / batches for k, v in totals.items()},
                **{f"pair_{k}": v for k, v in pair.items()},
                **{f"ordinary_{k}": v for k, v in ordinary.items()},
            }
            history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
            if all(ordinary_gates.values()) and score < best_ordinary_safe_score:
                best_ordinary_safe_score = score
                exploratory = dict(initial)
                exploratory_cfg = dict(exploratory["communicator_spec"])
                exploratory_cfg.update({
                    "deployment_selected_trust_scale": trust,
                    "remote_response_finetuned": True,
                    "ordinary_selection_clears": True,
                    "remote_selection_clears": all(response_gates.values()),
                    "exploratory_not_frozen": not all(response_gates.values()),
                    "test_subsets_opened": False,
                })
                exploratory.update({
                    "contract": CONTRACT, "phase": RUN_CONTRACT,
                    "epoch": epoch, "communicator": candidate.state_dict(),
                    "communicator_spec": exploratory_cfg,
                    "selected_metrics": row,
                    "selection_clears": all(response_gates.values()),
                    "test_subsets_opened": False, "args": vars(args),
                })
                torch.save(
                    exploratory,
                    args.output_dir / "best_ordinary_safe_exploratory.pt",
                )
            if clears and score < best:
                best = score; selected = row
                payload = dict(initial)
                cfg = dict(payload["communicator_spec"])
                cfg.update({
                    "deployment_selected_trust_scale": trust,
                    "remote_response_finetuned": True,
                    "relative_green_parameters": sum(
                        parameter.numel() for parameter in parameters
                    ),
                    "geometry_contextual_pair_gate": contextual,
                    "geometry_dynamic_low_rank_kernel": dynamic,
                    "trust_path": (
                        "relative_K_and_dynamic_output_layer_from_zero"
                        if dynamic else "all_relative_green_parameters_from_zero"
                    ),
                    "base_value_operator_frozen": True,
                    "test_subsets_opened": False,
                })
                payload.update({
                    "contract": CONTRACT, "phase": RUN_CONTRACT,
                    "epoch": epoch, "communicator": candidate.state_dict(),
                    "communicator_spec": cfg,
                    "initial_relative_green_checkpoint": str(
                        args.initial_checkpoint.resolve()
                    ),
                    "initial_relative_green_sha256": sha256(args.initial_checkpoint),
                    "ordinary_reference_checkpoint": str(
                        args.ordinary_reference_checkpoint.resolve()
                    ),
                    "ordinary_reference_sha256": sha256(
                        args.ordinary_reference_checkpoint
                    ),
                    "remote_pair_corpus": str(args.pairs.resolve()),
                    "remote_pair_corpus_sha256": sha256(args.pairs),
                    "selection_clears": True, "selected_metrics": row,
                    "test_subsets_opened": False, "args": vars(args),
                })
                torch.save(payload, args.output_dir / "checkpoint.pt")
        _set_scale(trust_parameters, trust_trained, 1.0)
        last = dict(initial); last.update({
            "contract": CONTRACT, "phase": RUN_CONTRACT, "epoch": epoch,
            "communicator": candidate.state_dict(), "history": history,
            "test_subsets_opened": False, "args": vars(args),
        })
        torch.save(last, args.output_dir / "last_checkpoint.pt")
    train_set.close(); pair_set.close(); ordinary_set.close()
    result = {
        "contract": RUN_CONTRACT, "selected": selected,
        "immutable_ordinary": immutable_ordinary,
        "baseline_pair": baseline_pair,
        "base_warm_start_max_abs_metric_delta": warm_delta,
        "geometry_contextual_pair_gate": contextual,
        "best_ordinary_safe_score": (
            None if not math.isfinite(best_ordinary_safe_score)
            else best_ordinary_safe_score
        ),
        "test_subsets_opened": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        flat = [{
            key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            for key, value in row.items()
        } for row in history]
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    if selected is None:
        raise RuntimeError("relative Green fine-tune did not clear frozen gates")
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
