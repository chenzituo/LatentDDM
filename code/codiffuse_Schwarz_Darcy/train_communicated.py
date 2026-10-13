"""Fine-tune the Darcy local endpoint with one learned tile communication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    COMMUNICATION_CONTRACTS,
    COMMUNICATION_STAGES,
    CommunicationStage,
    DarcyAdapter,
    DarcyStatistics,
    FaceDarcyAdapter,
    H5DarcyDataset,
    MODEL_SPEC,
    batch_to_transition,
    build_model,
    checkpoint_communication_stage,
    resolve_communication_stage,
    tiled_darcy_conditions,
    tiled_sni_conditions,
)
from .evaluate import move_batch, relative_l2, residual_rms
from .train import arbitrary_current, make_loader, seed_everything


Tensor = torch.Tensor


def matched_local_conditions(
    adapter: DarcyAdapter,
    batch: dict[str, Tensor],
    current: Tensor,
    cover: CartesianCover,
) -> tuple[Tensor, Tensor]:
    """Use the exact input contract on which the inherited endpoint was trained.

    The face-resolved local BVP consumes directional artificial/physical face
    data built by :func:`tiled_sni_conditions`.  Feeding that checkpoint through
    the legacy cellwise trace constructor changes the inherited Local POU
    endpoint and therefore is not a matched Patch Transformer control.
    """
    if isinstance(adapter, FaceDarcyAdapter):
        return tiled_sni_conditions(adapter, batch, current, cover)
    transition = batch_to_transition(
        batch, current, previous=current, target=batch["solution"],
    )
    return tiled_darcy_conditions(adapter, transition, cover)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


def objective(
    model: torch.nn.Module,
    adapter: DarcyAdapter,
    batch: dict[str, Any],
    current: Tensor,
    core: int,
    halo: int,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    transition = batch_to_transition(
        batch, current, previous=current, target=batch["solution"],
    )
    cover = CartesianCover(current.shape[-2], current.shape[-1], core, halo)
    physical, scalars = matched_local_conditions(
        adapter, batch, current, cover,
    )
    corrected, proposal = model.predict_fields(
        physical, scalars, cover.tile_count, cover.grid_shape,
    )
    assembled = cover.assemble_weighted(corrected, current.shape[0])
    target = adapter.normalized_target(transition)
    mask = batch["mask"].to(target.dtype)
    endpoint = masked_mse(assembled, target, mask)

    local_target = cover.tile(target)
    local_mask = cover.tile(mask) * cover.valid_tiles(
        current.shape[0], current.device, current.dtype,
    )
    local = masked_mse(corrected, local_target, local_mask)
    proposal_loss = masked_mse(proposal, local_target, local_mask)
    consensus = masked_mse(corrected, cover.tile(assembled), local_mask)
    boundary = torch.exp(-8.0 * batch["sdf"].abs()) * mask
    near_boundary = masked_mse(assembled, target, boundary)
    loss = endpoint + 0.35 * local + 0.20 * proposal_loss \
        + 0.25 * consensus + 0.25 * near_boundary
    prediction = adapter.apply_endpoint(transition, assembled)
    return loss, {
        "endpoint": endpoint, "local": local, "proposal": proposal_loss,
        "consensus": consensus, "near_boundary": near_boundary,
    }, prediction


def initialize_from_local(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    communication_stage: CommunicationStage,
) -> dict[str, int]:
    communicated = communication_stage != "none"
    incompatible = model.load_state_dict(
        checkpoint["model"], strict=not communicated,
    )
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"unexpected local checkpoint tensors: {unexpected}")
    if communicated and any("grid_communicator" not in name for name in missing):
        raise ValueError(f"non-communication tensors missing from local checkpoint: {missing}")
    expected_prefixes = {
        "proposal": ("proposal.grid_communicator",),
        "both": ("proposal.grid_communicator", "projector.grid_communicator"),
    }
    if communicated and any(
        not name.startswith(expected_prefixes[communication_stage])
        for name in missing
    ):
        raise ValueError(
            f"checkpoint initialization does not match {communication_stage!r}: {missing}"
        )
    return {
        "source_tensors": len(checkpoint["model"]),
        "new_communication_tensors": len(missing),
        "proposal_communication_tensors": sum(
            name.startswith("proposal.grid_communicator") for name in missing
        ),
            "projector_communication_tensors": sum(
            name.startswith("projector.grid_communicator") for name in missing
        ),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    adapter: DarcyAdapter,
    loader,
    device: torch.device,
    core: int,
    halo: int,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for host_batch in loader:
        batch = move_batch(host_batch, device)
        current = torch.zeros_like(batch["solution"])
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, terms, prediction = objective(
                model, adapter, batch, current, core, halo,
            )
        values: dict[str, Tensor] = {
            "loss": loss, **terms,
            "relative_l2": relative_l2(prediction, batch["solution"], batch["mask"]).mean(),
            "residual_rms": residual_rms(prediction, batch).mean(),
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
        if count >= max_batches:
            break
    return {name: value / max(count, 1) for name, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--communication-stage", choices=COMMUNICATION_STAGES,
        help=(
            "cross-tile communicator placement; proposal is the production "
            "single-message protocol, both is the historical two-stage ablation"
        ),
    )
    parser.add_argument("--disable-communication", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--communication-only", action="store_true",
        help=(
            "freeze every tensor inherited from --pretrained and optimize "
            "only the pre-field tile communicator (matched Patch Transformer control)"
        ),
    )
    parser.add_argument(
        "--zero-current", action="store_true",
        help="train the requested one-pass endpoint from the same zero state used at evaluation",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    source = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    source_checkpoint_sha256 = sha256(args.pretrained)
    statistics = DarcyStatistics.from_json(source["statistics"])
    adapter = (
        FaceDarcyAdapter(statistics)
        if source.get("contract") == "darcy_sni_face_boundary_to_solution_v1"
        else DarcyAdapter(statistics)
    )
    requested_stage = getattr(args, "communication_stage", None)
    disabled = bool(getattr(args, "disable_communication", False))
    if disabled and requested_stage not in (None, "none"):
        raise ValueError(
            "--disable-communication conflicts with --communication-stage "
            f"{requested_stage}"
        )
    # No explicit stage retains the historical CLI behavior: enabled meant
    # communicators in both proposal and projector. New production scripts
    # pass proposal explicitly.
    communication_stage = resolve_communication_stage(
        "none" if disabled else requested_stage,
        None if requested_stage is not None or disabled else True,
    )
    model = build_model(
        adapter, communication_stage=communication_stage,
    ).to(device)
    initialization = initialize_from_local(
        model, source, communication_stage,
    )
    if args.communication_only and communication_stage == "none":
        raise ValueError("--communication-only requires an enabled communicator")
    trainable_names = []
    for name, parameter in model.named_parameters():
        selected = (
            not args.communication_only
            or ".grid_communicator." in f".{name}"
        )
        parameter.requires_grad_(selected)
        if selected:
            trainable_names.append(name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("the requested training contract selected no parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr,
    )
    start_epoch, best, history = 0, math.inf, []
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        resumed_stage = checkpoint_communication_stage(resumed)
        if resumed_stage != communication_stage:
            raise ValueError(
                f"resume checkpoint uses {resumed_stage!r} communication, "
                f"requested {communication_stage!r}"
            )
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = int(resumed["epoch"])
        history = list(resumed.get("history", []))
        best = min((row["selection_score"] for row in history), default=best)
    train_set = H5DarcyDataset(args.data, "train")
    val_set = H5DarcyDataset(args.data, "validation")
    train_loader = make_loader(
        train_set, args.batch_size, True, args.workers, device, args.samples_per_epoch,
    )
    val_loader = make_loader(val_set, args.batch_size, False, min(args.workers, 2), device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    grid_size = int(train_set.attrs["grid_size"])
    if grid_size % args.core:
        raise ValueError("full Darcy grid must be divisible by the POU core")
    cover_grid_shape = (grid_size // args.core, grid_size // args.core)

    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        return {
            "contract": COMMUNICATION_CONTRACTS[communication_stage],
            "communication_stage": communication_stage,
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "statistics": statistics.to_json(),
            "model_spec": {
                **MODEL_SPEC, "communication_stage": communication_stage,
            },
            "training_contract": (
                "same_cover_patch_transformer_frozen_local_v1"
                if args.communication_only else
                "darcy_full_domain_communicated_finetune_v1"
            ),
            "local_input_contract": (
                "face_resolved_sni_directional_trace_v1"
                if isinstance(adapter, FaceDarcyAdapter) else
                "legacy_cellwise_artificial_trace_v1"
            ),
            "cover_contract": {
                "grid_shape_yx": cover_grid_shape,
                "subdomains": math.prod(cover_grid_shape),
                "core": args.core,
                "halo": args.halo,
            },
            "source_checkpoint": str(args.pretrained.resolve()),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "trainable_parameter_names": trainable_names,
            "args": vars(args), "history": history,
            "initialization": initialization,
        }

    # The residual communicator is initialized as an exact bypass.  Treat that
    # zero-message endpoint as epoch 0 in model selection; otherwise the
    # controlled Patch Transformer could be forced to report a trained model
    # that is worse than the identical frozen Local POU endpoint.
    if not args.resume:
        baseline = validate(
            model, adapter, val_loader, device, args.core, args.halo,
            args.validation_batches,
        )
        best = baseline["relative_l2"] + 0.1 * baseline["residual_rms"]
        history.append({
            "epoch": 0, "seconds": 0.0,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_score": best,
            **{f"validation_{name}": value for name, value in baseline.items()},
        })
        payload = checkpoint_payload(0)
        torch.save(payload, args.output_dir / "checkpoint.pt")
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    payload = checkpoint_payload(start_epoch)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        started = time.perf_counter()
        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            current = (
                torch.zeros_like(batch["solution"])
                if args.zero_current else
                arbitrary_current(batch, statistics, generator)
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss, terms, _ = objective(
                    model, adapter, batch, current, args.core, args.halo,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for name, value in {"loss": loss, **terms}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        scheduler.step()
        validation = validate(
            model, adapter, val_loader, device, args.core, args.halo,
            args.validation_batches,
        )
        score = validation["relative_l2"] + 0.1 * validation["residual_rms"]
        row = {
            "epoch": epoch, "seconds": time.perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"], "selection_score": score,
            **{f"train_{name}": value / max(batches, 1) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        payload = checkpoint_payload(epoch)
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        if score < best:
            best = score
            torch.save(payload, args.output_dir / "checkpoint.pt")
        print(json.dumps(row, sort_keys=True), flush=True)
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        # Epoch 0 has validation-only fields, while trained epochs also have
        # optimization terms.  Preserve the union instead of assuming every
        # record has the baseline schema.
        fieldnames = list(dict.fromkeys(
            key for row in history for key in row
        ))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(history)
    result = {
        "contract": payload["contract"], "best_selection_score": best,
        "communication_stage": communication_stage,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "training_contract": payload["training_contract"],
        "source_checkpoint": payload["source_checkpoint"],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "data": str(args.data.resolve()), "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
