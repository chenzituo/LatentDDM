"""Train a frozen-endpoint Darcy composer with semantic trace codes.

The local neural endpoint is called once.  A shared compatibility composer then
predicts one reduced artificial-boundary trace per tile through one global
relative-attention message and decodes locally generated response modes.  The
production path is operator-free; target trace codes are training/evaluation
supervision only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from codiffuse_Schwarz.compatibility_composer import (
    CompatibilityOutput, CompatibilityResponseComposer,
)
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyAdapter, DarcyStatistics, FaceDarcyAdapter, H5DarcyDataset, MODEL_SPEC,
    batch_to_transition, build_model, tiled_darcy_conditions,
    tiled_sni_conditions,
)
from .evaluate import move_batch, relative_l2
from .train import make_loader, seed_everything


Tensor = torch.Tensor
CONTRACT = "darcy_semantic_trace_one_message_pou_v1"


def masked_mse(prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1.0)


def semantic_trace_code(
    target: Tensor,
    mask: Tensor,
    cover: CartesianCover,
    bins_per_face: int,
) -> Tensor:
    """Project each target tile edge to fixed, gauge-identified bin averages."""
    if bins_per_face <= 0:
        raise ValueError("bins_per_face must be positive")
    tiled = cover.tile(target)
    active = cover.tile(mask.to(target.dtype))
    faces = (
        (tiled[..., 0, :], active[..., 0, :]),
        (tiled[..., -1, :], active[..., -1, :]),
        (tiled[..., :, 0], active[..., :, 0]),
        (tiled[..., :, -1], active[..., :, -1]),
    )
    coefficients = []
    for value, weight in faces:
        numerator = F.adaptive_avg_pool1d(value * weight, bins_per_face)
        denominator = F.adaptive_avg_pool1d(weight, bins_per_face)
        coefficients.append(numerator / denominator.clamp_min(1.0e-6))
    flat = torch.cat(coefficients, dim=1)
    return flat.reshape(target.shape[0], cover.tile_count, -1)


class SemanticDarcyComposer(torch.nn.Module):
    def __init__(
        self,
        local: torch.nn.Module,
        adapter: DarcyAdapter,
        *,
        bins_per_face: int,
        response_modes: int,
        token_width: int,
    ) -> None:
        super().__init__()
        self.local = local
        self.bins_per_face = bins_per_face
        self.composer = CompatibilityResponseComposer(
            field_channels=adapter.state_channels,
            physical_channels=adapter.spatial_condition_channels + 1,
            scalar_dim=adapter.scalar_condition_dim,
            dimension=2,
            code_width=4 * bins_per_face * adapter.state_channels,
            token_width=token_width,
            response_width=64,
            learned_modes=response_modes,
            heads=4,
            include_fixed_modes=True,
            maximum_coefficient=1.5,
        )

    def local_endpoint(
        self, physical: Tensor, scalar: Tensor, grid_shape: tuple[int, int],
    ) -> Tensor:
        local, _ = self.local.predict_fields(
            physical, scalar, math.prod(grid_shape), None,
        )
        return local

    def compose(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        *,
        code_override: Tensor | None = None,
        max_graph_radius: int | None = None,
    ) -> CompatibilityOutput:
        return self.composer(
            local, physical, scalar, valid, grid_shape,
            code_override=code_override,
            max_graph_radius=max_graph_radius,
        )


def prepare(
    model: SemanticDarcyComposer,
    adapter: DarcyAdapter,
    batch: dict[str, Any],
    core: int,
    halo: int,
    cover: CartesianCover | None = None,
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, CartesianCover, Any,
]:
    current = torch.zeros_like(batch["solution"])
    transition = batch_to_transition(batch, current, target=batch["solution"])
    if cover is None:
        cover = CartesianCover(current.shape[-2], current.shape[-1], core, halo)
    if isinstance(adapter, FaceDarcyAdapter):
        physical, scalar = tiled_sni_conditions(adapter, batch, current, cover)
    else:
        physical, scalar = tiled_darcy_conditions(adapter, transition, cover)
    valid = cover.valid_tiles(current.shape[0], current.device, current.dtype)
    local = model.local_endpoint(physical, scalar, cover.grid_shape)
    target = adapter.normalized_target(transition)
    return local, physical, scalar, valid, target, current, cover, transition


def assembled_prediction(
    adapter: DarcyAdapter,
    transition: Any,
    cover: CartesianCover,
    corrected: Tensor,
) -> tuple[Tensor, Tensor]:
    assembled = cover.assemble_weighted(corrected, transition.current.shape[0])
    return adapter.apply_endpoint(transition, assembled), assembled


def forward_variants(
    model: SemanticDarcyComposer,
    adapter: DarcyAdapter,
    batch: dict[str, Any],
    core: int,
    halo: int,
    *,
    cover: CartesianCover | None = None,
    include_shuffled: bool = False,
    max_graph_radius: int | None = None,
) -> tuple[dict[str, Tensor], dict[str, CompatibilityOutput | Tensor | CartesianCover]]:
    local, physical, scalar, valid, target, _, cover, transition = prepare(
        model, adapter, batch, core, halo, cover,
    )
    oracle_code = semantic_trace_code(
        target, batch["mask"], cover, model.bins_per_face,
    )
    predicted = model.compose(
        local, physical, scalar, valid, cover.grid_shape,
        max_graph_radius=max_graph_radius,
    )
    oracle = model.compose(
        local, physical, scalar, valid, cover.grid_shape,
        code_override=oracle_code,
        max_graph_radius=max_graph_radius,
    )
    zero_code = torch.zeros_like(predicted.predicted_code)
    local_output = model.compose(
        local, physical, scalar, valid, cover.grid_shape,
        code_override=zero_code,
        max_graph_radius=max_graph_radius,
    )
    outputs = {
        "local": local_output,
        "predicted": predicted,
        "oracle": oracle,
    }
    if include_shuffled:
        outputs["shuffled"] = model.compose(
            local, physical, scalar, valid, cover.grid_shape,
            code_override=torch.roll(predicted.predicted_code, shifts=1, dims=1),
            max_graph_radius=max_graph_radius,
        )
    predictions: dict[str, Tensor] = {}
    assemblies: dict[str, Tensor] = {}
    for name, output in outputs.items():
        prediction, assembled = assembled_prediction(
            adapter, transition, cover, output.corrected,
        )
        predictions[name] = prediction
        assemblies[name] = assembled
    detail: dict[str, CompatibilityOutput | Tensor | CartesianCover] = {
        **outputs,
        "target_normalized": target,
        "oracle_code": oracle_code,
        "valid": valid,
        "cover": cover,
        **{f"assembled_{name}": value for name, value in assemblies.items()},
    }
    return predictions, detail


def training_terms(
    predictions: dict[str, Tensor],
    detail: dict[str, CompatibilityOutput | Tensor | CartesianCover],
    batch: dict[str, Any],
) -> dict[str, Tensor]:
    target = detail["target_normalized"]
    valid = detail["valid"]
    cover = detail["cover"]
    assert isinstance(target, Tensor) and isinstance(valid, Tensor)
    assert isinstance(cover, CartesianCover)
    mask = batch["mask"].to(target.dtype)
    local_target = cover.tile(target)
    local_mask = cover.tile(mask) * valid
    predicted = detail["predicted"]
    oracle = detail["oracle"]
    assert isinstance(predicted, CompatibilityOutput)
    assert isinstance(oracle, CompatibilityOutput)
    assembled_predicted = detail["assembled_predicted"]
    assembled_oracle = detail["assembled_oracle"]
    oracle_code = detail["oracle_code"]
    assert isinstance(assembled_predicted, Tensor)
    assert isinstance(assembled_oracle, Tensor)
    assert isinstance(oracle_code, Tensor)
    return {
        "predicted_endpoint": masked_mse(assembled_predicted, target, mask),
        "oracle_endpoint": masked_mse(assembled_oracle, target, mask),
        "predicted_local": masked_mse(predicted.corrected, local_target, local_mask),
        "oracle_local": masked_mse(oracle.corrected, local_target, local_mask),
        "consensus": masked_mse(
            predicted.corrected, cover.tile(assembled_predicted), local_mask,
        ),
        "code": F.smooth_l1_loss(predicted.predicted_code, oracle_code),
        "coefficient_regularization": predicted.coefficients.square().mean(),
    }


def objective(terms: dict[str, Tensor], warmup: bool) -> Tensor:
    predicted_weight, oracle_weight = ((0.25, 1.0) if warmup else (1.0, 0.25))
    return (
        predicted_weight * terms["predicted_endpoint"]
        + oracle_weight * terms["oracle_endpoint"]
        + 0.20 * terms["predicted_local"]
        + 0.20 * terms["oracle_local"]
        + 0.10 * terms["consensus"]
        + 0.50 * terms["code"]
        + 1.0e-4 * terms["coefficient_regularization"]
    )


@torch.no_grad()
def validate(
    model: SemanticDarcyComposer,
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
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            predictions, detail = forward_variants(
                model, adapter, batch, core, halo, include_shuffled=True,
            )
            terms = training_terms(predictions, detail, batch)
        for name, prediction in predictions.items():
            value = relative_l2(prediction, batch["solution"], batch["mask"]).mean()
            totals[f"{name}_relative_l2"] = totals.get(
                f"{name}_relative_l2", 0.0,
            ) + float(value)
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
        if count >= max_batches:
            break
    return {name: value / max(count, 1) for name, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--oracle-warmup-epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--bins-per-face", type=int, default=4)
    parser.add_argument("--response-modes", type=int, default=12)
    parser.add_argument("--token-width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    source = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    if not bool(source.get("operator_free", False)):
        raise ValueError("semantic composer requires an operator-free local checkpoint")
    statistics = DarcyStatistics.from_json(source["statistics"])
    adapter = (
        FaceDarcyAdapter(statistics)
        if source.get("contract") == "darcy_sni_face_boundary_to_solution_v1"
        else DarcyAdapter(statistics, use_discrete_residual=False)
    )
    local = build_model(adapter, communication_stage="none")
    local.load_state_dict(source["model"], strict=True)
    model = SemanticDarcyComposer(
        local, adapter, bins_per_face=args.bins_per_face,
        response_modes=args.response_modes, token_width=args.token_width,
    ).to(device)
    for parameter in model.local.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.composer.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1.0e-6,
    )
    start_epoch, best, history = 0, math.inf, []
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resumed.get("contract") != CONTRACT:
            raise ValueError("resume checkpoint contract mismatch")
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = int(resumed["epoch"])
        history = list(resumed.get("history", []))
        best = min((float(row["selection_score"]) for row in history), default=best)

    train_set = H5DarcyDataset(args.data, "train")
    val_set = H5DarcyDataset(args.data, "validation")
    train_loader = make_loader(
        train_set, args.batch_size, True, args.workers, device,
        args.samples_per_epoch,
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, min(args.workers, 2), device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        model.local.eval()
        totals: dict[str, float] = {}
        count = 0
        started = time.perf_counter()
        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                predictions, detail = forward_variants(
                    model, adapter, batch, args.core, args.halo,
                )
                terms = training_terms(predictions, detail, batch)
                loss = objective(terms, epoch <= args.oracle_warmup_epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.composer.parameters(), 1.0)
            optimizer.step()
            for name, value in {"loss": loss, **terms}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            count += 1
        scheduler.step()
        validation = validate(
            model, adapter, val_loader, device, args.core, args.halo,
            args.validation_batches,
        )
        predicted = validation["predicted_relative_l2"]
        local_value = validation["local_relative_l2"]
        # The selection score cannot reward a candidate that is worse than its
        # exact zero-code/local ablation on the same validation samples.
        score = predicted + max(predicted - local_value, 0.0)
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "selection_score": score,
            "warmup": epoch <= args.oracle_warmup_epochs,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value / max(count, 1) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        payload = {
            "contract": CONTRACT,
            "operator_free": True,
            "communication_rounds": 1,
            "local_frozen": True,
            "semantic_code": "four artificial-face trace bin averages",
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "statistics": statistics.to_json(),
            "source_checkpoint": str(args.pretrained.resolve()),
            "source_contract": source.get("contract"),
            "model_spec": {
                **MODEL_SPEC,
                "bins_per_face": args.bins_per_face,
                "code_width": 4 * args.bins_per_face * adapter.state_channels,
                "response_modes": args.response_modes,
                "token_width": args.token_width,
                "absolute_tile_identifier": False,
            },
            "args": vars(args),
            "history": history,
        }
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        if score < best:
            best = score
            torch.save(payload, args.output_dir / "checkpoint.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    if not history:
        raise ValueError("training produced no epochs")
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    result = {
        "contract": CONTRACT,
        "best_selection_score": best,
        "operator_free": True,
        "communication_rounds": 1,
        "local_frozen": True,
        "source_checkpoint": str(args.pretrained.resolve()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
