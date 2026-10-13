"""Fine-tune only the homogeneous-q communicator on cached train pairs.

Local solver and response chart remain frozen.  For fixed geometry the
communicator is still a bias-free linear map of q; the pair loss teaches the
missing remote boundary-to-receiver derivative without adding source, target,
or absolute position to inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .adapter import FACE_DIRICHLET_FIELDS, REQUIRED_FIELDS, H5DarcyDataset
from .evaluate import move_batch
from .residual_port_composer import (
    SignedSourceLinearResidualPortCommunicator,
    SourceLinearResidualPortCommunicator,
)
from .train import seed_everything
from .train_residual_port_chart import (
    frozen_context, load_frozen_local, masked_mse, sha256,
)
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT, communicator_features, forward_composer,
    load_frozen_chart, objective, source_linear_signal_width, training_terms,
    validate as validate_ordinary,
)


Tensor = torch.Tensor
PAIR_CONTRACT = "darcy_remote_dirichlet_pair_train_v1"
CONTRACT_V2 = "darcy_residual_port_remote_response_finetune_v1"
ORDINARY_NONREGRESSION_FACTOR = 1.02
PAIR_IMPROVEMENT_FACTOR = 0.95


class RemotePairDataset(Dataset[dict[str, Any]]):
    def __init__(self, data: Path, pairs: Path, *, holdout: bool) -> None:
        self.base = H5DarcyDataset(data, "train")
        self.pairs_path = pairs.resolve()
        with h5py.File(self.pairs_path, "r") as handle:
            if handle.attrs.get("contract") != PAIR_CONTRACT:
                raise ValueError("remote-pair corpus contract differs")
            if handle.attrs.get("source_split") != "train" or bool(
                handle.attrs.get("validation_or_test_opened", True)
            ):
                raise ValueError("pair corpus is not train-only")
            if handle.attrs.get("source_data_sha256") != sha256(data):
                raise ValueError("pair corpus source-data hash differs")
            base_index = np.asarray(handle["train_pairs/base_index"])
        # Split by base case, not row, so two amplitudes never leak across the
        # train-pair/held-out-pair selection boundary.
        selector = (base_index % 10 == 0)
        self.rows = np.flatnonzero(selector if holdout else ~selector)
        if not len(self.rows):
            raise ValueError("empty remote-pair partition")
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
            self._handle = h5py.File(self.pairs_path, "r")
        group = self._handle["train_pairs"]
        row = int(self.rows[index])
        item = self.base[int(group["base_index"][row])]
        item["pair_delta_bc"] = torch.from_numpy(
            np.asarray(group["delta_bc"][row], dtype=np.float32),
        )
        item["pair_solution_response"] = torch.from_numpy(
            np.asarray(group["solution_response"][row], dtype=np.float32),
        )
        item["pair_target_tile"] = torch.from_numpy(
            np.asarray(group["target_tile"][row], dtype=np.int64),
        )
        item["pair_amplitude"] = torch.tensor(
            float(group["amplitude"][row]), dtype=torch.float32,
        )
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--pair-validation-batches", type=int, default=32)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ordinary-weight", type=float, default=0.25)
    parser.add_argument("--base-field-distill-weight", type=float, default=0.0)
    parser.add_argument("--base-code-distill-weight", type=float, default=0.0)
    parser.add_argument("--l2sp-weight", type=float, default=1.0e-4)
    parser.add_argument("--preserve-local-path", action="store_true")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--kernel-type", choices=("softmax", "signed_l1_tanh"),
        default="softmax",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _paired_batch(host: dict[str, Any], device: torch.device) -> tuple[
    dict[str, Tensor], Tensor, Tensor,
]:
    moved = move_batch(host, device)
    response = moved.pop("pair_solution_response")
    delta = moved.pop("pair_delta_bc")
    target_tile = moved.pop("pair_target_tile")
    moved.pop("pair_amplitude", None)
    batch_size = response.shape[0]
    paired: dict[str, Tensor] = {}
    for name in (*REQUIRED_FIELDS, *FACE_DIRICHLET_FIELDS, "case_index"):
        value = moved[name]
        paired[name] = torch.cat((value, value), dim=0)
    paired["solution"][batch_size:] += response[:, None]
    paired["bc_value"][batch_size:] += delta[:, None]
    paired["dirichlet_face_value"][batch_size:] += (
        delta[:, None] * paired["dirichlet_face_mask"][batch_size:]
    )
    return paired, response[:, None], target_tile


def _relative_square(error: Tensor, reference: Tensor, weight: Tensor) -> Tensor:
    # Ratio of aggregate energies, rather than a mean of per-case ratios.
    # Remote effects can be arbitrarily small for an individual geometry; a
    # per-case division would let near-zero physical responses dominate both
    # the loss and gradient with numerical gain rather than response energy.
    numerator = (error.square() * weight).sum()
    denominator = (reference.square() * weight).sum().clamp_min(1.0e-12)
    return numerator / denominator


def _mean_log_relative_square(
    error: Tensor, reference: Tensor, weight: Tensor,
) -> Tensor:
    """Robustly balance weak and strong remote responses across geometries."""
    axes = tuple(range(1, error.ndim))
    numerator = (error.square() * weight).sum(axes)
    denominator = (reference.square() * weight).sum(axes).clamp_min(1.0e-12)
    return torch.log1p(numerator / denominator).mean()


def pair_terms(
    communicator, chart, local, adapter, host, device, spec, scale, args,
    reference_communicator=None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    batch, exact_response, target_tile = _paired_batch(host, device)
    pair_count = exact_response.shape[0]
    context = frozen_context(local, adapter, batch, spec, args.core, args.halo)
    aligned = communicator_features(
        communicator, context, batch, spec, scale, adapter,
    )
    assembled, codes, _, _ = forward_composer(
        communicator, chart, context, scale, batch, aligned_features=aligned,
    )
    predicted_response = adapter.statistics.solution_std * (
        assembled[pair_count:] - assembled[:pair_count]
    )
    weight = batch["mask"][:pair_count]
    full = _relative_square(
        predicted_response - exact_response, exact_response, weight,
    )
    exact_tiles = context.cover.tile(exact_response)
    predicted_tiles = context.cover.tile(predicted_response)
    weight_tiles = context.cover.tile(weight)
    tile_count = context.cover.tile_count
    target_index = target_tile[:, 0] * context.cover.grid_shape[1] + target_tile[:, 1]
    flat_index = torch.arange(pair_count, device=device) * tile_count + target_index
    exact_receiver = exact_tiles[flat_index]
    predicted_receiver = predicted_tiles[flat_index]
    receiver_weight = weight_tiles[flat_index]
    receiver = _relative_square(
        predicted_receiver - exact_receiver, exact_receiver, receiver_weight,
    )
    receiver_balanced = _mean_log_relative_square(
        predicted_receiver - exact_receiver, exact_receiver, receiver_weight,
    )
    gain = (
        (predicted_receiver * exact_receiver * receiver_weight).sum()
        / (exact_receiver.square() * receiver_weight).sum().clamp_min(1.0e-12)
    )
    gain_loss = F.smooth_l1_loss(gain, torch.ones_like(gain))
    gain_range = (
        torch.relu(0.05 - gain).square() + torch.relu(gain - 1.95).square()
    ).mean()
    ordinary, _ = training_terms(
        communicator, chart, context, spec, scale, batch,
        adapter.statistics.solution_std, aligned,
    )
    pair = {
        "full_response_relative_square": full,
        "receiver_response_relative_square": receiver,
        "receiver_balanced_log_relative_square": receiver_balanced,
        "receiver_gain_loss": gain_loss,
        "receiver_gain_range_hinge": gain_range,
        "receiver_gain_mean": gain.mean(),
        "code_response_rms": (
            codes[pair_count:] - codes[:pair_count]
        ).square().mean().clamp_min(1.0e-12).sqrt(),
    }
    if reference_communicator is not None:
        with torch.no_grad():
            reference_assembled, reference_codes, _, _ = forward_composer(
                reference_communicator, chart, context, scale, batch,
                aligned_features=aligned,
            )
        pair["base_field_distillation"] = masked_mse(
            assembled[:pair_count], reference_assembled[:pair_count],
            batch["mask"][:pair_count],
        )
        pair["base_code_distillation"] = F.smooth_l1_loss(
            codes[:pair_count], reference_codes[:pair_count],
        )
    else:
        pair["base_field_distillation"] = assembled.new_zeros(())
        pair["base_code_distillation"] = assembled.new_zeros(())
    return pair, ordinary


def pair_objective(
    pair: dict[str, Tensor], ordinary: dict[str, Tensor], args,
) -> Tensor:
    return (
        pair["full_response_relative_square"]
        + 2.0 * pair["receiver_response_relative_square"]
        + 0.5 * pair["receiver_balanced_log_relative_square"]
        + 0.1 * pair["receiver_gain_loss"]
        + 0.1 * pair["receiver_gain_range_hinge"]
        + getattr(args, "ordinary_weight", 0.25) * objective(ordinary)
        + getattr(args, "base_field_distill_weight", 0.0)
        * pair["base_field_distillation"]
        + getattr(args, "base_code_distill_weight", 0.0)
        * pair["base_code_distillation"]
    )


@torch.no_grad()
def validate_pairs(
    loader, communicator, chart, local, adapter, device, spec, scale, args,
    reference_communicator=None,
):
    communicator.eval(); totals: dict[str, float] = {}; count = 0
    for host in loader:
        if count >= args.pair_validation_batches:
            break
        pair, ordinary = pair_terms(
            communicator, chart, local, adapter, host, device, spec, scale, args,
            reference_communicator,
        )
        values = {"objective": pair_objective(pair, ordinary, args), **pair}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
    if count == 0:
        raise RuntimeError("empty pair validation")
    return {name: value / count for name, value in totals.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    if initial.get("contract") != CONTRACT:
        raise ValueError("initial checkpoint is not a phase-separated communicator")
    cfg = initial["communicator_spec"]
    if cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
        raise ValueError("pair fine-tune requires homogeneous compatibility q")
    if initial["source_local_sha256"] != sha256(args.local_checkpoint) or \
            initial["source_chart_sha256"] != sha256(args.chart_checkpoint):
        raise ValueError("initial/local/chart checkpoint lineage differs")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    communicator_type = (
        SignedSourceLinearResidualPortCommunicator
        if args.kernel_type == "signed_l1_tanh"
        else SourceLinearResidualPortCommunicator
    )
    communicator = communicator_type(
        int(cfg["physical_channels"]), int(cfg["scalar_dim"]), spec.code_width,
        source_linear_signal_width(spec), token_width=int(cfg["token_width"]),
        heads=int(cfg["heads"]),
    ).to(device)
    mapped_context_keys: list[str] = []
    if args.kernel_type == "softmax":
        communicator.load_state_dict(initial["communicator"], strict=True)
    else:
        # The signed kernel has different normalization semantics. Transfer
        # only geometry/context scoring weights; value, output, local-value,
        # and code-head maps remain newly initialized and are never silently
        # interpreted as a signed-kernel checkpoint.
        target_state = communicator.state_dict()
        source_state = initial["communicator"]
        for target_name in tuple(target_state):
            source_name: str | None = None
            if target_name.startswith("geometry_analyzer.") or \
                    ".global_kernel.context_norm." in target_name or \
                    ".global_kernel.qk." in target_name:
                source_name = target_name
            elif ".global_kernel.relative_score." in target_name:
                source_name = target_name.replace(
                    ".global_kernel.relative_score.",
                    ".global_kernel.relative_bias.",
                )
            if source_name is not None and source_name in source_state and \
                    source_state[source_name].shape == target_state[target_name].shape:
                target_state[target_name] = source_state[source_name]
                mapped_context_keys.append(target_name)
        communicator.load_state_dict(target_state, strict=True)
    reference_communicator = None
    if args.base_field_distill_weight > 0.0 or args.base_code_distill_weight > 0.0:
        if args.kernel_type != "softmax":
            raise ValueError("base-output distillation currently requires softmax lineage")
        reference_communicator = SourceLinearResidualPortCommunicator(
            int(cfg["physical_channels"]), int(cfg["scalar_dim"]), spec.code_width,
            source_linear_signal_width(spec), token_width=int(cfg["token_width"]),
            heads=int(cfg["heads"]),
        ).to(device)
        reference_communicator.load_state_dict(initial["communicator"], strict=True)
        reference_communicator.eval()
        for parameter in reference_communicator.parameters():
            parameter.requires_grad_(False)
    local_head_reference = None
    if args.preserve_local_path:
        for parameter in communicator.geometry_analyzer.parameters():
            parameter.requires_grad_(False)
        for parameter in communicator.local_value.parameters():
            parameter.requires_grad_(False)
        local_head_reference = communicator.code_head.weight[:, :communicator.token_width] \
            .detach().clone()
    train_pairs = RemotePairDataset(args.data, args.pairs, holdout=False)
    heldout_pairs = RemotePairDataset(args.data, args.pairs, holdout=True)
    ordinary_val = H5DarcyDataset(args.data, "validation")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = torch.utils.data.RandomSampler(
        train_pairs, replacement=True, num_samples=args.samples_per_epoch,
        generator=generator,
    )
    train_loader = DataLoader(
        train_pairs, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    pair_val_loader = DataLoader(
        heldout_pairs, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers, 2), pin_memory=device.type == "cuda",
    )
    ordinary_loader = DataLoader(
        ordinary_val, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers, 2), pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in communicator.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=(0.0 if args.preserve_local_path else args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1.0e-6,
    )
    # The untouched remote-pair validation gate is intentionally not called
    # here. Pair selection uses only a held-out subset of train-split pairs.
    baseline_pair = validate_pairs(
        pair_val_loader, communicator, chart, local, adapter, device, spec, scale, args,
        reference_communicator,
    )
    baseline_ordinary = validate_ordinary(
        communicator, chart, local, adapter, ordinary_loader, device, spec, scale, args,
    )
    ordinary_keys = (
        "assembled", "energy", "residual", "value_interface", "flux_interface",
        "compatibility_cycle_relative_square",
    )
    initial_parameters = {
        name: value.detach().clone() for name, value in communicator.named_parameters()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best = math.inf; selected_epoch: int | None = None
    for epoch in range(1, args.epochs + 1):
        communicator.train(); totals: dict[str, float] = {}; batches = 0
        for host in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pair, ordinary = pair_terms(
                communicator, chart, local, adapter, host, device, spec, scale, args,
                reference_communicator,
            )
            drift = sum(
                (parameter - initial_parameters[name]).square().mean()
                for name, parameter in communicator.named_parameters()
            )
            loss = pair_objective(pair, ordinary, args) + args.l2sp_weight * drift
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite remote-pair fine-tune loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(communicator.parameters(), 1.0)
            optimizer.step(); batches += 1
            if local_head_reference is not None:
                with torch.no_grad():
                    communicator.code_head.weight[:, :communicator.token_width].copy_(
                        local_head_reference,
                    )
            for name, value in {"objective": loss, **pair}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
        scheduler.step()
        pair_val = validate_pairs(
            pair_val_loader, communicator, chart, local, adapter, device, spec, scale, args,
            reference_communicator,
        )
        ordinary = validate_ordinary(
            communicator, chart, local, adapter, ordinary_loader, device, spec, scale, args,
        )
        ordinary_gates = {
            key: ordinary[key] <= ORDINARY_NONREGRESSION_FACTOR * baseline_ordinary[key]
            for key in ordinary_keys
        }
        response_gates = {
            "full": pair_val["full_response_relative_square"]
                <= PAIR_IMPROVEMENT_FACTOR * baseline_pair["full_response_relative_square"],
            "receiver": pair_val["receiver_response_relative_square"]
                <= PAIR_IMPROVEMENT_FACTOR * baseline_pair["receiver_response_relative_square"],
            "receiver_balanced": pair_val["receiver_balanced_log_relative_square"]
                <= PAIR_IMPROVEMENT_FACTOR * baseline_pair["receiver_balanced_log_relative_square"],
            "gain": 0.05 <= pair_val["receiver_gain_mean"] <= 1.95,
        }
        clears = all(ordinary_gates.values()) and all(response_gates.values())
        score = (
            pair_val["receiver_response_relative_square"]
            + 0.25 * pair_val["receiver_balanced_log_relative_square"]
            + 0.5 * pair_val["full_response_relative_square"]
            + 0.1 * pair_val["receiver_gain_loss"]
        )
        row = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "selection_score": score, "clears": clears,
            "ordinary_gates": ordinary_gates, "response_gates": response_gates,
            **{f"train_{key}": value / batches for key, value in totals.items()},
            **{f"pair_validation_{key}": value for key, value in pair_val.items()},
            **{f"ordinary_validation_{key}": value for key, value in ordinary.items()},
        }
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        payload = dict(initial)
        communicator_spec = dict(cfg)
        communicator_spec.update({
            "kernel_type": args.kernel_type,
            "signed_global_weights": args.kernel_type == "signed_l1_tanh",
            "global_kernel_normalization": (
                "signed_tanh_l1" if args.kernel_type == "signed_l1_tanh"
                else "softmax"
            ),
        })
        payload.update({
            "contract": CONTRACT,
            "phase": "train_pair_remote_response_communicator_only",
            "epoch": epoch, "communicator": communicator.state_dict(),
            "communicator_spec": communicator_spec,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "initial_communicator_checkpoint": str(args.initial_checkpoint.resolve()),
            "initial_communicator_sha256": sha256(args.initial_checkpoint),
            "remote_pair_corpus": str(args.pairs.resolve()),
            "remote_pair_corpus_sha256": sha256(args.pairs),
            "remote_pair_contract": PAIR_CONTRACT,
            "training_splits": ["train", "validation"],
            "pair_supervision_split": "train only",
            "untouched_validation_remote_pairs_opened": False,
            "test_subsets_opened": False,
            "discrete_operator_used_at_inference": False,
            "fixed_context_q_linearity_preserved": True,
            "mapped_context_keys": mapped_context_keys,
            "ordinary_nonregression_factor": ORDINARY_NONREGRESSION_FACTOR,
            "pair_improvement_factor": PAIR_IMPROVEMENT_FACTOR,
            "baseline_pair": baseline_pair,
            "baseline_ordinary": baseline_ordinary,
            "selection_clears": clears, "history": history,
            "args": vars(args),
        })
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        if clears and score < best:
            best = score; selected_epoch = epoch
            torch.save(payload, args.output_dir / "checkpoint.pt")
    train_pairs.close(); heldout_pairs.close(); ordinary_val.close()
    result = {
        "contract": CONTRACT_V2, "selected_epoch": selected_epoch,
        "best_selection_score": None if selected_epoch is None else best,
        "baseline_pair": baseline_pair, "baseline_ordinary": baseline_ordinary,
        "test_subsets_opened": False,
        "untouched_validation_remote_pairs_opened": False,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    if selected_epoch is None:
        raise RuntimeError("no paired-response epoch cleared predeclared gates")
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
