"""Matched split-controlled evaluation of adaptive and global baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from codiffuse_Schwarz.global_vit import GlobalPatchTransformer

from codiffuse_Schwarz_Airfoil.compare_multitrajectory_models import (
    field_metrics,
    force_metrics,
)
from codiffuse_Schwarz_Airfoil.coupled_trajectory import CyclicWindowDataset
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics
from codiffuse_Schwarz_Airfoil.global_mpp import (
    load_fullfield_mpp,
    normalized_mpp_delta,
)
from codiffuse_Schwarz_Airfoil.interface_operator import (
    load_interface_trace_checkpoint,
)
from codiffuse_Schwarz_Airfoil.multiresolution import (
    AdaptiveGridController,
    PairedCartesianLevelDataset,
    conservative_restrict,
    constant_prolongation,
    decode_composite,
)
from codiffuse_Schwarz_Airfoil.multiresolution_inference import (
    composite_vorticity_relative_l2,
    predict_multiresolution_step,
)
from codiffuse_Schwarz_Airfoil.train_global_abupt import (
    INPUT_CHANNELS as ABUPT_INPUT_CHANNELS,
    make_inputs as make_abupt_inputs,
    physical_state,
    predict as abupt_predict,
)
from codiffuse_Schwarz_Airfoil.train_global_transolver import (
    TRANSOLVER_INPUT_CHANNELS,
    make_resolution_inputs,
)
from codiffuse_Schwarz_Airfoil.train_multiresolution_projection import batched
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import load_pretrained
from codiffuse_Schwarz_Airfoil.vorticity_projection import nonperiodic_vorticity
from src.neural_eos.abupt import ABUPTScalarOperator, DynamicABUPTOperator
from src.neural_eos.full_field import FullFieldFNO, FullFieldUNet
from src.neural_eos.transolver import StructuredTransolverOperator


DEFAULT_METHODS = ("codiffuse_amr", "transolver")
COLORS = {
    "codiffuse_amr": "#1769aa", "transolver": "#c62828", "abupt": "#2e7d32",
    "patch_transformer": "#7A4EAB", "fno": "#ef6c00",
    "unet": "#6a1b9a", "mpp": "#00838f",
}
LABELS = {
    "codiffuse_amr": "CoDiffuse-AMR",
    "transolver": "Structured Transolver",
    "patch_transformer": "Patch Transformer",
    "abupt": "AB-UPT",
    "fno": "FNO",
    "unet": "U-Net",
    "mpp": "Full-field MPP",
}


def global_method_name(checkpoint: dict) -> str:
    return (
        "patch_transformer"
        if checkpoint.get("model_name") == "global_patch_transformer"
        else "transolver"
    )


def load_global(checkpoint: dict, device: torch.device):
    if global_method_name(checkpoint) == "patch_transformer":
        if checkpoint["model_args"].get("position_encoding") != \
                "normalized_linear_fourier_v1":
            raise ValueError("global ViT uses an obsolete position encoding")
        model = GlobalPatchTransformer(**checkpoint["model_args"]).to(device)
    else:
        model = StructuredTransolverOperator(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.level_policy = checkpoint.get("level_policy", "balanced")
    return model.eval()


def load_transolver(checkpoint: dict, device: torch.device):
    """Backward-compatible loader retained for existing Transolver callers."""

    if global_method_name(checkpoint) != "transolver":
        raise ValueError("checkpoint is not a Transolver")
    return load_global(checkpoint, device)


def load_abupt(checkpoint: dict, device: torch.device):
    constructor = (
        DynamicABUPTOperator
        if checkpoint.get("model_name") == "dynamic_abupt"
        else ABUPTScalarOperator
    )
    model_args = checkpoint.get("model_args")
    if model_args is None:
        # Fixed-cadence AB-UPT source checkpoints produced before the explicit
        # constructor metadata was added remain exactly reconstructible from
        # their audited CLI and input-channel contract.
        args = checkpoint["args"]
        model_args = {
            "in_channels": len(checkpoint["input_channels"]),
            "d_model": int(args["d_model"]),
            "n_anchors": int(args["anchors"]),
            "depth": int(args["depth"]),
            "heads": int(args["heads"]),
            "out_channels": 3,
        }
    model = constructor(**model_args).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), int(checkpoint["args"]["context_stride"])


def load_fullfield(checkpoint: dict, device: torch.device):
    name = checkpoint["model_name"]
    if name == "mpp":
        return load_fullfield_mpp(checkpoint, device)
    constructors = {"fno": FullFieldFNO, "unet": FullFieldUNet}
    if name not in constructors:
        raise ValueError(f"unsupported full-field model {name!r}")
    model = constructors[name](**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def native_parent_mask(sample: dict, device: torch.device) -> torch.Tensor:
    fine = sample["native_refinement_mask_np1"][None].to(device).float()
    return F.max_pool2d(fine, kernel_size=2, stride=2) > 0.0


def composite_state_l2(
    coarse_prediction: torch.Tensor,
    fine_prediction: torch.Tensor,
    coarse_target: torch.Tensor,
    fine_target: torch.Tensor,
    parent_mask: torch.Tensor,
) -> float:
    coarse_weight = (~parent_mask).to(coarse_prediction.dtype)
    fine_weight = constant_prolongation(parent_mask.to(fine_prediction.dtype))
    numerator = (
        ((coarse_prediction - coarse_target).square() * coarse_weight).sum()
        + 0.25 * ((fine_prediction - fine_target).square() * fine_weight).sum()
    )
    denominator = (
        (coarse_target.square() * coarse_weight).sum()
        + 0.25 * (fine_target.square() * fine_weight).sum()
    ).clamp_min(1.0e-20)
    return float(torch.sqrt(numerator / denominator))


def native_composite(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    parent_mask: torch.Tensor,
) -> torch.Tensor:
    """Compose every method through the numerical solver's ownership mask."""
    return decode_composite(coarse, fine, parent_mask)


def protected_field_metrics(
    prediction: torch.Tensor, target: torch.Tensor, dx: float,
) -> dict[str, float]:
    """Field metrics with a physical, non-wrapping vorticity stencil."""
    metrics = field_metrics(prediction, target, dx)
    predicted_vorticity = nonperiodic_vorticity(prediction, dx)
    target_vorticity = nonperiodic_vorticity(target, dx)
    numerator = (predicted_vorticity - target_vorticity).square().sum()
    denominator = target_vorticity.square().sum().clamp_min(1.0e-20)
    metrics["vorticity_l2"] = float(torch.sqrt(numerator / denominator))
    return metrics


def model_input_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove held-out solver state and ownership metadata from inference."""
    result = {
        key: value for key, value in batch.items()
        if not key.startswith(("native_refinement", "right_ghost_"))
    }
    # Legacy transition helpers use q_np1 only to allocate endpoint tensors.
    # A zero sentinel preserves that shape without exposing the held-out field.
    result["q_np1"] = torch.zeros_like(batch["q_n"])
    return result


@torch.no_grad()
def transolver_step(
    model, batch, dataset, statistics, device,
) -> torch.Tensor:
    inputs, _ = make_resolution_inputs(batch, dataset, statistics, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        delta = model(inputs).float()
    return physical_state(delta, batch, statistics, device)


@torch.no_grad()
def transolver_level_pair(
    model, coarse_batch, fine_batch, coarse_dataset, fine_dataset,
    statistics, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict one hierarchy without independent levels in finest-only mode."""
    fine = transolver_step(model, fine_batch, fine_dataset, statistics, device)
    if getattr(model, "level_policy", "balanced") == "finest_available":
        return conservative_restrict(fine), fine
    coarse = transolver_step(
        model, coarse_batch, coarse_dataset, statistics, device,
    )
    return coarse, fine


@torch.no_grad()
def abupt_step(
    model, batch, dataset, statistics, context_stride, device,
) -> torch.Tensor:
    if isinstance(model, DynamicABUPTOperator):
        inputs, _ = make_resolution_inputs(batch, dataset, statistics, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            delta = model(inputs).float()
    else:
        input_width = model.context_embedding[0].in_features
        if input_width == len(TRANSOLVER_INPUT_CHANNELS):
            inputs, _ = make_resolution_inputs(batch, dataset, statistics, device)
        elif input_width == len(ABUPT_INPUT_CHANNELS):
            inputs, _ = make_abupt_inputs(batch, dataset, statistics, device)
        else:
            raise ValueError(f"unsupported AB-UPT input width {input_width}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            delta = abupt_predict(model, inputs, context_stride).float()
    return physical_state(delta, batch, statistics, device)


@torch.no_grad()
def fullfield_step(
    method, model, batch, dataset, statistics, device,
) -> torch.Tensor:
    if method == "mpp":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            delta = normalized_mpp_delta(
                model, batch, dataset, statistics, device,
            ).float()
    else:
        inputs, _ = make_resolution_inputs(batch, dataset, statistics, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            delta = model(inputs).float()
    return physical_state(delta, batch, statistics, device)


def row_metrics(
    method: str,
    coarse_prediction: torch.Tensor,
    fine_prediction: torch.Tensor,
    coarse_target: torch.Tensor,
    fine_target: torch.Tensor,
    parent_mask: torch.Tensor,
    coarse_dataset: CyclicWindowDataset,
    fine_dataset: CyclicWindowDataset,
    force_prediction: torch.Tensor,
    fine_batch: dict[str, torch.Tensor],
    reference_force_batch: dict[str, torch.Tensor] | None = None,
) -> dict[str, float | str]:
    return {
        "method": method,
        **{f"l5_{key}": value for key, value in protected_field_metrics(
            coarse_prediction, coarse_target, coarse_dataset.dx,
        ).items()},
        "amr_state_l2": composite_state_l2(
            coarse_prediction, fine_prediction, coarse_target, fine_target,
            parent_mask,
        ),
        "amr_vorticity_l2": float(composite_vorticity_relative_l2(
            coarse_prediction, fine_prediction, coarse_target, fine_target,
            parent_mask, coarse_dx=coarse_dataset.dx, fine_dx=fine_dataset.dx,
        )),
        **force_metrics(
            force_prediction, fine_target, fine_batch, fine_dataset,
            reference_batch=reference_force_batch,
        ),
    }


def timed_cuda(callable_):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable_()
    torch.cuda.synchronize()
    return result, 1000.0 * (time.perf_counter() - start)


@torch.no_grad()
def warmup_models(
    local_model, global_model, paired, coarse_dataset, fine_dataset,
    statistics, args, device, abupt_model=None, abupt_context_stride=None,
    fullfield_models=None,
) -> None:
    """Run one common untimed L5/L6 input through every inference path."""
    sample = paired[0]
    coarse_batch = model_input_batch(batched(sample, "coarse"))
    fine_batch = model_input_batch(batched(sample, "fine"))
    predict_multiresolution_step(
        local_model, coarse_batch, fine_batch, coarse_dataset, fine_dataset,
        statistics, fine_fraction=args.fine_fraction, halo=args.halo,
        seed=args.seed, device=device, fine_trust=args.fine_trust,
        vorticity_amplitude_fraction=args.vorticity_amplitude_fraction,
        placement_ensemble=args.placement_ensemble,
    )
    transolver_level_pair(
        global_model, coarse_batch, fine_batch, coarse_dataset, fine_dataset,
        statistics, device,
    )
    if abupt_model is not None:
        abupt_step(
            abupt_model, coarse_batch, coarse_dataset, statistics,
            abupt_context_stride, device,
        )
        abupt_step(
            abupt_model, fine_batch, fine_dataset, statistics,
            abupt_context_stride, device,
        )
    for method, model in (fullfield_models or {}).items():
        fullfield_step(method, model, coarse_batch, coarse_dataset, statistics, device)
        fullfield_step(method, model, fine_batch, fine_dataset, statistics, device)
    torch.cuda.synchronize()


@torch.no_grad()
def evaluate_one_step(
    local_model, global_model, pairs, coarse_sets, fine_sets,
    statistics, args, device, abupt_model=None, abupt_context_stride=None,
    fullfield_models=None, global_method: str = "transolver",
) -> list[dict]:
    fullfield_models = fullfield_models or {}
    rows = []
    for trajectory, paired in enumerate(pairs):
        count = min(args.one_step_samples_per_trajectory, len(paired))
        items = np.linspace(0, len(paired) - 1, count, dtype=int)
        for local, item in enumerate(items):
            sample = paired[int(item)]
            coarse_batch, fine_batch = batched(sample, "coarse"), batched(sample, "fine")
            coarse_model_batch = model_input_batch(coarse_batch)
            fine_model_batch = model_input_batch(fine_batch)
            local_prediction, local_ms = timed_cuda(lambda: predict_multiresolution_step(
                local_model, coarse_model_batch, fine_model_batch, coarse_sets[trajectory],
                fine_sets[trajectory], statistics, fine_fraction=args.fine_fraction,
                halo=args.halo, seed=args.seed + trajectory * 1_000_003 + local,
                device=device, fine_trust=args.fine_trust,
                vorticity_amplitude_fraction=args.vorticity_amplitude_fraction,
                placement_ensemble=args.placement_ensemble,
            ))
            (global_coarse, global_fine), global_ms = timed_cuda(lambda: (
                transolver_level_pair(
                    global_model, coarse_model_batch, fine_model_batch,
                    coarse_sets[trajectory], fine_sets[trajectory], statistics,
                    device,
                )
            ))
            abupt_outputs = None
            if abupt_model is not None:
                abupt_outputs, abupt_ms = timed_cuda(lambda: (
                    abupt_step(
                        abupt_model, coarse_model_batch, coarse_sets[trajectory],
                        statistics, abupt_context_stride, device,
                    ),
                    abupt_step(
                        abupt_model, fine_model_batch, fine_sets[trajectory],
                        statistics, abupt_context_stride, device,
                    ),
                ))
            fullfield_outputs = {}
            for method, model in fullfield_models.items():
                outputs, elapsed = timed_cuda(lambda method=method, model=model: (
                    fullfield_step(
                        method, model, coarse_model_batch, coarse_sets[trajectory],
                        statistics, device,
                    ),
                    fullfield_step(
                        method, model, fine_model_batch, fine_sets[trajectory],
                        statistics, device,
                    ),
                ))
                fullfield_outputs[method] = (*outputs, elapsed)
            coarse_target = coarse_batch["q_np1"].to(device)
            fine_target = fine_batch["q_np1"].to(device)
            parent_mask = native_parent_mask(sample, device)
            local_composite = native_composite(
                local_prediction.coarse, local_prediction.fine, parent_mask,
            )
            global_composite = native_composite(
                global_coarse, global_fine, parent_mask,
            )
            metadata = {
                "regime": "one_step", "trajectory": paired.coarse_path.stem,
                "item": int(item), "frame": int(sample["frame"]),
            }
            rows.append(metadata | row_metrics(
                "codiffuse_amr", local_prediction.coarse, local_prediction.fine,
                coarse_target, fine_target, parent_mask, coarse_sets[trajectory],
                fine_sets[trajectory], local_composite, fine_batch,
            ) | {
                "inference_ms": local_ms,
                "refined_parent_fraction": float(
                    local_prediction.refinement_mask.float().mean()
                ),
                "executed_fine_tile_fraction": float(
                    local_prediction.executed_fine_tiles.float().mean()
                ),
            })
            rows.append(metadata | row_metrics(
                global_method, global_coarse, global_fine, coarse_target,
                fine_target, parent_mask, coarse_sets[trajectory],
                fine_sets[trajectory], global_composite, fine_batch,
            ) | {"inference_ms": global_ms})
            if abupt_outputs is not None:
                abupt_coarse, abupt_fine = abupt_outputs
                abupt_composite = native_composite(
                    abupt_coarse, abupt_fine, parent_mask,
                )
                rows.append(metadata | row_metrics(
                    "abupt", abupt_coarse, abupt_fine, coarse_target,
                    fine_target, parent_mask, coarse_sets[trajectory],
                    fine_sets[trajectory], abupt_composite, fine_batch,
                ) | {"inference_ms": abupt_ms})
            for method, (coarse_prediction, fine_prediction, elapsed) in (
                fullfield_outputs.items()
            ):
                method_composite = native_composite(
                    coarse_prediction, fine_prediction, parent_mask,
                )
                rows.append(metadata | row_metrics(
                    method, coarse_prediction, fine_prediction, coarse_target,
                    fine_target, parent_mask, coarse_sets[trajectory],
                    fine_sets[trajectory], method_composite, fine_batch,
                ) | {"inference_ms": elapsed})
    return rows


@torch.no_grad()
def evaluate_rollouts(
    local_model, global_model, pairs, coarse_sets, fine_sets,
    statistics, args, device, abupt_model=None, abupt_context_stride=None,
    fullfield_models=None, global_method: str = "transolver",
) -> tuple[list[dict], list[dict]]:
    fullfield_models = fullfield_models or {}
    rows, endpoints = [], []
    for trajectory, paired in enumerate(pairs):
        maximum = len(paired) - args.horizon
        starts = sorted(set(np.linspace(
            0, maximum, args.rollout_starts_per_trajectory, dtype=int,
        ).tolist()))
        for start_index, start in enumerate(starts):
            initial = paired[start]
            local_history = {
                "coarse": (
                    initial["coarse_q_nm1"][None].to(device),
                    initial["coarse_q_n"][None].to(device),
                ),
                "fine": (
                    initial["fine_q_nm1"][None].to(device),
                    initial["fine_q_n"][None].to(device),
                ),
            }
            global_history = {
                level: tuple(value.clone() for value in local_history[level])
                for level in ("coarse", "fine")
            }
            abupt_history = (
                {
                    level: tuple(value.clone() for value in local_history[level])
                    for level in ("coarse", "fine")
                }
                if abupt_model is not None else None
            )
            fullfield_histories = {
                method: {
                    level: tuple(value.clone() for value in local_history[level])
                    for level in ("coarse", "fine")
                }
                for method in fullfield_models
            }
            controller = AdaptiveGridController(
                dx=coarse_sets[trajectory].dx, patch_cells=16,
                refine_amplitude_fraction=args.vorticity_amplitude_fraction,
            )
            for step in range(args.horizon):
                sample = paired[start + step]
                coarse_batch, fine_batch = batched(sample, "coarse"), batched(sample, "fine")
                local_batches = {
                    "coarse": model_input_batch(coarse_batch),
                    "fine": model_input_batch(fine_batch),
                }
                global_batches = {
                    level: dict(value) for level, value in local_batches.items()
                }
                abupt_batches = (
                    {level: dict(value) for level, value in local_batches.items()}
                    if abupt_history is not None else None
                )
                fullfield_batches = {
                    method: {
                        "coarse": dict(coarse_batch), "fine": dict(fine_batch),
                    }
                    for method in fullfield_models
                }
                for batches in fullfield_batches.values():
                    batches["coarse"] = model_input_batch(batches["coarse"])
                    batches["fine"] = model_input_batch(batches["fine"])
                for level in ("coarse", "fine"):
                    previous, current = local_history[level]
                    local_batches[level]["q_nm1"] = previous
                    local_batches[level]["q_n"] = current
                    previous, current = global_history[level]
                    global_batches[level]["q_nm1"] = previous
                    global_batches[level]["q_n"] = current
                    if abupt_batches is not None:
                        previous, current = abupt_history[level]
                        abupt_batches[level]["q_nm1"] = previous
                        abupt_batches[level]["q_n"] = current
                    for method in fullfield_models:
                        previous, current = fullfield_histories[method][level]
                        fullfield_batches[method][level]["q_nm1"] = previous
                        fullfield_batches[method][level]["q_n"] = current
                local_prediction, local_ms = timed_cuda(lambda: predict_multiresolution_step(
                    local_model, local_batches["coarse"], local_batches["fine"],
                    coarse_sets[trajectory], fine_sets[trajectory], statistics,
                    fine_fraction=args.fine_fraction, halo=args.halo,
                    seed=args.seed + trajectory * 10_000_019
                    + start_index * 100_003 + step, device=device,
                    fine_trust=args.fine_trust, grid_controller=controller,
                    placement_ensemble=args.placement_ensemble,
                ))
                (global_coarse, global_fine), global_ms = timed_cuda(lambda: (
                    transolver_level_pair(
                        global_model, global_batches["coarse"],
                        global_batches["fine"], coarse_sets[trajectory],
                        fine_sets[trajectory], statistics, device,
                    )
                ))
                abupt_outputs = None
                if abupt_batches is not None:
                    abupt_outputs, abupt_ms = timed_cuda(lambda: (
                        abupt_step(
                            abupt_model, abupt_batches["coarse"],
                            coarse_sets[trajectory], statistics,
                            abupt_context_stride, device,
                        ),
                        abupt_step(
                            abupt_model, abupt_batches["fine"],
                            fine_sets[trajectory], statistics,
                            abupt_context_stride, device,
                        ),
                    ))
                fullfield_outputs = {}
                for method, model in fullfield_models.items():
                    batches = fullfield_batches[method]
                    outputs, elapsed = timed_cuda(
                        lambda method=method, model=model, batches=batches: (
                            fullfield_step(
                                method, model, batches["coarse"],
                                coarse_sets[trajectory], statistics, device,
                            ),
                            fullfield_step(
                                method, model, batches["fine"],
                                fine_sets[trajectory], statistics, device,
                            ),
                        )
                    )
                    fullfield_outputs[method] = (*outputs, elapsed)
                local_history = {
                    "coarse": (local_history["coarse"][1], local_prediction.coarse),
                    "fine": (local_history["fine"][1], local_prediction.fine),
                }
                global_history = {
                    "coarse": (global_history["coarse"][1], global_coarse),
                    "fine": (global_history["fine"][1], global_fine),
                }
                if abupt_outputs is not None:
                    abupt_coarse, abupt_fine = abupt_outputs
                    abupt_history = {
                        "coarse": (abupt_history["coarse"][1], abupt_coarse),
                        "fine": (abupt_history["fine"][1], abupt_fine),
                    }
                for method, (coarse_prediction, fine_prediction, _) in (
                    fullfield_outputs.items()
                ):
                    fullfield_histories[method] = {
                        "coarse": (
                            fullfield_histories[method]["coarse"][1],
                            coarse_prediction,
                        ),
                        "fine": (
                            fullfield_histories[method]["fine"][1],
                            fine_prediction,
                        ),
                    }
                coarse_target = coarse_batch["q_np1"].to(device)
                fine_target = fine_batch["q_np1"].to(device)
                parent_mask = native_parent_mask(sample, device)
                local_composite = native_composite(
                    local_prediction.coarse, local_prediction.fine, parent_mask,
                )
                global_composite = native_composite(
                    global_coarse, global_fine, parent_mask,
                )
                metadata = {
                    "regime": "rollout", "trajectory": paired.coarse_path.stem,
                    "start": start, "step": step + 1,
                    "central_start": start_index == 1,
                }
                rows.append(metadata | row_metrics(
                    "codiffuse_amr", local_prediction.coarse, local_prediction.fine,
                    coarse_target, fine_target, parent_mask, coarse_sets[trajectory],
                    fine_sets[trajectory], local_composite,
                    local_batches["fine"], reference_force_batch=fine_batch,
                ) | {
                    "inference_ms": local_ms,
                    "refined_parent_fraction": float(
                        local_prediction.refinement_mask.float().mean()
                    ),
                    "executed_fine_tile_fraction": float(
                        local_prediction.executed_fine_tiles.float().mean()
                    ),
                })
                rows.append(metadata | row_metrics(
                    global_method, global_coarse, global_fine, coarse_target,
                    fine_target, parent_mask, coarse_sets[trajectory],
                    fine_sets[trajectory], global_composite, global_batches["fine"],
                    reference_force_batch=fine_batch,
                ) | {"inference_ms": global_ms})
                if abupt_outputs is not None:
                    abupt_composite = native_composite(
                        abupt_coarse, abupt_fine, parent_mask,
                    )
                    rows.append(metadata | row_metrics(
                        "abupt", abupt_coarse, abupt_fine, coarse_target,
                        fine_target, parent_mask, coarse_sets[trajectory],
                        fine_sets[trajectory], abupt_composite,
                        abupt_batches["fine"], reference_force_batch=fine_batch,
                    ) | {"inference_ms": abupt_ms})
                for method, (coarse_prediction, fine_prediction, elapsed) in (
                    fullfield_outputs.items()
                ):
                    method_composite = native_composite(
                        coarse_prediction, fine_prediction, parent_mask,
                    )
                    rows.append(metadata | row_metrics(
                        method, coarse_prediction, fine_prediction, coarse_target,
                        fine_target, parent_mask, coarse_sets[trajectory],
                        fine_sets[trajectory], method_composite,
                        fullfield_batches[method]["fine"],
                        reference_force_batch=fine_batch,
                    ) | {"inference_ms": elapsed})
                if step + 1 == args.horizon and start_index == 1:
                    target_composite = decode_composite(
                        coarse_target, fine_target, parent_mask,
                    )
                    endpoints.append({
                        "trajectory": paired.coarse_path.stem,
                        "target": target_composite.cpu(),
                        "codiffuse_amr": local_composite.cpu(),
                        global_method: global_composite.cpu(),
                        "dx": fine_sets[trajectory].dx,
                    } | (
                        {"abupt": abupt_composite.cpu()}
                        if abupt_outputs is not None else {}
                    ) | {
                        method: native_composite(
                            output[0], output[1], parent_mask,
                        ).cpu()
                        for method, output in fullfield_outputs.items()
                    })
    return rows, endpoints


def summarize(rows: list[dict], methods: tuple[str, ...] = DEFAULT_METHODS) -> dict:
    metrics = (
        "l5_state_l2", "l5_velocity_l2", "l5_pressure_gauge_l2",
        "l5_vorticity_l2", "amr_state_l2", "amr_vorticity_l2", "inference_ms",
    )
    result = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        result[method] = {}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            result[method][f"{metric}_mean"] = float(values.mean())
            result[method][f"{metric}_p90"] = float(np.quantile(values, 0.9))
            result[method][f"{metric}_max"] = float(values.max())
        for coefficient in (
            "c_drag", "c_lift", "c_drag_corrected", "c_lift_corrected",
        ):
            prediction = np.asarray([
                float(row[f"{coefficient}_prediction"]) for row in selected
            ])
            reference = np.asarray([
                float(row[f"{coefficient}_reference"]) for row in selected
            ])
            result[method][f"{coefficient}_relative_l2"] = float(
                np.linalg.norm(prediction - reference)
                / max(np.linalg.norm(reference), 1.0e-20)
            )
            result[method][f"{coefficient}_mae"] = float(
                np.mean(np.abs(prediction - reference))
            )
        for diagnostic in (
            "refined_parent_fraction", "executed_fine_tile_fraction",
        ):
            values = [float(row[diagnostic]) for row in selected if diagnostic in row]
            if values:
                result[method][f"{diagnostic}_mean"] = float(np.mean(values))
    return result


def paired_trajectory_analysis(
    rows: list[dict],
    methods: tuple[str, ...] = DEFAULT_METHODS,
    *,
    metric: str = "amr_vorticity_l2",
    reference: str = "codiffuse_amr",
    bootstrap_samples: int = 20_000,
    seed: int = 20260811,
) -> dict:
    """Aggregate correlated frames before paired trajectory bootstrapping."""
    if reference not in methods or bootstrap_samples <= 0:
        raise ValueError("invalid paired trajectory analysis contract")
    trajectories = sorted({str(row["trajectory"]) for row in rows})
    means: dict[str, dict[str, float]] = {}
    for trajectory in trajectories:
        means[trajectory] = {}
        for method in methods:
            values = [
                float(row[metric]) for row in rows
                if row["trajectory"] == trajectory and row["method"] == method
            ]
            if not values:
                raise ValueError(f"missing {method} rows for {trajectory}")
            means[trajectory][method] = float(np.mean(values))

    rng = np.random.default_rng(seed)
    count = len(trajectories)
    indices = rng.integers(0, count, size=(bootstrap_samples, count))
    comparisons = {}
    reference_values = np.asarray([
        means[trajectory][reference] for trajectory in trajectories
    ])
    for method in methods:
        if method == reference:
            continue
        values = np.asarray([means[trajectory][method] for trajectory in trajectories])
        differences = values - reference_values
        bootstrap = differences[indices].mean(axis=1)
        comparisons[method] = {
            "method_minus_codiffuse_mean": float(differences.mean()),
            "ci95_low": float(np.quantile(bootstrap, .025)),
            "ci95_high": float(np.quantile(bootstrap, .975)),
            "probability_method_lower": float(np.mean(bootstrap < 0.0)),
            "n_trajectories": count,
        }
    return {
        "metric": metric,
        "experimental_unit": "trajectory",
        "bootstrap_samples": bootstrap_samples,
        "trajectory_means": means,
        "method_minus_codiffuse": comparisons,
    }


def save_paired_trajectory_vorticity(
    analyses: dict[str, dict], output: Path, methods: tuple[str, ...],
) -> None:
    figure, axes = plt.subplots(1, len(analyses), figsize=(6.2 * len(analyses), 4.8))
    axes = np.atleast_1d(axes)
    for axis, (title, analysis) in zip(axes, analyses.items(), strict=True):
        trajectories = sorted(analysis["trajectory_means"])
        x = np.arange(len(methods))
        for trajectory in trajectories:
            values = [
                analysis["trajectory_means"][trajectory][method]
                for method in methods
            ]
            axis.plot(x, values, color="#bdbdbd", linewidth=.8, alpha=.75, zorder=1)
        for index, method in enumerate(methods):
            values = [
                analysis["trajectory_means"][trajectory][method]
                for trajectory in trajectories
            ]
            axis.scatter(
                np.full(len(values), index), values, s=28,
                color=COLORS[method], edgecolor="white", linewidth=.5, zorder=2,
            )
            axis.plot(
                [index - .22, index + .22], [np.mean(values)] * 2,
                color="#111111", linewidth=2.0, zorder=3,
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel("Trajectory-mean AMR vorticity relative L2")
        axis.set_xticks(
            x, [LABELS[method] for method in methods], rotation=32, ha="right",
        )
        axis.grid(True, axis="y", which="both", color="#dddddd")
    figure.suptitle("Paired evaluation trajectories; black bars are trajectory means")
    figure.tight_layout()
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)


def save_force_histories(
    rows: list[dict], output: Path, methods: tuple[str, ...] = DEFAULT_METHODS,
    *, corrected: bool = False,
) -> None:
    selected = [row for row in rows if row.get("central_start")]
    trajectories = sorted({row["trajectory"] for row in selected})
    figure, axes = plt.subplots(
        len(trajectories), 2, figsize=(11, 2.7 * len(trajectories)), squeeze=False,
    )
    for row_index, trajectory in enumerate(trajectories):
        subset = [row for row in selected if row["trajectory"] == trajectory]
        reference = sorted(
            (row for row in subset if row["method"] == methods[0]),
            key=lambda row: row["step"],
        )
        suffix = "_corrected" if corrected else ""
        for column, coefficient in enumerate((
            f"c_drag{suffix}", f"c_lift{suffix}",
        )):
            axis = axes[row_index, column]
            steps = [row["step"] for row in reference]
            axis.plot(
                steps, [row[f"{coefficient}_reference"] for row in reference],
                color="#222222", linewidth=2, label="AMReX",
            )
            for method in methods:
                method_rows = sorted(
                    (row for row in subset if row["method"] == method),
                    key=lambda row: row["step"],
                )
                axis.plot(
                    steps, [row[f"{coefficient}_prediction"] for row in method_rows],
                    color=COLORS[method], linewidth=1.5, label=LABELS[method],
                )
            axis.grid(True, color="#dddddd", linewidth=0.7)
            axis.set_xlabel("Autoregressive step")
            axis.set_ylabel(r"$C_D$" if column == 0 else r"$C_L$")
        axes[row_index, 0].text(
            0.01, 0.96, trajectory.replace("_level5_window", ""),
            transform=axes[row_index, 0].transAxes, va="top",
        )
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, frameon=False, ncol=len(methods) + 1, fontsize=8,
        loc="upper center", bbox_to_anchor=(0.5, 0.975),
    )
    qualifier = " with fictitious-fluid correction" if corrected else ""
    figure.suptitle(
        f"Evaluation central-start force histories{qualifier}", y=0.997,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def save_loss_curves(
    local_path: Path, global_path: Path, output: Path, abupt_path: Path | None = None,
) -> None:
    local = json.loads((local_path.parent / "summary.json").read_text())
    global_ = json.loads((global_path.parent / "summary.json").read_text())
    abupt = (
        json.loads((abupt_path.parent / "summary.json").read_text())
        if abupt_path is not None else None
    )
    if any("all_step_amr_vorticity_l2_mean" in row for row in local["history"]):
        columns = 4 if abupt is not None else 3
        figure, axes = plt.subplots(
            1, columns, figsize=(4.7 * columns, 4), constrained_layout=True,
        )
        local_training = [
            row for row in local["history"]
            if "train_composite_vorticity_l2" in row
        ]
        axes[0].plot(
            [row["epoch"] for row in local_training],
            [row["train_composite_vorticity_l2"] for row in local_training],
            color="#1769aa",
        )
        axes[0].set_title("CoDiffuse rollout training")
        for key, label, color in (
            ("step1_amr_vorticity_l2_mean", "step 1", "#2e7d32"),
            ("step4_amr_vorticity_l2_mean", "step 4", "#ef6c00"),
            ("step20_amr_vorticity_l2_mean", "step 20", "#1769aa"),
        ):
            selected = [row for row in local["history"] if key in row]
            axes[1].plot(
                [row["epoch"] for row in selected], [row[key] for row in selected],
                color=color, label=label,
            )
        axes[1].set_title("CoDiffuse rollout validation")
        axes[1].legend(frameon=False)
        axes[2].plot(
            [row["epoch"] for row in global_["history"]],
            [row["train_vorticity_l2"] for row in global_["history"]],
            color="#c62828", label="training",
        )
        axes[2].plot(
            [row["epoch"] for row in global_["history"]],
            [row["validation_balanced_vorticity_l2_mean"] for row in global_["history"]],
            color="#7b1fa2", label="one-step validation",
        )
        axes[2].set_title(f"{global_.get('model', 'Global model')} one-step training")
        axes[2].legend(frameon=False)
        if abupt is not None:
            axes[3].plot(
                [row["epoch"] for row in abupt["history"]],
                [row["train_vorticity_l2"] for row in abupt["history"]],
                color=COLORS["abupt"], label="training",
            )
            axes[3].plot(
                [row["epoch"] for row in abupt["history"]],
                [row["validation_balanced_vorticity_l2_mean"]
                 for row in abupt["history"]],
                color="#6a1b9a", label="one-step validation",
            )
            axes[3].set_title("AB-UPT one-step training")
            axes[3].legend(frameon=False)
        for axis in axes:
            axis.set_yscale("log"); axis.set_xlabel("Epoch")
            axis.set_ylabel("Vorticity relative L2")
            axis.grid(True, which="both", color="#dddddd")
        figure.savefig(output, dpi=180, facecolor="white")
        plt.close(figure)
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    series = (
        (local["history"], "CoDiffuse-AMR", "#1769aa",
         "train_composite_vorticity_l2", "composite_vorticity_mean"),
        (global_["history"], global_.get("model", "Global model"), "#c62828",
         "train_vorticity_l2", "validation_balanced_vorticity_l2_mean"),
    )
    if abupt is not None:
        series += ((
            abupt["history"], "AB-UPT", COLORS["abupt"],
            "train_vorticity_l2", "validation_balanced_vorticity_l2_mean",
        ),)
    for history, label, color, train_key, validation_key in series:
        training = [row for row in history if train_key in row]
        validation = [row for row in history if validation_key in row]
        axes[0].plot(
            [row["epoch"] for row in training],
            [row[train_key] for row in training], color=color, label=label,
        )
        axes[1].plot(
            [row["epoch"] for row in validation],
            [row[validation_key] for row in validation], color=color, label=label,
        )
    for axis, title in zip(axes, ("Training vorticity", "Validation vorticity")):
        axis.set_yscale("log"); axis.set_title(title); axis.set_xlabel("Epoch")
        axis.set_ylabel("Relative L2"); axis.grid(True, which="both", color="#dddddd")
        axis.legend(frameon=False)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def save_vorticity(
    endpoints: list[dict], output: Path,
    methods: tuple[str, ...] = DEFAULT_METHODS,
) -> None:
    if len(methods) > 3:
        columns = 1 + len(methods)
        figure, axes = plt.subplots(
            len(endpoints), columns,
            figsize=(3.25 * columns, 3.0 * len(endpoints)), squeeze=False,
        )
        for row, endpoint in enumerate(endpoints):
            fields = {
                key: nonperiodic_vorticity(endpoint[key], endpoint["dx"])[0]
                for key in ("target", *methods)
            }
            scale = float(fields["target"].abs().quantile(.995))
            panels = [(fields["target"], "AMReX reference")] + [
                (fields[method], LABELS[method]) for method in methods
            ]
            for column, (field, title) in enumerate(panels):
                axis = axes[row, column]
                image = axis.imshow(
                    field, origin="lower", aspect="auto", cmap="RdBu_r",
                    vmin=-scale, vmax=scale,
                )
                if row == 0:
                    axis.set_title(title)
                if column == 0:
                    axis.set_ylabel(
                        endpoint["trajectory"].replace("_level5_window", "")
                    )
                axis.set_xticks([]); axis.set_yticks([])
                figure.colorbar(image, ax=axis, fraction=.025, pad=.015)
        figure.suptitle("Evaluation central-start 20-step AMR vorticity rollout")
        figure.tight_layout()
        figure.savefig(output, dpi=180, facecolor="white")
        plt.close(figure)
        return
    columns = 1 + 2 * len(methods)
    figure, axes = plt.subplots(
        len(endpoints), columns,
        figsize=(3.5 * columns, 3.2 * len(endpoints)), squeeze=False,
    )
    for row, endpoint in enumerate(endpoints):
        fields = {
            key: nonperiodic_vorticity(endpoint[key], endpoint["dx"])[0]
            for key in ("target", *methods)
        }
        scale = float(fields["target"].abs().quantile(0.995))
        error_scale = max(float(
            (fields[method] - fields["target"]).abs().quantile(0.995)
        ) for method in methods)
        panels = [
            (fields["target"], "AMReX reference", "RdBu_r", -scale, scale),
        ]
        for method in methods:
            panels.extend((
                (fields[method], LABELS[method], "RdBu_r", -scale, scale),
                ((fields[method] - fields["target"]).abs(),
                 f"{LABELS[method]} error", "magma", 0.0, error_scale),
            ))
        for column, (field, title, cmap, low, high) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(
                field, origin="lower", aspect="auto", cmap=cmap, vmin=low, vmax=high,
            )
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(endpoint["trajectory"].replace("_level5_window", ""))
            axis.set_xticks([]); axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    figure.suptitle("Evaluation central-start 20-step AMR vorticity rollout")
    figure.tight_layout()
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def save_vorticity_errors(
    endpoints: list[dict], output: Path, methods: tuple[str, ...],
) -> None:
    figure, axes = plt.subplots(
        len(endpoints), len(methods),
        figsize=(3.25 * len(methods), 3.0 * len(endpoints)), squeeze=False,
    )
    for row, endpoint in enumerate(endpoints):
        target = nonperiodic_vorticity(endpoint["target"], endpoint["dx"])[0]
        errors = {
            method: (
                nonperiodic_vorticity(endpoint[method], endpoint["dx"])[0]
                - target
            ).abs()
            for method in methods
        }
        scale = max(float(error.quantile(.995)) for error in errors.values())
        for column, method in enumerate(methods):
            axis = axes[row, column]
            image = axis.imshow(
                errors[method], origin="lower", aspect="auto", cmap="magma",
                vmin=0.0, vmax=scale,
            )
            if row == 0:
                axis.set_title(f"{LABELS[method]} error")
            if column == 0:
                axis.set_ylabel(
                    endpoint["trajectory"].replace("_level5_window", "")
                )
            axis.set_xticks([]); axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=.025, pad=.015)
    figure.suptitle("Absolute step-20 vorticity error; shared scale per case")
    figure.tight_layout()
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def save_metric_bars(summary: dict, output: Path, methods: tuple[str, ...]) -> None:
    metrics = (
        ("one_step", "amr_vorticity_l2_mean", "One-step AMR vorticity"),
        ("rollout_all_steps", "amr_vorticity_l2_mean", "Steps 1--20 mean"),
        ("rollout_final_step", "amr_vorticity_l2_mean", "Step-20 AMR vorticity"),
        ("rollout_final_step", "amr_state_l2_mean", "Step-20 AMR state"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(17, 4.2))
    for axis, (regime, key, title) in zip(axes, metrics, strict=True):
        values = [summary[regime][method][key] for method in methods]
        bars = axis.bar(
            range(len(methods)), values,
            color=[COLORS[method] for method in methods],
        )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xticks(
            range(len(methods)), [LABELS[method] for method in methods],
            rotation=35, ha="right",
        )
        axis.grid(True, axis="y", which="both", color="#dddddd")
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom", fontsize=7,
            )
    figure.tight_layout()
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)


def save_global_loss_curves(paths: dict[str, Path], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method, checkpoint_path in paths.items():
        summary = json.loads((checkpoint_path.parent / "summary.json").read_text())
        history = summary["history"]
        training = [row for row in history if "train_vorticity_l2" in row]
        validation = [
            row for row in history
            if "validation_balanced_vorticity_l2_mean" in row
        ]
        axes[0].plot(
            [row["epoch"] for row in training],
            [row["train_vorticity_l2"] for row in training],
            color=COLORS[method], label=LABELS[method],
        )
        axes[1].plot(
            [row["epoch"] for row in validation],
            [row["validation_balanced_vorticity_l2_mean"] for row in validation],
            color=COLORS[method], label=LABELS[method],
        )
    for axis, title in zip(
        axes, ("Training vorticity", "One-step validation vorticity"), strict=True,
    ):
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Relative L2")
        axis.grid(True, which="both", color="#dddddd")
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--codiffuse", type=Path, required=True)
    parser.add_argument(
        "--global-model", type=Path,
        help="from-scratch Transolver or full-domain Patch Transformer checkpoint",
    )
    parser.add_argument(
        "--transolver", type=Path,
        help="deprecated alias for --global-model",
    )
    parser.add_argument("--abupt", type=Path)
    parser.add_argument("--fno", type=Path)
    parser.add_argument("--unet", type=Path)
    parser.add_argument("--mpp", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-split", choices=("validation", "test"),
        default="validation",
        help="checkpoint split to evaluate; paper tables use validation",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--one-step-samples-per-trajectory", type=int, default=32)
    parser.add_argument("--rollout-starts-per-trajectory", type=int, default=3)
    parser.add_argument("--fine-fraction", type=float, default=0.35)
    parser.add_argument("--fine-trust", type=float, default=0.5)
    parser.add_argument("--placement-ensemble", type=int, default=1)
    parser.add_argument("--vorticity-amplitude-fraction", type=float, default=.15)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.global_model is None) == (args.transolver is None):
        raise ValueError("provide exactly one of --global-model or --transolver")
    global_path = args.global_model or args.transolver
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    local_checkpoint = torch.load(args.codiffuse, map_location="cpu", weights_only=False)
    global_checkpoint = torch.load(global_path, map_location="cpu", weights_only=False)
    if local_checkpoint["split"] != global_checkpoint["split"]:
        raise RuntimeError("local and global checkpoints use different trajectory splits")
    if local_checkpoint["statistics"] != global_checkpoint["statistics"]:
        raise RuntimeError("local and global checkpoints use different normalization statistics")
    abupt_checkpoint = (
        torch.load(args.abupt, map_location="cpu", weights_only=False)
        if args.abupt is not None else None
    )
    if (
        abupt_checkpoint is not None
        and local_checkpoint["split"] != abupt_checkpoint["split"]
    ):
        raise RuntimeError("AB-UPT checkpoint uses a different trajectory split")
    if (
        abupt_checkpoint is not None
        and local_checkpoint["statistics"] != abupt_checkpoint["statistics"]
    ):
        raise RuntimeError("AB-UPT checkpoint uses different normalization statistics")
    fullfield_checkpoints = {
        method: torch.load(path, map_location="cpu", weights_only=False)
        for method, path in (
            ("fno", args.fno), ("unet", args.unet), ("mpp", args.mpp),
        )
        if path is not None
    }
    for method, checkpoint in fullfield_checkpoints.items():
        if local_checkpoint["split"] != checkpoint["split"]:
            raise RuntimeError(
                f"{LABELS[method]} checkpoint uses a different trajectory split"
            )
        if local_checkpoint["statistics"] != checkpoint["statistics"]:
            raise RuntimeError(
                f"{LABELS[method]} checkpoint uses different normalization statistics"
            )
    split = local_checkpoint["split"]
    statistics = FieldStatistics.from_json(local_checkpoint["statistics"])
    if bool(local_checkpoint.get("model_spec", {}).get(
        "interface_trace_operator", False,
    )):
        local_model, _ = load_interface_trace_checkpoint(local_checkpoint, device)
    else:
        local_model, _ = load_pretrained(local_checkpoint, device, history_steps=2)
    global_method = global_method_name(global_checkpoint)
    global_model = load_global(global_checkpoint, device)
    if abupt_checkpoint is not None:
        abupt_model, abupt_context_stride = load_abupt(abupt_checkpoint, device)
    else:
        abupt_model, abupt_context_stride = None, None
    fullfield_models = {
        method: load_fullfield(checkpoint, device)
        for method, checkpoint in fullfield_checkpoints.items()
    }
    methods = (
        ("codiffuse_amr", global_method)
        + (("abupt",) if abupt_model is not None else ())
        + tuple(fullfield_models)
    )
    if args.rollout_starts_per_trajectory < 1:
        raise ValueError("rollout starts per trajectory must be positive")
    names = split[args.evaluation_split]
    coarse_sets = [
        CyclicWindowDataset(args.data_root / name, "all", history_steps=2)
        for name in names
    ]
    fine_sets = [
        CyclicWindowDataset(
            args.data_root / name.replace("_level5_", "_level6_"),
            "all", history_steps=2,
        ) for name in names
    ]
    pairs = [
        PairedCartesianLevelDataset(
            args.data_root / name,
            args.data_root / name.replace("_level5_", "_level6_"), "all",
        ) for name in names
    ]
    warmup_models(
        local_model, global_model, pairs[0], coarse_sets[0], fine_sets[0],
        statistics, args, device, abupt_model, abupt_context_stride,
        fullfield_models,
    )
    one_step = evaluate_one_step(
        local_model, global_model, pairs, coarse_sets, fine_sets,
        statistics, args, device, abupt_model, abupt_context_stride,
        fullfield_models, global_method=global_method,
    )
    rollout, endpoints = evaluate_rollouts(
        local_model, global_model, pairs, coarse_sets, fine_sets,
        statistics, args, device, abupt_model, abupt_context_stride,
        fullfield_models, global_method=global_method,
    )
    rows = one_step + rollout
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    final_rollout = [row for row in rollout if row["step"] == args.horizon]
    paired = {
        "one_step": paired_trajectory_analysis(one_step, methods, seed=args.seed + 1),
        "rollout_all_steps": paired_trajectory_analysis(
            rollout, methods, seed=args.seed + 2,
        ),
        "rollout_final_step": paired_trajectory_analysis(
            final_rollout, methods, seed=args.seed + 3,
        ),
    }
    summary = {
        "evaluation_split": args.evaluation_split,
        "evaluation_trajectories": names,
        "protected_test_trajectories_opened": (
            len(names) if args.evaluation_split == "test" else 0
        ),
        "horizon": args.horizon,
        "placement_ensemble": args.placement_ensemble,
        "evaluation_contract": {
            "version": "native_amr_composite_v5_split_controlled",
            "one_step_samples_per_trajectory": args.one_step_samples_per_trajectory,
            "rollout_starts_per_trajectory": args.rollout_starts_per_trajectory,
            "model_owned_rollout": True,
            "amr_hierarchy_output_each_step": True,
            "global_level_policy": global_checkpoint.get(
                "level_policy", "balanced",
            ),
            "global_coarse_from_fine_restriction": (
                global_checkpoint.get("level_policy") == "finest_available"
            ),
            "native_ownership_for_fields_forces_figures": True,
            "native_ownership_is_model_input": False,
            "future_flow_state_is_model_input": False,
            "future_flow_state_placeholder": "zeros",
            "solver_boundary_ghost_is_model_input": False,
            "normalization_statistics_identical_across_methods": True,
            "one_untimed_l5_l6_warmup_per_method": True,
        },
        "training_budget_disclosure": {
            "version": "global_equal_optimizer_budget_v2",
            "global_epochs": 100,
            "global_samples_per_epoch": 1024,
            "global_validation_samples_per_trajectory": 24,
            "level_sampling": {
                global_method: global_checkpoint.get(
                    "level_policy", "balanced",
                ),
                **{
                    method: "balanced_l5_l6"
                    for method in ("abupt", "fno", "unet", "mpp")
                    if method in methods
                },
            },
            "codiffuse_final_stage_epochs": int(
                local_checkpoint["args"]["epochs"]
            ),
            "codiffuse_final_stage_sequences_per_epoch": int(
                local_checkpoint["args"]["sequences_per_epoch"]
            ),
            "global_models_optimizer_step_budget_matched": True,
            "global_models_level_distribution_matched": (
                global_checkpoint.get("level_policy", "balanced") == "balanced"
            ),
            "codiffuse_global_optimizer_budget_matched": False,
            "codiffuse_external_pretraining_weights_used": True,
            "global_models_external_pretraining_weights_used": False,
            "codiffuse_complete_training_lineage_reproducible": False,
        },
        "checkpoint_contracts": {
            "codiffuse_amr": local_checkpoint.get("training_contract"),
            global_method: global_checkpoint.get("training_contract"),
            **({
                "abupt": abupt_checkpoint.get("training_contract"),
            } if abupt_checkpoint is not None else {}),
            **{
                method: checkpoint.get("training_contract")
                for method, checkpoint in fullfield_checkpoints.items()
            },
        },
        "parameter_count": {
            "codiffuse_amr": sum(
                parameter.numel() for parameter in local_model.parameters()
            ),
            global_method: sum(
                parameter.numel() for parameter in global_model.parameters()
            ),
            **({
                "abupt": sum(
                    parameter.numel() for parameter in abupt_model.parameters()
                ),
            } if abupt_checkpoint is not None else {}),
            **{
                method: sum(
                    parameter.numel() for parameter in fullfield_models[method].parameters()
                )
                for method in fullfield_models
            },
        },
        "one_step": summarize(one_step, methods),
        "rollout_all_steps": summarize(rollout, methods),
        "rollout_final_step": summarize(final_rollout, methods),
        "paired_trajectory_analysis": paired,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    save_vorticity(endpoints, args.output_dir / "vorticity_fields.png", methods)
    if len(methods) > 3:
        save_vorticity_errors(
            endpoints, args.output_dir / "vorticity_errors.png", methods,
        )
    save_force_histories(
        rollout, args.output_dir / "force_coefficients.png", methods,
    )
    save_force_histories(
        rollout, args.output_dir / "force_coefficients_corrected.png", methods,
        corrected=True,
    )
    save_loss_curves(
        args.codiffuse, global_path, args.output_dir / "loss_curves.png",
        args.abupt,
    )
    save_metric_bars(summary, args.output_dir / "all_model_metrics.png", methods)
    save_paired_trajectory_vorticity({
        "One-step": paired["one_step"],
        "Step 20": paired["rollout_final_step"],
    }, args.output_dir / "paired_trajectory_vorticity.png", methods)
    save_global_loss_curves({
        global_method: global_path,
        **({"abupt": args.abupt} if args.abupt is not None else {}),
        **{
            method: getattr(args, method)
            for method in fullfield_models
        },
    }, args.output_dir / "global_loss_curves.png")
    for dataset in (*pairs, *coarse_sets, *fine_sets):
        dataset.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
