"""Estimate a validation-only residual-port response conditioning spectrum.

For each frozen local tile this constructs the standardized port transfer

    M_X = S^{-1} Q B_X Z_X S,

by applying the 16 coordinate perturbations to the learned Green-response
fields and projecting their traces with the same fixed degree-3 port basis.
No target solution is used in this diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from codiffuse_Schwarz.residual_port import project_residual_ports

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .residual_port_composer import (
    GeometryLinearResidualResponseChart, LinearResidualResponseChart,
    ResidualResponseChart,
)
from .train import make_loader
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-cases", type=int, default=64)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument(
        "--difference-step", type=float, default=0.1,
        help="centered Jacobian step for nonlinear charts",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def tile_transfer_matrices(
    chart: LinearResidualResponseChart | GeometryLinearResidualResponseChart | ResidualResponseChart,
    context, spec, scale: torch.Tensor, difference_step: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and standardized port-response matrices per tile."""
    tile_count, channels, height, spatial_width = context.local.shape
    width = spec.code_width
    if isinstance(chart, (LinearResidualResponseChart, GeometryLinearResidualResponseChart)):
        modes = chart._response_modes(
            context.local, context.physical, context.scalar, context.valid,
        )
        response = modes.permute(0, 2, 1, 3, 4).reshape(
            tile_count * width, channels, height, spatial_width,
        )
    else:
        if difference_step <= 0.0:
            raise ValueError("nonlinear centered-difference step must be positive")
        repeats = width * 2
        def repeated(value):
            return value[:, None].expand(
                tile_count, repeats, *value.shape[1:],
            ).reshape(tile_count * repeats, *value.shape[1:])
        identity = torch.eye(width, device=context.local.device,
                             dtype=context.local.dtype) * difference_step
        codes = torch.cat((identity, -identity), 0)[None].expand(
            tile_count, -1, -1,
        ).reshape(tile_count * repeats, width)
        output = chart(
            repeated(context.local), repeated(context.physical),
            repeated(context.scalar), repeated(context.valid), codes,
        ).correction.reshape(
            tile_count, repeats, channels, height, spatial_width,
        )
        response = (
            output[:, :width] - output[:, width:]
        ).div(2.0 * difference_step).reshape(
            tile_count * width, channels, height, spatial_width,
        )
    validity = context.valid[:, None].expand(
        tile_count, width, 1, height, spatial_width,
    ).reshape(tile_count * width, 1, height, spatial_width)
    projected = project_residual_ports(response, validity, spec).reshape(
        tile_count, width, width,
    )
    # Rows index input coordinates and columns output coordinates.  The raw
    # matrix is Q B_X Z_X for unit standardized input.  The second matrix also
    # standardizes the output coordinates by the train-only scale S.
    standardized = projected / scale.to(projected)[None, None, :]
    return projected, standardized


def _statistics(singular_values: torch.Tensor) -> dict[str, object]:
    values = singular_values.float().cpu()
    maximum = values[:, :1].clamp_min(1.0e-12)
    relative = values / maximum
    stable_rank = values.square().sum(1) / maximum[:, 0].square()
    effective_rank = (relative > 1.0e-2).sum(1).float()
    return {
        "mean_singular_values": values.mean(0).tolist(),
        "median_singular_values": values.median(0).values.tolist(),
        "p10_singular_values": torch.quantile(values, 0.1, dim=0).tolist(),
        "p90_singular_values": torch.quantile(values, 0.9, dim=0).tolist(),
        "mean_stable_rank": float(stable_rank.mean()),
        "mean_effective_rank_at_1e-2": float(effective_rank.mean()),
        "minimum_top_singular_value": float(values[:, 0].min()),
    }


def _supported_statistics(
    matrices: list[torch.Tensor],
    supports: list[torch.Tensor],
    physicals: list[torch.Tensor],
    empties: list[torch.Tensor],
) -> dict[str, object]:
    singular_values, support_counts = [], []
    physical_count = 0
    empty_count = 0
    for matrix_batch, support_batch, physical_batch, empty_batch in zip(
        matrices, supports, physicals, empties,
    ):
        for matrix, support in zip(matrix_batch, support_batch):
            count = int(support.sum())
            support_counts.append(count)
            if count:
                restricted = matrix[support][:, support]
                singular_values.append(torch.linalg.svdvals(restricted).cpu())
        physical_count += int(physical_batch.sum())
        empty_count += int(empty_batch.sum())
    if not singular_values:
        raise RuntimeError("no supported artificial-face port coordinates found")
    maximum = max(value.numel() for value in singular_values)
    padded = torch.full((len(singular_values), maximum), float("nan"))
    stable_ranks, effective_ranks, tops = [], [], []
    for row, value in enumerate(singular_values):
        padded[row, :value.numel()] = value
        top = value[0].clamp_min(1.0e-12)
        tops.append(value[0])
        stable_ranks.append(value.square().sum() / top.square())
        effective_ranks.append((value / top > 1.0e-2).sum())
    return {
        "mean_singular_values_by_supported_ordinal": torch.nanmean(padded, 0).tolist(),
        "median_singular_values_by_supported_ordinal": torch.nanmedian(padded, 0).values.tolist(),
        "mean_supported_coordinates": float(np.mean(support_counts)),
        "minimum_supported_coordinates": min(support_counts),
        "maximum_supported_coordinates": max(support_counts),
        "mean_stable_rank": float(torch.stack(stable_ranks).float().mean()),
        "mean_effective_rank_at_1e-2": float(
            torch.stack(effective_ranks).float().mean()
        ),
        "minimum_top_singular_value": float(torch.stack(tops).min()),
        "structural_physical_boundary_coordinates": physical_count,
        "structural_empty_coordinates": empty_count,
        "interpretation": (
            "SVD restricts input/output to artificial-face coordinates; "
            "physical-boundary and empty-face coordinates are structural nulls"
        ),
    }


def port_coordinate_support(context, spec) -> tuple[Tensor, Tensor, Tensor]:
    if context.physical.shape[1] < 28:
        raise ValueError("supported-port audit requires FaceDarcyAdapter roles")
    artificial_face = (
        context.physical[:, 24:28] * context.valid
    ).flatten(2).sum(2) > 0
    physical_face = (
        context.physical[:, 20:24] * context.valid
    ).flatten(2).sum(2) > 0
    face_present = (
        context.physical[:, 12:16] * context.valid
    ).flatten(2).sum(2) > 0
    empty_face = ~face_present
    repeat = spec.field_channels * spec.modes_per_face
    return tuple(
        value.repeat_interleave(repeat, dim=1)
        for value in (artificial_face, physical_face, empty_face)
    )


def analyze(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    if chart_checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("linear chart and local checkpoint identities differ")
    dataset = H5DarcyDataset(args.data, "validation")
    if not dataset.has_face_dirichlet:
        raise ValueError("spectrum diagnostic requires face-resolved validation data")
    loader = make_loader(
        dataset, args.batch_size, False, args.workers, device,
    )
    raw_values, standardized_values = [], []
    raw_matrices, standardized_matrices = [], []
    supports, physicals, empties = [], [], []
    cases = 0
    for host_batch in loader:
        if cases >= args.maximum_cases:
            break
        batch = move_batch(host_batch, device)
        context = frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )
        raw, standardized = tile_transfer_matrices(
            chart, context, spec, scale, args.difference_step,
        )
        raw_values.append(torch.linalg.svdvals(raw).cpu())
        standardized_values.append(torch.linalg.svdvals(standardized).cpu())
        support, physical, empty = port_coordinate_support(context, spec)
        raw_matrices.append(raw.cpu())
        standardized_matrices.append(standardized.cpu())
        supports.append(support.cpu())
        physicals.append(physical.cpu())
        empties.append(empty.cpu())
        cases += batch["solution"].shape[0]
    if not raw_values:
        raise RuntimeError("spectrum diagnostic saw no validation cases")
    raw_singular = torch.cat(raw_values)
    standardized_singular = torch.cat(standardized_values)
    result = {
        "contract": "darcy_response_chart_port_spectrum_v1",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "operator": (
            f"exact Q B_X Z_X by all degree-{spec.polynomial_degree} "
            "port-coordinate perturbations"
            if isinstance(chart, (LinearResidualResponseChart, GeometryLinearResidualResponseChart))
            else f"centered Q dB_X/dz Z_X by degree-{spec.polynomial_degree} "
            "port perturbations"
        ),
        "chart_type": chart_checkpoint["chart_spec"].get(
            "chart_type", "nonlinear_12mode",
        ),
        "difference_step": (
            None if isinstance(chart, (LinearResidualResponseChart, GeometryLinearResidualResponseChart))
            else args.difference_step
        ),
        "port_width": spec.code_width,
        "cases": min(cases, args.maximum_cases),
        "tiles": int(raw_singular.shape[0]),
        "source_local_sha256": sha256(args.local_checkpoint),
        "source_chart_sha256": sha256(args.chart_checkpoint),
        "raw_output_coordinates": _statistics(raw_singular),
        "standardized_output_coordinates": _statistics(standardized_singular),
        "supported_artificial_coordinates_raw": _supported_statistics(
            raw_matrices, supports, physicals, empties,
        ),
        "supported_artificial_coordinates_standardized": _supported_statistics(
            standardized_matrices, supports, physicals, empties,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2))
