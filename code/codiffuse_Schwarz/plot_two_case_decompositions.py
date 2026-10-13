"""Plot the exact Darcy and validation-recorded Airfoil paper covers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codiffuse_mplconfig")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DARCY = ROOT / "codiffuse_Schwarz_Darcy/data/darcy_domain_extension_128_v1.h5"
OUTPUT = (
    ROOT
    / "codiffuse_Schwarz/workshop/submission/Two_case/figures/"
    "two_case_decompositions.pdf"
)
PURPLE = "#5B21B6"


def verify_cover_artifact(path: Path, payload: dict[str, object]) -> dict[str, object]:
    """Check that a cover comes from a sealed validation evaluation.

    The decomposition drawing is independent of rollout horizon and method set,
    so it should not be coupled to the legacy native-H20 verifier.  It does,
    however, inherit the same split-integrity requirements as a paper result.
    """

    if payload.get("split") != "validation":
        raise ValueError(f"Airfoil cover must come from validation, got {payload.get('split')!r}")
    if int(payload.get("protected_test_trajectories_opened", -1)) != 0:
        raise ValueError("Airfoil cover artifact opened protected test trajectories")
    if int(payload.get("protected_test_field_arrays_opened", -1)) != 0:
        raise ValueError("Airfoil cover artifact opened protected test fields")
    cover = payload.get("local_cover_contract")
    if not isinstance(cover, dict) or not cover.get("records"):
        raise ValueError("Airfoil evaluation does not contain a local-cover record")
    return {
        "contract": "validation_cover_verification_v1",
        "artifact": str(path.resolve()),
        "split": payload["split"],
        "protected_test_trajectories_opened": 0,
        "protected_test_field_arrays_opened": 0,
    }


def make_figure(
    airfoil_evaluation: Path,
    output: Path,
    airfoil_file: Path | None = None,
) -> dict[str, object]:
    payload = json.loads(airfoil_evaluation.read_text(encoding="utf-8"))
    verification = verify_cover_artifact(airfoil_evaluation, payload)
    cover_record = payload["local_cover_contract"]["records"][0]
    trajectory = int(cover_record["trajectory"])
    start = int(cover_record["start"])
    if airfoil_file is None:
        local_payload = torch.load(
            payload["local_checkpoint"], map_location="cpu", weights_only=False,
        )
        data_root = Path(local_payload["args"]["data_root"])
        coarse_name = payload["split_trajectory_files"][trajectory]
        fine_name = coarse_name.replace("_level4_", "_level5_")
        airfoil = data_root / fine_name
    else:
        airfoil = airfoil_file

    with h5py.File(DARCY, "r") as handle:
        mask = np.asarray(handle["val/mask"][39, 0], dtype=bool)
        log_permeability = np.asarray(
            handle["val/logk"][39, 0], dtype=np.float32
        )
    with h5py.File(airfoil, "r") as handle:
        signed_distance = np.asarray(handle["sdf"][start, 0], dtype=np.float32)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.15),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 2.0)},
    )

    axes[0].imshow(
        np.ma.array(log_permeability, mask=~mask),
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        rasterized=True,
    )
    for boundary in (31.5, 63.5, 95.5):
        axes[0].axhline(boundary, color="white", lw=0.7, ls="--", alpha=0.9)
        axes[0].axvline(boundary, color="white", lw=0.7, ls="--", alpha=0.9)
    axes[0].contour(mask.astype(float), levels=(0.5,), colors="black", linewidths=0.6)
    axes[0].set_title("Darcy", fontsize=9)
    axes[0].set_aspect("equal")

    height, width = signed_distance.shape
    axes[1].set_facecolor("#F7F7FA")
    axes[1].contourf(
        np.arange(width),
        np.arange(height),
        signed_distance < 0,
        levels=(-0.5, 0.5, 1.5),
        colors=("#F7F7FA", "#262626"),
    )
    axes[1].contour(signed_distance, levels=(0,), colors="black", linewidths=0.6)
    core = int(cover_record["core"])
    offset_y = int(cover_record["offset_y"])
    offset_x = int(cover_record["offset_x"])
    rows, columns = (int(value) for value in cover_record["grid_shape"])
    for index in range(columns + 1):
        boundary = index * core - offset_x
        if 0 <= boundary <= width:
            axes[1].axvline(boundary, color=PURPLE, lw=0.32, alpha=0.52)
    for index in range(rows + 1):
        boundary = index * core - offset_y
        if 0 <= boundary <= height:
            axes[1].axhline(boundary, color=PURPLE, lw=0.32, alpha=0.52)
    axes[1].set_xlim(0, width - 1)
    axes[1].set_ylim(0, height - 1)
    axes[1].set_aspect("equal")
    axes[1].set_title("Pitching Airfoil", fontsize=9)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.65)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        output.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    metadata: dict[str, object] = {
        "contract": "two_case_decomposition_visual_v2",
        "darcy": {"grid_shape": [4, 4], "tiles": 16, "core": 32, "halo": 4},
        "airfoil": {
            "trajectory": trajectory,
            "start": start,
            "grid_shape_yx": [rows, columns],
            "tiles": int(cover_record["tile_count"]),
            "core": core,
            "halo": int(cover_record["halo"]),
            "offset_yx": [offset_y, offset_x],
            "source": str(airfoil.resolve()),
        },
        "airfoil_evaluation": str(airfoil_evaluation.resolve()),
        "verification": verification,
        "output_pdf": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--airfoil-evaluation",
        "--airfoil-h20",
        dest="airfoil_evaluation",
        type=Path,
        required=True,
        help="sealed validation JSON containing local_cover_contract",
    )
    parser.add_argument(
        "--airfoil-file",
        type=Path,
        help="optional current Airfoil HDF5 file used only as the SDF backdrop",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(
        make_figure(args.airfoil_evaluation, args.output, args.airfoil_file),
        indent=2,
    ))


if __name__ == "__main__":
    main()
