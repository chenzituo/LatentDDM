"""Render deterministic eight-sample galleries for the two paper datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codiffuse_dataset_gallery_mpl")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "codiffuse_Schwarz/workshop/submission/Two_case/figures"
DARCY_ROOT = ROOT / "codiffuse_Schwarz_Darcy/data"
AIRFOIL_ROOT = Path(
    "/orcd/scratch/orcd/003/zituo/AMR_Diffusion_Foundation/codiffuse_airfoil"
)
LAYOUT = ROOT / "codiffuse_Schwarz_Airfoil/layouts/pitch_rate_ood_v1"


def _finish(figure: plt.Figure, stem: str, metadata: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    (OUTPUT / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plt.close(figure)


def darcy_gallery() -> None:
    specifications = (
        ("3x3 source", DARCY_ROOT / "darcy_full_domain_face_96_v2.h5", (62, 187)),
        ("4x4 target", DARCY_ROOT / "darcy_domain_extension_fixed_dx_4x4_v1.h5", (16, 48)),
        ("8x8 target", DARCY_ROOT / "darcy_domain_extension_fixed_dx_8x8_v1.h5", (16, 48)),
        ("16x16 target", DARCY_ROOT / "darcy_domain_extension_fixed_dx_16x16_v1.h5", (16, 48)),
    )
    records: list[dict] = []
    active_values: list[np.ndarray] = []
    for label, path, indices in specifications:
        with h5py.File(path, "r") as handle:
            group = handle["val"]
            for index in indices:
                solution = np.asarray(group["solution"][index, 0], dtype=np.float32)
                mask = np.asarray(group["mask"][index, 0], dtype=np.float32) > 0.5
                active_values.append(solution[mask])
                records.append(
                    {
                        "cover": label,
                        "file": path.name,
                        "split": "val",
                        "index": int(index),
                        "seed": int(group["seed"][index]),
                        "family": group["family"][index].decode("utf-8"),
                        "solution": solution,
                        "mask": mask,
                    }
                )
    values = np.concatenate(active_values)
    vmin, vmax = np.quantile(values, (0.005, 0.995))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("white")
    figure, axes = plt.subplots(2, 4, figsize=(5.40, 2.76))
    figure.subplots_adjust(left=0.025, right=0.91, top=0.91, bottom=0.05,
                           wspace=0.08, hspace=0.18)
    image = None
    ordered = [records[i] for i in (0, 2, 4, 6, 1, 3, 5, 7)]
    for panel, (axis, record) in enumerate(zip(axes.flat, ordered)):
        field = np.ma.masked_where(~record["mask"], record["solution"])
        image = axis.imshow(
            field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="bilinear", rasterized=True,
        )
        axis.contour(record["mask"].astype(float), levels=(0.5,), colors="#202020",
                     linewidths=0.35, origin="lower")
        axis.set_title(f"({chr(97 + panel)}) {record['cover']}", fontsize=6.4, pad=2)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#777777")
    color_axis = figure.add_axes((0.925, 0.18, 0.014, 0.64))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("pressure $u$", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    metadata = {
        "contract": "darcy_dataset_gallery_v1",
        "selection": "fixed validation indices at approximately 25% and 75% of each stored split",
        "solution_or_prediction_used_for_selection": False,
        "display_quantiles": [0.005, 0.995],
        "display_limits": [float(vmin), float(vmax)],
        "samples": [
            {key: value for key, value in record.items() if key not in {"solution", "mask"}}
            for record in ordered
        ],
    }
    _finish(figure, "darcy_dataset_eight_samples", metadata)


def _airfoil_candidates(manifest: str, regime: str) -> list[dict]:
    names = json.loads((LAYOUT / manifest).read_text(encoding="utf-8"))["train"]
    rows = []
    for name in names:
        path = AIRFOIL_ROOT / name
        with h5py.File(path, "r") as handle:
            physical = json.loads(handle.attrs["source_manifest_json"])["physical_parameters"]
            pitch_rate = float(np.max(np.abs(handle["motion"][:, 1])))
        rows.append(
            {
                "regime": regime,
                "name": name,
                "path": path,
                "pitch_rate": pitch_rate,
                "reynolds": float(physical["reynolds"]),
                "thickness": float(physical["thickness"]),
            }
        )
    return sorted(rows, key=lambda row: (row["pitch_rate"], row["name"]))


def airfoil_gallery() -> None:
    quantiles = (0.10, 0.35, 0.65, 0.90)
    selected: list[dict] = []
    for manifest, regime in (("source_split.json", "ID"), ("target_split.json", "OOD")):
        candidates = _airfoil_candidates(manifest, regime)
        for quantile in quantiles:
            index = int(round(quantile * (len(candidates) - 1)))
            record = dict(candidates[index])
            record["selection_quantile"] = quantile
            selected.append(record)
    target_phase = 2.5 * np.pi
    vortices: list[np.ndarray] = []
    for record in selected:
        with h5py.File(record["path"], "r") as handle:
            phase = np.asarray(handle["motion"][:, 3])
            frame = int(np.argmin(np.abs(phase - target_phase)))
            state = np.asarray(handle["state"][frame], dtype=np.float32)
            sdf = np.asarray(handle["sdf"][frame, 0], dtype=np.float32)
            dx = float(handle.attrs["dx"])
            window = np.asarray(handle.attrs["window"], dtype=float)
        vorticity = np.gradient(state[1], dx, axis=-1) - np.gradient(state[0], dx, axis=-2)
        vorticity = vorticity.astype(np.float32)
        vorticity[sdf < 0.0] = np.nan
        vorticity[:3] = np.nan
        vorticity[-3:] = np.nan
        vorticity[:, :3] = np.nan
        vorticity[:, -3:] = np.nan
        record.update(
            {
                "frame": frame,
                "phase": float(phase[frame]),
                "time": None,
                "sdf": sdf,
                "window": window,
                "vorticity": vorticity,
            }
        )
        vortices.append(vorticity)
    scale = max(float(np.nanpercentile(np.abs(value), 99.5)) for value in vortices)
    figure, axes = plt.subplots(2, 4, figsize=(5.40, 1.80))
    figure.subplots_adjust(left=0.025, right=0.91, top=0.90, bottom=0.06,
                           wspace=0.08, hspace=0.28)
    image = None
    for panel, (axis, record) in enumerate(zip(axes.flat, selected)):
        xmin, xmax, ymin, ymax = record["window"]
        image = axis.imshow(
            record["vorticity"], origin="lower", extent=(xmin, xmax, ymin, ymax),
            aspect="equal", cmap="RdBu_r", vmin=-scale, vmax=scale,
            interpolation="bilinear", rasterized=True,
        )
        axis.contour(
            np.linspace(xmin, xmax, record["sdf"].shape[1]),
            np.linspace(ymin, ymax, record["sdf"].shape[0]),
            record["sdf"], levels=(0.0,), colors="black", linewidths=0.7,
        )
        axis.set_title(
            f"({chr(97 + panel)}) {record['regime']}, $q_{{pitch}}$={record['pitch_rate']:.3f}",
            fontsize=6.2, pad=2,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#777777")
    color_axis = figure.add_axes((0.925, 0.14, 0.014, 0.72))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("vorticity $\\omega$", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    metadata = {
        "contract": "airfoil_dataset_gallery_v1",
        "selection": "four training trajectories per regime at fixed pitch-rate quantiles",
        "selection_quantiles": list(quantiles),
        "field_or_appearance_used_for_selection": False,
        "snapshot_phase_target": target_phase,
        "display_absolute_percentile": 99.5,
        "display_limit": scale,
        "samples": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in record.items()
                if key not in {"sdf", "window", "vorticity", "path", "time"}
            }
            for record in selected
        ],
    }
    _finish(figure, "airfoil_dataset_eight_samples", metadata)


if __name__ == "__main__":
    darcy_gallery()
    airfoil_gallery()
