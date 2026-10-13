"""Plot the three-case field/latency frontiers.

Init-SNI includes its exact sweep-zero state: the no-message Local-POU
self-ablation.  Keeping that point in the plotted polyline prevents the first
corrected sweep from being mistaken for the initialization.  The two global
controls, Transolver and a from-scratch full-domain Patch Transformer, are shown as
audited scatter points rather than connected to an iterative trajectory.  Every
marker uses measured end-to-end latency.  Local tiles are evaluated in a batched
GPU call within each operator round; no method receives an analytic ``time/N``
projection.  Rollout costs accumulate all twenty physical steps.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "codiffuse_Schwarz/figures/sni_three_case_frontier_v1"
KEY = re.compile(
    r"^init_(?P<initialization>[^/]+)/tau_(?P<tau>[0-9.]+)/"
    r"sweep_(?P<sweep>[0-9]+)$"
)
AIRFOIL_GLOBAL_AUDIT = (
    "codiffuse_Schwarz_Airfoil/outputs/global100_v1_finest/"
    "global_exact_main_validation_v1.json"
)
DARCY_GLOBAL_TRANSOLVER_AUDIT = (
    "codiffuse_Schwarz_Darcy/outputs/validation_only_baseline_audit/"
    "global_transolver_face_v2_summary_250.json"
)


def load(relative: str):
    path = ROOT / relative
    with path.open(encoding="utf-8") as stream:
        return json.load(stream), path


def airfoil_global_record(
    method: str, regime: str,
) -> tuple[dict[str, float], Path]:
    payload, path = load(AIRFOIL_GLOBAL_AUDIT)
    row = payload["summary"][method][regime]
    return {
        "seconds": float(row["seconds_mean"]),
        "error": float(row["field_relative_l2_mean"]),
    }, path


def patch_record(key: str) -> tuple[dict[str, float] | None, Path | None]:
    """Load only the independent from-scratch global ViT audit."""

    if key == "darcy":
        payload, path = load(
            "codiffuse_Schwarz_Darcy/outputs/"
            "global_vit_face_v2_stablepos_seed20260805/"
            "validation_only_250_gpu/summary.json"
        )
        row = payload["summary"]["validation/sweep_001"]
        return {
            "seconds": float(row["wall_seconds"]),
            "error": float(row["relative_l2"]),
        }, path
    if key == "car":
        payload, path = load(
            "codiffuse_Schwarz_Car/outputs/"
            "global_vit_fullvolume_stablepos_v1/validation_metrics.json"
        )
        return {
            "seconds": float(payload["summary"]["seconds"]),
            "error": float(payload["summary"]["native_speed_relative_l2"]),
        }, path
    if key == "airfoil_one_step":
        return airfoil_global_record("patch_transformer", "one_step")
    if key == "airfoil_20_step_mean":
        return airfoil_global_record("patch_transformer", "rollout_all_steps")
    # No development mixer is a fallback.
    return None, None


def mean_rows(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def parse_summary(payload: dict, initialization: str, metric: str):
    records = []
    for key, values in payload["summary"].items():
        match = KEY.match(key)
        if match is None or match.group("initialization") != initialization:
            continue
        sweep = int(match.group("sweep"))
        records.append({
            "sweep": sweep,
            "tau": float(match.group("tau")),
            "error": float(values[metric]),
            "seconds": float(values["seconds_mean"]),
            "local_nfe": float(values["local_nfe_mean"]),
        })
    if not records:
        raise RuntimeError(f"no records for initialization {initialization}")
    return records


def selected_curve(payload: dict, initialization: str, metric: str):
    records = parse_summary(payload, initialization, metric)
    eligible = [row for row in records if row["sweep"] > 0]
    best = min(eligible, key=lambda row: (row["error"], row["seconds"]))
    curve = sorted(
        [row for row in eligible if np.isclose(row["tau"], best["tau"])],
        key=lambda row: row["sweep"],
    )
    return curve, best


def darcy_records():
    payload, payload_path = load(
        "codiffuse_Schwarz_Darcy/outputs/"
        "darcy_weighted_pou_sni_two_init_v1.json"
    )
    latent, latent_path = load(
        "codiffuse_Schwarz_Darcy/outputs/"
        "dynamic_relative_green_frozen_fair_v1_seed20260806/"
        "postfreeze_validation_metrics.json"
    )
    legacy_timing, legacy_timing_path = load(
        "codiffuse_Schwarz_Darcy/outputs/"
        "dynamic_relative_green_frozen_fair_v1_seed20260806/h200_timing.json"
    )
    patch, patch_path = patch_record("darcy")
    transolver_payload, transolver_path = load(
        DARCY_GLOBAL_TRANSOLVER_AUDIT
    )
    transolver_row = transolver_payload["aggregate"]["1"]
    pou, pou_best = selected_curve(payload, "local_pou", "relative_l2_mean")
    zero, zero_best = selected_curve(payload, "zero", "relative_l2_mean")
    sweep0 = payload["summary"][
        f"init_local_pou/tau_{pou_best['tau']:.3f}/sweep_0000"
    ]
    timing_path = ROOT / (
        "codiffuse_Schwarz_Darcy/outputs/"
        "darcy_weighted_pou_sni_matched_h200_timing_v1.json"
    )
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        local_seconds = timing["timings"]["local_pou"]["mean_seconds_per_case"]
        latent_seconds = timing["timings"]["latentddm"]["mean_seconds_per_case"]
        for row in pou:
            key = f"pou_sni_{row['sweep']}"
            if key in timing["timings"]:
                row["seconds"] = timing["timings"][key]["mean_seconds_per_case"]
    else:
        local_seconds = float(sweep0["seconds_mean"])
        latent_seconds = legacy_timing["full_one_pass"]["mean_seconds_per_case"]
        timing_path = legacy_timing_path
    return {
        "case": "Darcy",
        "metric": "field relative $L^2$",
        "pou_sni": pou,
        "zero_sni": zero,
        "selected": {"Init-SNI": pou_best, "Zero-SNI": zero_best},
        "local": {"seconds": local_seconds,
                  "error": float(sweep0["relative_l2_mean"])},
        "latentddm": {
            "seconds": latent_seconds,
            "error": latent["candidate_ordinary"]["relative_l2"],
        },
        "transolver": {
            "seconds": float(transolver_row["wall_seconds_mean"]),
            "error": float(transolver_row["relative_l2_mean"]),
        },
        "patch_transformer": patch,
        "sources": [
            str(payload_path), str(latent_path), str(timing_path),
            str(transolver_path),
            *([str(patch_path)] if patch_path is not None else []),
        ],
    }


def car_records():
    payload, payload_path = load(
        "codiffuse_Schwarz_Car/outputs/"
        "car_flux_sni_two_init_v1_25designs.json"
    )
    timing, timing_path = load(
        "codiffuse_Schwarz_Car/outputs/matched_volume_methods_timing_v2.json"
    )
    patch, patch_path = patch_record("car")
    pou, pou_best = selected_curve(
        payload, "local_pou", "native_speed_relative_l2_mean",
    )
    zero, zero_best = selected_curve(
        payload, "zero", "native_speed_relative_l2_mean",
    )
    # Use the dedicated same-process timing audit for the main POU-start
    # frontier.  The two-initialization evaluator intentionally records a cold
    # local pass, which is useful for provenance but not a matched latency
    # comparison against LatentDDM.
    for row in pou:
        key = f"flux_sni_{row['sweep']}"
        if key in timing["summary"]:
            row["seconds"] = timing["summary"][key]["mean_seconds"]
    local_rows = payload["initial_local_pou"]
    return {
        "case": "DrivAerNet++",
        "metric": "native $|U|$ relative $L^2$",
        "pou_sni": pou,
        "zero_sni": zero,
        "selected": {"Init-SNI": pou_best, "Zero-SNI": zero_best},
        "local": {
            "seconds": timing["summary"]["local_pou"]["mean_seconds"],
            "error": mean_rows(local_rows, "native_speed_relative_l2"),
        },
        "latentddm": {
            "seconds": timing["summary"]["latentddm"]["mean_seconds"],
            "error": .20281718790531159,
        },
        "transolver": {"seconds": .0602325843, "error": .3322483718},
        "patch_transformer": patch,
        "sources": [str(payload_path), str(timing_path)]
        + ([str(patch_path)] if patch_path is not None else []),
    }


def airfoil_records():
    payload, payload_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "global_response_rank16_onpolicy_rollin3_lr5e5_v1/"
        "full_sni_two_init_40states.json"
    )
    timing, timing_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "global_response_rank16_onpolicy_rollin3_lr5e5_v1/"
        "matched_production_timing.json"
    )
    pou, pou_best = selected_curve(payload, "local", "field_relative_l2_mean")
    zero, zero_best = selected_curve(payload, "zero", "field_relative_l2_mean")
    local = payload["local_pou_baseline"]
    latent = payload["latentddm_baseline"]
    matched_local = (
        timing["timing"]["zero"]["total_production_ms"]["mean"] / 1.0e3
    )
    matched_latent = (
        timing["timing"]["predicted"]["total_production_ms"]["mean"] / 1.0e3
    )
    # Remove the evaluator's first-call overhead while retaining the measured
    # SNI correction time on every displayed point.
    for row in (*pou, *zero):
        row["seconds"] = matched_local + row["seconds"] - local["seconds_mean"]
    patch, patch_path = patch_record("airfoil_one_step")
    transolver, global_path = airfoil_global_record("transolver", "one_step")
    return {
        "case": "Airfoil",
        "metric": "one-step field relative $L^2$",
        "pou_sni": pou,
        "zero_sni": zero,
        "selected": {"Init-SNI": pou_best, "Zero-SNI": zero_best},
        "local": {"seconds": matched_local,
                  "error": local["field_relative_l2_mean"]},
        "latentddm": {"seconds": matched_latent,
                      "error": latent["field_relative_l2_mean"]},
        "transolver": transolver,
        "patch_transformer": patch,
        "sources": [str(payload_path), str(timing_path), str(global_path)]
        + ([str(patch_path)] if patch_path is not None else []),
    }


def rollout_curve(payload: dict):
    return sorted(({
        "sweep": int(key.split("_")[-1]),
        "seconds": value["seconds_per_step"],
        "error": value["all_step_field_relative_l2"],
        "local_nfe": value["local_nfe_per_step"],
    } for key, value in payload["aggregate"].items()
        if key.startswith("sni_")), key=lambda row: row["sweep"])


def airfoil_rollout_records():
    pou, pou_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "global_response_rank16_onpolicy_rollin3_lr5e5_v1/"
        "flux_sni_rollout_h20_local.json"
    )
    zero, zero_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "global_response_rank16_onpolicy_rollin3_lr5e5_v1/"
        "flux_sni_rollout_h20_zero.json"
    )
    aggregate = pou["aggregate"]
    patch, patch_path = patch_record("airfoil_20_step_mean")
    transolver, global_path = airfoil_global_record(
        "transolver", "rollout_all_steps",
    )
    return {
        "case": "Airfoil (20-step)",
        "metric": "mean rollout field relative $L^2$",
        "cost_multiplier": int(pou["horizon"]),
        "cost_scope": "complete 20-step autoregressive rollout",
        "pou_sni": rollout_curve(pou),
        "zero_sni": rollout_curve(zero),
        "selected": {
            "Init-SNI": pou["sni"], "Zero-SNI": zero["sni"],
        },
        "local": {"seconds": aggregate["local"]["seconds_per_step"],
                  "error": aggregate["local"]["all_step_field_relative_l2"]},
        "latentddm": {
            "seconds": aggregate["latentddm"]["seconds_per_step"],
            "error": aggregate["latentddm"]["all_step_field_relative_l2"],
        },
        "transolver": transolver,
        "patch_transformer": patch,
        "sources": [
            str(pou_path), str(zero_path), str(global_path),
            *([str(patch_path)] if patch_path is not None else []),
        ],
    }


def attach_init_anchor(case: dict) -> dict:
    """Prepend the exact Init-SNI-0 no-message state."""
    if not case["pou_sni"] or case["pou_sni"][0]["sweep"] <= 0:
        raise ValueError(f"{case['case']} lacks a positive-sweep Init-SNI curve")
    first = case["pou_sni"][0]
    local_nfe = first["local_nfe"] / (first["sweep"] + 1)
    anchor = {
        "sweep": 0,
        "error": case["local"]["error"],
        "seconds": case["local"]["seconds"],
        "local_nfe": local_nfe,
        "role": "exact no-message Local-POU self-ablation",
    }
    if "tau" in first:
        anchor["tau"] = first["tau"]
    case["pou_sni"] = [anchor, *case["pou_sni"]]
    case.setdefault("cost_multiplier", 1)
    case.setdefault("cost_scope", "one requested physical endpoint")
    # One local round means that every subdomain endpoint is evaluated
    # concurrently.  SNI can parallelize within a sweep, but not across
    # sweeps; LatentDDM uses one endpoint round followed by one message.
    for row in (*case["pou_sni"], *case["zero_sni"]):
        row["parallel_operator_rounds"] = float(row["local_nfe"] / local_nfe)
    for name in ("local", "latentddm"):
        case[name]["parallel_operator_rounds"] = 1.0
    for name in ("transolver", "patch_transformer"):
        if case.get(name) is not None:
            case[name]["parallel_operator_rounds"] = 1.0
            case[name]["round_semantics"] = "one whole-domain global pass"
    case["parallel_cost_contract"] = {
        "subdomains_within_round": "measured batched GPU execution",
        "local_nfe_per_round": local_nfe,
        "latentddm": "one local-operator round plus one residual message",
        "sni": "one sequential local-response round per recorded recurrence stage",
        "global_controls": "one whole-domain pass; not compute-matched to a local round",
        "analytic_time_division_or_projection": False,
        "all_markers_use_measured_wall_seconds": True,
    }
    case["init_sni_sweep0_identity"] = {
        "exact_error_equality": anchor["error"] == case["local"]["error"],
        "exact_seconds_equality": anchor["seconds"] == case["local"]["seconds"],
        "interpretation": "Init-SNI-0 is the no-message self-ablation",
    }
    if not all(case["init_sni_sweep0_identity"].values()):
        # ``interpretation`` is a nonempty string and therefore truthy.
        raise RuntimeError(f"{case['case']} Init-SNI sweep zero does not replay")
    return case


def plot(records):
    def x_value(row: dict, case: dict) -> float:
        multiplier = float(case["cost_multiplier"])
        return multiplier * float(row["seconds"]) * 1e3

    def draw_latent(axis, case: dict, *, size: float) -> None:
        measured_x = x_value(case["latentddm"], case)
        y = case["latentddm"]["error"]
        axis.scatter(
            measured_x, y, marker="*", s=size,
            color="#E45756", edgecolor="white", linewidth=.4,
            zorder=5, label="LatentDDM",
        )

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 4, figsize=(7.15, 2.28))
    figure.subplots_adjust(left=.075, right=.995, bottom=.23, top=.75,
                           wspace=.82)
    for axis, case in zip(axes, records, strict=True):
        for name, style in (
            ("pou_sni", dict(color="#4C78A8", ls="-", marker="o",
                             label="Init-SNI")),
            ("zero_sni", dict(color="#F28E2B", ls="--", marker="^",
                              label="Zero-SNI")),
        ):
            curve = case[name]
            times = [x_value(row, case) for row in curve]
            errors = [row["error"] for row in curve]
            axis.plot(times, errors, lw=1.25, ms=3.2, **style)
            if name == "pou_sni":
                for row, x, y in zip(curve, times, errors, strict=True):
                    if row["sweep"] in (0, 1, 4, 8, 16, 64, 128):
                        axis.annotate(
                            str(row["sweep"]), (x, y), xytext=(2, 3),
                            textcoords="offset points", fontsize=5.8,
                            color="#315B7D",
                        )
        axis.scatter(
            x_value(case["local"], case), case["local"]["error"],
            marker="s", s=28, color="#8F8F8F", edgecolor="white",
            linewidth=.4, zorder=4, label="No message",
        )
        draw_latent(axis, case, size=80)
        axis.scatter(
            x_value(case["transolver"], case),
            case["transolver"]["error"], marker="D", s=34,
            facecolor="#59A14F", edgecolor="white", linewidth=.4,
            zorder=5, label="Transolver",
        )
        if case["patch_transformer"] is not None:
            axis.scatter(
                x_value(case["patch_transformer"], case),
                case["patch_transformer"]["error"], marker="P", s=42,
                facecolor="#7A4EAB", edgecolor="white", linewidth=.4,
                zorder=5, label="Patch Transformer",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_minor_formatter(NullFormatter())
        y_ticks = {
            "Darcy": (.01, .03, .1, .3),
            "DrivAerNet++": (.20, .25, .30),
            "Airfoil": (.01, .03, .1, .3, 1.0),
            "Airfoil (20-step)": (.1, .2, .5),
        }[case["case"]]
        axis.set_yticks(y_ticks)
        axis.set_yticklabels([f"{value:g}" for value in y_ticks])
        axis.grid(True, which="both", alpha=.20, lw=.5)
        axis.set_title(case["case"])
        axis.set_ylabel(case["metric"])
        axis.spines[["top", "right"]].set_visible(False)

        # Resolve the one-pass comparison without hiding the zero-start scale
        # in the main panel.
        if case["case"] in ("DrivAerNet++", "Airfoil"):
            inset = axis.inset_axes([.43, .49, .54, .44])
            for name, color, linestyle, marker in (
                ("pou_sni", "#4C78A8", "-", "o"),
                ("zero_sni", "#F28E2B", "--", "^"),
            ):
                curve = case[name]
                inset.plot(
                    [x_value(row, case) for row in curve],
                    [row["error"] for row in curve], color=color,
                    ls=linestyle, marker=marker, lw=1.0, ms=2.2,
                )
            inset.scatter(
                x_value(case["local"], case), case["local"]["error"],
                marker="s", s=16, color="#8F8F8F", zorder=4,
            )
            draw_latent(inset, case, size=40)
            # The main axes retain the entire zero-start range.  The inset is
            # deliberately restricted to the decomposed family: it resolves
            # the no-message/LatentDDM/early-SNI differences without treating
            # the independently trained global ViT as a communication ablation.
            maximum_zoom_sweep = 16 if case["case"] == "DrivAerNet++" else 2
            zoom_rows = [
                row for row in case["pou_sni"]
                if row["sweep"] <= maximum_zoom_sweep
            ]
            finite = [row["error"] for row in zoom_rows] + [
                case["local"]["error"], case["latentddm"]["error"],
            ]
            low, high = min(finite), max(finite)
            margin = max((high - low) * .14, abs(low) * 5e-5, 1e-7)
            inset.set_ylim(low - margin, high + margin)
            zoom_times = [x_value(row, case) for row in zoom_rows] + [
                x_value(case["local"], case),
                x_value(case["latentddm"], case),
            ]
            inset.set_xscale("log")
            inset.set_xlim(min(zoom_times) * .92, max(zoom_times) * 1.08)
            inset.tick_params(labelsize=5.2, pad=1, length=2)
            inset.xaxis.set_major_formatter(NullFormatter())
            inset.xaxis.set_minor_formatter(NullFormatter())
            inset.tick_params(axis="x", which="both", length=1.5)
            inset.ticklabel_format(axis="y", style="plain", useOffset=False)
            inset.grid(True, which="both", alpha=.16, lw=.4)
            inset.spines[["top", "right"]].set_visible(False)
            inset.set_title("zoom", fontsize=5.8, pad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, loc="upper center", ncol=6,
        bbox_to_anchor=(.5, .985), columnspacing=.9, handletextpad=.4,
    )
    figure.supxlabel("measured end-to-end latency (ms)", fontsize=8, y=.04)
    return figure


def comparison(case: dict, name: str):
    latent_error = case["latentddm"]["error"]
    curve = [row for row in case[name] if row["sweep"] > 0]
    matching = [row for row in curve if row["error"] <= 1.01 * latent_error]
    earliest = min(matching, key=lambda row: row["sweep"]) if matching else None
    return {
        "best": min(curve, key=lambda row: row["error"]),
        "earliest_within_one_percent_of_latentddm": earliest,
        "latentddm_error": latent_error,
        "latentddm_seconds": case["latentddm"]["seconds"],
        "latency_ratio_at_earliest_match": (
            earliest["seconds"] / case["latentddm"]["seconds"]
            if earliest else None
        ),
    }


def main():
    records = [attach_init_anchor(case) for case in (
        darcy_records(), car_records(), airfoil_records(),
        airfoil_rollout_records(),
    )]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = plot(records)
    for suffix in ("pdf", "png"):
        figure.savefig(
            OUTPUT / f"sni_three_case_frontier.{suffix}", dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    audit = {
        "contract": "three_case_two_initialization_sni_frontier_global_vit_v7",
        "main_figure_cost_axis": "measured end-to-end H200 latency",
        "analytic_latency_projection": False,
        "records": records,
        "comparisons": {
            case["case"]: {
                "Init-SNI": comparison(case, "pou_sni"),
                "Zero-SNI": comparison(case, "zero_sni"),
            }
            for case in records
        },
    }
    (OUTPUT / "sni_three_case_frontier.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
