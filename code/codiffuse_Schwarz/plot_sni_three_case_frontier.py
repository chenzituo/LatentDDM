"""Plot the matched three-case and Airfoil-rollout Flux-SNI frontiers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "codiffuse_Schwarz" / "figures" / "sni_three_case_frontier_v1"


def load_json(relative: str):
    path = ROOT / relative
    with path.open(encoding="utf-8") as stream:
        return json.load(stream), path


def darcy_records():
    sni, sni_path = load_json(
        "codiffuse_Schwarz_Darcy/outputs/sni_face_local_seed20260806/"
        "validation_huang_extended/summary.json"
    )
    latent, latent_path = load_json(
        "codiffuse_Schwarz_Darcy/outputs/dynamic_relative_green_frozen_fair_v1_seed20260806/"
        "postfreeze_validation_metrics.json"
    )
    timing, timing_path = load_json(
        "codiffuse_Schwarz_Darcy/outputs/dynamic_relative_green_frozen_fair_v1_seed20260806/"
        "h200_timing.json"
    )
    visual, visual_path = load_json(
        "codiffuse_Schwarz_Darcy/outputs/frozen_latentddm_visual_v1/"
        "darcy_frozen_latentddm_median_case.json"
    )
    true_local_error = sum(
        row["local"] for row in visual["validation_cases"]
    ) / len(visual["validation_cases"])
    global_audit_path = (
        ROOT / "codiffuse_Schwarz/figures/global_operator_audit_v1/"
        "low_data_transolver_car_darcy.json"
    )
    sweeps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    curve = []
    for sweep in sweeps:
        row = sni["summary"][f"tau_0.1000/checkpoint_{sweep:04d}"]
        curve.append({
            "sweep": sweep,
            "seconds": row["wall_seconds"],
            "error": row["relative_l2"],
            "local_nfe": row["local_nfe"],
        })
    return {
        "case": "Darcy",
        "metric": "field relative $L^2$",
        "sni": curve,
        "local": {
            "seconds": timing["endpoint"]["mean_seconds_per_case"],
            "error": true_local_error,
        },
        "latentddm": {
            "seconds": timing["full_one_pass"]["mean_seconds_per_case"],
            "error": latent["candidate_ordinary"]["relative_l2"],
        },
        "transolver": {
            "seconds": 0.007787,
            "error": 0.132792687,
            "protocol": (
                "three-seed first-60 validation diagnostic with selection overlap; "
                "same PDE family, not a paired per-case frontier"
            ),
        },
        "sources": [str(sni_path), str(latent_path), str(timing_path),
                    str(visual_path),
                    str(global_audit_path)],
    }


def car_records():
    sni, sni_path = load_json(
        "codiffuse_Schwarz_Car/outputs/"
        "car_flux_conditioned_sni_validation_v2_frozen_u_25designs.json"
    )
    # Use the single-process, same-design H200 benchmark whenever available.
    # The v2 artifact times every point on the displayed 1/2/4/8-sweep curve;
    # v1 is a checked fallback for the two tabulated endpoints while v2 runs.
    timing_path = next(
        path for path in (
            ROOT / "codiffuse_Schwarz_Car/outputs/"
            "matched_volume_methods_timing_v2.json",
            ROOT / "codiffuse_Schwarz_Car/outputs/"
            "matched_volume_methods_timing_v1.json",
        ) if path.exists()
    )
    with timing_path.open(encoding="utf-8") as stream:
        timing = json.load(stream)
    matched = timing["summary"]
    sweeps = [1, 2, 4, 8]
    curve = []
    for sweep in sweeps:
        row = sni["summary"][f"tau_0.250/sweep_{sweep:03d}"]
        curve.append({
            "sweep": sweep,
            "seconds": matched.get(
                f"flux_sni_{sweep}", {"mean_seconds": row["seconds_mean"]}
            )["mean_seconds"],
            "error": row["native_speed_relative_l2_mean"],
            "local_nfe": 420 * (sweep + 1),
        })
    # These one-pass values are the complete-field H200 audit in
    # CAR_FINAL_VALIDATION_AUDIT.md; the native error is also recorded in the
    # matched 25-design volume comparison JSON.
    audit_path = ROOT / "codiffuse_Schwarz_Car/CAR_FINAL_VALIDATION_AUDIT.md"
    comparison_path = ROOT / (
        "codiffuse_Schwarz_Car/outputs/car_volume_global_validation_comparison_v1.json"
    )
    transolver_path = ROOT / (
        "codiffuse_Schwarz_Car/outputs/global_point_v1/validation_metrics_v1.json"
    )
    return {
        "case": "DrivAerNet++",
        "metric": "native $|U|$ relative $L^2$",
        "sni": curve,
        "local": {
            "seconds": matched["local_pou"]["mean_seconds"],
            "error": 0.20292723834514617,
        },
        "latentddm": {
            "seconds": matched["latentddm"]["mean_seconds"],
            "error": 0.2028171879,
            "protocol": (
                "promoted state-gated checkpoint; direct same-process complete-"
                "field timing on the identical five validation designs"
            ),
        },
        "transolver": {
            "seconds": 0.0602325843,
            "error": 0.3322483718,
            "protocol": (
                "same 25 cars and 131072 native queries; latency is direct-query "
                "time and is not comparable to complete Cartesian-field timing"
            ),
        },
        "sources": [str(sni_path), str(timing_path), str(audit_path),
                    str(comparison_path),
                    str(transolver_path)],
    }


def airfoil_records():
    summary, summary_path = load_json(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "airfoil_flux_conditioned_sni_validation_v4_centered_40states_v2.json"
    )
    transolver_path = ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/global100_v3_finest/head_to_head/summary.json"
    )
    sweeps = [1, 2, 4, 8, 16]
    curve = []
    for sweep in sweeps:
        row = summary["summary"][f"init_local/tau_0.500/sweep_{sweep:03d}"]
        curve.append({
            "sweep": sweep,
            "seconds": row["seconds_mean"],
            "error": row["field_relative_l2_mean"],
            "local_nfe": 325 * (sweep + 1),
        })
    local = summary["local_pou_baseline"]
    latent = summary["latentddm_baseline"]
    return {
        "case": "Airfoil",
        "metric": "one-step field relative $L^2$",
        "sni": curve,
        "local": {
            "seconds": local["seconds_mean"],
            "error": local["field_relative_l2_mean"],
        },
        "latentddm": {
            "seconds": latent["seconds_mean"],
            "error": latent["field_relative_l2_mean"],
        },
        "transolver": {
            "seconds": 0.1594475145,
            "error": 0.0098134631,
            "protocol": (
                "separate common-protocol 20-step test campaign; displayed for "
                "global-operator context, not as a paired point on the SNI frontier"
            ),
        },
        "sources": [str(summary_path), str(transolver_path)],
    }


def airfoil_rollout_records():
    rollout, rollout_path = load_json(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "airfoil_flux_conditioned_sni_rollout_h20_v1.json"
    )
    aggregate = rollout["aggregate"]
    transolver_path = ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/global100_v3_finest/"
        "head_to_head/summary.json"
    )
    return {
        "case": "Airfoil (20-step)",
        "metric": "mean rollout field relative $L^2$",
        "sni": [
            {
                "sweep": sweep,
                "seconds": aggregate[f"sni_{sweep}"]["seconds_per_step"],
                "error": aggregate[f"sni_{sweep}"]["all_step_field_relative_l2"],
                "local_nfe": aggregate[f"sni_{sweep}"]["local_nfe_per_step"],
            }
            for sweep in (1, 16)
        ],
        "local": {
            "seconds": aggregate["local"]["seconds_per_step"],
            "error": aggregate["local"]["all_step_field_relative_l2"],
        },
        "latentddm": {
            "seconds": aggregate["latentddm"]["seconds_per_step"],
            "error": aggregate["latentddm"]["all_step_field_relative_l2"],
        },
        "transolver": {
            "seconds": 0.1594475145,
            "error": 0.12531,
            "protocol": "separate common-protocol 20-step campaign",
        },
        "sources": [str(rollout_path), str(transolver_path)],
    }


def plot(records):
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 4, figsize=(7.15, 2.25))
    figure.subplots_adjust(left=.075, right=.995, bottom=.23, top=.75,
                           wspace=.82)
    for axis, case in zip(axes, records):
        times = [row["seconds"] * 1e3 for row in case["sni"]]
        errors = [row["error"] for row in case["sni"]]
        axis.plot(times, errors, "o-", color="#4C78A8", lw=1.35, ms=3.5,
                  label="Flux-SNI sweeps")
        for row, x, y in zip(case["sni"], times, errors):
            if row["sweep"] in (1, 4, 8, 16, 64, 128, 512):
                axis.annotate(str(row["sweep"]), (x, y), xytext=(2, 3),
                              textcoords="offset points", fontsize=6,
                              color="#315B7D")
        axis.scatter(case["local"]["seconds"] * 1e3, case["local"]["error"],
                     marker="s", s=28, color="#9C9C9C", edgecolor="white",
                     linewidth=.4, zorder=4, label="Local POU")
        axis.scatter(case["latentddm"]["seconds"] * 1e3,
                     case["latentddm"]["error"], marker="*", s=80,
                     color="#E45756", edgecolor="white", linewidth=.4,
                     zorder=5, label="LatentDDM")
        axis.scatter(case["transolver"]["seconds"] * 1e3,
                     case["transolver"]["error"], marker="D", s=38,
                     facecolor="white", edgecolor="#2CA02C", linewidth=1.4,
                     zorder=5, label="Transolver audit")
        axis.set_xscale("log")
        if case["case"] == "Darcy":
            axis.set_xticks((1, 10, 100, 1000),
                            ("1", "10", "$10^2$", "$10^3$"))
        elif case["case"] == "DrivAerNet++":
            axis.set_xticks((100, 1000), ("100", "$10^3$"))
        else:
            axis.set_xticks((100, 200, 400), ("100", "200", "400"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="both", alpha=.20, lw=.5)
        axis.set_title(case["case"])
        axis.set_ylabel(case["metric"])
        axis.spines[["top", "right"]].set_visible(False)
        if case["case"] in ("DrivAerNet++", "Airfoil"):
            inset = axis.inset_axes([.43, .49, .54, .44])
            # Retain the figure-wide millisecond unit inside the zoom.
            inset_times = times
            inset.plot(inset_times, errors, "o-", color="#4C78A8", lw=1.0, ms=2.4)
            inset.scatter(case["local"]["seconds"] * 1e3,
                          case["local"]["error"], marker="s", s=18,
                          color="#9C9C9C", edgecolor="white", linewidth=.3,
                          zorder=4)
            inset.scatter(case["latentddm"]["seconds"] * 1e3,
                          case["latentddm"]["error"], marker="*", s=44,
                          color="#E45756", edgecolor="white", linewidth=.3,
                          zorder=5)
            if case["case"] == "DrivAerNet++":
                # Keep the promoted state-gated point visible and make the
                # small but paired-resolved Local--Latent separation legible.
                inset.set_ylim(.20278, .20308)
                inset.set_xlim(650, 5000)
                inset.set_xticks((1000, 3000, 5000), ("1k", "3k", "5k"))
            else:
                inset.set_ylim(.01482, .01491)
                inset.set_xlim(100, 460)
                inset.set_xticks((100, 250, 400))
            inset.tick_params(labelsize=5.2, pad=1, length=2)
            inset.ticklabel_format(axis="y", style="plain", useOffset=False)
            inset.grid(True, which="both", alpha=.16, lw=.4)
            inset.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper center", ncol=4,
                  bbox_to_anchor=(0.5, .98), columnspacing=1.1,
                  handletextpad=0.45)
    figure.supxlabel("end-to-end latency (ms, log)", fontsize=8, y=.04)
    return figure


def comparison(case):
    latent_error = case["latentddm"]["error"]
    latent_seconds = case["latentddm"]["seconds"]
    best = min(case["sni"], key=lambda row: row["error"])
    matching = [
        row for row in case["sni"] if row["error"] <= 1.01 * latent_error
    ]
    earliest = min(matching, key=lambda row: row["sweep"]) if matching else None
    return {
        "latentddm_error": latent_error,
        "latentddm_seconds": latent_seconds,
        "best_reported_sni": best,
        "earliest_sni_within_one_percent": earliest,
        "speedup_over_earliest_matching_sni": (
            earliest["seconds"] / latent_seconds if earliest else None
        ),
        "deepest_sni_to_latentddm_speed_ratio": (
            max(case["sni"], key=lambda row: row["sweep"])["seconds"]
            / latent_seconds
        ),
    }


def main():
    records = [darcy_records(), car_records(), airfoil_records(),
               airfoil_rollout_records()]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = plot(records)
    for suffix in ("pdf", "png"):
        figure.savefig(OUTPUT / f"sni_three_case_frontier.{suffix}", dpi=300,
                       bbox_inches="tight")
    plt.close(figure)
    with (OUTPUT / "sni_three_case_frontier.json").open("w", encoding="utf-8") as stream:
        json.dump({"contract": "matched_three_case_flux_sni_frontier_v2",
                   "records": records,
                   "comparisons": {
                       case["case"]: comparison(case) for case in records
                   }}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
