"""Build the cross-case audit explaining when learned SNI contracts.

The audit separates three questions: whether the frozen local response family
contains a better answer, whether target-free LatentDDM selects a better
response, and whether one deployable SNI sweep descends from Local/Core.  The
capacity oracle uses a held-out target only after model selection; neither
LatentDDM nor SNI uses that target at inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "codiffuse_Schwarz/figures/unified_report/sni_contractivity_audit.json"
ROWS = ROOT / "codiffuse_Schwarz/workshop/submission/figures/sni_contractivity_rows.tex"


def load(relative: str) -> tuple[dict[str, Any], Path]:
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")), path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_audit() -> dict[str, Any]:
    darcy_chart, darcy_chart_path = load(
        "codiffuse_Schwarz_Darcy/outputs/input_correction_boundary_chart_capacity_v1.json"
    )
    darcy_sni, darcy_sni_path = load(
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "matched_normalized_overlap_sni_128.json"
    )
    car_oracle, car_oracle_path = load(
        "codiffuse_Schwarz_Car/outputs/"
        "input_correction_face4_order3_depth4_pressurectx_parentgate_seed20260811/"
        "validation_response_oracle25_steps8_matched_globals_v2.json"
    )
    car_sni, car_sni_path = load(
        "codiffuse_Schwarz_Car/outputs/"
        "input_correction_response_selected_v1/"
        "matched_validation_sni_v1.json"
    )
    air_oracle, air_oracle_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "state_input_selected_depth4_resp_w0005_fw2_seed20260812/"
        "validation_oracle_gap40_v1.json"
    )
    air_sni, air_sni_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "final_holdout_h20_five_starts_v3.json"
    )

    def row(
        case: str, *, oracle: float, oracle_local: float,
        sni: dict[str, Any], sni_metric: str,
    ) -> dict[str, float | str]:
        sweep_zero = float(sni["summary"]["init_sni_0"][sni_metric])
        sweep_one = float(sni["summary"]["init_sni_1"][sni_metric])
        latent = float(sni["summary"]["latentddm"][sni_metric])
        return {
            "case": case,
            "oracle_response_error_over_local": oracle / oracle_local,
            "latentddm_error_over_local": latent / sweep_zero,
            "one_sni_sweep_error_over_sweep_zero": sweep_one / sweep_zero,
        }

    darcy_capacity = darcy_chart["field_relative_l2"]
    car_oracle_mean = float(car_oracle["comparisons"]["Point"]["oracle_mean"])
    car_local_mean = float(car_sni["summary"]["local_core"]["velocity_relative_l2_mean"])
    air_capacity = air_oracle["mean_relative_l2"]
    records = [
        row(
            "Darcy",
            oracle=float(darcy_capacity["exact_trace_oracle"]["mean"]),
            oracle_local=float(darcy_capacity["local_core"]["mean"]),
            sni=darcy_sni, sni_metric="relative_l2_mean",
        ),
        row(
            "Car", oracle=car_oracle_mean, oracle_local=car_local_mean,
            sni=car_sni, sni_metric="velocity_relative_l2_mean",
        ),
        row(
            "Airfoil-20",
            oracle=float(air_capacity["oracle"]["field_relative_l2"]),
            oracle_local=float(air_capacity["parent"]["field_relative_l2"]),
            sni=air_sni, sni_metric="all_step_field_mean",
        ),
    ]
    sources = (
        darcy_chart_path, darcy_sni_path, car_oracle_path,
        car_sni_path, air_oracle_path, air_sni_path,
    )
    return {
        "contract": "three_case_sni_contractivity_audit_v3_frozen_final",
        "interpretation": {
            "oracle_response_error_over_local": (
                "target-trace oracle diagnostic; lower means the frozen local "
                "response family can use the correct boundary state"
            ),
            "latentddm_error_over_local": (
                "target-free LatentDDM field error divided by its exact "
                "Local/Core error on the locked reporting split"
            ),
            "one_sni_sweep_error_over_sweep_zero": (
                "deployable SNI field error after one sweep divided by the exact "
                "Local/Core sweep-zero error; below one is descent"
            ),
        },
        "records": records,
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sources
        ],
    }


def main() -> None:
    audit = make_audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = []
    for record in audit["records"]:
        lines.append(
            f"{record['case']} & "
            f"{record['oracle_response_error_over_local']:.3f} & "
            f"{record['latentddm_error_over_local']:.3f} & "
            f"{record['one_sni_sweep_error_over_sweep_zero']:.3f} \\\\"
        )
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    ROWS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
