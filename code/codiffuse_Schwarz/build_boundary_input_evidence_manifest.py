"""Rebuild the production evidence manifest for the final boundary-input paper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "codiffuse_Schwarz/workshop/submission"
OUTPUT = ROOT / (
    "codiffuse_Schwarz/workshop/submission_audits/"
    "PRODUCTION_EVIDENCE_MANIFEST.json"
)


PATHS = {
    "darcy_matched": ROOT / (
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "matched_normalized_overlap_sni_128.json"
    ),
    "darcy_matched_fine": ROOT / (
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "matched_normalized_overlap_sni_128_fine.json"
    ),
    "darcy_adaptation": ROOT / (
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "aggregate.json"
    ),
    "car_selection": ROOT / (
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "selection.json"
    ),
    "car_matched": ROOT / (
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "matched_validation_sni_v1.json"
    ),
    "car_frontier": ROOT / (
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "paper_boundary_input_frontier_v1.json"
    ),
    "car_timing": ROOT / (
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "timing_validation3_repeats5_v1.json"
    ),
    "car_oracle_global": ROOT / (
        "codiffuse_Schwarz_Car/outputs/"
        "input_correction_face4_order3_depth4_pressurectx_parentgate_seed20260811/"
        "validation_response_oracle25_steps8_matched_globals_v2.json"
    ),
    "air_selection": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "selection.json"
    ),
    "air_manifest": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/"
        "state_input_final_holdout_v3_manifest.json"
    ),
    "air_reported": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "final_holdout_reported_v3.json"
    ),
    "air_local": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "final_holdout_h20_five_starts_v3.json"
    ),
    "air_global": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "global_final_holdout_h20_five_starts_v3.json"
    ),
    "air_uncertainty": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "paired_uncertainty_h20_v3.json"
    ),
    "air_timing": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "timing_validation_h20_two_windows_repeats5_v1.json"
    ),
    "local_bvp_contracts": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/local_bvp_contracts_v1.json"
    ),
    "sni_contractivity": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/sni_contractivity_audit.json"
    ),
    "local_position_prior": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/"
        "local_position_prior_audit_v2.json"
    ),
    "car_identical_cube_downstream": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/"
        "car_identical_cube_downstream_audit_v1.json"
    ),
    "car_response_oracle": ROOT / (
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "response_oracle_amortization_v1.json"
    ),
    "air_response_oracle": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/"
        "state_input_selected_depth4_resp_w0005_fw2_seed20260812/"
        "validation_oracle_gap40_v1.json"
    ),
    "figure2_metadata": ROOT / "codiffuse_Schwarz/figures/latentddm_claim_summary_v3.json",
    "figure2": ROOT / "codiffuse_Schwarz/figures/latentddm_claim_summary_v3.pdf",
    "figure3_metadata": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/"
        "darcy_input_correction_seven_method.json"
    ),
    "figure3": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/"
        "darcy_input_correction_seven_method.pdf"
    ),
    "figure4_metadata": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/car_six_method_volume.json"
    ),
    "figure4": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/car_six_method_volume.pdf"
    ),
    "figure5_metadata": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/"
        "airfoil_six_method_final_holdout_rollout.json"
    ),
    "figure5": ROOT / (
        "codiffuse_Schwarz/figures/unified_report/"
        "airfoil_six_method_final_holdout_rollout.pdf"
    ),
    "manuscript": SUBMISSION / "main.tex",
    "compiled_pdf": SUBMISSION / "main.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> dict[str, Any]:
    path = PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build() -> dict[str, Any]:
    darcy = load("darcy_matched")
    darcy_fine = load("darcy_matched_fine")
    car_selection = load("car_selection")
    car = load("car_matched")
    car_frontier = load("car_frontier")
    car_timing = load("car_timing")
    air_selection = load("air_selection")
    air_manifest = load("air_manifest")
    air_reported = load("air_reported")
    air = load("air_local")
    air_global = load("air_global")
    air_uncertainty = load("air_uncertainty")
    air_timing = load("air_timing")
    local_bvp_contracts = load("local_bvp_contracts")
    sni_contractivity = load("sni_contractivity")
    local_position_prior = load("local_position_prior")
    car_identical_cube_downstream = load("car_identical_cube_downstream")
    car_response_oracle = load("car_response_oracle")
    air_response_oracle = load("air_response_oracle")
    figure2 = load("figure2_metadata")
    figure3 = load("figure3_metadata")
    figure4 = load("figure4_metadata")
    figure5 = load("figure5_metadata")

    require(darcy.get("assembly") == "normalized_overlap", "Darcy assembly drift")
    require(darcy.get("zero_message_local_core_max_abs") == 0.0, "Darcy zero-message drift")
    require(darcy_fine.get("composer_checkpoint_sha256") == darcy.get("composer_checkpoint_sha256"), "Darcy SNI lineage drift")
    require(car_selection.get("contract") == "car_response_amortization_validation_selection_v1", "Car selection drift")
    require(car_selection.get("oracle_used_for_selection") is False, "Car oracle leaked into selection")
    require(car.get("assembly") == "normalized_overlap", "Car assembly drift")
    require(car_frontier.get("patch_size") == [16, 16, 16], "Car Patch scale drift")
    require(car_timing.get("timing_subset") is True and int(car_timing.get("repeats", 0)) == 5, "Car timing is not repeated subset")
    require(air_selection.get("contract") == "airfoil_boundary_composer_validation_selection_v1", "Air selection drift")
    require(air_manifest.get("contract") == "airfoil_unopened_boundary_final_holdout_v3", "Air prospective manifest drift")
    require(air_reported.get("contract") == "airfoil_boundary_final_holdout_v3_reported", "Air finalization drift")
    require(air_reported.get("selection_preceded_holdout_opening") is True, "Air chronology drift")
    require(air.get("assembly") == "normalized_overlap" and air.get("split") == "prospective_final_holdout", "Air local result drift")
    require(air_global.get("patch_size") == [32, 32], "Air Patch scale drift")
    require(air_uncertainty.get("windows") == 60 and air_uncertainty.get("trajectories") == 12, "Air uncertainty units drift")
    require(air_timing.get("timing_subset") is True and int(air_timing.get("repeats", 0)) == 5, "Air timing is not repeated subset")
    require(
        local_bvp_contracts.get("contract")
        == "latentddm_three_case_local_bvp_contract_v1",
        "local-BVP distribution contract drift",
    )
    require(
        sni_contractivity.get("contract")
        == "three_case_sni_contractivity_audit_v3_frozen_final",
        "SNI contractivity audit drift",
    )
    car_position = local_position_prior.get("car", {})
    require(
        local_position_prior.get("contract")
        == "three_case_local_position_prior_audit_v1"
        and car_position.get("explicit_absolute_coordinate_channels") is False
        and int(car_position.get("feature_free_tiles", 0)) == 40
        and float(car_position.get("feature_free_input_variation_rms", -1.0)) == 0.0
        and float(car_position.get("feature_free_endpoint_variation_rms", -1.0)) == 0.0,
        "Car coordinate-free endpoint audit drift",
    )
    downstream_variation = car_identical_cube_downstream.get("variation_rms", {})
    downstream_pair = car_identical_cube_downstream.get(
        "maximally_separated_local_core_pair", {},
    )
    require(
        car_identical_cube_downstream.get("contract")
        == "car_identical_cube_downstream_audit_v1"
        and car_identical_cube_downstream.get("coordinate_channels_used") is False
        and int(car_identical_cube_downstream.get(
            "exactly_identical_feature_free_tiles", 0,
        )) == 40
        and float(downstream_variation.get("coordinate_free_input_x_i", -1.0)) == 0.0
        and float(downstream_variation.get("raw_endpoint_q_i", -1.0)) == 0.0
        and float(downstream_variation.get("face_role_mask", -1.0)) == 0.0
        and float(downstream_variation.get("initialized_trace_b_i_0", 0.0)) > 0.0
        and float(downstream_variation.get("local_core_candidate", 0.0)) > 0.0
        and float(downstream_pair.get("input_max_abs_difference", -1.0)) == 0.0
        and float(downstream_pair.get("raw_endpoint_max_abs_difference", -1.0)) == 0.0,
        "Car identical-cube downstream audit drift",
    )
    require(
        car_response_oracle.get("contract")
        == "car_response_oracle_amortization_audit_v1"
        and car_response_oracle.get("oracle_used_by_candidate_training") is False,
        "Car response oracle chronology drift",
    )
    require(
        air_response_oracle.get("contract")
        == "airfoil_state_input_learned_to_oracle_gap_v1"
        and air_response_oracle.get("target_used_by_candidate_inference") is False,
        "Airfoil response oracle chronology drift",
    )
    require(figure2.get("contract") == "latentddm_boundary_input_claim_summary_v6_frozen_final", "Figure 2 contract drift")
    require(figure3.get("patch_size") == [32, 32], "Figure 3 Patch scale drift")
    require(figure4.get("patch_size") == [16, 16, 16], "Figure 4 Patch scale drift")
    require(figure5.get("patch_size") == [32, 32], "Figure 5 Patch scale drift")

    result = {
        "contract": "latentddm_boundary_input_production_evidence_v1",
        "paper_method_contract": {
            "method_order": [
                "Local/Core", "LatentDDM", "Init-SNI",
                "Point", "Slice", "Patch",
            ],
            "local_core_dependent_rounds": 1,
            "latentddm_dependent_rounds": 2,
            "sni_initialization": "exact Local/Core",
            "assembly": "normalized_overlap",
            "local_operator_frozen_during_composition": True,
            "pde_residual_used_by_local_family_inference": False,
            "patch_owned_core_sizes": {
                "Darcy": [32, 32], "Car": [16, 16, 16],
                "Airfoil": [32, 32],
            },
        },
        "protected_splits": {
            "Darcy_test_cases_opened": int(darcy["protected_test_cases_opened"]),
            "Car_test_designs_opened": int(car["protected_test_designs_opened"]),
            "Airfoil_original_test_trajectories_opened": int(air["protected_test_trajectories_opened"]),
            "Airfoil_prospective_trajectories_reported": int(air["final_holdout_trajectories_opened"]),
        },
        "selected_checkpoints": {
            "Darcy": darcy["composer_checkpoint_sha256"],
            "Car": car_selection["copied_checkpoint_sha256"],
            "Airfoil": air_selection["copied_checkpoint_sha256"],
        },
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in PATHS.items()
        },
    }
    return result


def main() -> None:
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
