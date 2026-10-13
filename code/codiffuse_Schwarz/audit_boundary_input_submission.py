"""Release audit for the normalized-overlap boundary-input submission.

This intentionally replaces the historical disjoint-core audit.  It checks
method identity, protected-split declarations, table provenance, packaged
figure hashes, TeX self-containment, and the compiled page boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "codiffuse_Schwarz/workshop/submission"
OUTPUT = SUBMISSION / "boundary_input_release_audit.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    require(path.is_file(), f"missing evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value: float, places: int = 6) -> str:
    rendered = f"{value:.{places}f}"
    return rendered[1:] if rendered.startswith("0.") else rendered


def table_block(manuscript: str, label: str) -> str:
    match = re.search(
        rf"\\label\{{{re.escape(label)}\}}(.*?)\\end\{{table\}}",
        manuscript, flags=re.DOTALL,
    )
    require(match is not None, f"missing table {label}")
    return match.group(1)


def table_row(block: str, case: str) -> str:
    match = re.search(
        rf"(?:^|\n)\s*{re.escape(case)}\s*&.*?\\\\",
        block, flags=re.DOTALL,
    )
    require(match is not None, f"missing {case} row in unified table")
    return match.group(0)


def main() -> None:
    tex = SUBMISSION / "main.tex"
    log = SUBMISSION / "main.log"
    pdf = SUBMISSION / "main.pdf"
    for path in (tex, log, pdf, SUBMISSION / "references.bib", SUBMISSION / "neurips_2026.sty"):
        require(path.is_file(), f"missing submission input: {path}")
    manuscript = tex.read_text(encoding="utf-8")
    require("Figure 1 placeholder" in manuscript, "Figure 1 is not the requested placeholder")
    require("/orcd/" not in manuscript and "../" not in manuscript,
            "submission TeX contains an external path")
    for stale in ("Zero-SNI", "TripNet", "Flux-SNI", "BC-SNI", "partition of unity", "POU"):
        require(stale not in manuscript, f"stale method label remains: {stale}")
    for required in (
        "normalized overlap", "Init-SNI", "Point, Slice, and Patch",
        "same frozen", "Scope and limitations", "Figure 1 placeholder",
    ):
        require(required in manuscript, f"missing method/story invariant: {required}")

    figure_sources = {
        "latentddm_claim_summary.pdf": ROOT / "codiffuse_Schwarz/figures/latentddm_claim_summary_v3.pdf",
        "darcy_input_correction_seven_method.pdf": ROOT / "codiffuse_Schwarz/figures/unified_report/darcy_input_correction_seven_method.pdf",
        "car_six_method_volume.pdf": ROOT / "codiffuse_Schwarz/figures/unified_report/car_six_method_volume.pdf",
        "airfoil_six_method_final_holdout_rollout.pdf": ROOT / "codiffuse_Schwarz/figures/unified_report/airfoil_six_method_final_holdout_rollout.pdf",
        "three_case_decompositions.pdf": ROOT / "codiffuse_Schwarz/figures/unified_report/three_case_decompositions.pdf",
    }
    included = set(re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}", manuscript,
    ))
    require(included == set(figure_sources),
            f"unexpected included figures: {sorted(included ^ set(figure_sources))}")
    figure_hashes = {}
    for name, source in figure_sources.items():
        packaged = SUBMISSION / "figures" / name
        require(source.is_file() and packaged.is_file(), f"missing figure {name}")
        require(sha256(source) == sha256(packaged), f"packaged figure is stale: {name}")
        figure_hashes[name] = sha256(source)

    darcy = read(
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "matched_normalized_overlap_sni_128.json"
    )
    darcy_fine = read(
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "matched_normalized_overlap_sni_128_fine.json"
    )
    require(
        darcy.get("assembly") == "normalized_overlap"
        and darcy.get("protected_test_cases_opened") == 0
        and darcy.get("pde_residual_used_by_inference") is False
        and darcy.get("zero_message_local_core_max_abs") == 0.0,
        "Darcy evidence violates the shared boundary-input contract",
    )
    require(
        darcy_fine.get("assembly") == darcy.get("assembly")
        and darcy_fine.get("local_checkpoint_sha256")
        == darcy.get("local_checkpoint_sha256")
        and darcy_fine.get("composer_checkpoint_sha256")
        == darcy.get("composer_checkpoint_sha256"),
        "Darcy fine SNI sweep does not match the frozen endpoint",
    )
    car_selection = read(
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "selection.json"
    )
    car = read(
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "paper_boundary_input_frontier_v1.json"
    )
    matched_car = read(
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "matched_validation_sni_v1.json"
    )
    car_timing = read(
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "timing_validation3_repeats5_v1.json"
    )
    require(
        car_selection.get("contract")
        == "car_response_amortization_validation_selection_v1"
        and car_selection.get("oracle_used_for_selection") is False
        and car_selection.get("protected_test_designs_opened") == 0
        and car.get("contract") == "car_input_correction_six_method_frontier_v1"
        and car.get("protected_test_designs_opened") == 0
        and car.get("design_count") == 25
        and car.get("patch_size") == [16, 16, 16]
        and matched_car.get("assembly") == "normalized_overlap"
        and matched_car.get("pde_residual_used_by_inference") is False
        and car_timing.get("timing_subset") is True
        and car_timing.get("repeats") == 5,
        "Car evidence violates the shared boundary-input contract",
    )
    local_rows = {
        (row["design_id"], row["method"]): row for row in matched_car["rows"]
    }
    for design in car["design_ids"]:
        local = local_rows[(design, "local_core")]
        zero = local_rows[(design, "init_sni_0")]
        for metric in (
            "velocity_relative_l2", "pressure_relative_l2",
            "wake_magnitude_relative_l2", "divergence_rms",
        ):
            require(local[metric] == zero[metric],
                    f"Car SNI(0) differs from Local/Core for {design}: {metric}")

    air_selection = read(
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "selection.json"
    )
    air_manifest = read(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "state_input_final_holdout_v3_manifest.json"
    )
    air_reported = read(
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "final_holdout_reported_v3.json"
    )
    air = read(
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "final_holdout_h20_five_starts_v3.json"
    )
    air_timing = read(
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "timing_validation_h20_two_windows_repeats5_v1.json"
    )
    require(
        air_selection.get("contract")
        == "airfoil_boundary_composer_validation_selection_v1"
        and air_manifest.get("contract")
        == "airfoil_unopened_boundary_final_holdout_v3"
        and air_reported.get("contract")
        == "airfoil_boundary_final_holdout_v3_reported"
        and air_reported.get("selection_preceded_holdout_opening") is True
        and air.get("assembly") == "normalized_overlap"
        and air.get("split") == "prospective_final_holdout"
        and air.get("protected_test_trajectories_opened") == 0
        and air.get("final_holdout_trajectories_opened") == 12
        and air.get("target_used_by_inference") is False
        and air.get("pde_residual_used_by_inference") is False
        and air.get("same_frozen_endpoint_for_all_local_rounds") is True
        and air_timing.get("timing_subset") is True
        and air_timing.get("repeats") == 5,
        "Airfoil evidence violates the prospective boundary-input contract",
    )

    local_contracts = read(
        "codiffuse_Schwarz/figures/unified_report/local_bvp_contracts_v1.json"
    )
    require(
        local_contracts.get("contract")
        == "latentddm_three_case_local_bvp_contract_v1"
        and set(local_contracts.get("records", {}))
        == {"Darcy", "Car", "Airfoil"}
        and all(
            row.get("assembly") == "normalized_overlap"
            for row in local_contracts["records"].values()
        ),
        "local-BVP distribution manifest drift",
    )
    sni_contractivity = read(
        "codiffuse_Schwarz/figures/unified_report/sni_contractivity_audit.json"
    )
    require(
        sni_contractivity.get("contract")
        == "three_case_sni_contractivity_audit_v3_frozen_final",
        "SNI contractivity audit drift",
    )
    local_position_prior = read(
        "codiffuse_Schwarz/figures/unified_report/"
        "local_position_prior_audit_v2.json"
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
    car_identical_cube_downstream = read(
        "codiffuse_Schwarz/figures/unified_report/"
        "car_identical_cube_downstream_audit_v1.json"
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
    sni_table = table_block(manuscript, "tab:sni-contractivity")
    for record in sni_contractivity["records"]:
        row = table_row(sni_table, record["case"])
        expected = (
            record["oracle_response_error_over_local"],
            record["latentddm_error_over_local"],
            record["one_sni_sweep_error_over_sweep_zero"],
        )
        require(
            all(f"{value:.3f}" in row for value in expected),
            f"stale SNI contractivity row for {record['case']}: {expected}",
        )

    unified = table_block(manuscript, "tab:unified-frontier")
    global_darcy = read(
        "codiffuse_Schwarz_Darcy/outputs/field_token_domain_extension_degree1_v1/"
        "depth4/comparison_aggregate_with_point.json"
    )
    darcy_global_names = {
        "Point Transformer full fine-tune": "Point",
        "Transolver full fine-tune": "Slice",
    }
    darcy_values = {
        "Local/Core": darcy["summary"]["local_core"]["relative_l2_mean"],
        "LatentDDM": darcy["summary"]["latentddm"]["relative_l2_mean"],
        "Init-SNI": darcy_fine["summary"]["init_sni_20"]["relative_l2_mean"],
    }
    for row in global_darcy["rows"]:
        if int(row["budget"]) == 64 and row["method"] in darcy_global_names:
            darcy_values[darcy_global_names[row["method"]]] = row["adapted_relative_l2_mean"]
    patch_values = []
    for seed in (20260810, 20260811, 20260812):
        patch = read(
            "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1/"
            "patch_core32_capacitymatched_v1/budget64/"
            f"patch_seed{seed}/validation/summary.json"
        )
        require(
            patch["model_spec"]["patch_size"] == [32, 32],
            "Darcy Patch adaptation is not matched to the 32x32 local core",
        )
        patch_values.append(
            patch["summary"]["validation/sweep_001"]["relative_l2"]
        )
    darcy_values["Patch"] = float(np.mean(patch_values))
    car_values = {
        method: car["summary"][method]["velocity_relative_l2"]["mean"]
        for method in car["method_order"]
    }
    air_local = air["summary"]
    air_global_artifact = read(
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "global_final_holdout_h20_five_starts_v3.json"
    )
    require(
        air_global_artifact.get("patch_size") == [32, 32],
        "Airfoil Patch Transformer is not owned-core-size matched",
    )
    air_global = air_global_artifact["summary"]
    air_values = {
        "Local/Core": air_local["local_core"]["all_step_field_mean"],
        "LatentDDM": air_local["latentddm"]["all_step_field_mean"],
        "Init-SNI": air_local["init_sni_1"]["all_step_field_mean"],
        "Point": air_global["point_anchor"]["rollout_all_steps"]["field_relative_l2_mean"],
        "Slice": air_global["transolver"]["rollout_all_steps"]["field_relative_l2_mean"],
        "Patch": air_global["patch_transformer"]["rollout_all_steps"]["field_relative_l2_mean"],
    }
    for case, row_name, values in (
        ("Darcy", "Darcy", darcy_values),
        ("Car", "Car", car_values),
        ("Airfoil", "Airfoil--20", air_values),
    ):
        row = table_row(unified, row_name)
        require(all(numeric(value) in row for value in values.values()),
                f"{case} unified row is stale: {values}")

    claim = read("codiffuse_Schwarz/figures/latentddm_claim_summary_v3.json")
    car_figure = read("codiffuse_Schwarz/figures/unified_report/car_six_method_volume.json")
    darcy_figure = read("codiffuse_Schwarz/figures/unified_report/darcy_input_correction_seven_method.json")
    air_figure = read("codiffuse_Schwarz/figures/unified_report/airfoil_six_method_final_holdout_rollout.json")
    require(
        claim.get("contract")
        == "latentddm_boundary_input_claim_summary_v6_frozen_final"
        and claim.get("darcy_adaptation", {}).get("Patch", {}).get(
            "patch_size"
        ) == [32, 32],
        "Figure 2 has the wrong contract or Darcy Patch scale",
    )
    require(
        car_figure.get("contract") == "car_input_correction_six_method_visual_v1"
        and car_figure.get("assembly") == "normalized_overlap"
        and car_figure.get("protected_test_designs_opened") == 0
        and car_figure.get("patch_size") == [16, 16, 16],
        "Car visualization has the wrong contract",
    )
    require(
        car_figure.get("checkpoint_sha256", {}).get("latentddm")
        == car_selection.get("copied_checkpoint_sha256"),
        "Car visualization does not use the validation-selected composer",
    )
    require(
        np.isclose(car_figure["visualization_planes"]["horizontal_z_m"], .5)
        and "no method field inspected"
        in car_figure["visualization_planes"]["selection"],
        "Car visualization plane was not selected by the frozen neutral rule",
    )
    require(
        darcy_figure.get("contract") == "darcy_input_correction_seven_method_figure_v1"
        and darcy_figure.get("assembly") == "normalized_overlap"
        and darcy_figure.get("protected_test_cases_opened") == 0,
        "Darcy visualization has the wrong contract",
    )
    require(
        air_figure.get("contract") == "airfoil_state_input_six_method_final_visual_v1"
        and air_figure.get("assembly") == "normalized_overlap"
        and air_figure.get("patch_size") == [32, 32]
        and air_figure.get("snapshot_steps") == [0, 1, 20]
        and air_figure.get("rollout_windows") == 60,
        "Airfoil visualization has the wrong contract",
    )
    require(
        air_figure.get("checkpoint_sha256", {}).get("latentddm")
        == air_selection.get("copied_checkpoint_sha256"),
        "Airfoil visualization does not use the validation-selected composer",
    )

    log_text = log.read_text(encoding="utf-8", errors="replace")
    for warning in ("undefined references", "Citation `", "Reference `"):
        require(warning not in log_text, f"unresolved TeX warning: {warning}")
    newest_input = max(
        [tex.stat().st_mtime, (SUBMISSION / "references.bib").stat().st_mtime,
         (SUBMISSION / "neurips_2026.sty").stat().st_mtime]
        + [(SUBMISSION / "figures" / name).stat().st_mtime for name in figure_sources]
    )
    require(pdf.stat().st_mtime >= newest_input, "compiled PDF predates an input")
    document = fitz.open(pdf)
    reference_pages = [
        index + 1 for index, page in enumerate(document)
        if "References" in page.get_text("text")
    ]
    require(reference_pages and min(reference_pages) <= 8,
            f"main text exceeds eight pages: references at {reference_pages}")
    result = {
        "contract": "latentddm_boundary_input_submission_release_audit_v1",
        "passed": True,
        "submission_pdf_sha256": sha256(pdf),
        "pages": len(document),
        "references_begin_page": min(reference_pages),
        "included_figures": sorted(included),
        "figure_sha256": figure_hashes,
        "car_selected_sni_sweeps": car["selected_sni_sweeps"],
        "car_selected_communication_depth": int(
            car_selection["selected"]["configuration"]["communication_depth"]
        ),
        "car_selection_score": car_selection["selected"]["selection_score"],
        "protected_splits": {
            "Darcy_test_opened": 0, "Car_test_opened": 0,
            "Airfoil_original_test_opened": 0,
            "Airfoil_prospective_final_holdout_opened": 12,
        },
        "tests": "run separately with pytest",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
