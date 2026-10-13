"""Build one identity-checked evidence manifest for the workshop submission.

The repository contains a long autoresearch history. This script deliberately
reads only the selected normalized-overlap local-family experiments and the
matched Point/Slice/Patch audits. It records every source hash so figures,
tables, and prose can be regenerated without mixing superseded disjoint-core
or native-speed-only numbers into the current paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / (
    "codiffuse_Schwarz/workshop/submission_audits/"
    "PRODUCTION_EVIDENCE_MANIFEST.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(relative: str) -> tuple[dict[str, Any], Path]:
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")), path


def source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def darcy_evidence() -> dict[str, Any]:
    matched, matched_path = load(
        "codiffuse_Schwarz_Darcy/outputs/"
        "input_correction_modes4_spatial4_trace0p5_depth4_warm_seed20260811/"
        "matched_local_latent_sni.json"
    )
    accuracy, accuracy_path = load(
        "codiffuse_Schwarz_Darcy/outputs/"
        "input_correction_modes4_spatial4_trace0p5_depth4_warm_seed20260811/"
        "matched_accuracy_sni_grid.json"
    )
    adaptation, adaptation_path = load(
        "codiffuse_Schwarz_Darcy/outputs/"
        "input_correction_domain_extension_v1/aggregate.json"
    )
    require(
        matched.get("contract") == "darcy_input_correction_matched_validation_v1"
        and matched.get("assembly") == "normalized_overlap"
        and matched.get("cases") == 96
        and matched.get("protected_test_cases_opened") == 0
        and matched.get("target_used_by_inference") is False
        and matched.get("pde_residual_used_by_inference") is False
        and matched.get("zero_message_local_core_max_abs") == 0.0
        and matched.get("sni_sweeps_are_sequential") is True,
        "selected Darcy comparison violates the common local-family contract",
    )
    require(
        accuracy.get("contract") == matched.get("contract")
        and accuracy.get("assembly") == matched.get("assembly")
        and accuracy.get("cases") == matched.get("cases")
        and accuracy.get("local_checkpoint_sha256")
        == matched.get("local_checkpoint_sha256")
        and accuracy.get("composer_checkpoint_sha256")
        == matched.get("composer_checkpoint_sha256")
        and accuracy.get("zero_message_local_core_max_abs") == 0.0,
        "Darcy fine SNI sweep is not tied to the selected local endpoint",
    )
    require(
        adaptation.get("contract")
        == "darcy_boundary_input_domain_extension_summary_v1"
        and adaptation.get("frozen_local_bvp") is True
        and adaptation.get("source_composer_sha256")
        == matched.get("composer_checkpoint_sha256")
        and adaptation.get("local_bvp_sha256")
        == matched.get("local_checkpoint_sha256")
        and set(adaptation.get("budgets", {})) == {"4", "16", "64"},
        "Darcy target-domain adaptation has inconsistent lineage",
    )
    global_adaptation, global_adaptation_path = load(
        "codiffuse_Schwarz_Darcy/outputs/"
        "field_token_domain_extension_degree1_v1/depth4/"
        "comparison_aggregate_with_point.json"
    )
    require(
        global_adaptation.get("contract")
        == "darcy_field_token_domain_extension_comparison_v1"
        and global_adaptation.get("protected_test_cases_opened") == 0
        and "Point Transformer full fine-tune"
        in {row["method"] for row in global_adaptation.get("rows", [])},
        "Darcy global adaptation audit is not validation-only",
    )

    adapted: dict[str, dict[str, Any]] = {
        "LatentDDM boundary-input composer": {
            budget: {
                "relative_l2_mean": float(values["relative_l2_mean"]),
                "relative_l2_sample_std": float(
                    values["relative_l2_sample_std"]
                ),
                "trainable_parameters": int(values["trainable_parameters"]),
                "training_wall_seconds_mean": float(
                    values["training_wall_seconds_mean"]
                ),
            }
            for budget, values in adaptation["budgets"].items()
        }
    }
    for row in global_adaptation["rows"]:
        method = row["method"]
        if method in (
            "LatentDDM field-token composer",
            "Patch Transformer full fine-tune",
        ):
            continue
        adapted.setdefault(method, {})[str(int(row["budget"]))] = {
            "relative_l2_mean": float(row["adapted_relative_l2_mean"]),
            "relative_l2_sample_std": float(
                row["adapted_relative_l2_sample_std"]
            ),
            "trainable_parameters": int(row["trainable_parameters"]),
            "training_wall_seconds_mean": float(row["training_wall_seconds_mean"]),
        }
    patch_sources: list[Path] = []
    adapted["Patch Transformer full fine-tune"] = {}
    for budget in (4, 16, 64):
        values, wall_seconds, parameters = [], [], set()
        for seed in (20260810, 20260811, 20260812):
            payload, path = load(
                "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1/"
                f"patch_core32_capacitymatched_v1/budget{budget}/"
                f"patch_seed{seed}/validation/summary.json"
            )
            training, training_path = load(
                "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1/"
                f"patch_core32_capacitymatched_v1/budget{budget}/"
                f"patch_seed{seed}/summary.json"
            )
            require(
                payload.get("contract") == "darcy_global_full_domain_evaluation_v3"
                and payload.get("model_spec", {}).get("patch_size") == [32, 32]
                and int(payload.get("evaluated_records", 0)) == 128
                and training.get("initial_checkpoint", "").endswith(
                    "global_vit_core32_face_v2_capacitymatched_seed20260805/"
                    "checkpoint.pt"
                ),
                "Darcy Patch adaptation is not the capacity-matched 32x32 model",
            )
            values.append(float(
                payload["summary"]["validation/sweep_001"]["relative_l2"]
            ))
            wall_seconds.append(float(training["wall_seconds"]))
            parameters.add(int(training["trainable_parameter_count"]))
            patch_sources.extend((path, training_path))
        require(len(parameters) == 1, "Darcy Patch parameter count changed across seeds")
        adapted["Patch Transformer full fine-tune"][str(budget)] = {
            "relative_l2_mean": statistics.mean(values),
            "relative_l2_sample_std": statistics.stdev(values),
            "trainable_parameters": parameters.pop(),
            "training_wall_seconds_mean": statistics.mean(wall_seconds),
            "patch_size": [32, 32],
        }

    return {
        "status": "selected_complete",
        "method_order": [
            "Local/Core", "LatentDDM", "SNI", "Point", "Slice", "Patch",
        ],
        "assembly": "normalized_overlap",
        "evaluation_cases": int(matched["cases"]),
        "local_checkpoint_sha256": matched["local_checkpoint_sha256"],
        "composer_checkpoint_sha256": matched["composer_checkpoint_sha256"],
        "local_core_equals_init_sni_0_max_abs": float(
            matched["zero_message_local_core_max_abs"]
        ),
        "local_core_relative_l2": float(
            matched["summary"]["local_core"]["relative_l2_mean"]
        ),
        "latentddm_relative_l2": float(
            matched["summary"]["latentddm"]["relative_l2_mean"]
        ),
        "communication_depth": int(matched["communication_depth"]),
        "trainable_composer_parameters": int(
            adaptation["budgets"]["64"]["trainable_parameters"]
        ),
        "sequential_local_rounds": {
            "local_core": float(
                matched["summary"]["local_core"]["sequential_local_rounds_mean"]
            ),
            "latentddm": float(
                matched["summary"]["latentddm"]["sequential_local_rounds_mean"]
            ),
        },
        "sni": {
            "matched_sweeps": 13,
            "matched_relative_l2": float(
                accuracy["summary"]["init_sni_13"]["relative_l2_mean"]
            ),
            "sni_16_relative_l2": float(
                accuracy["summary"]["init_sni_16"]["relative_l2_mean"]
            ),
            "initialization": "local_core",
            "sweep0_equals_local_core_max_abs": float(
                matched["zero_message_local_core_max_abs"]
            ),
        },
        "same_process_seconds": {
            name: float(row["seconds_mean"])
            for name, row in matched["summary"].items()
        },
        "domain_extension": adapted,
        "sources": [
            source(path) for path in (
                matched_path, accuracy_path, adaptation_path,
                global_adaptation_path,
                *patch_sources,
            )
        ],
    }


def car_evidence() -> dict[str, Any]:
    frontier, frontier_path = load(
        "codiffuse_Schwarz_Car/outputs/field_token_selected/"
        "paper_common_grid_frontier.json"
    )
    require(
        frontier.get("contract") == "car_complete_cartesian_six_method_frontier_v2"
        and frontier.get("method_order") == [
            "Local/Core", "LatentDDM", "SNI", "Point", "Slice", "Patch",
        ]
        and frontier["paper_promotion_gate"]["passed"] is True
        and frontier.get("protected_test_designs_opened") == 0,
        "Car paper promotion gate is not valid",
    )
    frontier_inputs = []
    for raw_path in frontier.get("sources", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        require(path.is_file(), f"missing Car frontier source: {path}")
        frontier_inputs.append(path)
    require(
        len(frontier_inputs) == 6,
        "Car frontier no longer identifies all local/SNI/global source audits",
    )
    timing, timing_path = load(
        "codiffuse_Schwarz_Car/outputs/field_token_selected/"
        "same_process_core_init_sni_timing_v2.json"
    )
    require(
        timing.get("contract") == "car_field_token_same_process_init_sni_timing_v2"
        and timing.get("assembly") == "disjoint_owned_core"
        and timing.get("sni_initialization") == "local_core"
        and timing.get("tau") == 0.5
        and set(timing.get("summary", {})) == {
            "local_core", "latentddm", "init_sni_1", "init_sni_2",
            "init_sni_4",
        }
        and timing.get("analytic_time_division_or_projection") is False
        and timing.get("velocity_endpoint_replay_passed") is True,
        "Car timing is not the exact selected owned-core audit",
    )
    compute, compute_path = load(
        "codiffuse_Schwarz_Car/outputs/compute_anatomy_field_token_v2.json"
    )
    require(
        compute.get("contract") == "car_uniform_compute_anatomy_field_token_v2"
        and compute.get("protected_test_accessed") is False
        and compute.get("device") == "NVIDIA H200",
        "Car compute anatomy is not the selected field-token H200 audit",
    )
    compute_methods = {row["method"]: row for row in compute["methods"]}
    require(
        set(compute_methods) == {
            "LatentDDM", "Matched AB-UPT adaptation", "Transolver",
            "Patch Transformer",
        },
        "Car compute-anatomy family set changed",
    )
    selected_checkpoint = ROOT / (
        "codiffuse_Schwarz_Car/outputs/field_token_selected/checkpoint.pt"
    )
    require(
        compute_methods["LatentDDM"]["field_token_checkpoint_sha256"]
        == sha256(selected_checkpoint)
        and compute_methods["LatentDDM"]["assembly"]
        == "disjoint_owned_core",
        "Car compute anatomy does not profile the selected checkpoint",
    )
    mechanism_path = ROOT / (
        "codiffuse_Schwarz_Car/outputs/field_token_selected/"
        "mechanism_controls_validation.json"
    )
    mechanism = None
    if mechanism_path.exists():
        mechanism_payload = json.loads(
            mechanism_path.read_text(encoding="utf-8")
        )
        require(
            mechanism_payload.get("contract")
            == "car_field_token_mechanism_validation_v2"
            and mechanism_payload.get("split") == "validation"
            and mechanism_payload.get("assembly") == "disjoint_owned_core"
            and mechanism_payload.get("protected_test_designs_opened") == 0
            and mechanism_payload.get("checkpoint_sha256")
            == sha256(selected_checkpoint),
            "Car mechanism controls do not use the frozen selected model",
        )
        mechanism = mechanism_payload["mechanism_analysis"]
    distributed_path = ROOT / (
        "codiffuse_Schwarz_Car/outputs/field_token_selected/"
        "distributed_scaling_v1/summary.json"
    )
    distributed = None
    distributed_sources: list[Path] = []
    if distributed_path.exists():
        distributed_payload = json.loads(
            distributed_path.read_text(encoding="utf-8")
        )
        require(
            distributed_payload.get("contract")
            == "car_field_token_distributed_scaling_summary_v1"
            and distributed_payload.get("split") == "validation"
            and distributed_payload.get("protected_test_designs_opened") == 0
            and distributed_payload.get("checkpoint_sha256")
            == sha256(selected_checkpoint)
            and distributed_payload.get("method_contract", {}).get(
                "assembly"
            ) == "disjoint_owned_core"
            and distributed_payload.get("method_contract", {}).get(
                "uses_partition_of_unity"
            ) is False
            and distributed_payload.get("rank_invariance", {}).get("passed")
            is True,
            "Car distributed scaling violates the selected method contract",
        )
        require(
            [int(row["world_size"]) for row in distributed_payload["rows"]]
            == [1, 2, 4],
            "Car distributed scaling is not the declared 1/2/4-GPU audit",
        )
        distributed = {
            "device": distributed_payload["device"],
            "rows": distributed_payload["rows"],
            "rank_invariance": distributed_payload["rank_invariance"],
        }
        distributed_sources = [
            distributed_path,
            *(distributed_path.parent / f"p{world}.json"
              for world in (1, 2, 4)),
        ]
    methods = {}
    for name, row in frontier["methods"].items():
        methods[name] = {
            metric: {
                "mean": float(values["mean"]),
                "paired_design_bootstrap_ci95": [
                    float(value)
                    for value in values["paired_design_bootstrap_ci95"]
                ],
            }
            for metric, values in row.items()
            if isinstance(values, dict) and "mean" in values
        }
    sources = [
        source(frontier_path),
        *(source(path) for path in frontier_inputs),
        source(timing_path),
        source(compute_path),
    ]
    if mechanism is not None:
        sources.append(source(mechanism_path))
    sources.extend(source(path) for path in distributed_sources)
    return {
        "status": "selected_complete",
        "method_order": list(frontier["method_order"]),
        "assembly": "disjoint_owned_core",
        "methods": methods,
        "latent_minus_patch_velocity": frontier[
            "paired_full_vector_differences"
        ]["Patch"],
        "same_process_seconds": {
            name: float(row["mean_seconds"])
            for name, row in timing["summary"].items()
        },
        "compute_anatomy": {
            name: {
                key: row[key]
                for key in (
                    "total_stage_seconds", "coupled_compute_fraction",
                    "cross_region_state_bytes", "peak_allocated_gib",
                    "global_dependency_depth",
                    "adaptation_trainable_parameter_count",
                )
            }
            for name, row in compute_methods.items()
        },
        "mechanism_controls_status": (
            "complete" if mechanism is not None else "pending"
        ),
        "mechanism_controls": mechanism,
        "distributed_scaling_status": (
            "complete" if distributed is not None else "pending"
        ),
        "distributed_scaling": distributed,
        "sources": sources,
    }


def airfoil_evidence() -> dict[str, Any]:
    global_audit, global_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/global100_v1_finest/"
        "global_exact_point_slice_patch_validation_v2.json"
    )
    require(
        global_audit.get("contract") == "airfoil_global_exact_main_validation_v1"
        and global_audit.get("protected_test_trajectories_opened") == 0,
        "Airfoil global audit is not the exact validation protocol",
    )
    global_names = {
        "point_anchor": "point",
        "transolver": "slice",
        "patch_transformer": "patch",
    }
    globals_summary = {
        global_names[family]: {
            metric: float(row["rollout_all_steps"][source_key])
            for metric, source_key in (
                ("field", "field_relative_l2_mean"),
                ("vorticity", "vorticity_relative_l2_mean"),
                ("interface", "interface_relative_l2_mean"),
                ("divergence", "divergence_rms_mean"),
            )
        }
        for family, row in global_audit["summary"].items()
    }
    globals_one_step = {
        global_names[family]: {
            metric: float(row["one_step"][source_key])
            for metric, source_key in (
                ("field", "field_relative_l2_mean"),
                ("vorticity", "vorticity_relative_l2_mean"),
                ("interface", "interface_relative_l2_mean"),
                ("divergence", "divergence_rms_mean"),
                ("force", "force_relative_l2_mean"),
            )
        }
        for family, row in global_audit["summary"].items()
    }
    selection_path = ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/field_token_matched_selected/"
        "selection.json"
    )
    result: dict[str, Any] = {
        "status": "matched_selection_pending",
        "method_order": [
            "Local/Core", "LatentDDM", "SNI", "Point", "Slice", "Patch",
        ],
        "global_h20": globals_summary,
        "global_one_step": globals_one_step,
        "sources": [source(global_path)],
    }
    # Selection-ineligible capacity audit for the theorem's response-manifold
    # term.  It is recorded independently of which deployable checkpoint wins
    # the reporting gate, and may never be used as a method row.
    expanded_run = ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/"
        "field_token_core_joint_depth4_response32_spatial3_extra32_"
        "conforminghelmholtz4_code75_balanced4_seed20260810"
    )
    expanded_checkpoint = expanded_run / "checkpoint_epoch001.pt"
    expanded_audit_path = expanded_run / "exact_parent_audit.json"
    expanded_oracle_path = expanded_run / (
        "rollout_h20_epoch001_oracle_balanced6_steps64_validation.json"
    )
    expanded_audit = json.loads(
        expanded_audit_path.read_text(encoding="utf-8")
    )
    expanded_oracle = json.loads(
        expanded_oracle_path.read_text(encoding="utf-8")
    )
    expanded_optimization = expanded_oracle.get(
        "oracle_port_diagnostic_complete_code_optimization", {}
    )
    require(
        expanded_audit.get("contract")
        == "airfoil_conforming_helmholtz_exact_parent_audit_v1"
        and expanded_audit.get("parent_tensor_identity") is True
        and expanded_audit.get("nonzero_message_parent_identity") is True
        and expanded_audit.get(
            "conforming_helmholtz_only_training_scope_verified"
        ) is True
        and expanded_audit.get("conforming_coefficient_nonzero_count") == 0
        and int(expanded_audit.get("conforming_subcells", 0)) == 4
        and int(expanded_audit.get("conforming_code_width", 0)) == 75
        and int(expanded_audit.get("complete_response_code_width", 0)) == 123
        and expanded_audit.get("protected_test_trajectories_opened") == 0,
        "expanded Airfoil response diagnostic lacks exact-parent evidence",
    )
    require(
        expanded_oracle.get("contract")
        == "airfoil_field_token_rollout_h20_validation_v2"
        and expanded_oracle.get("split") == "validation"
        and expanded_oracle.get("checkpoint_sha256")
        == sha256(expanded_checkpoint)
        and expanded_oracle.get("oracle_port_diagnostic_included") is True
        and expanded_oracle.get(
            "oracle_port_diagnostic_uses_current_target"
        ) is True
        and expanded_oracle.get(
            "oracle_port_diagnostic_selection_eligible"
        ) is False
        and int(expanded_optimization.get("steps", 0)) == 64
        and expanded_optimization.get("objective", "").startswith(
            "equal Local/Core-normalized field"
        )
        and expanded_oracle.get("protected_test_trajectories_opened") == 0,
        "expanded Airfoil response-manifold oracle violates its audit contract",
    )
    result["expanded_response_manifold_oracle"] = {
        metric: float(expanded_oracle["aggregate"]["oracle_port"][
            f"all_step_{metric}"
        ])
        for metric in (
            "field", "vorticity", "interface", "divergence", "drag", "lift",
        )
    }
    result["expanded_response_manifold_oracle"].update({
        "base_response_code_width": 48,
        "expanded_response_code_width": 123,
        "selection_eligible": False,
        "uses_current_target": True,
    })
    result["sources"].extend((
        source(expanded_checkpoint),
        source(expanded_audit_path),
        source(expanded_oracle_path),
    ))
    if not selection_path.exists():
        return result
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    require(
        selection.get("contract") in {
            "airfoil_matched_field_token_rollout_selection_v1",
            "airfoil_spatial_port_field_token_rollout_selection_v2",
            "airfoil_helmholtz_field_token_rollout_selection_v3",
            "airfoil_coarse_helmholtz_field_token_rollout_selection_v4",
            "airfoil_conforming_helmholtz_field_token_rollout_selection_v5",
            "airfoil_expanded_conforming_field_token_rollout_selection_v6",
            "airfoil_validation_reporting_field_token_rollout_selection_v7",
        }
        and selection.get("protected_test_trajectories_opened") == 0,
        "Airfoil matched selector has the wrong contract",
    )
    result["selection"] = {
        "passed": bool(selection["selection_passed"]),
        "strict_passed": bool(selection.get(
            "strict_selection_passed", selection["selection_passed"],
        )),
        "claim_tier": selection.get(
            "claim_tier",
            "strict_global_frontier" if selection["selection_passed"] else "none",
        ),
        "status": selection["status"],
        "admissible_candidate_count": int(
            selection["admissible_candidate_count"]
        ),
        "reporting_admissible_candidate_count": int(selection.get(
            "reporting_admissible_candidate_count",
            selection["admissible_candidate_count"],
        )),
        "selected_diagnostic_candidate": selection["selected"],
    }
    result["sources"].append(source(selection_path))
    if not selection["selection_passed"]:
        result["status"] = "matched_gate_failed"
        return result
    selected_arm = str(selection["selected"]["arm"])
    result["selected_architecture"] = selected_arm
    if "potential32" in selected_arm:
        potential_audit_path = ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/"
            "field_token_core_joint_depth4_response32_spatial3_extra32_"
            "potential32_balanced4_seed20260810/exact_parent_audit.json"
        )
        potential_audit = json.loads(
            potential_audit_path.read_text(encoding="utf-8")
        )
        require(
            potential_audit.get("contract")
            == "airfoil_potential_response_exact_parent_audit_v1"
            and potential_audit.get("parent_tensor_identity") is True
            and potential_audit.get("nonzero_message_parent_identity") is True
            and potential_audit.get("potential_only_training_scope_verified")
            is True
            and potential_audit.get("potential_coefficient_nonzero_count") == 0
            and potential_audit.get("potential_response_rank") == 32,
            "selected Airfoil potential child lacks exact-parent evidence",
        )
        result["potential_response_exact_parent_audit"] = potential_audit
        result["sources"].append(source(potential_audit_path))
    if "stream32" in selected_arm:
        stream_audit_path = ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/"
            "field_token_core_joint_depth4_response32_spatial3_extra32_"
            "potential32_stream32_balanced4_seed20260810/"
            "exact_parent_audit.json"
        )
        stream_audit = json.loads(
            stream_audit_path.read_text(encoding="utf-8")
        )
        require(
            stream_audit.get("contract")
            == "airfoil_stream_response_exact_parent_audit_v1"
            and stream_audit.get("parent_tensor_identity") is True
            and stream_audit.get("nonzero_message_parent_identity") is True
            and stream_audit.get(
                "streamfunction_only_training_scope_verified"
            ) is True
            and stream_audit.get("stream_coefficient_nonzero_count") == 0
            and stream_audit.get("stream_response_rank") == 32,
            "selected Airfoil streamfunction child lacks exact-parent evidence",
        )
        result["stream_response_exact_parent_audit"] = stream_audit
        result["sources"].append(source(stream_audit_path))
    if "conforming_helmholtz_4" in selected_arm:
        expanded = "conforming_helmholtz_4_code75" in selected_arm
        run_suffix = (
            "conforminghelmholtz4_code75_balanced4_seed20260810"
            if expanded
            else "conforminghelmholtz4_balanced4_seed20260810"
        )
        conforming_audit_path = ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/"
            "field_token_core_joint_depth4_response32_spatial3_extra32_"
            f"{run_suffix}/exact_parent_audit.json"
        )
        conforming_audit = json.loads(
            conforming_audit_path.read_text(encoding="utf-8")
        )
        require(
            conforming_audit.get("contract")
            == "airfoil_conforming_helmholtz_exact_parent_audit_v1"
            and conforming_audit.get("parent_tensor_identity") is True
            and conforming_audit.get("nonzero_message_parent_identity") is True
            and conforming_audit.get(
                "conforming_helmholtz_only_training_scope_verified"
            ) is True
            and conforming_audit.get(
                "conforming_coefficient_nonzero_count"
            ) == 0
            and int(conforming_audit.get("conforming_subcells", 0)) == 4
            and conforming_audit.get("protected_test_trajectories_opened") == 0,
            "selected Airfoil conforming child lacks exact-parent evidence",
        )
        if expanded:
            require(
                int(conforming_audit.get("conforming_code_width", 0)) == 75
                and int(conforming_audit.get(
                    "complete_response_code_width", 0,
                )) == 123,
                "expanded Airfoil conforming child lacks the declared 123D code",
            )
        result["conforming_response_exact_parent_audit"] = conforming_audit
        result["sources"].append(source(conforming_audit_path))
    rollout_path = Path(selection["selected"]["rollout"])
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    require(
        rollout.get("contract") == "airfoil_field_token_rollout_h20_validation_v2"
        and rollout.get("assembly") == "disjoint_owned_core"
        and rollout.get("protected_test_trajectories_opened") == 0,
        "selected Airfoil rollout violates the paper contract",
    )
    result["status"] = "selected_test_pending"
    result["local_h20"] = {
        name: {
            metric: float(row[f"all_step_{metric}"])
            for metric in ("field", "vorticity", "interface", "divergence")
        }
        for name, row in rollout["aggregate"].items()
        if name in ("local_core", "latentddm", "shuffled")
    }
    result["sources"].append(source(rollout_path))
    oracle_path = selection_path.parent / (
        "rollout_h20_response_oracle_balanced6_validation.json"
    )
    if oracle_path.exists():
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        optimization = oracle.get(
            "oracle_port_diagnostic_complete_code_optimization", {}
        )
        require(
            oracle.get("contract")
            == "airfoil_field_token_rollout_h20_validation_v2"
            and oracle.get("split") == "validation"
            and oracle.get("checkpoint_sha256")
            == selection["copied_checkpoint_sha256"]
            and oracle.get("oracle_port_diagnostic_included") is True
            and oracle.get("oracle_port_diagnostic_uses_current_target") is True
            and oracle.get("oracle_port_diagnostic_selection_eligible") is False
            and int(optimization.get("steps", 0)) == 64
            and optimization.get("objective", "").startswith(
                "equal Local/Core-normalized field"
            )
            and oracle.get("protected_test_trajectories_opened") == 0,
            "Airfoil response-manifold oracle is not the sealed selected audit",
        )
        result["response_manifold_oracle"] = {
            metric: float(oracle["aggregate"]["oracle_port"][
                f"all_step_{metric}"
            ])
            for metric in (
                "field", "vorticity", "interface", "divergence",
                "drag", "lift",
            )
        }
        result["response_manifold_oracle"]["selection_eligible"] = False
        result["response_manifold_oracle"]["uses_current_target"] = True
        result["sources"].append(source(oracle_path))
    one_step_path = selection_path.parent / "one_step_core_validation.json"
    if one_step_path.exists():
        one_step = json.loads(one_step_path.read_text(encoding="utf-8"))
        require(
            one_step.get("contract")
            == "airfoil_field_token_one_step_validation_v1"
            and one_step.get("assembly") == "disjoint_owned_core"
            and one_step.get("uses_partition_of_unity") is False
            and one_step.get("examples") == 40
            and one_step.get("protected_test_trajectories_opened") == 0
            and one_step.get("checkpoint_sha256")
            == selection["copied_checkpoint_sha256"],
            "selected Airfoil one-step audit violates the paper contract",
        )
        result["one_step"] = {
            "examples": int(one_step["examples"]),
            "summary": {
                name: values for name, values in one_step["summary"].items()
                if name in (
                    "local_core", "latentddm", "shuffled", "init_sni_1",
                )
            },
            "sni_operating_point": one_step["sni_operating_points"][
                "init_sni_1"
            ],
        }
        result["sources"].append(source(one_step_path))
    sni_path = selection_path.parent / "rollout_h20_core_sni_validation.json"
    if sni_path.exists():
        sni = json.loads(sni_path.read_text(encoding="utf-8"))
        require(
            sni.get("contract") == "airfoil_field_token_rollout_h20_validation_v2"
            and sni.get("assembly") == "disjoint_owned_core"
            and sni.get("protected_test_trajectories_opened") == 0,
            "selected Airfoil SNI rollout violates the paper contract",
        )
        require(
            "init_sni_1" in sni["aggregate"],
            "selected Airfoil SNI rollout omits the Local/Core initialization",
        )
        result["sni_h20"] = {
            name: {
                metric: float(row[f"all_step_{metric}"])
                for metric in ("field", "vorticity", "interface", "divergence")
            }
            for name, row in sni["aggregate"].items()
            if name.startswith("init_sni_")
        }
        result["sources"].append(source(sni_path))
    timing_path = selection_path.parent / "same_process_core_timing.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        require(
            timing.get("contract")
            == "airfoil_same_process_init_sni_timing_v2"
            and timing.get("assembly") == "core_ras"
            and timing.get("sni_initialization") == "local_core"
            and timing.get("tau") == 0.125
            and set(timing.get("summary", {})) == {
                "local_core", "latentddm", "init_sni_1", "init_sni_2",
                "init_sni_4",
            }
            and timing.get("analytic_time_division_or_projection") is False,
            "selected Airfoil timing uses an analytic projection",
        )
        result["same_process_seconds"] = {
            name: float(row["mean_seconds"])
            for name, row in timing["summary"].items()
        }
        result["sources"].append(source(timing_path))
    test_path = ROOT / (
        "codiffuse_Schwarz/figures/unified_report/"
        "airfoil_six_method_test_rollout.json"
    )
    if test_path.exists():
        test = json.loads(test_path.read_text(encoding="utf-8"))
        require(
            test.get("contract")
            == "airfoil_six_method_test_rollout_cluster_band_v1"
            and test.get("split") == "test"
            and test.get("selection_split") == "validation"
            and test.get("checkpoint_frozen_before_test") is True
            and test.get("field_token_checkpoint_sha256")
            == selection["copied_checkpoint_sha256"]
            and test.get("selection", {}).get("claim_tier")
            == result["selection"]["claim_tier"]
            and test.get("selection", {}).get("strict_selection_passed")
            == result["selection"]["strict_passed"]
            and test.get("protected_test_trajectories_opened") == 6
            and len(test.get("test_trajectories", [])) == 6
            and test.get("rollout_windows") == 60
            and test.get("horizon") == 20,
            "Airfoil final rollout is not the frozen-checkpoint held-out audit",
        )
        method_order = [
            "local_core", "latentddm", "sni", "point", "slice", "patch",
        ]
        require(
            test.get("method_order") == method_order
            and set(test.get("summary", {})) == set(method_order)
            and set(test.get("records", {})) == set(method_order),
            "Airfoil final rollout does not contain exactly the six paper methods",
        )
        uncertainty = test.get("uncertainty", {})
        require(
            uncertainty.get("cluster_unit") == "held-out trajectory"
            and uncertainty.get("windows_retained_within_cluster") is True
            and float(uncertainty.get("confidence_level", 0.0)) == 0.95
            and int(uncertainty.get("replicates", 0)) >= 4000,
            "Airfoil uncertainty is not the declared trajectory-cluster band",
        )
        result["test_h20"] = {
            name: {
                metric: float(test["summary"][name][f"all_step_{metric}"])
                for metric in (
                    "field", "vorticity", "interface", "divergence",
                    "force", "drag", "lift", "seconds",
                )
                if f"all_step_{metric}" in test["summary"][name]
            }
            for name in method_order
        }
        result["test_uncertainty"] = uncertainty
        result["test_rollout_windows"] = int(test["rollout_windows"])
        result["test_trajectories"] = list(test["test_trajectories"])
        result["status"] = "selected_complete"
        result["sources"].append(source(test_path))
    return result


def build() -> dict[str, Any]:
    return {
        "contract": "latentddm_workshop_production_evidence_manifest_v2",
        "paper_method_contract": {
            "method_order": [
                "Local/Core", "LatentDDM", "SNI", "Point", "Slice", "Patch",
            ],
            "local_endpoint_calls": 1,
            "composer": "centered stacked field-token communication",
            "assembly": "disjoint_owned_core",
            "uses_partition_of_unity": False,
            "global_families": ["point", "slice", "patch"],
            "sni_initialization": "local_core",
        },
        "darcy": darcy_evidence(),
        "car": car_evidence(),
        "airfoil": airfoil_evidence(),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "darcy": payload["darcy"]["status"],
        "car": payload["car"]["status"],
        "airfoil": payload["airfoil"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
