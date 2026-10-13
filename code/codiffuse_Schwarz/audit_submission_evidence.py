"""Read-only audit of the promoted LatentDDM workshop evidence.

The production manifest is the sole numerical source of truth.  This audit
verifies its hashes and method identities, checks that settled claims have not
regressed to POU/residual-era artifacts, and validates the self-contained
submission package.  The default is release-strict; ``--allow-pending-airfoil``
is provided only for development while the declared Airfoil selector runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "codiffuse_Schwarz/workshop/submission"
MAIN = SUBMISSION / "main.tex"
PDF = SUBMISSION / "main.pdf"
MANIFEST = ROOT / (
    "codiffuse_Schwarz/workshop/submission_audits/"
    "PRODUCTION_EVIDENCE_MANIFEST.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-pending-airfoil", action="store_true",
        help="development-only: audit settled cases while Airfoil is pending",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(observed: float, expected: float, atol: float = 5.0e-7) -> None:
    require(
        math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=atol),
        f"numeric evidence drift: expected {expected}, observed {observed}",
    )


def audit_sources(manifest: dict[str, Any]) -> int:
    count = 0
    for case in ("darcy", "car", "airfoil"):
        for item in manifest[case]["sources"]:
            path = Path(item["path"])
            require(path.is_file(), f"missing evidence source: {path}")
            require(sha256(path) == item["sha256"], f"source hash drift: {path}")
            count += 1
    return count


def audit_darcy(row: dict[str, Any]) -> dict[str, Any]:
    require(row["status"] == "selected_complete", "Darcy is not promoted")
    require(row["assembly"] == "disjoint_owned_core", "Darcy assembly drifted")
    close(row["local_core_relative_l2"], 0.5394316837191582)
    close(row["latentddm_relative_l2"], 0.04873154265806079)
    close(row["seam_error"], 0.05969602253753692)
    require(
        row["seam_error_change_fraction"] < -0.20,
        "Darcy seam-safe continuation no longer reduces interface jumps",
    )
    require(row["communication_depth"] == 1, "Darcy selected depth changed")
    require(
        row["latentddm_relative_l2"] < row["local_core_relative_l2"],
        "Darcy communication no longer improves Local/Core",
    )
    close(row["sni"]["sni_128_best_relative_l2"], 0.012968188621016452)
    close(row["sni"]["sni_9_relative_l2"], 0.04233491581908311)
    require(
        row["sni"]["initialization"] == "local_core",
        "Darcy production SNI is not initialized from Local/Core",
    )
    timing = row["same_process_seconds"]
    require(
        timing["local_core"] < timing["latentddm"] < timing["init_sni_1"]
        < timing["init_sni_4"] < timing["init_sni_8"]
        < timing["init_sni_9"],
        "Darcy same-process timing order drifted",
    )
    families = row["global_three_seed"]
    require(
        set(families) == {"point", "slice", "patch"},
        "Darcy global family set drifted",
    )
    require(all(len(value["seeds"]) == 3 for value in families.values()),
            "Darcy global seed count drifted")
    adapted = row["domain_extension"]
    for budget in ("4", "16", "64"):
        latent = adapted["LatentDDM field-token composer"][budget]["relative_l2_mean"]
        globals_ = (
            adapted["Point Transformer full fine-tune"][budget]["relative_l2_mean"],
            adapted["Transolver full fine-tune"][budget]["relative_l2_mean"],
            adapted["Patch Transformer full fine-tune"][budget]["relative_l2_mean"],
        )
        require(latent < min(globals_), f"Darcy adaptation lost at {budget} labels")
    return {
        "local": row["local_core_relative_l2"],
        "latentddm": row["latentddm_relative_l2"],
        "depth": row["communication_depth"],
    }


def audit_car(
    row: dict[str, Any], *, allow_pending_mechanism: bool,
) -> dict[str, Any]:
    require(row["status"] == "selected_complete", "Car is not promoted")
    require(row["assembly"] == "disjoint_owned_core", "Car assembly drifted")
    methods = row["methods"]
    expected = {
        "Local/Core", "LatentDDM", "SNI", "Point", "Slice", "Patch",
    }
    require(set(methods) == expected, "Car comparison method set drifted")
    velocity = {
        name: values["velocity_rel_l2"]["mean"] for name, values in methods.items()
    }
    close(velocity["LatentDDM"], 0.014150551743805409)
    require(
        velocity["LatentDDM"] < min(
            velocity[name] for name in expected - {"LatentDDM"}
        ),
        "LatentDDM is no longer the best Car full-vector model",
    )
    ci = row["latent_minus_patch_velocity"]["paired_design_bootstrap_ci95"]
    require(ci[1] < 0.0, "Car LatentDDM-minus-Patch interval crosses zero")
    timing = row["same_process_seconds"]
    require(
        timing["local_core"] < timing["latentddm"] < timing["init_sni_1"]
        < timing["init_sni_2"] < timing["init_sni_4"],
        "Car same-process timing order drifted",
    )
    compute = row["compute_anatomy"]
    require(
        set(compute) == {
            "LatentDDM", "Matched AB-UPT adaptation", "Transolver",
            "Patch Transformer",
        },
        "Car compute-anatomy family set drifted",
    )
    latent_compute = compute["LatentDDM"]
    close(latent_compute["total_stage_seconds"], 0.9369238382950424, atol=1.0e-6)
    close(latent_compute["coupled_compute_fraction"], 0.002951241038323461, atol=1.0e-8)
    require(
        latent_compute["cross_region_state_bytes"]
        < min(
            compute[name]["cross_region_state_bytes"]
            for name in compute if name != "LatentDDM"
        ),
        "LatentDDM no longer has the smallest Car cross-region state",
    )
    require(
        latent_compute["coupled_compute_fraction"]
        < min(
            compute[name]["coupled_compute_fraction"]
            for name in compute if name != "LatentDDM"
        ),
        "LatentDDM no longer has the smallest Car coupled fraction",
    )
    if row.get("mechanism_controls_status") != "complete":
        require(
            allow_pending_mechanism,
            "selected Car communication controls are still pending",
        )
    else:
        mechanism = row["mechanism_controls"]
        require(
            mechanism["radius0_local_core_max_abs"] <= 2.0e-6,
            "Car edge removal does not replay Local/Core",
        )
        velocity_mechanism = mechanism["metrics"]["velocity_rel_l2"]
        require(
            velocity_mechanism[
                "latentddm_minus_local_paired_design_bootstrap_ci95"
            ][1] < 0.0,
            "Car communication gain over Local/Core is not paired-resolved",
        )
        shuffled = velocity_mechanism["controls"]["shuffled"]
        require(
            shuffled[
                "control_minus_latent_paired_design_bootstrap_ci95"
            ][0] > 0.0,
            "Car token derangement does not significantly lose the selected gain",
        )
        require(
            math.isclose(
                velocity_mechanism["controls"]["radius0"][
                    "fraction_of_latentddm_gain_lost"
                ],
                1.0,
                rel_tol=0.0,
                abs_tol=2.0e-4,
            ),
            "Car radius-zero control is not the exact Local/Core mechanism anchor",
        )
    if row.get("distributed_scaling_status") != "complete":
        require(
            allow_pending_mechanism,
            "selected Car no-POU distributed scaling is still pending",
        )
    else:
        distributed = row["distributed_scaling"]
        require(
            isinstance(distributed.get("device"), str)
            and bool(distributed["device"].strip()),
            "Car distributed audit does not identify its GPU model",
        )
        require(
            [int(value["world_size"]) for value in distributed["rows"]]
            == [1, 2, 4],
            "Car distributed audit is not the declared 1/2/4-GPU sequence",
        )
        require(
            distributed["rank_invariance"]["passed"] is True,
            "Car distributed outputs are not rank invariant",
        )
        require(
            all(
                float(value["wall_seconds"]) > 0.0
                and float(value["gpu_seconds"]) > 0.0
                and float(value["speedup"]) > 0.0
                for value in distributed["rows"]
            ),
            "Car distributed timing contains a nonpositive measurement",
        )
    return {
        "velocity": velocity,
        "latent_minus_patch_ci95": ci,
        "compute_anatomy": latent_compute,
        "mechanism_controls_status": row.get("mechanism_controls_status"),
        "distributed_scaling_status": row.get("distributed_scaling_status"),
    }


def audit_airfoil(
    row: dict[str, Any], *, allow_pending: bool,
) -> dict[str, Any]:
    require(
        set(row["global_h20"]) == {
            "point", "slice", "patch",
        },
        "Airfoil global family set drifted",
    )
    require(
        set(row["global_one_step"]) == {
            "point", "slice", "patch",
        },
        "Airfoil one-step global family set drifted",
    )
    expanded_oracle = row.get("expanded_response_manifold_oracle")
    require(
        isinstance(expanded_oracle, dict)
        and expanded_oracle.get("selection_eligible") is False
        and expanded_oracle.get("uses_current_target") is True
        and int(expanded_oracle.get("base_response_code_width", 0)) == 48
        and int(expanded_oracle.get("expanded_response_code_width", 0)) == 123
        and all(
            math.isfinite(float(expanded_oracle[metric]))
            and float(expanded_oracle[metric]) >= 0.0
            for metric in (
                "field", "vorticity", "interface", "divergence",
                "drag", "lift",
            )
        ),
        "Airfoil expanded response-manifold diagnostic is missing or invalid",
    )
    if row["status"] != "selected_complete":
        require(allow_pending, f"Airfoil release gate is {row['status']}")
        return {"status": row["status"]}
    require(row["selection"]["passed"] is True, "Airfoil selector did not pass")
    claim_tier = row["selection"].get("claim_tier")
    require(
        claim_tier in {"strict_global_frontier", "complementary_validation"},
        "Airfoil reporting checkpoint has no declared validation claim tier",
    )
    require(
        row["selection"].get("strict_passed") is
        (claim_tier == "strict_global_frontier"),
        "Airfoil strict-frontier result was relabeled during reporting selection",
    )
    require(
        row["selection"]["selected_diagnostic_candidate"].get(
            "improves_drag_and_lift_over_local_and_shuffled"
        ) is True,
        "selected Airfoil model sacrifices integrated drag or lift",
    )
    if "potential32" in str(row.get("selected_architecture", "")):
        parent = row.get("potential_response_exact_parent_audit", {})
        require(
            parent.get("parent_tensor_identity") is True
            and parent.get("nonzero_message_parent_identity") is True
            and parent.get("potential_only_training_scope_verified") is True
            and parent.get("potential_coefficient_nonzero_count") == 0
            and parent.get("potential_response_rank") == 32,
            "selected Airfoil potential response lost its exact-parent audit",
        )
    architecture = str(row.get("selected_architecture", ""))
    if "conforming_helmholtz_4" in architecture:
        parent = row.get("conforming_response_exact_parent_audit", {})
        require(
            parent.get("parent_tensor_identity") is True
            and parent.get("nonzero_message_parent_identity") is True
            and parent.get(
                "conforming_helmholtz_only_training_scope_verified"
            ) is True
            and parent.get("conforming_coefficient_nonzero_count") == 0
            and int(parent.get("conforming_subcells", 0)) == 4
            and parent.get("protected_test_trajectories_opened") == 0,
            "selected Airfoil conforming response lost its exact-parent audit",
        )
        if "conforming_helmholtz_4_code75" in architecture:
            require(
                int(parent.get("conforming_code_width", 0)) == 75
                and int(parent.get("complete_response_code_width", 0)) == 123,
                "selected Airfoil expanded response lost its 123D-code audit",
            )
    local = row["local_h20"]["local_core"]
    latent = row["local_h20"]["latentddm"]
    shuffled = row["local_h20"]["shuffled"]
    for metric in ("field", "vorticity", "interface", "divergence"):
        require(latent[metric] < local[metric], f"Airfoil lost Local on {metric}")
        require(latent[metric] < shuffled[metric], f"Airfoil shuffled gate failed on {metric}")
        if claim_tier == "strict_global_frontier":
            require(
                latent[metric]
                < min(values[metric] for values in row["global_h20"].values()),
                f"Airfoil global frontier gate failed on {metric}",
            )
    oracle = row.get("response_manifold_oracle")
    require(
        isinstance(oracle, dict)
        and oracle.get("selection_eligible") is False
        and oracle.get("uses_current_target") is True
        and all(
            math.isfinite(float(oracle[metric])) and float(oracle[metric]) >= 0.0
            for metric in (
                "field", "vorticity", "interface", "divergence",
                "drag", "lift",
            )
        ),
        "selected Airfoil response-manifold diagnostic is missing or invalid",
    )
    require("same_process_seconds" in row, "selected Airfoil timing is missing")
    require("one_step" in row, "selected Airfoil owned-core one-step audit is missing")
    one_step = row["one_step"]
    require(one_step["examples"] == 40, "Airfoil one-step sample count drifted")
    require(
        set(one_step["summary"]) == {
            "local_core", "latentddm", "shuffled", "init_sni_1",
        },
        "Airfoil one-step local method set drifted",
    )
    for metric in ("field", "vorticity", "interface", "divergence"):
        require(
            float(one_step["summary"]["latentddm"][metric])
            < float(one_step["summary"]["local_core"][metric])
            and float(one_step["summary"]["latentddm"][metric])
            < float(one_step["summary"]["shuffled"][metric]),
            f"Airfoil one-step {metric} communication gate failed",
        )
    # Teacher-forced wall loads and model-owned rollout loads answer different
    # questions.  The selected composer improves the four instantaneous field
    # diagnostics above, but Local/Core and one-sweep SNI retain the lower
    # teacher-forced integrated-force error.  The H20 rollout gate is audited
    # separately from the selected validation artifact.
    require(
        float(one_step["summary"]["local_core"]["force"])
        < float(one_step["summary"]["latentddm"]["force"])
        and float(one_step["summary"]["init_sni_1"]["force"])
        < float(one_step["summary"]["latentddm"]["force"]),
        "Airfoil teacher-forced force distinction drifted",
    )
    require(
        one_step["sni_operating_point"] == {
            "mode": "init_sni_1", "sweeps": 1,
            "initialization": "local", "tau": 0.125,
        },
        "Airfoil one-step Flux-SNI operating points drifted",
    )
    require(
        "init_sni_1" in row.get("sni_h20", {}),
        "selected Airfoil local-initialized SNI rollout is missing",
    )
    test = row.get("test_h20", {})
    require(
        set(test) == {"local_core", "latentddm", "sni", "point", "slice", "patch"},
        "Airfoil held-out comparison does not contain the six paper methods",
    )
    require(
        row.get("test_rollout_windows") == 60
        and len(row.get("test_trajectories", [])) == 6
        and row.get("test_uncertainty", {}).get("cluster_unit")
        == "held-out trajectory"
        and row.get("test_uncertainty", {}).get("confidence_level") == 0.95,
        "Airfoil held-out uncertainty contract drifted",
    )
    return {"status": row["status"], "local": local, "latentddm": latent}


def audit_manifest(*, allow_pending_airfoil: bool) -> dict[str, Any]:
    manifest = load(MANIFEST)
    require(
        manifest.get("contract") == "latentddm_workshop_production_evidence_manifest_v2",
        "wrong production manifest contract",
    )
    method = manifest["paper_method_contract"]
    require(method["local_endpoint_calls"] == 1, "paper no longer uses one endpoint call")
    require(method["assembly"] == "disjoint_owned_core", "paper assembly drifted")
    require(method["uses_partition_of_unity"] is False, "POU leaked into production")
    require(
        method["global_families"] == ["point", "slice", "patch"],
        "global family declaration drifted",
    )
    report = {
        "darcy": audit_darcy(manifest["darcy"]),
        "car": audit_car(
            manifest["car"],
            allow_pending_mechanism=allow_pending_airfoil,
        ),
        "airfoil": audit_airfoil(
            manifest["airfoil"], allow_pending=allow_pending_airfoil,
        ),
    }
    report["verified_source_hashes"] = audit_sources(manifest)
    return report


def same_file(source: Path, copied: Path) -> None:
    require(source.is_file(), f"missing figure source: {source}")
    require(copied.is_file(), f"missing submission figure: {copied}")
    require(sha256(source) == sha256(copied), f"figure copy drift: {copied.name}")


def row_decimals(row: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(r"(?<![A-Za-z0-9_])(?:\d+\.\d+|\.\d+)", row)
    ]


def require_numeric_row(
    manuscript: str,
    label_pattern: str,
    expected: list[float],
    *,
    atol: float = 5.1e-7,
) -> None:
    match = re.search(rf"^{label_pattern}\s*&.*$", manuscript, flags=re.MULTILINE)
    require(match is not None, f"missing quantitative row: {label_pattern}")
    observed = row_decimals(match.group(0))
    require(
        len(observed) >= len(expected),
        f"quantitative row is incomplete: {label_pattern}: {observed}",
    )
    for index, (actual, target) in enumerate(zip(observed, expected)):
        require(
            math.isclose(actual, target, rel_tol=0.0, abs_tol=atol),
            f"{label_pattern} column {index} drifted: {actual} != {target}",
        )


def audit_submission(*, allow_pending_airfoil: bool) -> dict[str, Any]:
    manuscript = MAIN.read_text(encoding="utf-8")
    # LaTeX can recover from a mismatched float environment and still emit a
    # PDF, so make structural balance an explicit release invariant rather
    # than relying only on the compiler exit status.
    environment_stack: list[str] = []
    for action, environment in re.findall(
        r"\\(begin|end)\{(figure\*?|table\*?|tabularx?|equation\*?|align\*?)\}",
        manuscript,
    ):
        if action == "begin":
            environment_stack.append(environment)
            continue
        require(
            bool(environment_stack) and environment_stack[-1] == environment,
            f"mismatched LaTeX environment: end{{{environment}}} after "
            f"{environment_stack[-1] if environment_stack else 'empty stack'}",
        )
        environment_stack.pop()
    require(
        not environment_stack,
        f"unclosed LaTeX environments: {environment_stack}",
    )
    settled_legacy = (
        "Local POU", "partition of unity", "Value+flux", ".553140",
        ".060665", ".202929", ".202817", "0.811/1.027",
        "same-cover patch", "pre-head mixer", "Zero-SNI", "TripNet",
        "Car surface &",
    )
    found = [token for token in settled_legacy if token.lower() in manuscript.lower()]
    require(not found, f"settled legacy evidence remains in manuscript: {found}")
    manifest = load(MANIFEST)
    darcy = manifest["darcy"]
    require_numeric_row(
        manuscript,
        r"Darcy",
        [
            darcy["local_core_relative_l2"],
            darcy["latentddm_relative_l2"],
            darcy["sni"]["sni_9_relative_l2"],
            darcy["global_three_seed"]["point"][
                "one_global_call_relative_l2_mean"
            ],
            darcy["global_three_seed"]["slice"][
                "one_global_call_relative_l2_mean"
            ],
            darcy["global_three_seed"]["patch"][
                "one_global_call_relative_l2_mean"
            ],
        ],
    )
    car = manifest["car"]["methods"]
    require_numeric_row(
        manuscript,
        r"Car",
        [
            car[name]["velocity_rel_l2"]["mean"]
            for name in (
                "Local/Core", "LatentDDM", "SNI",
                "Point", "Slice", "Patch",
            )
        ],
    )
    four_gpu = next(
        row for row in manifest["car"]["distributed_scaling"]["rows"]
        if int(row["world_size"]) == 4
    )
    rounded_speedup = f"{float(four_gpu['speedup']):.2f}\\times"
    rounded_efficiency = (
        f"{100.0 * float(four_gpu['parallel_efficiency']):.0f}\\%"
    )
    require(
        rounded_speedup in manuscript and rounded_efficiency in manuscript,
        "Car distributed scaling prose drifted from the measured 4-GPU audit",
    )
    if not allow_pending_airfoil:
        air_legacy = ("airfoil_sni_rollout_step20.pdf",)
        found = [token for token in air_legacy if token in manuscript]
        require(not found, f"legacy Airfoil evidence remains in manuscript: {found}")
        evidence = manifest["airfoil"]
        if evidence["selection"].get("claim_tier") == "complementary_validation":
            airfoil_section = re.search(
                r"\\subsection\{Moving [Aa]irfoil:[^}]*\}"
                r"(.*?)\\begin\{figure\}",
                manuscript,
                flags=re.DOTALL,
            )
            airfoil_claim = re.sub(
                r"\s+", " ", airfoil_section.group(1)
            ) if airfoil_section is not None else ""
            require(
                airfoil_section is not None
                and "best vorticity, interface, and force errors" in airfoil_claim
                and "Patch gives the best field error" in airfoil_claim
                and "Point the best divergence" in airfoil_claim,
                "complementary Airfoil selection is stated as a dominant result",
            )
        one = evidence["one_step"]["summary"]
        require_numeric_row(
            manuscript,
            r"Airfoil",
            [
                one["local_core"]["field"],
                one["latentddm"]["field"],
                one["init_sni_1"]["field"],
                evidence["global_one_step"]["point"]["field"],
                evidence["global_one_step"]["slice"]["field"],
                evidence["global_one_step"]["patch"]["field"],
            ],
        )
        test_h20 = evidence["test_h20"]
        require_numeric_row(
            manuscript,
            r"Airfoil--20",
            [
                test_h20[name]["field"]
                for name in (
                    "local_core", "latentddm", "sni", "point", "slice", "patch",
                )
            ],
        )
        appendix_match = re.search(
            r"\\label\{tab:airfoil-global-app\}(.*?)\\end\{table\}",
            manuscript,
            flags=re.DOTALL,
        )
        require(appendix_match is not None, "Airfoil H20 appendix table is missing")
        appendix = appendix_match.group(1)
        h20_rows = {
            "Local/Core": test_h20["local_core"],
            r"\lddm{}": test_h20["latentddm"],
            "SNI": test_h20["sni"],
            "Point": test_h20["point"],
            "Slice": test_h20["slice"],
            "Patch": test_h20["patch"],
        }
        for label, values in h20_rows.items():
            require_numeric_row(
                appendix,
                re.escape(label),
                [values[key] for key in (
                    "field", "vorticity", "interface", "divergence", "force",
                )],
                atol=5.1e-6,
            )
        oracle_match = re.search(
            r"\\label\{tab:airfoil-response-oracle-app\}"
            r"(.*?)\\end\{table\}",
            manuscript,
            flags=re.DOTALL,
        )
        require(
            oracle_match is not None,
            "Airfoil response-chart diagnostic table is missing",
        )
        oracle_table = oracle_match.group(1)
        oracle_rows = {
            r"Deployed \\lddm\{\} \(48D\)": evidence["local_h20"]["latentddm"],
            "48D response oracle": evidence["response_manifold_oracle"],
            "123D response oracle": evidence["expanded_response_manifold_oracle"],
            "Point": evidence["global_h20"]["point"],
        }
        for label, values in oracle_rows.items():
            require_numeric_row(
                oracle_table,
                label,
                [values[key] for key in (
                    "field", "vorticity", "interface", "divergence",
                )],
                atol=5.1e-6,
            )

    required = (
        "disjoint owned", "z_{G,i}^0", "field-first", "Point Transformer",
        "SNI", "Scope and limitations",
    )
    missing = [token for token in required if token not in manuscript]
    require(not missing, f"method/story fragments missing: {missing}")

    figure_sources = {
        "latentddm_claim_summary.pdf": ROOT / "codiffuse_Schwarz/figures/latentddm_claim_summary_v2/latentddm_claim_summary.pdf",
        "darcy_six_method_comparison.pdf": ROOT / "codiffuse_Schwarz/figures/unified_report/darcy_six_method_comparison.pdf",
        "car_six_method_volume.pdf": ROOT / "codiffuse_Schwarz/figures/unified_report/car_six_method_volume.pdf",
        "three_case_decompositions.pdf": ROOT / "codiffuse_Schwarz/figures/unified_report/three_case_decompositions.pdf",
        "low_data_transolver_car_darcy.pdf": ROOT / "codiffuse_Schwarz/figures/global_operator_audit_v1/low_data_transolver_car_darcy.pdf",
        "airfoil_six_method_test_rollout.pdf": ROOT / "codiffuse_Schwarz/figures/unified_report/airfoil_six_method_test_rollout.pdf",
    }
    for name, source in figure_sources.items():
        if (
            allow_pending_airfoil
            and name == "airfoil_six_method_test_rollout.pdf"
            and not source.is_file()
        ):
            continue
        same_file(source, SUBMISSION / "figures" / name)

    included = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}", manuscript))
    require(included == set(figure_sources),
            f"figure package/include mismatch: included={sorted(included)}")
    packaged = {path.name for path in (SUBMISSION / "figures").glob("*.pdf")}
    expected_packaged = set(figure_sources)
    if allow_pending_airfoil and not figure_sources[
        "airfoil_six_method_test_rollout.pdf"
    ].is_file():
        expected_packaged.remove("airfoil_six_method_test_rollout.pdf")
    require(packaged == expected_packaged,
            f"stale or missing packaged figures: {sorted(packaged ^ expected_packaged)}")

    darcy_meta = load(ROOT / "codiffuse_Schwarz/figures/unified_report/darcy_six_method_comparison.json")
    car_meta = load(ROOT / "codiffuse_Schwarz/figures/unified_report/car_six_method_volume.json")
    require(
        darcy_meta.get("contract") == "darcy_six_method_seven_column_visual_v3"
        and darcy_meta.get("assembly") == "disjoint_owned_core"
        and darcy_meta.get("overlap_averaging_or_pou") is False
        and darcy_meta.get("method_order") == [
            "Local/Core", "LatentDDM", "SNI", "Point", "Slice", "Patch",
        ]
        and darcy_meta.get("protected_test_cases_opened") == 0
        and darcy_meta.get("seam_selection", {}).get(
            "visual_appearance_used_for_selection"
        ) is False
        and darcy_meta.get("seam_selection", {}).get(
            "seam_error_change_fraction", 0.0
        ) < -0.20
        and darcy_meta.get("sni", {}).get("sweeps") == 9
        and darcy_meta.get("sni", {}).get("sweep0_is_exact_local_core") is True
        and darcy_meta.get("residual_visualization", {}).get("quantity")
        == "signed prediction error on one shared symmetric-log scale",
        "Darcy figure has the wrong method identity",
    )
    require(
        car_meta.get("contract") == "car_six_method_common_grid_seven_column_visual_v10"
        and car_meta.get("assembly") == "disjoint_owned_core"
        and car_meta.get("uses_partition_of_unity") is False
        and car_meta.get("method_order") == [
            "Local/Core", "LatentDDM", "SNI", "Point", "Slice", "Patch",
        ]
        and car_meta.get("protected_test_designs_opened") == 0,
        "Car figure has the wrong method identity",
    )
    require(
        car_meta.get("design_id") == "N_S_WWS_WM_328"
        and car_meta.get("wake_metric_region")
        == "valid, 4 <= x <= 12 m, sdf < 3 m"
        and car_meta.get("qualitative_selection", {}).get(
            "visual_appearance_used_for_selection"
        ) is False,
        "Car figure is not tied to the deterministic representative-design protocol",
    )
    if not allow_pending_airfoil:
        air_meta = load(ROOT / (
            "codiffuse_Schwarz/figures/unified_report/"
            "airfoil_six_method_test_rollout.json"
        ))
        require(
            air_meta.get("contract")
            == "airfoil_six_method_test_rollout_cluster_band_v1"
            and air_meta.get("assembly") == "disjoint_owned_core"
            and air_meta.get("uses_partition_of_unity") is False
            and air_meta.get("split") == "test"
            and air_meta.get("protected_test_trajectories_opened") == 6,
            "Airfoil figure has the wrong method identity",
        )
        require(
            air_meta.get("rollout_windows") == 60
            and air_meta.get("prediction_rows_per_method") == 1200
            and air_meta.get("rollouts_per_trajectory_requested") == 10,
            "Airfoil uncertainty figure does not cover the declared 60 windows",
        )
        require(
            air_meta.get("snapshot_steps") == [0, 1, 20],
            "Airfoil figure does not display the declared 0/1/20 sequence",
        )
        require(
            air_meta.get("plot_curve_metrics")
            == ["field", "vorticity", "force"]
            and air_meta.get("plot_step20_zoom") == "airfoil and near wake",
            "Airfoil figure does not use the declared separated diagnostics and step-20 zoom",
        )
        snapshot = air_meta.get("snapshot", {})
        require(
            snapshot.get("selection_role") == "qualitative display only"
            and snapshot.get("visual_appearance_used_for_selection") is False
            and snapshot.get("quantitative_summary_uses_all_windows") is True,
            "Airfoil qualitative window was not selected independently of its rendering",
        )
        expected_air_methods = {
            "local_core", "latentddm", "sni", "point", "slice", "patch",
        }
        require(
            set(air_meta.get("records", {})) == expected_air_methods,
            "Airfoil figure method set drifted",
        )
        require(
            all(
                len(air_meta.get("bands", {}).get(method, {}).get(metric, []))
                == 20
                for method in expected_air_methods
                for metric in (
                    "field", "vorticity", "interface", "divergence", "force",
                )
            ),
            "Airfoil colored uncertainty bands are incomplete",
        )
        require(
            air_meta.get("selection", {}).get("source")
            == "accepted validation-only selection",
            "Airfoil figure is not tied to the accepted selector",
        )
        claim_meta = load(ROOT / (
            "codiffuse_Schwarz/figures/latentddm_claim_summary_v2/"
            "latentddm_claim_summary.json"
        ))
        require(
            claim_meta.get("contract") == "latentddm_four_claim_summary_v3"
            and claim_meta.get("analytic_latency_projection") is False
            and claim_meta.get("airfoil_candidate_status")
            == "frozen-checkpoint held-out test",
            "Figure 2 is not the measured held-out evidence summary",
        )
        adaptation = claim_meta.get("adaptation", {})
        adaptation_names = {
            "LatentDDM composer": "LatentDDM field-token composer",
            "Point Transformer full": "Point Transformer full fine-tune",
            "Slice/Transolver full": "Transolver full fine-tune",
            "Patch Transformer full": "Patch Transformer full fine-tune",
        }
        require(
            set(adaptation) == set(adaptation_names),
            "Figure 2 Darcy reuse family set drifted",
        )
        manifest_adaptation = manifest["darcy"]["domain_extension"]
        for label, source_name in adaptation_names.items():
            require(
                {int(row["budget"]) for row in adaptation[label]}
                == {4, 16, 64},
                f"Figure 2 Darcy reuse budgets drifted for {label}",
            )
            for row in adaptation[label]:
                budget = str(int(row["budget"]))
                source = manifest_adaptation[source_name][budget]
                close(
                    row["adapted_relative_l2_mean"],
                    source["relative_l2_mean"], atol=1.0e-10,
                )
                close(
                    row["adapted_relative_l2_sample_std"],
                    source["relative_l2_sample_std"], atol=1.0e-10,
                )

        normalized = claim_meta.get(
            "airfoil_rollout_normalized_to_local_core", {}
        )
        air_names = {
            "Local/Core": "local_core", "LatentDDM": "latentddm",
            "SNI": "sni", "Point": "point", "Slice": "slice",
            "Patch": "patch",
        }
        require(
            set(normalized) == set(air_names),
            "Figure 2 Airfoil method set drifted",
        )
        local_test = manifest["airfoil"]["test_h20"]["local_core"]
        for label, source_name in air_names.items():
            source = manifest["airfoil"]["test_h20"][source_name]
            require(
                set(normalized[label]) == {
                    "all_step_field", "all_step_vorticity",
                    "all_step_interface", "all_step_divergence",
                    "all_step_force",
                },
                f"Figure 2 Airfoil metric set drifted for {label}",
            )
            for metric in (
                "field", "vorticity", "interface", "divergence", "force",
            ):
                close(
                    normalized[label][f"all_step_{metric}"],
                    source[metric] / local_test[metric],
                    atol=1.0e-10,
                )
        error_latency = claim_meta.get("error_latency", {})
        require(
            set(error_latency) == {"Darcy", "Car", "Airfoil-20"},
            "Figure 2 error--latency case set drifted",
        )
        timing_cases = {
            "Darcy": ("darcy", 1, (0, 1, 4, 8, 9)),
            "Car": ("car", 1, (0, 1, 2, 4)),
            "Airfoil-20": ("airfoil", 20, (0, 1, 2, 4)),
        }
        for label, (manifest_case, multiplier, sweeps) in timing_cases.items():
            record = error_latency[label]
            require(
                int(record.get("accumulated_steps", 0)) == multiplier,
                f"Figure 2 accumulated-step contract drifted for {label}",
            )
            latent_seconds = multiplier * manifest[manifest_case][
                "same_process_seconds"
            ]["latentddm"]
            close(record["latentddm"]["seconds"], latent_seconds, atol=1e-9)
            points = record["sni_path"]
            require(
                tuple(int(point["sweep"]) for point in points) == sweeps,
                f"Figure 2 SNI sweeps drifted for {label}",
            )
            for point in points:
                sweep = int(point["sweep"])
                timing_name = "local_core" if sweep == 0 else f"init_sni_{sweep}"
                expected_seconds = multiplier * manifest[manifest_case][
                    "same_process_seconds"
                ][timing_name]
                close(point["seconds"], expected_seconds, atol=1e-9)
                close(
                    point["latency_over_latentddm"],
                    point["seconds"] / record["latentddm"]["seconds"],
                    atol=1e-12,
                )
                close(
                    point["error_over_latentddm"],
                    point["field_error"] / record["latentddm"]["field_error"],
                    atol=1e-12,
                )

    if (
        allow_pending_airfoil
        and not figure_sources["airfoil_six_method_test_rollout.pdf"].is_file()
    ):
        return {
            "status": "development_airfoil_pending",
            "manuscript_references_all_release_figures": True,
            "settled_figure_copies_verified": len(expected_packaged),
            "release_pdf_audit_deferred": True,
        }

    require(PDF.is_file(), "submission PDF has not been compiled")
    compile_inputs = [
        MAIN, SUBMISSION / "references.bib", SUBMISSION / "neurips_2026.sty",
        *(SUBMISSION / "figures" / name for name in figure_sources),
    ]
    require(
        PDF.stat().st_mtime_ns
        >= max(path.stat().st_mtime_ns for path in compile_inputs),
        "submission PDF predates a TeX, bibliography, style, or figure input",
    )
    document = fitz.open(PDF)
    references_pages = [
        index + 1 for index, page in enumerate(document)
        if "References" in page.get_text("text")
    ]
    require(
        len(references_pages) == 1 and references_pages[0] <= 9,
        f"references must begin after at most eight main pages: {references_pages}",
    )
    log = (SUBMISSION / "main.log").read_text(encoding="utf-8")
    forbidden_log = ("undefined references", "undefined citations", "multiply defined", "Overfull")
    found_log = [token for token in forbidden_log if token.lower() in log.lower()]
    require(not found_log, f"LaTeX release warnings remain: {found_log}")
    return {
        "pdf_pages": len(document),
        "main_pages": references_pages[0] - 1,
        "references_page": references_pages[0],
        "figures_verified": len(figure_sources),
    }


def main() -> None:
    args = parse_args()
    report = {
        "manifest": audit_manifest(allow_pending_airfoil=args.allow_pending_airfoil),
        "submission": audit_submission(allow_pending_airfoil=args.allow_pending_airfoil),
        "release_strict": not args.allow_pending_airfoil,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
