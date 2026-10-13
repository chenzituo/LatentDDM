"""Build the frozen local-operator and SNI distribution contract.

The paper compares the same *execution order* in all three cases, but the
state presented to the frozen local map is case specific.  This manifest
records those differences explicitly, including whether local pretraining
ever saw self-generated interface states.  It is evidence metadata; it does
not run an experiment or open a protected split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / (
    "codiffuse_Schwarz/figures/unified_report/local_bvp_contracts_v1.json"
)

PATHS = {
    "darcy_local": ROOT / (
        "codiffuse_Schwarz_Darcy/outputs/sni_face_local_seed20260806/"
        "checkpoint.pt"
    ),
    "darcy_composer": ROOT / (
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "budget64/seed20260811/checkpoint_best.pt"
    ),
    "darcy_matched": ROOT / (
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "matched_normalized_overlap_sni_128.json"
    ),
    "car_local": ROOT / (
        "codiffuse_Schwarz_Car/outputs/"
        "car_boundary_local_state_face4_warm_seed20260811/checkpoint_best.pt"
    ),
    "car_composer": ROOT / (
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "checkpoint_best.pt"
    ),
    "car_matched": ROOT / (
        "codiffuse_Schwarz_Car/outputs/input_correction_response_selected_v1/"
        "matched_validation_sni_v1.json"
    ),
    "air_local": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/"
        "expanded52_shared_projection_recurrent_h4/checkpoint.pt"
    ),
    "air_composer": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "checkpoint_best.pt"
    ),
    "air_matched": ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3/"
        "final_holdout_h20_five_starts_v3.json"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(name: str) -> dict[str, Any]:
    path = PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a checkpoint dictionary")
    return value


def artifact(name: str) -> dict[str, Any]:
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
    darcy_local = checkpoint("darcy_local")
    darcy_composer = checkpoint("darcy_composer")
    darcy_matched = artifact("darcy_matched")
    car_local = checkpoint("car_local")
    car_composer = checkpoint("car_composer")
    car_matched = artifact("car_matched")
    air_local = checkpoint("air_local")
    air_composer = checkpoint("air_composer")
    air_matched = artifact("air_matched")

    require(
        darcy_local.get("contract") == "darcy_sni_face_boundary_to_solution_v1",
        "Darcy local contract drift",
    )
    require(
        darcy_composer.get("local_checkpoint_sha256")
        == sha256(PATHS["darcy_local"])
        == darcy_matched.get("local_checkpoint_sha256"),
        "Darcy local lineage drift",
    )
    require(
        car_local.get("sni_recurrence_used_for_training") is False,
        "Car local BVP unexpectedly used recurrent SNI training",
    )
    require(
        car_composer.get("boundary_local_checkpoint_sha256")
        == sha256(PATHS["car_local"]),
        "Car local lineage drift",
    )
    require(
        car_matched.get("boundary_local_checkpoint")
        == str(PATHS["car_local"].resolve()),
        "Car matched evaluator uses another local BVP",
    )
    require(
        air_composer.get("response_checkpoint_sha256") is not None
        and air_matched.get("composer_checkpoint_sha256")
        == sha256(PATHS["air_composer"]),
        "Airfoil selected composer lineage drift",
    )
    require(
        darcy_matched.get("assembly")
        == car_matched.get("assembly")
        == air_matched.get("assembly")
        == "normalized_overlap",
        "assembly drift across cases",
    )

    records = {
        "Darcy": {
            "local_training_task": "face-resolved local Dirichlet BVP",
            "training_boundary_distribution": (
                "directional physical and artificial face values recorded from "
                "train/validation global solutions"
            ),
            "local_core_initialization": (
                "zero interior state; physical outer boundary is enforced and "
                "artificial interior faces start from zero"
            ),
            "self_generated_interface_states_in_local_pretraining": False,
            "self_generated_coverage": "none",
            "core": 32,
            "halo": 4,
            "face_support": "full directional face fields",
            "correction_chart": "four tangential modes per face",
            "sni_relaxation": float(darcy_matched["tau"]),
            "assembly": "normalized_overlap",
        },
        "Car": {
            "local_training_task": "state-trace-conditioned cube BVP",
            "training_boundary_distribution": (
                "target velocity/pressure traces from 121 training cars"
            ),
            "local_core_initialization": (
                "assembled frozen endpoint field; the first SNI trace is "
                "extracted from that Local/Core prediction"
            ),
            "self_generated_interface_states_in_local_pretraining": False,
            "self_generated_coverage": (
                "none in local-BVP optimization; this target-trace to "
                "self-trace shift is measured by the response-oracle audit"
            ),
            "core": int(car_composer["configuration"]["core"]),
            "halo": int(car_composer["configuration"]["halo"]),
            "face_support": (
                f"{int(car_local['args']['face_width'])}-cell face bands"
            ),
            "correction_chart": (
                f"order-{int(car_composer['configuration']['trace_order'])} "
                "tangential polynomial state update"
            ),
            "sni_relaxation": float(car_matched["sni_tau"]),
            "assembly": "normalized_overlap",
        },
        "Airfoil": {
            "local_training_task": "state-conditioned local transition",
            "training_boundary_distribution": (
                "one-step states and short model-owned roll-ins from the 41 "
                "training trajectories in the fixed 52-trajectory split"
            ),
            "local_core_initialization": (
                "current and historical predicted state; SNI sweep zero is the "
                "same frozen transition as Local/Core"
            ),
            "self_generated_interface_states_in_local_pretraining": True,
            "self_generated_coverage": (
                f"recurrent horizon {max(air_local['args']['curriculum_horizons'])} "
                f"with BPTT {int(air_local['args']['bptt_steps'])}; the composer "
                "also uses train-only short roll-ins"
            ),
            "core": 32,
            "halo": int(air_composer["configuration"]["halo"]),
            "face_support": (
                f"{int(air_composer['configuration']['face_width'])}-cell "
                "oriented disagreement bands"
            ),
            "correction_chart": "48-dimensional state-input response chart",
            "sni_relaxation": float(air_matched["sni_tau"]),
            "assembly": "normalized_overlap",
        },
    }
    return {
        "contract": "latentddm_three_case_local_bvp_contract_v1",
        "purpose": (
            "separate local response-family capacity from distribution shift "
            "and recurrence contractivity"
        ),
        "shared_execution": {
            "local_core": "one parallel frozen-local round",
            "latentddm": "local round, communication, corrected local round",
            "init_sni": "Local/Core followed by dependent boundary updates",
            "assembly": "normalized_overlap",
            "pde_residual_used_at_inference": False,
        },
        "records": records,
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in PATHS.items()
        },
    }


def main() -> None:
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
