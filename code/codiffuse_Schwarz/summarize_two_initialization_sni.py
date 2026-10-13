"""Summarize the matched POU-start and zero-start SNI validation runs.

This script deliberately keeps initialization separate from the recurrence.
For each case it selects relaxation and stopping depth using the declared
validation field metric, retains the complete curve at the selected
relaxation, and also reports a same-(tau, sweep) initialization contrast.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "codiffuse_Schwarz/figures/"
    "sni_two_initialization_audit_v1/summary.json"
)

CASES = {
    "Darcy": {
        "path": ROOT / (
            "codiffuse_Schwarz_Darcy/outputs/"
            "darcy_weighted_pou_sni_two_init_v1.json"
        ),
        "metric": "relative_l2_mean",
        "row_metric": "relative_l2",
        "pou": "local_pou",
        "zero": "zero",
        "identity": "case_index",
    },
    "DrivAerNet++": {
        "path": ROOT / (
            "codiffuse_Schwarz_Car/outputs/"
            "car_flux_sni_two_init_v1_25designs.json"
        ),
        "metric": "native_speed_relative_l2_mean",
        "row_metric": "native_speed_relative_l2",
        "pou": "local_pou",
        "zero": "zero",
        "identity": "design",
    },
    "Airfoil": {
        "path": ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/"
            "airfoil_flux_sni_patch_endpoint_two_init_v1_40states.json"
        ),
        "metric": "field_relative_l2_mean",
        "row_metric": "field_relative_l2",
        "pou": "local",
        "zero": "zero",
        "identity": ("trajectory", "index"),
    },
}

KEY = re.compile(
    r"^init_(?P<initialization>[^/]+)/tau_(?P<tau>[0-9.]+)/"
    r"sweep_(?P<sweep>[0-9]+)$"
)


def parse_key(key: str) -> dict[str, str | float | int]:
    match = KEY.match(key)
    if match is None:
        raise ValueError(f"unexpected SNI summary key: {key}")
    return {
        "initialization": match.group("initialization"),
        "tau": float(match.group("tau")),
        "sweep": int(match.group("sweep")),
    }


def configurations(payload: dict, metric: str, initialization: str) -> list[dict]:
    result = []
    for key, values in payload["summary"].items():
        config = parse_key(key)
        if config["initialization"] != initialization:
            continue
        result.append({
            **config,
            "key": key,
            "error": float(values[metric]),
            "seconds": float(values["seconds_mean"]),
            "local_nfe": float(values["local_nfe_mean"]),
            "batched_local_calls": float(values["batched_local_calls_mean"]),
        })
    return sorted(result, key=lambda row: (row["tau"], row["sweep"]))


def row_identity(row: dict, fields: str | tuple[str, ...]):
    if isinstance(fields, str):
        return row[fields]
    return tuple(row[field] for field in fields)


def paired_initialization_contrast(
    payload: dict,
    specification: dict,
    tau: float,
    sweep: int,
) -> dict:
    metric = specification["row_metric"]
    identity = specification["identity"]
    by_initialization = {}
    for initialization in (specification["pou"], specification["zero"]):
        by_initialization[initialization] = {
            row_identity(row, identity): float(row[metric])
            for row in payload["rows"]
            if row["initialization"] == initialization
            and np.isclose(float(row["tau"]), tau)
            and int(row["sweep"]) == sweep
        }
    shared = sorted(
        set(by_initialization[specification["pou"]])
        & set(by_initialization[specification["zero"]]),
        key=str,
    )
    differences = np.asarray([
        by_initialization[specification["zero"]][item]
        - by_initialization[specification["pou"]][item]
        for item in shared
    ], dtype=np.float64)
    if not len(differences):
        raise RuntimeError("no paired initialization rows")
    rng = np.random.default_rng(20260808)
    bootstrap = differences[
        rng.integers(0, len(differences), size=(10_000, len(differences)))
    ].mean(axis=1)
    return {
        "tau": tau,
        "sweep": sweep,
        "pairs": len(shared),
        "mean_zero_minus_pou": float(differences.mean()),
        "bootstrap_95_interval": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "fraction_zero_better": float(np.mean(differences < 0.0)),
    }


def summarize_case(name: str, specification: dict) -> dict:
    path = specification["path"]
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("protected_test_cases_opened", 0) not in (0, False):
        raise ValueError(f"{name}: protected Darcy test data were opened")
    if payload.get("protected_test_designs_opened", 0) not in (0, False):
        raise ValueError(f"{name}: protected Car test data were opened")
    if payload.get("protected_test_trajectories_opened", 0) not in (0, False):
        raise ValueError(f"{name}: protected Airfoil test data were opened")

    curves = {}
    best = {}
    for label, initialization in (
        ("Init-SNI", specification["pou"]),
        ("Zero-SNI", specification["zero"]),
    ):
        candidates = [
            row for row in configurations(
                payload, specification["metric"], initialization,
            )
            if row["sweep"] > 0
        ]
        selected = min(candidates, key=lambda row: row["error"])
        best[label] = selected
        curves[label] = [
            row for row in candidates
            if np.isclose(row["tau"], selected["tau"])
        ]

    # Hold tau and depth fixed at the validation-optimal Init-SNI point.  This
    # is the clean causal contrast for initialization; the separate best rows
    # above give each baseline its strongest validation setting.
    matched = paired_initialization_contrast(
        payload, specification,
        float(best["Init-SNI"]["tau"]), int(best["Init-SNI"]["sweep"]),
    )
    return {
        "source": str(path),
        "contract": payload["contract"],
        "selection_split": payload["split"],
        "primary_metric": specification["metric"],
        "assembly": payload.get("assembly")
        or payload.get("configuration", {}).get("assembly"),
        "cost_definition": payload["cost_definition"],
        "best_validation_configuration": best,
        "selected_tau_curves": curves,
        "matched_initialization_contrast_at_pou_optimum": matched,
    }


def main() -> None:
    missing = [str(spec["path"]) for spec in CASES.values()
               if not spec["path"].exists()]
    if missing:
        raise FileNotFoundError("pending SNI outputs:\n" + "\n".join(missing))
    result = {
        "contract": "three_case_weighted_pou_sni_initialization_audit_v1",
        "selection_policy": (
            "tau and depth minimize the declared field metric on validation; "
            "the matched contrast fixes both at the Init-SNI optimum"
        ),
        "protected_test_examples_opened": 0,
        "cases": {
            name: summarize_case(name, specification)
            for name, specification in CASES.items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        name: values["best_validation_configuration"]
        for name, values in result["cases"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
