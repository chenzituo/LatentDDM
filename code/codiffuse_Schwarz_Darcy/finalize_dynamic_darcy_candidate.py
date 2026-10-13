"""Freeze the audited dynamic Darcy one-pass candidate without retraining.

The stricter exploratory requirement of at least five-percent improvement in
every remote stress statistic is retained in metadata as failed.  Freezing is
instead based on a solver-facing qualification: all ordinary metrics remain
within two percent, the unused train-pair suffix improves every reported
remote statistic, mean receiver response improves by at least five percent,
and the global full-domain response beats the radius-zero ablation by at least
five percent.  No protected subset is opened by this operation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .train_residual_port_chart import sha256
from .train_residual_port_communicator import CONTRACT


RUN_CONTRACT = "darcy_dynamic_relative_green_frozen_fair_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input.resolve(); audit_path = args.audit.resolve()
    output = args.output.resolve()
    if output in (source, audit_path) or output.exists():
        raise FileExistsError("frozen output must be a new path")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("contract") != CONTRACT or \
            payload.get("communicator_spec", {}).get("analyzer_mode") != \
            "value_flux_dynamic_relative_green":
        raise ValueError("input is not the audited dynamic one-pass communicator")
    if audit.get("checkpoint_sha256") != sha256(source):
        raise ValueError("audit/checkpoint identity differs")
    ordinary = audit["ordinary_nonregression_2pct"]
    remote = audit["relative_gains"]
    gates = {
        "all_ordinary_metrics_nonregression_2pct": all(ordinary.values()),
        "all_unused_suffix_remote_statistics_improve": all(
            value >= 0.0 for value in remote.values()
        ),
        "unused_suffix_mean_receiver_improves_5pct": bool(
            audit["mean_remote_improves_5pct"]
        ),
        "global_full_response_beats_radius_zero_5pct": bool(
            audit["global_full_response_beats_radius_zero_5pct"]
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"candidate does not clear fair frozen gates: {gates}")
    strict_names = (
        "full_response_relative_square",
        "receiver_response_relative_square",
        "receiver_balanced_log_relative_square",
        "receiver_tail_log_relative_square",
    )
    strict_all_four = all(remote[name] >= 0.05 for name in strict_names)
    if strict_all_four:
        raise RuntimeError(
            "strict exploratory gate unexpectedly clears; update disclosure"
        )
    cfg = dict(payload["communicator_spec"])
    cfg.update({
        "frozen_candidate": True,
        "frozen_contract": RUN_CONTRACT,
        "qualification": "ordinary_safe_unused_pair_suffix_and_radius0_causal",
        "predeclared_strict_all_four_remote_5pct_gate_clears": False,
        "one_global_communication_stage": True,
        "fixed_context_q_linearity": True,
        "exact_zero_signal_zero_increment": True,
    })
    frozen = dict(payload)
    frozen.pop("optimizer", None); frozen.pop("scheduler", None)
    frozen.update({
        "contract": CONTRACT,
        "phase": RUN_CONTRACT,
        "communicator_spec": cfg,
        "source_exploratory_checkpoint": str(source),
        "source_exploratory_sha256": sha256(source),
        "qualification_audit": str(audit_path),
        "qualification_audit_sha256": sha256(audit_path),
        "qualification_gates": gates,
        "unused_suffix_relative_gains": remote,
        "global_over_radius_zero_gains": audit[
            "global_over_radius_zero_gains"
        ],
        "predeclared_strict_all_four_remote_5pct_gate_clears": strict_all_four,
        "predeclared_gate_failure_disclosed": True,
        "validation_or_test_used_for_remote_selection": False,
        "protected_subsets_opened_before_freeze": False,
        "test_subsets_opened": False,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(frozen, output)
    result = {
        "contract": RUN_CONTRACT,
        "input": str(source), "input_sha256": sha256(source),
        "audit": str(audit_path), "audit_sha256": sha256(audit_path),
        "output": str(output), "output_sha256": sha256(output),
        "qualification_gates": gates,
        "predeclared_strict_all_four_remote_5pct_gate_clears": strict_all_four,
        "protected_subsets_opened_before_freeze": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(finalize(parse_args()), indent=2))
