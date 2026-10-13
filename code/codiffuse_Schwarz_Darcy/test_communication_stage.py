from __future__ import annotations

import pytest
import torch

from codiffuse_Schwarz_Darcy.adapter import (
    COMMUNICATION_CONTRACTS,
    DarcyAdapter,
    DarcyStatistics,
    build_model,
    checkpoint_communication_stage,
    resolve_communication_stage,
)
from codiffuse_Schwarz_Darcy.train_communicated import initialize_from_local
from codiffuse_Schwarz_Darcy.evaluate import checkpoint_local_model_state


def adapter() -> DarcyAdapter:
    return DarcyAdapter(DarcyStatistics(
        solution_mean=0.0, solution_std=1.0,
        logk_mean=0.0, logk_std=1.0,
        source_mean=0.0, source_std=1.0,
        bc_mean=0.0, bc_std=1.0, sampled_cases=1,
    ))


@pytest.mark.parametrize(
    ("stage", "proposal", "projector"),
    (("none", False, False), ("proposal", True, False), ("both", True, True)),
)
def test_explicit_communication_placement(
    stage: str, proposal: bool, projector: bool,
) -> None:
    model = build_model(adapter(), communication_stage=stage)
    assert (model.proposal.grid_communicator is not None) is proposal
    assert model.projector is not None
    assert (model.projector.grid_communicator is not None) is projector


def test_legacy_boolean_maps_to_historical_both_stage() -> None:
    assert resolve_communication_stage(None, True) == "both"
    assert resolve_communication_stage(None, False) == "none"
    model = build_model(adapter(), grid_communication=True)
    assert model.proposal.grid_communicator is not None
    assert model.projector is not None
    assert model.projector.grid_communicator is not None
    with pytest.raises(ValueError, match="conflicts"):
        build_model(
            adapter(), communication_stage="proposal", grid_communication=True,
        )


def test_local_initialization_adds_only_requested_communicator() -> None:
    local = build_model(adapter(), communication_stage="none")
    checkpoint = {
        "contract": "darcy_arbitrary_current_direct_endpoint_v1",
        "model": local.state_dict(),
    }
    proposal_model = build_model(adapter(), communication_stage="proposal")
    report = initialize_from_local(proposal_model, checkpoint, "proposal")
    assert report["proposal_communication_tensors"] > 0
    assert report["projector_communication_tensors"] == 0

    both_model = build_model(adapter(), communication_stage="both")
    report = initialize_from_local(both_model, checkpoint, "both")
    assert report["proposal_communication_tensors"] > 0
    assert report["projector_communication_tensors"] > 0


def test_old_two_stage_checkpoint_remains_loadable_and_relabelled() -> None:
    old_model = build_model(adapter(), communication_stage="both")
    old_checkpoint = {
        "contract": "darcy_full_domain_relative_communication_v2",
        "model": old_model.state_dict(),
    }
    assert checkpoint_communication_stage(old_checkpoint) == "both"
    rebuilt = build_model(
        adapter(), communication_stage=checkpoint_communication_stage(old_checkpoint),
    )
    rebuilt.load_state_dict(old_checkpoint["model"])
    assert COMMUNICATION_CONTRACTS["both"] \
        == "darcy_full_domain_both_stage_communication_v3"


def test_new_proposal_checkpoint_stage_is_explicit() -> None:
    checkpoint = {
        "contract": COMMUNICATION_CONTRACTS["proposal"],
        "communication_stage": "proposal",
        "model_spec": {"communication_stage": "proposal"},
    }
    assert checkpoint_communication_stage(checkpoint) == "proposal"


def test_composer_evaluation_extracts_exact_embedded_local_endpoint() -> None:
    checkpoint = {
        "contract": "darcy_operator_only_one_message_pou_v1",
        "model": {"local.block.weight": torch.ones(2),
                  "composer.block.weight": torch.zeros(2)},
    }
    stage, state, source = checkpoint_local_model_state(checkpoint)
    assert stage == "none"
    assert list(state) == ["block.weight"]
    assert source == "embedded_finetuned_local_endpoint"
