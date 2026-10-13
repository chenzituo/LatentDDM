"""Opt-in Flow-Matching and PnP-Schwarz research path.

These APIs preserve the original generative implementation, but they are not
the inference mechanism used by the retained airfoil checkpoint.  The package
root intentionally exports the deterministic direct-endpoint path instead.
"""

from .hierarchy import integrate_multilevel_flow, multilevel_velocity, pnp_multilevel_sample
from .model import (
    CoDiffuseOperator,
    ExperimentalFlowOperator,
    LocalFlowOperator,
    NeuralEndpointProjector,
)
from .schwarz import (
    FlowSample,
    integrate_flow,
    local_velocity,
    pnp_schwarz_sample,
    schwarz_velocity,
)
from .training import LossReport, combined_loss, flow_matching_loss, predict_next_state


__all__ = [
    "CoDiffuseOperator",
    "ExperimentalFlowOperator",
    "FlowSample",
    "LocalFlowOperator",
    "LossReport",
    "NeuralEndpointProjector",
    "combined_loss",
    "flow_matching_loss",
    "integrate_flow",
    "integrate_multilevel_flow",
    "local_velocity",
    "multilevel_velocity",
    "pnp_multilevel_sample",
    "pnp_schwarz_sample",
    "predict_next_state",
    "schwarz_velocity",
]
