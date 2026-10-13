from __future__ import annotations

from argparse import Namespace

import pytest

from .evaluate_global import evaluate_global
from .evaluate_input_correction_latentddm import evaluate as evaluate_latent
from .evaluate_weighted_pou_sni_initializations import evaluate as evaluate_sni


def test_latent_evaluator_fails_closed_before_data_access(tmp_path) -> None:
    args = Namespace(
        output=tmp_path / "result.json",
        sni_sweeps=(0,), batch_size=1, max_cases=1, tau=1.0,
        split="test", allow_protected_test=False,
    )
    with pytest.raises(ValueError, match="allow-protected-test"):
        evaluate_latent(args)


def test_sni_evaluator_fails_closed_before_data_access(tmp_path) -> None:
    args = Namespace(
        output=tmp_path / "result.json",
        checkpoints=(0,), max_sweeps=0, taus=(1.0,),
        initializations=("local_core",), split="test",
        allow_protected_test=False,
    )
    with pytest.raises(ValueError, match="allow-protected-test"):
        evaluate_sni(args)


def test_global_evaluator_fails_closed_before_checkpoint_access() -> None:
    args = Namespace(
        sweeps=1, relaxation=1.0, groups=("test",),
        allow_protected_test=False,
    )
    with pytest.raises(ValueError, match="allow-protected-test"):
        evaluate_global(args)
