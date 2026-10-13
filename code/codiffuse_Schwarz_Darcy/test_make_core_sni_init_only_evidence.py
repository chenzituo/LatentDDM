import json

from .make_core_sni_init_only_evidence import CONTRACT, build


def test_build_keeps_only_local_core_initialization() -> None:
    source = {
        "contract": "darcy_core_ras_sni_two_initializations_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "cases": 1,
        "data": "/data.h5",
        "local_checkpoint": "/local.pt",
        "local_checkpoint_sha256": "abc",
        "assembly": "core_ras",
        "taus": [1.0],
        "checkpoints": [0, 1],
        "max_sweeps": 1,
        "local_initialization_equals_sni_sweep0_max_abs": 0.0,
        "summary": {
            "init_local_core/tau_1.000/sweep_0000": {
                "relative_l2_mean": 0.5,
            },
            "init_local_core/tau_1.000/sweep_0001": {
                "relative_l2_mean": 0.1,
            },
            "init_zero/tau_1.000/sweep_0001": {
                "relative_l2_mean": 0.5,
            },
        },
        "rows": [
            {"initialization": "local_core", "relative_l2": 0.5},
            {"initialization": "local_core", "relative_l2": 0.1},
            {"initialization": "zero", "relative_l2": 0.5},
        ],
    }
    result = build(source, "source-hash")
    assert result["contract"] == CONTRACT
    assert result["initialization"] == "local_core"
    assert set(result["summary"]) == {
        "init_local_core/tau_1.000/sweep_0000",
        "init_local_core/tau_1.000/sweep_0001",
    }
    assert all(row["initialization"] == "local_core" for row in result["rows"])
    assert "init_zero" not in json.dumps(result)
    assert result["best_validation_field"]["relative_l2_mean"] == 0.1
