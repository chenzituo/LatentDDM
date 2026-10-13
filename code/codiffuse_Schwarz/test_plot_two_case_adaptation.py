import json

import pytest

from .plot_two_case_adaptation import selected_validation_field_mean


def selection_payload() -> dict[str, object]:
    return {
        "contract": "airfoil_pitch_rate_ood_adaptation_selection_v1",
        "selection_partition": "target_validation",
        "protected_test_trajectories_opened": 0,
        "seeds": [1, 2, 3],
        "selections": {
            "latent": {
                "budget16": {
                    "1": {
                        "decision": "adapted",
                        "source_field_h20": 0.4,
                        "adapted_field_h20": 0.3,
                    },
                    "2": {
                        "decision": "source_retained",
                        "source_field_h20": 0.4,
                        "adapted_field_h20": 0.5,
                    },
                    "3": {
                        "decision": "adapted",
                        "source_field_h20": 0.4,
                        "adapted_field_h20": 0.2,
                    },
                }
            }
        },
    }


def test_selected_validation_mean_follows_adapt_or_retain(tmp_path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection_payload()), encoding="utf-8")
    assert selected_validation_field_mean(
        path, method="latent", budget=16,
    ) == pytest.approx(0.3)


def test_selected_validation_mean_rejects_protected_access(tmp_path) -> None:
    payload = selection_payload()
    payload["protected_test_trajectories_opened"] = 1
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="audit contract"):
        selected_validation_field_mean(path, method="latent", budget=16)
