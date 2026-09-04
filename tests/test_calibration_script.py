"""The long calibration harness must fail honestly and retain its controls."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "calibration_script", ROOT / "scripts" / "calibration.py"
)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)


def _row(*, feature_rate=0.04, study_rate=0.04, support_rate=1.0, blocked=True):
    return {
        "generator": "time_correlated",
        "tau": 20.0,
        "metric": "jsd",
        "alpha": 0.05,
        "feature_rate": feature_rate,
        "study_rate": study_rate,
        "support_rate": support_rate,
        "blocked": blocked,
    }


def test_a_supported_correlated_null_inside_the_band_passes():
    gates = CALIBRATION.calibration_gates({"correlation": [_row()]})

    assert gates["passed"]
    assert gates["rows"][0]["maximum_false_positive_rate"] == 0.10


def test_withheld_inference_cannot_masquerade_as_zero_false_positives():
    gates = CALIBRATION.calibration_gates(
        {"correlation": [_row(feature_rate=0.0, study_rate=0.0, support_rate=0.9)]}
    )

    assert not gates["passed"]


def test_the_known_frame_permutation_failure_is_only_a_negative_control():
    gates = CALIBRATION.calibration_gates(
        {
            "correlation": [
                _row(),
                _row(feature_rate=0.95, study_rate=1.0, blocked=False),
            ]
        }
    )

    assert gates["passed"]
    control = gates["rows"][1]
    assert control["role"] == "negative_control"
    assert control["passed"] is None


def test_an_inflated_corrected_null_fails():
    gates = CALIBRATION.calibration_gates(
        {"correlation": [_row(feature_rate=0.11, study_rate=0.08)]}
    )

    assert not gates["passed"]
