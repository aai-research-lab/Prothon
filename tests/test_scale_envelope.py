"""Portable performance gates must fail on the original regression shape."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "scale_envelope_script", ROOT / "scripts" / "scale_envelope.py"
)
assert SPEC is not None and SPEC.loader is not None
SCALE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCALE)


def _measurement(*, task="cbcn", frames=8000, gb=0.65, status="ok"):
    return {
        "axis": "frames",
        "residues": 100,
        "frames": frames,
        "task": task,
        "status": status,
        "seconds": 1.0 if status == "ok" else None,
        "gb": gb if status == "ok" else None,
        "wall_seconds": 1.1,
        "error": None if status == "ok" else status,
    }


def _scaling(*, frame_cbcn=1.01, frame_compare=0.20, residue_cbcn=2.03, residue_compare=1.18):
    return {
        "frames": {
            "cbcn": {"exponent": frame_cbcn, "points": 3},
            "compare": {"exponent": frame_compare, "points": 3},
            "mmd": {"exponent": 0.50, "points": 3},
            "c2st": {"exponent": 0.40, "points": 3},
        },
        "residues": {
            "cbcn": {"exponent": residue_cbcn, "points": 3},
            "compare": {"exponent": residue_compare, "points": 4},
            "mmd": {"exponent": 1.0, "points": 4},
            "c2st": {"exponent": 1.2, "points": 4},
        },
    }


def test_fit_exponent_recovers_a_known_power_law():
    exponent, points = SCALE.fit_exponent([10, 20, 40], [0.2, 0.8, 3.2])

    assert points == 3
    assert abs(exponent - 2.0) < 1e-12


def test_the_measured_healthy_envelope_passes():
    measurements = [
        _measurement(task=task) for task in ("cbcn", "compare", "mmd", "c2st")
    ]

    assert SCALE.performance_gates(measurements, _scaling())["passed"]


def test_the_original_superlinear_frame_regression_fails():
    measurements = [
        _measurement(task=task) for task in ("cbcn", "compare", "mmd", "c2st")
    ]

    gates = SCALE.performance_gates(measurements, _scaling(frame_cbcn=1.38))

    assert not gates["passed"]
    assert not next(
        check for check in gates["checks"] if check["name"] == "cbcn scaling against frames"
    )["passed"]


def test_the_original_memory_regression_fails():
    measurements = [
        _measurement(task="cbcn", gb=1.24),
        _measurement(task="compare"),
        _measurement(task="mmd"),
        _measurement(task="c2st"),
    ]

    gates = SCALE.performance_gates(measurements, _scaling())

    assert not gates["passed"]


def test_a_timeout_cannot_disappear_from_the_report():
    measurements = [
        _measurement(task="cbcn"),
        _measurement(task="compare", frames=500, status="timed_out"),
        _measurement(task="mmd"),
        _measurement(task="c2st"),
    ]

    gates = SCALE.performance_gates(measurements, _scaling())

    assert not gates["passed"]
    assert gates["checks"][0]["actual"] == 1


def test_a_missing_or_underresolved_fit_fails():
    measurements = [
        _measurement(task=task) for task in ("cbcn", "compare", "mmd", "c2st")
    ]
    scaling = _scaling()
    scaling["frames"]["cbcn"] = {"exponent": None, "points": 1}

    assert not SCALE.performance_gates(measurements, scaling)["passed"]
