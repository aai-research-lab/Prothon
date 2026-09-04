"""The scheduled scientific workflow must enforce every declared release gate."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "scientific-calibration.yml"


def _matrix_rows():
    document = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    job = document["jobs"]["calibrate"]
    return job, job["strategy"]["matrix"]["include"]


def test_every_scientific_release_gate_runs_in_the_scheduled_workflow():
    _, rows = _matrix_rows()
    commands = {row["gate"]: row["command"] for row in rows}

    assert set(commands) == {"local-null", "default-path", "floors", "joint"}
    assert "calibration.py --quick --study correlation" in commands["local-null"]
    assert "calibration.py --quick --study default_path" in commands["default-path"]
    assert "floor_calibration.py --replicates 60" in commands["floors"]
    assert "joint_calibration.py" in commands["joint"]


def test_slow_gates_run_independently_and_keep_distinct_evidence():
    job, rows = _matrix_rows()
    source = WORKFLOW.read_text(encoding="utf-8")

    assert job["strategy"]["fail-fast"] == "false"
    assert len({row["gate"] for row in rows}) == len(rows)
    assert "scientific-calibration-evidence-${{ matrix.gate }}" in source
    assert "if: always()" in source
    assert 'run: test "$GATE_OUTCOME" = success' in source
