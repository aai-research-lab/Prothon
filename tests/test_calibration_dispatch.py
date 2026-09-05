"""Branch calibration must be reachable through a registered workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TESTS_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
CALIBRATION_WORKFLOW = ROOT / ".github" / "workflows" / "scientific-calibration.yml"


def _load(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_registered_workflow_calls_branch_calibration_on_manual_dispatch():
    tests = _load(TESTS_WORKFLOW)
    calibration = _load(CALIBRATION_WORKFLOW)
    bridge = tests["jobs"]["scientific-calibration"]

    assert "workflow_dispatch" in tests["on"]
    assert "workflow_call" in calibration["on"]
    assert bridge["if"] == "github.event_name == 'workflow_dispatch'"
    assert bridge["uses"] == "./.github/workflows/scientific-calibration.yml"
