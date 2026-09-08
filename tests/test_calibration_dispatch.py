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
    """Reachable by hand, and wired to the reusable workflow.

    This asserted the whole condition by string equality, which pinned the
    gate to `workflow_dispatch` alone. That was the defect rather than the
    contract: the calibration never ran on a tag, so a release was not gated on
    it while two gates were ticked off in the roadmap. Adding the tag trigger
    broke this test, which is the wrong way round.

    The intent survives -- manual dispatch must still work and the bridge must
    still point at the reusable workflow -- without freezing every other
    trigger out.
    """
    tests = _load(TESTS_WORKFLOW)
    calibration = _load(CALIBRATION_WORKFLOW)
    bridge = tests["jobs"]["scientific-calibration"]

    assert "workflow_dispatch" in tests["on"]
    assert "workflow_call" in calibration["on"]
    assert "workflow_dispatch" in bridge["if"]
    assert bridge["uses"] == "./.github/workflows/scientific-calibration.yml"


def test_the_bridge_also_fires_on_a_tag():
    """A gate that runs only when somebody remembers is not a gate.

    `publish.yml` calls `tests.yml` as its `quality` job and everything else
    needs it, so this condition is what decides whether a release waits for the
    calibration.
    """
    bridge = _load(TESTS_WORKFLOW)["jobs"]["scientific-calibration"]
    assert "refs/tags/v" in bridge["if"]
