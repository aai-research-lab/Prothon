"""PyPI must receive the same wheel and sdist that passed the artifact gates."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"


def _workflow():
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_each_built_distribution_runs_every_artifact_gate():
    job = _workflow()["jobs"]["test-distributions"]
    steps = "\n".join(step.get("run", "") for step in job["steps"])

    assert job["strategy"]["matrix"]["distribution"] == ["wheel", "sdist"]
    assert '"${PROTHON_PACKAGE}[dev,docs]"' in steps
    assert "python -I -m pytest" in steps
    assert "python -I -m pytest -m network" in steps
    assert "ruff check" in steps
    assert "sphinx-build" in steps
    assert "pip-audit" in steps
    assert "cyclonedx-json" in steps
    assert "sha256sum" in steps
    assert "prothon info" in steps


def test_artifact_import_is_isolated_from_the_checkout_and_evidence_is_retained():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "workspace in location.parents" in source
    assert "import resolved to checkout instead of artifact" in source
    assert "release-evidence-${{ matrix.distribution }}" in source
    assert "Enforce every installed-artifact gate" in source


def test_publish_requires_the_artifact_test_matrix():
    workflow = _workflow()
    publish = workflow["jobs"]["publish"]

    assert set(publish["needs"]) == {"build", "test-distributions"}
    assert publish["if"] == "startsWith(github.ref, 'refs/tags/v')"


def test_a_branch_can_dry_run_every_gate_without_publishing():
    workflow = _workflow()
    build_steps = workflow["jobs"]["build"]["steps"]
    version_gate = next(
        step for step in build_steps
        if step.get("name") == "Verify the built version matches the release tag"
    )

    assert "workflow_dispatch" in workflow["on"]
    assert version_gate["if"] == "startsWith(github.ref, 'refs/tags/v')"


class TestTheCalibrationGateRunsOnARelease:
    """A gate that runs only when somebody remembers is not a gate.

    `scientific-calibration` was conditioned on `workflow_dispatch` alone, so
    it never ran on a push, a tag or a release. The measurements existed and
    the jobs passed when triggered by hand, and two release gates were ticked
    off in the roadmap on that basis. Nothing enforced them at the moment they
    were supposed to bite.
    """

    import pathlib

    WORKFLOWS = pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows"

    def _tests_workflow(self):
        return (self.WORKFLOWS / "tests.yml").read_text(encoding="utf-8")

    def test_it_runs_on_a_tag(self):
        text = self._tests_workflow()
        block = text[text.index("scientific-calibration:") :]
        block = block[: block.index("\n\n")] if "\n\n" in block else block
        assert "refs/tags/v" in block, (
            "the calibration gate must run on a tag, or a release is not gated "
            "on it"
        )

    def test_publish_waits_for_the_tests_workflow(self):
        """`publish.yml` calls `tests.yml`, and everything else needs it.

        That is what makes the tag condition above bite: the calibration runs
        inside the workflow the release is blocked on.
        """
        publish = (self.WORKFLOWS / "publish.yml").read_text(encoding="utf-8")
        assert "uses: ./.github/workflows/tests.yml" in publish
        assert "needs: quality" in publish

    def test_it_does_not_run_on_every_push(self):
        """Four jobs at up to three hours. Statistics change when the
        statistics change, not when a docstring does."""
        text = self._tests_workflow()
        block = text[text.index("scientific-calibration:") :]
        block = block[: block.index("uses:")]
        assert "workflow_dispatch" in block
        assert "github.event_name == 'push'" not in block
