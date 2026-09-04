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
