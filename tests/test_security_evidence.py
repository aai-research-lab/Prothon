"""The release security gates keep their reviewed exceptions narrow."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "review_secret_scan", ROOT / "scripts" / "review_secret_scan.py"
)
assert SPEC is not None and SPEC.loader is not None
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)

NODE24_ACTIONS = {
    "actions/upload-artifact": (
        "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "v7.0.1",
        4,
    ),
    "actions/download-artifact": (
        "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "v8.0.1",
        2,
    ),
    "codecov/codecov-action": (
        "fb8b3582c8e4def4969c97caa2f19720cb33a72f",
        "v7.0.0",
        1,
    ),
}


def test_workflow_secret_inheritance_is_explicitly_reviewed():
    publish = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    declaration = next(
        line.strip() for line in publish.splitlines()
        if line.strip().startswith("secrets:")
    )

    assert declaration.endswith("# pragma: allowlist secret")


def test_workflow_keeps_the_raw_scan_then_reviews_it():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
        encoding="utf-8"
    )

    assert (
        "detect-secrets scan --no-verify > security-evidence/detect-secrets.json"
        in workflow
    )
    assert (
        "python scripts/review_secret_scan.py security-evidence/detect-secrets.json"
        in workflow
    )
    assert "--exclude-lines" not in workflow
    assert "--exclude-files" not in workflow


def test_dependency_audit_excludes_only_the_unpublished_root_distribution():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
        encoding="utf-8"
    )

    assert "> security-evidence/runtime-requirements.txt" in workflow
    assert workflow.count("--exclude prothon-ensembles") == 1
    assert "> security-evidence/audit-requirements.txt" in workflow
    assert "--requirement security-evidence/audit-requirements.txt" in workflow
    assert "--no-deps" in workflow
    assert "--disable-pip" in workflow
    assert '--path "$RUNTIME_SITE"' not in workflow


def test_artifact_and_coverage_actions_use_reviewed_node24_releases():
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    )

    for action, (commit, version, count) in NODE24_ACTIONS.items():
        references = re.findall(
            rf"{re.escape(action)}@([0-9a-f]{{40}}) # (v[^\s]+)", workflows
        )
        assert references == [(commit, version)] * count


def test_real_data_corpus_uses_an_immutable_reviewed_revision():
    source = (ROOT / "tests" / "test_real_data.py").read_text(encoding="utf-8")
    revision = "9bac885848417ec1c257af0191c4186f82f87c8f"

    assert f'_CORPUS_REVISION = "{revision}"' in source
    assert "mdtraj/mdtraj/master/tests/data" not in source
    assert 'f"{_CORPUS_REVISION}/tests/data"' in source
    assert "cache = cache_root / _CORPUS_REVISION" in source


def test_the_public_recipe_digest_is_reviewed():
    recipe = (ROOT / "recipes" / "prothon" / "recipe.yaml").read_text(
        encoding="utf-8"
    )
    line_number = next(
        number
        for number, line in enumerate(recipe.splitlines(), start=1)
        if line.strip().startswith("sha256:")
    )
    report = {
        "results": {
            "recipes/prothon/recipe.yaml": [{"line_number": line_number}]
        }
    }

    unreviewed, reviewed_count = REVIEW.partition_findings(report, ROOT)

    assert unreviewed == {}
    assert reviewed_count == 1


def test_the_same_value_outside_the_recipe_is_not_reviewed(tmp_path):
    report = {
        "results": {
            "elsewhere.yaml": [{"line_number": 1}]
        }
    }

    unreviewed, reviewed_count = REVIEW.partition_findings(report, tmp_path)

    assert unreviewed == report["results"]
    assert reviewed_count == 0


def test_a_non_digest_line_inside_the_recipe_is_not_reviewed(tmp_path):
    recipe = tmp_path / "recipes" / "prothon" / "recipe.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("token: " + "a" * 64 + "\n", encoding="utf-8")
    report = {
        "results": {
            "recipes/prothon/recipe.yaml": [{"line_number": 1}]
        }
    }

    unreviewed, reviewed_count = REVIEW.partition_findings(report, tmp_path)

    assert unreviewed == report["results"]
    assert reviewed_count == 0


def test_the_corpus_revision_is_reviewed():
    """The same category as an action pin, from a different declaration.

    Pinning the real-data corpus put its MDTraj commit in two files: the
    constant that declares it and the test that asserts the pin. Both tripped
    the scanner, for the same reason the action SHAs did.
    """
    source = (ROOT / "tests" / "test_real_data.py").read_text(encoding="utf-8")
    revision = re.search(r'_CORPUS_REVISION\s*=\s*"([0-9a-f]{40})"', source)
    assert revision is not None, "the corpus revision is no longer declared"
    assert revision.group(1) in REVIEW._pinned_action_shas(ROOT)


def test_every_declared_pin_is_reviewed():
    """Whatever the sources declare, the reviewer accepts -- and only that."""
    found = REVIEW._pinned_action_shas(ROOT)
    assert found, "no pinned revisions found at all"
    assert all(len(sha) == 40 for sha in found)


def test_a_pinned_action_sha_is_reviewed():
    """A pinned action SHA is public by construction.

    Workflows pin actions to immutable commits, which is the recommended
    practice. `detect-secrets` sees forty hex characters and reports high
    entropy, and the three SHAs at the top of this file failed the release gate
    for it.
    """
    pinned = REVIEW._pinned_action_shas(ROOT)
    assert pinned, "no pinned action SHAs found in the workflows"
    for _, (sha, _, _) in NODE24_ACTIONS.items():
        assert sha in pinned, f"{sha} is asserted here but pinned nowhere"


def test_the_exception_is_narrow(tmp_path):
    """A forty-hex string that is not a pinned SHA is still a finding.

    The value of the reviewed-exception design is that it names one thing. A
    rule that excused every hex string in every file would pass this test file
    and also pass a leaked token pasted beside it.
    """
    source = tmp_path / "sample.py"
    source.write_text('LEAKED = "' + "a1" * 20 + '"\n', encoding="utf-8")
    match = {"line_number": 1}
    pinned = REVIEW._pinned_action_shas(ROOT)
    assert not REVIEW._is_reviewed_pinned_action("sample.py", match, tmp_path, pinned)


def test_removing_a_pin_removes_the_exception(tmp_path):
    """The list is read from the workflows, not written down twice.

    A SHA is excused only while a workflow actually pins to it, so deleting the
    pin withdraws the exception without anyone remembering to.
    """
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    assert REVIEW._pinned_action_shas(tmp_path) == set()
