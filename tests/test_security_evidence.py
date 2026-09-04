"""The release security gates keep their reviewed exceptions narrow."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "review_secret_scan", ROOT / "scripts" / "review_secret_scan.py"
)
assert SPEC is not None and SPEC.loader is not None
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


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
