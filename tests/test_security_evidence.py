"""The release secret scan keeps its reviewed exceptions narrow."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIGEST_EXCLUSION = r"^\s*sha256:\s*[0-9a-f]{64}\s*$"


def test_workflow_secret_inheritance_is_explicitly_reviewed():
    publish = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    declaration = next(
        line.strip() for line in publish.splitlines()
        if line.strip().startswith("secrets:")
    )

    assert declaration.endswith("# pragma: allowlist secret")


def test_scanner_excludes_only_a_complete_sha256_metadata_field():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
        encoding="utf-8"
    )

    assert f"--exclude-lines '{DIGEST_EXCLUSION}'" in workflow
    assert workflow.count("--exclude-lines") == 1
    assert "--exclude-files" not in workflow
    assert re.fullmatch(DIGEST_EXCLUSION, "  sha256: " + "a" * 64)
    assert not re.fullmatch(DIGEST_EXCLUSION, "  token: " + "a" * 64)
