"""Release metadata must identify what users can actually install."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "check_conda_sync", ROOT / "scripts" / "check_conda_sync.py"
)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def _recipe(version: str, digest: str) -> str:
    return (
        "schema_version: 1\n"
        "context:\n"
        f'  version: "{version}"\n'
        "source:\n"
        f"  sha256: {digest}\n"
    )


def _pypi(version: str, digest: str) -> dict:
    return {
        "info": {"version": version},
        "releases": {
            version: [
                {"packagetype": "sdist", "digests": {"sha256": digest}}
            ]
        },
    }


def test_the_conda_checker_accepts_one_identical_published_recipe(tmp_path):
    digest = "a" * 64
    recipe = _recipe("2.3.2", digest)
    recipe_path = tmp_path / "recipe.yaml"
    feedstock_path = tmp_path / "feedstock.yaml"
    pypi_path = tmp_path / "pypi.json"
    recipe_path.write_text(recipe, encoding="utf-8")
    feedstock_path.write_text(recipe, encoding="utf-8")
    pypi_path.write_text(json.dumps(_pypi("2.3.2", digest)), encoding="utf-8")

    status = CHECKER.main(
        [
            "--recipe",
            str(recipe_path),
            "--feedstock-url",
            feedstock_path.as_uri(),
            "--pypi-url",
            pypi_path.as_uri(),
        ]
    )

    assert status == 0


def test_the_conda_checker_reports_recipe_version_and_hash_drift(tmp_path, capsys):
    old_digest = "a" * 64
    new_digest = "b" * 64
    recipe_path = tmp_path / "recipe.yaml"
    feedstock_path = tmp_path / "feedstock.yaml"
    pypi_path = tmp_path / "pypi.json"
    recipe_path.write_text(_recipe("2.3.1", old_digest), encoding="utf-8")
    feedstock_path.write_text(_recipe("2.3.2", new_digest), encoding="utf-8")
    pypi_path.write_text(json.dumps(_pypi("2.3.2", new_digest)), encoding="utf-8")

    status = CHECKER.main(
        [
            "--recipe",
            str(recipe_path),
            "--feedstock-url",
            feedstock_path.as_uri(),
            "--pypi-url",
            pypi_path.as_uri(),
        ]
    )

    errors = capsys.readouterr().err
    assert status == 1
    assert "live feedstock recipes differ" in errors
    assert "not PyPI's latest" in errors
    assert "does not match PyPI sdist" in errors


def test_an_unversioned_source_snapshot_does_not_claim_an_old_release():
    source = (ROOT / "src" / "prothon" / "__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "0+unknown"' in source
    assert '__version__ = "2.1.0.dev0"' not in source
