#!/usr/bin/env python3
"""Check that the reference recipe, feedstock and latest PyPI sdist agree."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RECIPE = ROOT / "recipes" / "prothon" / "recipe.yaml"
FEEDSTOCK_URL = (
    "https://raw.githubusercontent.com/conda-forge/prothon-feedstock/"
    "main/recipe/recipe.yaml"
)
PYPI_URL = "https://pypi.org/pypi/prothon-ensembles/json"

_VERSION = re.compile(r'^\s{2}version:\s*"([^"]+)"\s*$', re.MULTILINE)
_SHA256 = re.compile(r"^\s{2}sha256:\s*([0-9a-f]{64})\s*$", re.MULTILINE)


def _read_url(url: str) -> str:
    """Read UTF-8 text from an HTTPS or test ``file:`` URL."""
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        return response.read().decode("utf-8")


def recipe_identity(text: str) -> tuple[str, str]:
    """Return the one literal version and source digest from a v1 recipe."""
    versions = _VERSION.findall(text)
    digests = _SHA256.findall(text)
    if len(versions) != 1 or len(digests) != 1:
        raise ValueError(
            "recipe must contain exactly one quoted context version and one SHA-256"
        )
    return versions[0], digests[0]


def sync_problems(local: str, feedstock: str, pypi: dict[str, Any]) -> list[str]:
    """Describe every disagreement instead of stopping at the first one."""
    problems: list[str] = []
    version, digest = recipe_identity(local)

    if local != feedstock:
        diff = "\n".join(
            difflib.unified_diff(
                local.splitlines(),
                feedstock.splitlines(),
                fromfile="recipes/prothon/recipe.yaml",
                tofile="conda-forge/prothon-feedstock/recipe/recipe.yaml",
                lineterm="",
            )
        )
        problems.append("the in-repository and live feedstock recipes differ:\n" + diff)

    latest = str(pypi.get("info", {}).get("version", ""))
    if latest != version:
        problems.append(f"recipe version {version!r} is not PyPI's latest {latest!r}")

    files = pypi.get("releases", {}).get(version, [])
    sdists = [item for item in files if item.get("packagetype") == "sdist"]
    if len(sdists) != 1:
        problems.append(
            f"PyPI has {len(sdists)} source distributions for recipe version {version!r}"
        )
    else:
        published = str(sdists[0].get("digests", {}).get("sha256", ""))
        if published != digest:
            problems.append(
                f"recipe SHA-256 {digest} does not match PyPI sdist {published}"
            )
    return problems


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--feedstock-url", default=FEEDSTOCK_URL)
    parser.add_argument("--pypi-url", default=PYPI_URL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        local = args.recipe.read_text(encoding="utf-8")
        feedstock = _read_url(args.feedstock_url)
        pypi = json.loads(_read_url(args.pypi_url))
        problems = sync_problems(local, feedstock, pypi)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"conda synchronisation check could not run: {exc}", file=sys.stderr)
        return 2

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    version, digest = recipe_identity(local)
    print(f"reference recipe, feedstock and PyPI agree on {version} ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
