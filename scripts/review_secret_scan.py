#!/usr/bin/env python3
"""Fail when a detect-secrets report contains an unreviewed finding."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIGEST_PATH = "recipes/prothon/recipe.yaml"
PUBLIC_DIGEST_LINE = re.compile(r"\s*sha256:\s*[0-9a-f]{64}\s*")


def _normalise_path(filename: str) -> str:
    """Use the repository-relative spelling emitted on every runner OS."""
    normalised = filename.replace("\\", "/")
    return normalised[2:] if normalised.startswith("./") else normalised


def _line_number(match: dict[str, Any]) -> int:
    line_number = match.get("line_number")
    if isinstance(line_number, bool) or not isinstance(line_number, int):
        raise ValueError("a detect-secrets finding has no integer line_number")
    if line_number < 1:
        raise ValueError("a detect-secrets finding has an invalid line_number")
    return line_number


def _is_reviewed_public_digest(
    filename: str,
    match: dict[str, Any],
    root: Path,
) -> bool:
    """Recognise the one public digest without hiding other 64-hex values."""
    filename = _normalise_path(filename)
    if filename != PUBLIC_DIGEST_PATH:
        return False

    lines = (root / filename).read_text(encoding="utf-8").splitlines()
    line_number = _line_number(match)
    if line_number > len(lines):
        raise ValueError(f"finding points beyond the end of {filename}")
    return PUBLIC_DIGEST_LINE.fullmatch(lines[line_number - 1]) is not None


def partition_findings(
    report: dict[str, Any],
    root: Path = ROOT,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Return unreviewed findings and the number of reviewed findings."""
    results = report.get("results")
    if not isinstance(results, dict):
        raise ValueError("detect-secrets report has no results object")

    unreviewed: dict[str, list[dict[str, Any]]] = {}
    reviewed_count = 0
    for filename, matches in results.items():
        if not isinstance(filename, str) or not isinstance(matches, list):
            raise ValueError("detect-secrets report contains malformed results")
        for match in matches:
            if not isinstance(match, dict):
                raise ValueError("detect-secrets report contains a malformed finding")
            _line_number(match)
            if _is_reviewed_public_digest(filename, match, root):
                reviewed_count += 1
            else:
                unreviewed.setdefault(filename, []).append(match)
    return unreviewed, reviewed_count


def _annotation_property(value: str) -> str:
    """Escape a value used in a GitHub workflow-command property."""
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail on detect-secrets findings except reviewed public metadata."
    )
    parser.add_argument("report", type=Path, help="detect-secrets JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("detect-secrets report is not a JSON object")
        unreviewed, reviewed_count = partition_findings(report)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"secret scan review could not run: {error}", file=sys.stderr)
        return 2

    for filename, matches in sorted(unreviewed.items()):
        line_numbers = [_line_number(match) for match in matches]
        lines = ", ".join(str(line_number) for line_number in line_numbers)
        print(
            f"::error file={_annotation_property(filename)},line={line_numbers[0]}::"
            f"possible secret(s) on line(s) {lines}"
        )
    if unreviewed:
        print("secret scan found material that must be removed or reviewed")
        return 1

    print(
        "No unreviewed possible secrets found "
        f"({reviewed_count} reviewed public metadata finding(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
