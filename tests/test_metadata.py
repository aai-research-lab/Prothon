"""The metadata a citation is built from, kept honest.

Three files claim to describe the same software, and nothing makes them agree:
``pyproject.toml`` goes to PyPI, ``CITATION.cff`` is what GitHub's "Cite this
repository" button reads, and the git tags are what a Zenodo deposit is cut
from. The contact address in ``pyproject.toml`` was two institutions out of
date, and a reader who found the package before the paper would have written
to an address the author no longer uses.

A version in ``CITATION.cff`` is the specific failure this guards against. The
package version comes from ``setuptools-scm`` and cannot go stale, which is why
``pyproject.toml`` deliberately carries no literal. A citation file has to name
a release, so it does carry one, and a literal that has to be edited by hand at
release time is a literal that will eventually be forgotten.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

yaml = pytest.importorskip("yaml")

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9-3.10
    tomllib = pytest.importorskip("tomli")

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def citation() -> dict:
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def _tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def _latest_tag() -> str | None:
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    tag = result.stdout.strip()
    return tag or None


class TestTheFilesAgreeWithEachOther:
    def test_the_two_contact_addresses_are_deliberately_different(
        self, citation, pyproject
    ):
        """They are not supposed to match, and a test once assumed they were.

        The package metadata carries a personal address and the citation an
        institutional one. That is a choice: the software outlives any one
        affiliation, and correspondence about the paper should reach the
        institution. An earlier version of this test asserted the two agreed
        and would have blocked putting them back.

        What is worth checking is that both are present and well formed, not
        that they are the same.
        """
        cff = {a["email"] for a in citation["authors"] if "email" in a}
        project = {a["email"] for a in pyproject["project"]["authors"]}
        assert cff, "CITATION.cff should carry a contact address"
        assert project, "pyproject.toml should carry a contact address"
        for address in cff | project:
            assert "@" in address and "." in address.split("@")[-1], address

    def test_maintainers_and_authors_do_not_disagree(self, pyproject):
        authors = {a["email"] for a in pyproject["project"]["authors"]}
        maintainers = {a["email"] for a in pyproject["project"]["maintainers"]}
        assert maintainers <= authors

    def test_the_licence_matches(self, citation, pyproject):
        assert citation["license"] == pyproject["project"]["license"]["text"]

    def test_the_repository_url_matches(self, citation, pyproject):
        urls = set(pyproject["project"].get("urls", {}).values())
        assert citation["repository-code"] in urls


class TestTheCitationVersionIsNotBehindARelease:
    """It may run ahead of the tags -- that is the state before a release.

    It may not run behind one. A tag exists, someone can cite it, and a
    citation file naming an earlier version sends them to the wrong artifact.
    """

    def test_it_is_not_behind_the_latest_tag(self, citation):
        tag = _latest_tag()
        if tag is None:
            pytest.skip("no tags in this checkout")
        assert _tuple(citation["version"]) >= _tuple(tag), (
            f"CITATION.cff says {citation['version']} but {tag} is tagged; "
            f"update the version field as part of cutting a release"
        )

    def test_the_version_is_a_release_number_and_not_a_development_one(
        self, citation
    ):
        assert re.fullmatch(r"\d+\.\d+\.\d+", str(citation["version"])), (
            "CITATION.cff should name a release, not a development version"
        )


class TestTheCitationIsUsable:
    def test_the_published_method_is_still_the_preferred_citation(self, citation):
        preferred = citation["preferred-citation"]
        assert preferred["doi"] == "10.1021/acs.jcim.3c00145"
        assert preferred["year"] == 2023

    def test_every_orcid_is_a_resolvable_url(self, citation):
        for author in citation["authors"]:
            if "orcid" in author:
                assert re.fullmatch(
                    r"https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]",
                    author["orcid"],
                )


class TestZenodoAgreesWithTheCitation:
    """Three files now describe this software and they must not diverge.

    `CITATION.cff` is read by GitHub, `.zenodo.json` by Zenodo when it mints a
    DOI, and `pyproject.toml` by PyPI. A DOI is permanent, so a title or an
    author list that was wrong on the day of the release stays wrong.
    """

    @staticmethod
    def _zenodo():
        import json
        import pathlib

        path = pathlib.Path(__file__).resolve().parent.parent / ".zenodo.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_title_matches(self, citation):
        assert self._zenodo()["title"] == citation["title"]

    def test_the_licence_matches(self, citation):
        assert self._zenodo()["license"] == citation["license"]

    def test_the_author_list_matches_in_order(self, citation):
        """Order is authorship, not presentation."""
        zenodo = [c["name"] for c in self._zenodo()["creators"]]
        cff = [
            f"{a['family-names']}, {a['given-names']}"
            for a in citation["authors"]
        ]
        assert zenodo == cff

    def test_the_method_paper_is_linked(self, citation):
        """The DOI must point back at what it is supplementary to."""
        doi = citation["preferred-citation"]["doi"]
        related = {
            r["identifier"] for r in self._zenodo()["related_identifiers"]
        }
        assert doi in related, f"{doi} is not among the related identifiers"

    def test_the_orcid_is_bare_in_zenodo_and_a_url_in_the_citation(self):
        """The two schemas want it written differently, which is easy to get
        wrong by copying one into the other."""
        citation_orcid = "0000-0002-8215-7452"
        creator = self._zenodo()["creators"][0]
        assert creator["orcid"] == citation_orcid
        assert not creator["orcid"].startswith("http")
