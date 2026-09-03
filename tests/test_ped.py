"""Tests for the PED loader.

The parsing tests use a synthetic archive shaped like the one PED serves,
because the real download for a single entry runs to tens of megabytes and the
thing being tested is the unwrapping, not the network. The tests that need PED
itself are marked ``network``.
"""

from __future__ import annotations

import gzip
import io
import tarfile

import numpy as np
import pytest

from prothon.ingest import Ensemble
from prothon.ingest.ped import (
    PedUnavailable,
    _extract_pdb,
    _normalise,
    ped_ensemble,
    ped_entry,
)


def tiny_pdb(n_models=3) -> str:
    """A multi-model PDB with three alanines, in PED's layout."""
    lines = []
    for model in range(1, n_models + 1):
        lines.append(f"MODEL     {model:>4}")
        serial = 1
        for residue in range(1, 4):
            for name, dx in (("N", 0.0), ("CA", 1.5), ("C", 3.0)):
                lines.append(
                    f"ATOM  {serial:>5}  {name:<3} ALA A{residue:>4}    "
                    f"{residue * 3.8 + dx:>8.3f}{model * 0.1:>8.3f}{0.0:>8.3f}"
                    f"  1.00  0.00           C"
                )
                serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


def as_ped_tarball(text: str, name="PED99999e001.pdb") -> bytes:
    """Wrap text the way the API does: a gzipped tar holding one PDB."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        payload = text.encode()
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return gzip.compress(raw.getvalue())


class TestAccessions:
    @pytest.mark.parametrize(
        "given", ["PED00024", "ped00024", "PED24", "24", 24, " PED00024 "]
    )
    def test_forms_that_should_work(self, given):
        assert _normalise(given) == "PED00024"

    def test_something_else_is_refused(self):
        with pytest.raises(ValueError, match="Not a PED accession"):
            _normalise("ubiquitin")


class TestUnwrapping:
    def test_a_gzipped_tar_is_the_case_that_matters(self):
        """What PED actually serves. The endpoint is called `ensemble-pdb` and
        returns `ensemble.tar` gzipped, so decompressing and parsing directly
        interleaves tar headers with coordinates."""
        text = _extract_pdb(as_ped_tarball(tiny_pdb()), "PED99999")
        assert text.startswith("MODEL")
        assert "ustar" not in text          # no tar header leaked through
        assert text.count("MODEL") == 3

    def test_a_plain_gzipped_pdb_also_works(self):
        """A fallback, so that a change in what PED serves surfaces as a PDB
        parse error rather than as a confusing tar failure."""
        text = _extract_pdb(gzip.compress(tiny_pdb().encode()), "PED99999")
        assert text.count("MODEL") == 3

    def test_plain_text_also_works(self):
        text = _extract_pdb(tiny_pdb().encode(), "PED99999")
        assert text.count("MODEL") == 3

    def test_an_empty_archive_is_refused(self):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w"):
            pass
        with pytest.raises(ValueError, match="no files"):
            _extract_pdb(gzip.compress(raw.getvalue()), "PED99999")


class TestLoading:
    def test_a_cached_entry_is_not_downloaded(self, tmp_path, monkeypatch):
        """Entries run to tens of megabytes and a benchmark may want the same
        one repeatedly."""
        (tmp_path / "PED99999e001.pdb").write_text(tiny_pdb(4), encoding="utf-8")

        def refuse(*args, **kwargs):
            raise AssertionError("the network was used despite a cache hit")

        monkeypatch.setattr("prothon.ingest.ped._get", refuse)
        ensemble = ped_ensemble("PED99999", cache_dir=str(tmp_path))
        assert ensemble.n_frames == 4
        assert ensemble.provenance["kind"] == "ped"
        assert ensemble.provenance["sampling_kind"] == "iid"
        assert ensemble.provenance["accession"] == "PED99999"

    def test_a_download_is_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "prothon.ingest.ped._get", lambda url, **kw: as_ped_tarball(tiny_pdb(5))
        )
        ped_ensemble("PED99999", cache_dir=str(tmp_path))
        assert (tmp_path / "PED99999e001.pdb").exists()

    def test_it_works_without_a_cache_directory(self, monkeypatch):
        monkeypatch.setattr(
            "prothon.ingest.ped._get", lambda url, **kw: as_ped_tarball(tiny_pdb(3))
        )
        assert ped_ensemble("PED99999").n_frames == 3

    def test_conformers_are_uniformly_weighted(self, monkeypatch):
        """PED publishes no populations, so an ensemble from it is uniform.
        That is a fact about the database rather than a default chosen here,
        and the provenance says where the ensemble came from."""
        monkeypatch.setattr(
            "prothon.ingest.ped._get", lambda url, **kw: as_ped_tarball(tiny_pdb(6))
        )
        ensemble = ped_ensemble("PED99999")
        assert ensemble.weights is None
        np.testing.assert_allclose(ensemble.frame_weights, 1 / 6)

    def test_the_constructor_on_ensemble_reaches_the_same_place(self, monkeypatch):
        monkeypatch.setattr(
            "prothon.ingest.ped._get", lambda url, **kw: as_ped_tarball(tiny_pdb(3))
        )
        assert Ensemble.from_ped("PED99999").n_frames == 3

    def test_a_missing_entry_names_the_problem(self, monkeypatch):
        import urllib.error

        def not_found(url, **kwargs):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        monkeypatch.setattr("prothon.ingest.ped.urllib.request.urlopen", not_found)
        with pytest.raises(ValueError, match="404"):
            ped_entry("PED99999")


@pytest.mark.network
class TestAgainstPed:
    """Against the database itself. Deselected by default.

    These skip when PED is unreachable or answers 5xx, and only then. A server
    error is theirs and says nothing about this code; a 404, a malformed
    archive or a count that does not match the metadata is ours and fails.

    Without the distinction a PED outage turns the build red, which trains
    everyone to ignore a red build.
    """

    @staticmethod
    def _skip_if_ped_is_down(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except PedUnavailable as error:
            pytest.skip(f"PED is unavailable: {error}")

    def test_an_entry_lists_its_ensembles(self):
        entry = self._skip_if_ped_is_down(ped_entry, "PED00001")
        assert entry["entry_id"] == "PED00001"
        # Three separate determinations, which must not be merged.
        assert len(entry["ensembles"]) == 3
        assert {e["ensemble_id"] for e in entry["ensembles"]} == {"e001", "e002", "e003"}

    def test_a_small_entry_loads(self, tmp_path):
        ensemble = self._skip_if_ped_is_down(
            ped_ensemble, "PED00001", "e001", cache_dir=str(tmp_path)
        )
        assert ensemble.n_frames == 11
        assert ensemble.sequence
        assert ensemble.provenance["n_models_loaded"] == 11

    def test_the_loaded_count_matches_the_metadata(self, tmp_path):
        """PED00024 reports 576 models and a naive line count finds 575,
        because the first MODEL record shares a line with a tar header. The
        parsed count is the one to trust, and it should agree with the
        metadata."""
        entry = self._skip_if_ped_is_down(ped_entry, "PED00001")
        reported = entry["ensembles"][0]["models"]
        loaded = ped_ensemble("PED00001", "e001", cache_dir=str(tmp_path)).n_frames
        assert loaded == reported

    def test_a_loaded_ensemble_can_be_compared(self, tmp_path):
        from prothon.represent.order_parameters import compute_representation

        ensemble = self._skip_if_ped_is_down(
            ped_ensemble, "PED00001", "e001", cache_dir=str(tmp_path)
        )
        matrix = compute_representation(ensemble.trajectory, "cacn")
        assert matrix.shape[0] == ensemble.n_frames
        assert matrix.shape[1] > 10


class TestAnOutageIsNotABadAccession:
    """Two failures that call for opposite responses.

    A 404 means the accession is wrong and will be wrong tomorrow. A 502, a
    timeout or a refused connection means PED is down and the same call may
    succeed later. Both raised the same `ValueError`, so a caller could not
    tell them apart and the live tests could not skip on one without skipping
    on the other.

    This is not hypothetical: a 502 from PED turned the build red, for a
    service outage that says nothing about this code.
    """

    @staticmethod
    def _raising(exc):
        def fake(url, timeout=None):
            raise exc
        return fake

    def test_a_server_error_is_unavailable(self, monkeypatch):
        import urllib.error

        monkeypatch.setattr(
            "urllib.request.urlopen",
            self._raising(urllib.error.HTTPError("u", 502, "Bad Gateway", {}, None)),
        )
        with pytest.raises(PedUnavailable):
            ped_entry("PED00001")

    def test_an_unreachable_host_is_unavailable(self, monkeypatch):
        import urllib.error

        monkeypatch.setattr(
            "urllib.request.urlopen",
            self._raising(urllib.error.URLError("no route to host")),
        )
        with pytest.raises(PedUnavailable):
            ped_entry("PED00001")

    def test_a_missing_entry_is_not(self, monkeypatch):
        """A 404 must stay an ordinary ValueError, or a wrong accession would
        skip the test that was meant to catch it."""
        import urllib.error

        monkeypatch.setattr(
            "urllib.request.urlopen",
            self._raising(urllib.error.HTTPError("u", 404, "Not Found", {}, None)),
        )
        with pytest.raises(ValueError) as caught:
            ped_entry("PED00001")
        assert not isinstance(caught.value, PedUnavailable)

    def test_it_is_still_a_value_error(self):
        """So `except ValueError` in existing code keeps working."""
        assert issubclass(PedUnavailable, ValueError)
