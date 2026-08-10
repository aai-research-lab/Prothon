"""Prothon run on real proteins, in real file formats.

Every other test in this suite builds its own trajectory: a poly-alanine chain
with Gaussian jitter, a dozen residues long. That is enough to check that an
algorithm does what its docstring says, and not enough to check that the
software works. A real topology has residues nobody planned for, chains that
break, atoms that are missing, and enough residues to reach code paths a
fixture never approaches — the contact-number chunking needs more than about
ninety residues, and no fixture has fourteen.

The structures here come from the MDTraj test corpus, which is public, small,
and stable: an NMR ensemble of a real protein, single structures of three more,
one RNA hairpin, and a short molecular dynamics trajectory written in four
formats. They are downloaded once and cached, so this file needs the network.
Run it with ``pytest -m network``; it is deselected by default so that the
ordinary suite stays offline and fast.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from prothon.core.dissimilarity import dissimilarity
from prothon.core.metrics import METRICS
from prothon.core.precision_recall import precision_recall
from prothon.core.representation import (
    _PAIR_BLOCK,
    MEASURES,
    compute_representation,
)
from prothon.ingest import Ensemble, reconcile, sequence_of

pytestmark = pytest.mark.network

_SOURCE = "https://raw.githubusercontent.com/mdtraj/mdtraj/master/tests/data"

#: What each file is for. Keeping the reason beside the name means a future
#: reader can tell whether a replacement would still test the same thing.
_FILES = {
    "2EQQ.pdb": "a real NMR ensemble: 20 models of a 28-residue protein",
    "1bpi.pdb": "BPTI, 58 residues, a real sequence to align",
    "1vii.pdb": "villin headpiece, 36 residues",
    "1am7_protein.pdb": "158 residues -- past the contact-number block size",
    "2koc.pdb": "an RNA hairpin, which is not a protein",
    "native.pdb": "topology for the trajectory below",
    "frame0.xtc": "501 frames of real dynamics",
    "frame0.dcd": "the same frames, written as DCD",
    "frame0.nc": "the same frames, written as AMBER NetCDF",
}


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> dict[str, str]:
    """Download the structures once per session, or reuse a local cache."""
    cache = Path(os.environ.get("PROTHON_TEST_DATA", tmp_path_factory.mktemp("corpus")))
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in _FILES:
        target = cache / name
        if not target.exists():
            try:
                urllib.request.urlretrieve(f"{_SOURCE}/{name}", target)
            except Exception as error:  # pragma: no cover - network
                pytest.skip(f"could not fetch {name}: {error}")
        paths[name] = str(target)
    return paths


# ---------------------------------------------------------------------------
# A real ensemble
# ---------------------------------------------------------------------------
class TestRealNmrEnsemble:
    """2EQQ: twenty experimentally determined models of one protein."""

    @pytest.mark.parametrize("measure", sorted(MEASURES))
    def test_every_measure_computes(self, corpus, measure):
        traj = md.load(corpus["2EQQ.pdb"])
        matrix = compute_representation(traj, measure)
        assert matrix.shape[0] == traj.n_frames == 20
        assert matrix.shape[1] > 0
        assert np.isfinite(matrix).all()

    def test_contact_numbers_are_physically_plausible(self, corpus):
        """A folded protein's residues have neighbours, and not too many. A
        number outside this range means the cutoff or the units are wrong."""
        traj = md.load(corpus["2EQQ.pdb"])
        cbcn = compute_representation(traj, "cbcn")
        mean = cbcn.mean()
        assert 0.5 < mean < 15, f"mean C-beta contact number of {mean:.1f}"
        assert cbcn.min() >= 0

    def test_torsions_span_the_circle(self, corpus):
        traj = md.load(corpus["2EQQ.pdb"])
        cata = compute_representation(traj, "cata")
        assert cata.min() >= -np.pi - 1e-6
        assert cata.max() <= np.pi + 1e-6

    def test_sasa_is_non_negative_and_not_absurd(self, corpus):
        traj = md.load(corpus["2EQQ.pdb"])
        sasa = compute_representation(traj, "sasa")
        assert sasa.min() >= 0
        # A residue's accessible surface is under about 3 nm^2 for any amino
        # acid; anything larger means the units are not nm^2.
        assert sasa.max() < 3.0

    def test_a_real_ensemble_reconciles_with_itself(self, corpus):
        traj = md.load(corpus["2EQQ.pdb"])
        first = Ensemble(traj[:10], label="models 1-10")
        second = Ensemble(traj[10:], label="models 11-20")
        correspondence = reconcile(first, second)
        assert correspondence.is_identical
        assert correspondence.identity == 1.0
        assert correspondence.n_aligned == len(first.sequence)

    def test_a_real_sequence_is_read(self, corpus):
        traj = md.load(corpus["1bpi.pdb"])
        sequence, indices = sequence_of(traj.topology)
        # BPTI, whose sequence is known.
        assert sequence.startswith("RPDFCLEPPYTGPCKA")
        assert len(sequence) == len(indices) == 58

    def test_halves_of_one_ensemble_are_not_called_different(self, corpus):
        """Ten models against ten models of the same protein. The sampling is
        far too thin to resolve anything, and the software should say so rather
        than find a difference."""
        traj = md.load(corpus["2EQQ.pdb"])
        rep = compute_representation(traj, "cacn")
        with pytest.warns(UserWarning, match="independent conformations"):
            result = dissimilarity(
                rep[::2], rep[1::2], rep.min(), rep.max(),
                x_num=60, s_num=2, random_state=0,
            )
        assert result.n_significant == 0


# ---------------------------------------------------------------------------
# Real dynamics, and the formats it arrives in
# ---------------------------------------------------------------------------
class TestRealTrajectory:
    def test_every_format_reads_the_same_frames(self, corpus):
        for name in ("frame0.xtc", "frame0.dcd", "frame0.nc"):
            traj = md.load(corpus[name], top=corpus["native.pdb"])
            assert traj.n_frames == 501
            assert traj.n_atoms == 22

    def test_storage_precision_changes_the_representation_slightly(self, corpus):
        """Formats do not all store coordinates to the same precision, and
        solvent accessible surface amplifies what they lose.

        In this corpus the NetCDF file carries coordinates rounded to about
        0.001 nm where the XTC and DCD agree to 1e-7. That thousandth of a
        nanometre becomes a 1.2% difference in per-residue accessible surface:
        the geometry is a sum over a sphere of test points, so a rounding at
        the input is magnified about tenfold at the output.

        Worth knowing, and worth bounding, because somebody will one day
        compare an ensemble stored one way against an ensemble stored another.
        """
        matrices = {
            name: compute_representation(
                md.load(corpus[name], top=corpus["native.pdb"]), "sasa"
            )
            for name in ("frame0.xtc", "frame0.dcd", "frame0.nc")
        }
        # XTC and DCD hold the same numbers.
        np.testing.assert_allclose(
            matrices["frame0.xtc"], matrices["frame0.dcd"], rtol=1e-5
        )
        # NetCDF differs, but by a bounded amount.
        relative = np.abs(
            matrices["frame0.nc"] - matrices["frame0.xtc"]
        ) / np.maximum(matrices["frame0.xtc"], 1e-12)
        assert relative.max() < 0.05

    def test_storage_precision_does_not_manufacture_a_difference(self, corpus):
        """The claim that matters. A 1.2% wobble in the representation must not
        become a reported difference between ensembles -- and it does not,
        because it is far below the noise floor."""
        first = compute_representation(
            md.load(corpus["frame0.nc"], top=corpus["native.pdb"]), "sasa"
        )
        second = compute_representation(
            md.load(corpus["frame0.xtc"], top=corpus["native.pdb"]), "sasa"
        )
        low = min(first.min(), second.min())
        high = max(first.max(), second.max())
        result = dissimilarity(
            first, second, low, high, x_num=80, s_num=3, random_state=0
        )
        assert not result.resolved
        assert result.n_significant == 0

    def test_alternate_frames_of_one_trajectory_are_not_called_different(
        self, corpus
    ):
        """The sharpest calibration test available on real data.

        Even-numbered and odd-numbered frames of one trajectory are drawn from
        the same distribution by construction, but adjacent frames are as
        correlated in time as any two frames can be. If time correlation broke
        the permutation null, it would break here first.
        """
        traj = md.load(corpus["frame0.xtc"], top=corpus["native.pdb"])
        rep = compute_representation(traj, "sasa")

        # Confirm the premise: the frames really are correlated.
        centred = rep - rep.mean(axis=0)
        lag_one = (centred[:-1] * centred[1:]).sum(axis=0) / (centred**2).sum(axis=0)
        assert lag_one.max() > 0.2, "these frames are not correlated enough to test"

        resolved = 0
        for seed in range(5):
            result = dissimilarity(
                rep[::2], rep[1::2], rep.min(), rep.max(),
                x_num=80, s_num=3, random_state=seed,
            )
            resolved += int(result.resolved)
        assert resolved == 0


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------
class TestARealisticProtein:
    def test_the_block_path_runs_and_agrees_with_the_direct_computation(
        self, corpus
    ):
        """158 residues is past the block size, so the chunked accumulation
        finally executes. No fixture in this suite reaches it.
        """
        traj = md.load(corpus["1am7_protein.pdb"])
        indices = traj.topology.select("name CB")
        residues = np.array(
            [traj.topology.atom(int(a)).residue.index for a in indices]
        )
        i, j = np.triu_indices(len(indices), 1)
        keep = np.abs(residues[i] - residues[j]) >= 3
        assert keep.sum() > _PAIR_BLOCK, "this protein does not trigger chunking"

        chunked = compute_representation(traj, "cbcn")

        # The same quantity computed in one pass, without blocking.
        pairs = np.column_stack([indices[i[keep]], indices[j[keep]]])
        distances = md.compute_distances(traj, pairs, periodic=False).astype(np.float64)
        weights = 1.0 / (1.0 + np.exp(np.clip(50.0 * (distances - 1.0), -700, 700)))
        direct = np.zeros((traj.n_frames, len(indices)))
        np.add.at(direct.T, i[keep], weights.T)
        np.add.at(direct.T, j[keep], weights.T)

        np.testing.assert_allclose(chunked, direct, rtol=1e-9, atol=1e-12)

    def test_a_realistic_protein_measures_in_reasonable_time(self, corpus):
        import time

        traj = md.load(corpus["1am7_protein.pdb"])
        start = time.perf_counter()
        matrix = compute_representation(traj, "cbcn")
        elapsed = time.perf_counter() - start
        assert matrix.shape[1] > 100
        assert elapsed < 30, f"one frame of 158 residues took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Things that are not what was expected
# ---------------------------------------------------------------------------
class TestNonProtein:
    def test_rna_is_refused_rather_than_measured(self, corpus):
        """2KOC is an RNA hairpin. It has no C-beta atoms and no amino acids,
        and the failure should name that rather than surface as an array error
        somewhere downstream."""
        traj = md.load(corpus["2koc.pdb"])
        with pytest.raises(ValueError, match="C-beta"):
            compute_representation(traj, "cbcn")

    def test_rna_has_no_sequence_to_align(self, corpus):
        traj = md.load(corpus["2koc.pdb"])
        sequence, _ = sequence_of(traj.topology)
        assert sequence == ""

    def test_reconciling_a_protein_with_rna_is_refused(self, corpus):
        protein = Ensemble(md.load(corpus["1vii.pdb"]), label="villin")
        rna = Ensemble(md.load(corpus["2koc.pdb"]), label="RNA hairpin")
        with pytest.raises(ValueError):
            reconcile(protein, rna)


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------
class TestEveryMetricOnRealData:
    @pytest.mark.parametrize("metric", sorted(METRICS))
    def test_metric_runs_on_a_real_ensemble(self, corpus, metric):
        traj = md.load(corpus["frame0.xtc"], top=corpus["native.pdb"])
        rep = compute_representation(traj, "sasa")
        result = dissimilarity(
            rep[:250], rep[251:], rep.min(), rep.max(),
            x_num=60, s_num=2, metric=metric, random_state=0,
        )
        assert np.isfinite(result.global_dissimilarity)
        assert result.noise_floor >= 0

    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_joint_methods_run_on_a_real_ensemble(self, corpus, method):
        from prothon.core.ensemble_metrics import distinguishability

        traj = md.load(corpus["frame0.xtc"], top=corpus["native.pdb"])
        rep = compute_representation(traj, "sasa")
        result = distinguishability(rep[::2], rep[1::2], method, random_state=0)
        assert 0.0 <= result.p_value <= 1.0

    def test_coverage_and_fidelity_run_on_a_real_ensemble(self, corpus):
        traj = md.load(corpus["frame0.xtc"], top=corpus["native.pdb"])
        rep = compute_representation(traj, "sasa")
        result = precision_recall(rep[::2], rep[1::2], random_state=0)
        assert 0.0 <= result.mean_precision <= 1.0
        assert 0.0 <= result.mean_recall <= 1.0
        # Alternate frames of one trajectory: nothing should be flagged.
        assert result.missed().size == 0
        assert result.invented().size == 0
