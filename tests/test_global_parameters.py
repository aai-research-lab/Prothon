"""Tests for order parameters describing the whole molecule.

These give one number per conformation rather than one per residue, so the
representation has a single column. Everything downstream already copes --
what needed checking is that the physics is right and that the reporting does
not say "1/1 residues differ".
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest
from test_ingest import as_residues, build_chains

from prothon import Prothon
from prothon.represent.order_parameters import (
    ORDER_PARAMETERS,
    compute_asph,
    compute_nu,
    compute_ree,
    compute_representation,
    compute_rg,
    resolve_order_parameter,
)

SEQ = "ACDEFHIKLMNPQRSTVWYACDEFHIKLM"      # 29 residues


def shaped(kind: str, n_frames: int = 300, seed: int = 0, seq: str = SEQ):
    """A chain with a genuinely different overall shape.

    The usual fixture is a near-straight chain, on which asphericity is 1 and
    the scaling exponent is 1 whatever else changes -- correct, and useless
    for testing a shape parameter.
    """
    rng = np.random.default_rng(seed)
    top = md.Topology()
    chain = top.add_chain()
    for name in as_residues(seq):
        residue = top.add_residue(name, chain)
        for atom in ("N", "CA", "C", "O", "CB"):
            top.add_atom(atom, md.element.carbon, residue)

    n = len(as_residues(seq))
    xyz = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
    for frame in range(n_frames):
        if kind == "coil":                       # an ideal random walk
            positions = np.cumsum(rng.normal(0, 0.38, (n, 3)), axis=0)
        elif kind == "globule":                  # collapsed
            positions = rng.normal(0, 0.38 * n ** (1 / 3) / 2, (n, 3))
        elif kind == "rod":
            positions = np.zeros((n, 3))
            positions[:, 0] = np.arange(n) * 0.38
            positions += rng.normal(0, 0.01, (n, 3))
        else:
            raise ValueError(kind)
        for atom in top.atoms:
            xyz[frame, atom.index] = positions[atom.residue.index] + rng.normal(
                0, 0.02, 3
            )
    return md.Trajectory(xyz, top)


def with_nonprotein_outliers(traj):
    """Add distant water and calcium without changing the protein."""
    top = md.Topology()
    chain = top.add_chain()
    water = top.add_residue("HOH", chain)
    top.add_atom("O", md.element.oxygen, water)
    calcium = top.add_residue("CAL", chain)
    top.add_atom("CA", md.element.calcium, calcium)
    xyz = np.empty((traj.n_frames, 2, 3), dtype=np.float32)
    xyz[:, 0] = [50.0, 0.0, 0.0]
    xyz[:, 1] = [-50.0, 0.0, 0.0]
    return traj.stack(md.Trajectory(xyz, top))


class TestTheyAreRegistered:
    @pytest.mark.parametrize("name", ["rg", "ree", "asph", "nu"])
    def test_each_is_available_for_comparison(self, name):
        """They were computable in `validate` and not comparable, which is an
        asymmetry with no reason behind it."""
        assert name in ORDER_PARAMETERS
        assert resolve_order_parameter(name).is_global

    def test_the_local_ones_are_not_marked_global(self):
        for name in ("cbcn", "cacn", "caba", "cata", "sasa"):
            assert not resolve_order_parameter(name).is_global

    @pytest.mark.parametrize("name", ["rg", "ree", "asph", "nu"])
    def test_one_column_per_conformation(self, name):
        traj = shaped("coil", 40, seed=1)
        matrix = compute_representation(traj, name)
        assert matrix.shape == (40, 1)
        assert np.isfinite(matrix).all()


class TestThePhysicsIsRight:
    def test_asphericity_is_zero_for_a_sphere_and_one_for_a_rod(self):
        """The two ends of the scale, where the answer is known exactly."""
        assert compute_representation(shaped("globule", 100, seed=2), "asph").mean() < 0.15
        assert compute_representation(shaped("rod", 100, seed=3), "asph").mean() > 0.95

    def test_asphericity_is_bounded(self):
        for kind in ("coil", "globule", "rod"):
            values = compute_representation(shaped(kind, 60, seed=4), "asph")
            assert (values >= 0).all() and (values <= 1).all()

    def test_the_per_conformation_exponent_is_below_the_ensemble_one(self):
        """They are different quantities, and the documentation says so.

        The ensemble fit takes log of the averaged distance; the
        per-conformation fit averages the log. Jensen's inequality makes the
        second smaller, by about 0.03 on an ideal chain, and the gap does not
        shrink with chain length. Anyone quoting the mean of `nu` as a Flory
        exponent would be about 0.03 low.
        """
        rng = np.random.default_rng(20)
        walks = np.cumsum(rng.normal(0, 1, (400, 200, 3)), axis=1)
        separations = np.arange(3, 100)

        per_conformation = np.empty((400, separations.size))
        for k, s in enumerate(separations):
            delta = walks[:, s:] - walks[:, :-s]
            per_conformation[:, k] = np.sqrt((delta**2).sum(-1).mean(axis=1))

        logs = np.log(separations)
        ensemble = np.polyfit(
            logs, np.log(np.sqrt((per_conformation**2).mean(axis=0))), 1
        )[0]
        centred = logs - logs.mean()
        log_profile = np.log(per_conformation)
        each = (
            centred @ (log_profile - log_profile.mean(axis=1, keepdims=True)).T
        ) / (centred @ centred)

        assert ensemble == pytest.approx(0.5, abs=0.03)      # the true value
        assert each.mean() < ensemble - 0.01                 # systematically below
        assert each.mean() == pytest.approx(0.47, abs=0.03)

    def test_the_exponent_is_in_the_physical_range(self):
        """Whatever it is called, it should land where a chain lands."""
        nu = compute_representation(shaped("coil", 300, seed=5), "nu")
        assert 0.35 < nu.mean() < 0.65

    def test_a_collapsed_chain_scales_more_slowly_than_a_coil(self):
        coil = compute_representation(shaped("coil", 200, seed=6), "nu").mean()
        globule = compute_representation(shaped("globule", 200, seed=7), "nu").mean()
        assert globule < coil - 0.2

    def test_a_collapsed_chain_is_smaller(self):
        coil = compute_representation(shaped("coil", 200, seed=8), "rg").mean()
        globule = compute_representation(shaped("globule", 200, seed=9), "rg").mean()
        assert globule < coil

    def test_end_to_end_is_larger_than_the_radius_of_gyration(self):
        traj = shaped("coil", 100, seed=10)
        assert (
            compute_representation(traj, "ree").mean()
            > compute_representation(traj, "rg").mean()
        )

    def test_a_chain_too_short_for_a_scaling_exponent_is_refused(self):
        with pytest.raises(ValueError, match="scaling exponent needs"):
            compute_representation(shaped("coil", 10, seed=11, seq="ACDEF"), "nu")

    def test_shape_defaults_ignore_solvent_and_ions(self):
        protein = shaped("coil", 20, seed=14)
        mixed = with_nonprotein_outliers(protein)
        np.testing.assert_allclose(compute_rg(mixed), compute_rg(protein))
        np.testing.assert_allclose(compute_asph(mixed), compute_asph(protein))
        assert compute_rg(mixed, selection="all").mean() > 3 * compute_rg(mixed).mean()

    @pytest.mark.parametrize("name", ["ree", "nu"])
    @pytest.mark.parametrize(
        "chains",
        [
            [("A", SEQ[:12]), ("B", SEQ[:12])],
            [("A", SEQ[:12]), ("B", SEQ[:13])],
        ],
        ids=["homomer", "heteromer"],
    )
    def test_chain_level_parameters_refuse_an_ambiguous_complex(self, name, chains):
        complex_traj = build_chains(chains, n_frames=3)
        with pytest.raises(ValueError, match="2 protein chains"):
            compute_representation(complex_traj, name)

    def test_chain_level_parameters_accept_an_explicit_chain(self):
        complex_traj = build_chains([("A", SEQ[:12]), ("B", SEQ[:13])], n_frames=3)
        assert compute_ree(complex_traj, chain="A").shape == (3, 1)
        assert compute_nu(complex_traj, chain=1).shape == (3, 1)
        with pytest.raises(ValueError, match="either selection or chain"):
            compute_ree(complex_traj, selection="name CA", chain="A")


class TestThroughAStudy:
    @pytest.fixture(scope="class")
    def files(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("global")
        shaped("coil", 400, seed=12).save_dcd(str(d / "coil.dcd"))
        shaped("globule", 400, seed=13).save_dcd(str(d / "globule.dcd"))
        shaped("coil", 1, seed=12)[0].save_pdb(str(d / "top.pdb"))
        return d

    def test_a_study_on_a_global_parameter(self, files):
        study = Prothon(
            ensembles=[str(files / "coil.dcd"), str(files / "globule.dcd")],
            topology=str(files / "top.pdb"), random_state=0,
        )
        result = study.compare("rg", s_num=3)["rg"][0]
        assert result.resolved
        assert result.local_dissimilarity.size == 1

    def test_the_summary_does_not_count_residues(self, files):
        """"1/1 residues differ" is true of a radius of gyration and says
        nothing anybody wants."""
        study = Prothon(
            ensembles=[str(files / "coil.dcd"), str(files / "globule.dcd")],
            topology=str(files / "top.pdb"), random_state=0,
        )
        study.compare("rg", s_num=3)
        summary = study.summary()
        assert "residues differ" not in summary
        assert "differs" in summary

    def test_local_parameters_still_count_residues(self, files):
        study = Prothon(
            ensembles=[str(files / "coil.dcd"), str(files / "globule.dcd")],
            topology=str(files / "top.pdb"), random_state=0,
        )
        study.compare("cbcn", s_num=3)
        assert "residues differ" in study.summary()

    def test_global_and_local_in_one_run(self, files):
        study = Prothon(
            ensembles=[str(files / "coil.dcd"), str(files / "globule.dcd")],
            topology=str(files / "top.pdb"), random_state=0,
        )
        results = study.compare("rg,nu,cbcn", s_num=3)
        assert set(results) == {"rg", "nu", "cbcn"}

    def test_the_null_is_still_calibrated_with_one_column(self):
        """A single feature is the smallest possible study, and the
        multiplicity correction over one test is a no-op. The rate should
        still be near the threshold."""
        from prothon.compare.dissimilarity import dissimilarity

        flagged = 0
        for seed in range(30):
            rng = np.random.default_rng(500 + seed)
            a, b = rng.normal(size=(800, 1)), rng.normal(size=(800, 1))
            flagged += dissimilarity(
                a, b, -4, 4, x_num=60, s_num=2, random_state=seed,
                block_permutation=False,
            ).n_significant
        assert flagged <= 4          # nominal 5% of 30
