"""Tests for the order parameters and the ensemble matrices built from them."""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from prothon.core.representation import (
    MEASURES,
    compute_caba,
    compute_cacn,
    compute_cata,
    compute_cbcn,
    compute_ensemble_representation,
    compute_sasa,
    describe_measure,
    resolve_measure,
)


class TestMeasureRegistry:
    def test_every_measure_is_computable(self):
        from prothon.core.representation import _COMPUTE

        assert set(MEASURES) == set(_COMPUTE)

    def test_only_torsions_are_circular(self):
        circular = {name for name, spec in MEASURES.items() if spec.circular}
        assert circular == {"cata"}

    def test_resolve_is_case_insensitive(self):
        assert resolve_measure("CBCN").name == "cbcn"
        assert resolve_measure("  sasa ").name == "sasa"

    def test_unknown_measure_suggests_a_neighbour(self):
        with pytest.raises(ValueError, match="Did you mean cbcn"):
            resolve_measure("cbnc")

    def test_unknown_measure_lists_the_options(self):
        with pytest.raises(ValueError, match="cacn"):
            resolve_measure("zzzz")

    def test_describe_includes_units(self):
        assert "nm^2" in describe_measure("sasa")


class TestContactNumbers:
    def test_shape_is_frames_by_residues(self, ensemble_files, topology_file):
        traj = md.load(ensemble_files[0], top=topology_file)
        result = compute_cbcn(traj)
        assert result.shape == (traj.n_frames, traj.topology.n_residues)

    def test_values_are_non_negative_and_finite(self, ensemble_files, topology_file):
        traj = md.load(ensemble_files[0], top=topology_file)
        result = compute_cacn(traj)
        assert np.isfinite(result).all()
        assert (result >= 0).all()

    def test_matches_the_reference_implementation(self, ensemble_files, topology_file):
        """The vectorised version must reproduce the original loop exactly.

        This is the whole warrant for the rewrite: same numbers, less time.
        """
        from itertools import combinations

        traj = md.load(ensemble_files[0], top=topology_file)[:20]
        indices = traj.topology.select("name CB")

        pairs = np.array(
            [
                (i, j)
                for i, j in combinations(indices, 2)
                if abs(
                    traj.topology.atom(i).residue.index
                    - traj.topology.atom(j).residue.index
                )
                > 2
            ]
        )
        expected = []
        for idx in indices:
            selected = pairs[[idx in pair for pair in pairs]]
            distances = md.compute_distances(traj, selected)
            argument = np.clip(50 * (distances.astype(np.float64) - 1), -700, 700)
            expected.append(np.sum(1.0 / (1 + np.exp(argument)), axis=1))
        expected = np.transpose(np.array(expected))

        # Agreement to float32 precision: the inputs are float32 coordinates,
        # and the two implementations accumulate the same terms in a different
        # order.
        np.testing.assert_allclose(compute_cbcn(traj), expected, rtol=1e-6, atol=1e-9)

    def test_compaction_raises_contact_number_where_expected(
        self, ensemble_files, topology_file
    ):
        # Ensemble c has its tail pulled inward from residue 7 onward.
        extended = compute_cbcn(md.load(ensemble_files[0], top=topology_file)).mean(0)
        compact = compute_cbcn(md.load(ensemble_files[2], top=topology_file)).mean(0)
        assert compact[8:].mean() > extended[8:].mean()

    def test_missing_atoms_fail_with_a_usable_message(self, topology_file):
        traj = md.load(topology_file)
        stripped = traj.atom_slice(traj.topology.select("name CA"))
        with pytest.raises(ValueError, match="No C-beta atoms found"):
            compute_cbcn(stripped)


class TestAngles:
    def test_bond_angle_count_and_range(self, ensemble_files, topology_file):
        traj = md.load(ensemble_files[0], top=topology_file)
        result = compute_caba(traj)
        n_ca = len(traj.topology.select("name CA"))
        assert result.shape == (traj.n_frames, n_ca - 2)
        assert (result >= 0).all() and (result <= np.pi).all()

    def test_torsion_count_and_range(self, ensemble_files, topology_file):
        traj = md.load(ensemble_files[0], top=topology_file)
        result = compute_cata(traj)
        n_ca = len(traj.topology.select("name CA"))
        assert result.shape == (traj.n_frames, n_ca - 3)
        assert (result >= -np.pi - 1e-6).all() and (result <= np.pi + 1e-6).all()

    def test_short_chains_are_refused(self, topology_file):
        traj = md.load(topology_file)
        short = traj.atom_slice(
            [a.index for a in traj.topology.atoms if a.residue.index < 2]
        )
        with pytest.raises(ValueError, match="at least 4 C-alpha"):
            compute_cata(short)


class TestSasa:
    def test_shape_and_non_negativity(self, ensemble_files, topology_file):
        traj = md.load(ensemble_files[0], top=topology_file)[:10]
        result = compute_sasa(traj)
        assert result.shape == (10, traj.topology.n_residues)
        assert (result >= 0).all()


class TestEnsembleRepresentation:
    def test_one_matrix_per_file(self, ensemble_files, topology_file):
        reps = compute_ensemble_representation(ensemble_files, topology_file, "cbcn")
        assert len(reps) == len(ensemble_files)
        assert len({rep.shape[1] for rep in reps}) == 1

    def test_unknown_measure_is_refused(self, ensemble_files, topology_file):
        with pytest.raises(ValueError, match="Unknown measure"):
            compute_ensemble_representation(ensemble_files, topology_file, "nope")
