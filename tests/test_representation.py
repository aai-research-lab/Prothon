"""Tests for the order parameters and the ensemble matrices built from them."""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from prothon.ingest import feature_residues
from prothon.represent.order_parameters import (
    ORDER_PARAMETERS,
    compute_caba,
    compute_cacn,
    compute_cata,
    compute_cbcn,
    compute_ensemble_representation,
    compute_representation,
    compute_sasa,
    describe_order_parameter,
    resolve_order_parameter,
)


def mixed_system(
    chain_lengths=(4,),
    *,
    n_frames=2,
    break_after=None,
    numbering_gap_after=None,
    extras=False,
):
    """Protein chains plus optional ligand, water, and a calcium named CA."""
    top = md.Topology()
    atoms_by_residue = []
    for chain_index, length in enumerate(chain_lengths):
        chain = top.add_chain()
        chain.chain_id = chr(ord("A") + chain_index)
        previous_c = None
        for position in range(length):
            gap = (
                8
                if numbering_gap_after == (chain_index, position - 1)
                or (
                    numbering_gap_after is not None
                    and numbering_gap_after[0] == chain_index
                    and position - 1 > numbering_gap_after[1]
                )
                else 0
            )
            residue = top.add_residue("ALA", chain, resSeq=position + 1 + gap)
            atoms = {
                name: top.add_atom(name, element, residue)
                for name, element in (
                    ("N", md.element.nitrogen),
                    ("CA", md.element.carbon),
                    ("C", md.element.carbon),
                    ("O", md.element.oxygen),
                    ("CB", md.element.carbon),
                )
            }
            top.add_bond(atoms["N"], atoms["CA"])
            top.add_bond(atoms["CA"], atoms["C"])
            top.add_bond(atoms["C"], atoms["O"])
            top.add_bond(atoms["CA"], atoms["CB"])
            if previous_c is not None and break_after != (chain_index, position - 1):
                top.add_bond(previous_c, atoms["N"])
            previous_c = atoms["C"]
            atoms_by_residue.append((chain_index, position, atoms))

    extra_atoms = {}
    if extras:
        extra_chain = top.add_chain()
        ligand = top.add_residue("LIG", extra_chain, resSeq=1)
        extra_atoms["ligand"] = top.add_atom("C1", md.element.carbon, ligand)
        water = top.add_residue("HOH", extra_chain, resSeq=2)
        extra_atoms["water"] = top.add_atom("O", md.element.oxygen, water)
        calcium = top.add_residue("CAL", extra_chain, resSeq=3)
        extra_atoms["calcium"] = top.add_atom("CA", md.element.calcium, calcium)

    xyz = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
    offsets = {"N": -0.12, "CA": 0.0, "C": 0.12, "O": 0.2, "CB": 0.15}
    for chain_index, position, atoms in atoms_by_residue:
        for name, atom in atoms.items():
            xyz[:, atom.index] = [position * 0.38, chain_index * 0.4, offsets[name]]
    for offset, atom in enumerate(extra_atoms.values(), start=1):
        xyz[:, atom.index] = [5.0 + offset, 0.0, 0.0]
    return md.Trajectory(xyz, top)


class TestMeasureRegistry:
    def test_every_measure_is_computable(self):
        from prothon.represent.order_parameters import _COMPUTE

        assert set(ORDER_PARAMETERS) == set(_COMPUTE)

    def test_only_torsions_are_circular(self):
        circular = {name for name, spec in ORDER_PARAMETERS.items() if spec.circular}
        assert circular == {"cata"}

    def test_resolve_is_case_insensitive(self):
        assert resolve_order_parameter("CBCN").name == "cbcn"
        assert resolve_order_parameter("  sasa ").name == "sasa"

    def test_an_unknown_name_suggests_a_neighbour(self):
        with pytest.raises(ValueError, match="Did you mean cbcn"):
            resolve_order_parameter("cbnc")

    def test_the_2x_names_still_import(self):
        """`measure` collided with `metric`, which means something else here,
        so the registry took the term the paper uses. Published code keeps
        importing the old names."""
        from prothon.represent.order_parameters import (
            MEASURES,
            Measure,
            OrderParameter,
            describe_measure,
            resolve_measure,
        )

        assert MEASURES is ORDER_PARAMETERS
        assert Measure is OrderParameter
        assert resolve_measure("cbcn").name == "cbcn"
        assert "cbcn" in describe_measure("cbcn")

    def test_unknown_name_lists_the_options(self):
        with pytest.raises(ValueError, match="cacn"):
            resolve_order_parameter("zzzz")

    def test_describe_includes_units(self):
        assert "nm^2" in describe_order_parameter("sasa")


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

    def test_calcium_named_ca_is_not_a_protein_feature(self):
        traj = mixed_system(extras=True)
        result = compute_cacn(traj)
        assert result.shape == (traj.n_frames, 4)
        assert feature_residues(traj.topology, "cacn") == [
            (0,), (1,), (2,), (3,)
        ]

    def test_inter_chain_contacts_are_kept(self):
        traj = mixed_system((2, 2))
        result = compute_cacn(traj)
        # Every atom sees the two atoms of the other chain. All within-chain
        # pairs are below the minimum sequence separation and are excluded.
        assert result.shape == (traj.n_frames, 4)
        assert (result > 1.9).all()


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

    def test_windows_do_not_cross_chain_boundaries(self):
        traj = mixed_system((4, 4))
        assert compute_caba(traj).shape == (traj.n_frames, 4)
        assert compute_cata(traj).shape == (traj.n_frames, 2)
        assert feature_residues(traj.topology, "caba") == [
            (0, 1, 2), (1, 2, 3), (4, 5, 6), (5, 6, 7)
        ]

    def test_windows_do_not_cross_a_missing_peptide_bond(self):
        traj = mixed_system((5,), break_after=(0, 1))
        assert compute_caba(traj).shape == (traj.n_frames, 1)
        assert feature_residues(traj.topology, "caba") == [(2, 3, 4)]
        with pytest.raises(ValueError, match="no valid windows"):
            compute_cata(traj)

    def test_windows_do_not_cross_a_missing_residue_gap(self):
        traj = mixed_system((5,), numbering_gap_after=(0, 1))
        assert compute_caba(traj).shape == (traj.n_frames, 1)
        assert feature_residues(traj.topology, "caba") == [(2, 3, 4)]


class TestSasa:
    def test_shape_and_non_negativity(self, ensemble_files, topology_file):
        traj = md.load(ensemble_files[0], top=topology_file)[:10]
        result = compute_sasa(traj)
        assert result.shape == (10, traj.topology.n_residues)
        assert (result >= 0).all()

    def test_only_protein_residues_are_reported_but_ligand_still_shields(self):
        traj = mixed_system((1,), extras=True)
        protein_ca = next(
            atom for atom in traj.topology.atoms
            if atom.residue.name == "ALA" and atom.name == "CA"
        )
        ligand = next(
            atom for atom in traj.topology.atoms if atom.residue.name == "LIG"
        )
        traj.xyz[0, ligand.index] = traj.xyz[0, protein_ca.index] + [0.25, 0.0, 0.0]
        # Keep the unbound ligand away from every other atom. MDTraj's native
        # SASA routine terminates the process for exactly coincident atoms;
        # water sits at x=7 in this fixture.
        traj.xyz[1, ligand.index] = [20.0, 0.0, 0.0]

        result = compute_sasa(traj)
        assert result.shape == (2, 1)
        assert result[0, 0] < result[1, 0]
        assert feature_residues(traj.topology, "sasa") == [(0,)]

        ligand_only = compute_sasa(traj, report_selection="resname LIG")
        assert ligand_only.shape == (2, 1)


class TestEnsembleRepresentation:
    @pytest.mark.parametrize("name", ["cbcn", "cacn", "caba", "cata", "sasa"])
    def test_mixed_system_feature_identity_matches_every_matrix(self, name):
        traj = mixed_system((5, 4), extras=True)
        matrix = compute_representation(traj, name)
        assert len(feature_residues(traj.topology, name)) == matrix.shape[1]

    def test_one_matrix_per_file(self, ensemble_files, topology_file):
        reps = compute_ensemble_representation(ensemble_files, topology_file, "cbcn")
        assert len(reps) == len(ensemble_files)
        assert len({rep.shape[1] for rep in reps}) == 1

    def test_unknown_measure_is_refused(self, ensemble_files, topology_file):
        with pytest.raises(ValueError, match="Unknown order parameter"):
            compute_ensemble_representation(ensemble_files, topology_file, "nope")
