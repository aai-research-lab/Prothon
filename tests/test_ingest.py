"""Tests for the ingest layer.

The interesting cases are the ones where a naive column-for-column comparison
would give a plausible, wrong answer: a mutation to or from glycine, which
removes a C-beta and shifts every ``cbcn`` column after it, and a deletion,
which breaks the windows the angular measures are defined on.
"""

from __future__ import annotations

import json
import warnings

import mdtraj as md
import numpy as np
import pytest

from prothon.ingest import (
    Ensemble,
    align,
    feature_identity,
    feature_residues,
    reconcile,
    same_topology,
    sequence_of,
    topology_fingerprint,
)
from prothon.ingest.sequence import THREE_TO_ONE, residue_letter
from prothon.represent.order_parameters import compute_cbcn, compute_ensemble_representation

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def build(residue_names, n_frames=30, seed=0, compact_from=None):
    """A chain with the given residues. Glycines get no C-beta, as in reality."""
    rng = np.random.default_rng(seed)
    top = md.Topology()
    chain = top.add_chain()
    for name in residue_names:
        residue = top.add_residue(name, chain)
        atoms = ["N", "CA", "C", "O"] + ([] if name in ("GLY", "G") else ["CB"])
        for atom in atoms:
            top.add_atom(atom, md.element.carbon, residue)

    xyz = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
    for atom in top.atoms:
        i = atom.residue.index
        offset = {"N": -0.12, "CA": 0.0, "C": 0.12, "O": 0.2, "CB": 0.15}[atom.name]
        base = np.array([i * 0.38, offset, 0.0])
        if compact_from is not None and i >= compact_from:
            base = base - np.array([0.5 * (i - compact_from) * 0.38, 0.0, 0.0])
        xyz[:, atom.index, :] = base
    xyz += rng.normal(0, 0.04, xyz.shape).astype(np.float32)
    return md.Trajectory(xyz, top)


def build_chains(chains, n_frames=2):
    """A small labelled complex for topology and chain-mapping tests."""
    top = md.Topology()
    for chain_id, sequence in chains:
        chain = top.add_chain()
        chain.chain_id = chain_id
        for name in as_residues(sequence):
            residue = top.add_residue(name, chain)
            atoms = ["N", "CA", "C", "O"] + ([] if name == "GLY" else ["CB"])
            for atom in atoms:
                top.add_atom(atom, md.element.carbon, residue)
    return md.Trajectory(
        np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32), top
    )


ONE_TO_THREE = {v: k for k, v in reversed(list(THREE_TO_ONE.items()))}


def as_residues(sequence):
    return [ONE_TO_THREE[letter] for letter in sequence]


class TestResidueNaming:
    @pytest.mark.parametrize(
        "name,letter",
        [("ALA", "A"), ("HIE", "H"), ("HIP", "H"), ("HSD", "H"), ("CYX", "C"),
         ("ASH", "D"), ("GLH", "E"), ("LYN", "K"), ("MSE", "M")],
    )
    def test_force_field_dialects_resolve(self, name, letter):
        """mdtraj returns None for AMBER's HIE and CYX.

        A sequence read straight from an AMBER-prepared topology would come
        back with holes, and every alignment built on it would be wrong.
        """
        traj = build([name], n_frames=1)
        assert residue_letter(next(traj.topology.residues)) == letter

    def test_unknown_residue_becomes_x_not_a_hole(self):
        traj = build(["ALA", "ALA"], n_frames=1)
        top = traj.topology
        list(top.residues)[1].name = "ZZZ"
        sequence, indices = sequence_of(top)
        assert sequence == "AX"
        assert len(indices) == 2

    def test_sequence_carries_residue_indices(self):
        traj = build(as_residues("ACDE"), n_frames=1)
        sequence, indices = sequence_of(traj.topology)
        assert sequence == "ACDE"
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])


class TestAlignment:
    def test_identical_sequences_align_fully(self):
        result = align(UBIQUITIN, UBIQUITIN)
        assert result.n_aligned == len(UBIQUITIN)
        assert result.identity == 1.0

    def test_point_mutation_is_located(self):
        mutant = UBIQUITIN[:47] + "R" + UBIQUITIN[48:]
        result = align(UBIQUITIN, mutant)
        differing = [i for i, j in result.columns if UBIQUITIN[i] != mutant[j]]
        assert differing == [47]

    def test_terminal_truncation_is_free(self):
        short = UBIQUITIN[6:-4]
        result = align(UBIQUITIN, short)
        assert result.identity == 1.0
        assert tuple(result.columns[0]) == (6, 0)
        assert tuple(result.columns[-1]) == (len(UBIQUITIN) - 5, len(short) - 1)

    def test_internal_deletion_is_one_gap_not_several(self):
        # The whole point of affine penalties: a missing loop is one event.
        deleted = UBIQUITIN[:40] + UBIQUITIN[45:]
        result = align(UBIQUITIN, deleted)
        runs = sum(
            1 for k, c in enumerate(result.gapped_b)
            if c == "-" and (k == 0 or result.gapped_b[k - 1] != "-")
        )
        assert result.gapped_b.count("-") == 5
        assert runs == 1

    def test_empty_sequence_is_refused(self):
        with pytest.raises(ValueError, match="empty sequence"):
            align("", "ACDE")


class TestEnsemble:
    def test_weights_are_normalised(self, tmp_path):
        traj = build(as_residues("ACDEF"), n_frames=10)
        ens = Ensemble(traj, label="x", weights=np.full(10, 3.0))
        assert ens.weights.sum() == pytest.approx(1.0)

    def test_uniform_weights_when_none_given(self):
        ens = Ensemble(build(as_residues("ACDEF"), n_frames=8), label="x")
        assert ens.weights is None
        np.testing.assert_allclose(ens.frame_weights, 1 / 8)

    def test_negative_weights_are_refused(self):
        traj = build(as_residues("ACDEF"), n_frames=4)
        with pytest.raises(ValueError, match="log-weights"):
            Ensemble(traj, weights=[0.5, -0.1, 0.3, 0.3])

    def test_wrong_number_of_weights_is_refused(self):
        traj = build(as_residues("ACDEF"), n_frames=4)
        with pytest.raises(ValueError, match="exactly one conformation"):
            Ensemble(traj, weights=[0.5, 0.5])

    def test_empty_ensemble_is_refused(self):
        traj = build(as_residues("ACDEF"), n_frames=4)
        with pytest.raises(ValueError, match="no conformations"):
            Ensemble(traj[0:0], label="empty")

    def test_subsample_carries_and_renormalises_weights(self):
        traj = build(as_residues("ACDEF"), n_frames=20)
        ens = Ensemble(traj, weights=np.arange(1, 21, dtype=float))
        small = ens.subsample(5, random_state=0)
        assert small.n_frames == 5
        assert small.weights.sum() == pytest.approx(1.0)
        assert small.provenance["subsampled_to"] == 5

    def test_subsample_is_reproducible(self):
        ens = Ensemble(build(as_residues("ACDEF"), n_frames=50))
        a = ens.subsample(10, random_state=3).trajectory.xyz
        b = ens.subsample(10, random_state=3).trajectory.xyz
        np.testing.assert_array_equal(a, b)

    def test_quality_finds_a_chain_break(self):
        traj = build(as_residues("ACDEFGHIK"), n_frames=3)
        # Push the tail away, breaking the chain between residues 4 and 5.
        moved = traj.xyz.copy()
        tail = [a.index for a in traj.topology.atoms if a.residue.index >= 5]
        moved[:, tail, 0] += 2.0
        report = Ensemble(md.Trajectory(moved, traj.topology), label="broken").quality()
        assert 4 in report.chain_breaks
        assert any("chain break" in note for note in report.warnings())

    def test_from_pdb_models_reads_a_directory(self, tmp_path):
        traj = build(as_residues("ACDEF"), n_frames=5)
        for i in range(5):
            traj[i].save_pdb(str(tmp_path / f"sample_{i}.pdb"))
        ens = Ensemble.from_pdb_models(str(tmp_path), label="generated")
        assert ens.n_frames == 5
        assert ens.provenance["n_files"] == 5
        assert ens.provenance["sampling_kind"] == "iid"

    def test_from_pdb_models_refuses_a_mismatched_model(self, tmp_path):
        build(as_residues("ACDEF"), n_frames=1)[0].save_pdb(str(tmp_path / "a.pdb"))
        build(as_residues("ACDE"), n_frames=1)[0].save_pdb(str(tmp_path / "b.pdb"))
        with pytest.raises(ValueError, match="same molecule"):
            Ensemble.from_pdb_models(str(tmp_path))

    def test_from_pdb_models_refuses_an_equal_count_mutant(self, tmp_path):
        """Joining models cannot use atom count as molecule identity."""
        build(as_residues("ACDEF"), n_frames=1)[0].save_pdb(str(tmp_path / "a.pdb"))
        build(as_residues("ACNEF"), n_frames=1)[0].save_pdb(str(tmp_path / "b.pdb"))
        with pytest.raises(ValueError, match="atom count alone"):
            Ensemble.from_pdb_models(str(tmp_path))

    def test_missing_file_is_named(self):
        with pytest.raises(FileNotFoundError, match="nowhere.xtc"):
            Ensemble.from_trajectory("nowhere.xtc", "nowhere.pdb")


class TestTopologyIdentity:
    def test_identical_independent_copies_have_one_fingerprint(self):
        a = build(as_residues("ACDE"), n_frames=1)
        b = build(as_residues("ACDE"), n_frames=1, seed=9)
        assert a.topology is not b.topology
        assert topology_fingerprint(a) == topology_fingerprint(b)
        assert same_topology(a, b)
        assert topology_fingerprint(a).hexdigest() == topology_fingerprint(b).hexdigest()

    def test_equal_atom_count_sequences_are_not_identical(self):
        a = build(as_residues("ACDE"), n_frames=1)
        b = build(as_residues("ACNE"), n_frames=1)
        assert a.n_atoms == b.n_atoms
        assert not same_topology(a, b)

    def test_atom_name_and_element_are_identity(self):
        named = build(as_residues("ACDE"), n_frames=1)
        renamed = build(as_residues("ACDE"), n_frames=1)
        renamed.topology.atom(0).name = "P"
        renamed.topology.atom(0).element = md.element.phosphorus
        assert named.n_atoms == renamed.n_atoms
        assert not same_topology(named, renamed)

    def test_bond_connectivity_distinguishes_equal_atom_isomers(self):
        linear = build(as_residues("ACDE"), n_frames=1)
        branched = build(as_residues("ACDE"), n_frames=1)
        linear.topology.add_bond(linear.topology.atom(0), linear.topology.atom(1))
        branched.topology.add_bond(branched.topology.atom(0), branched.topology.atom(2))
        assert linear.n_atoms == branched.n_atoms
        assert not same_topology(linear, branched)

    def test_a_missing_atom_changes_the_fingerprint(self):
        full = build(as_residues("ACDE"), n_frames=1)
        keep = [atom.index for atom in full.topology.atoms if atom.index != 3]
        missing = full.atom_slice(keep)
        assert not same_topology(full, missing)

    def test_chain_order_is_identity(self):
        ordered = build_chains((("A", "ACDE"), ("B", "HIKL")))
        swapped = build_chains((("B", "HIKL"), ("A", "ACDE")))
        assert ordered.n_atoms == swapped.n_atoms
        assert not same_topology(ordered, swapped)

    def test_manifest_form_is_compact_and_deterministic(self):
        ensemble = Ensemble(build(as_residues("ACDE"), n_frames=1))
        digest = ensemble.to_dict()["topology_fingerprint"]
        assert len(digest) == 64
        assert digest == ensemble.topology_fingerprint.hexdigest()


class TestFeatureIdentity:
    def test_cbcn_does_not_renumber_around_glycine(self):
        traj = build(as_residues("AGV"), n_frames=1)
        index, labels = feature_identity(traj.topology, "cbcn")
        np.testing.assert_array_equal(index, [1, 3])
        assert labels.tolist() == ["1", "3"]

    def test_multichain_labels_are_unambiguous(self):
        traj = build_chains((("A", "AC"), ("B", "DE")))
        index, labels = feature_identity(traj.topology, "cacn")
        np.testing.assert_array_equal(index, [1, 2, 3, 4])
        assert labels.tolist() == ["A:1", "A:2", "B:1", "B:2"]


class TestReconcile:
    def test_identical_ensembles_reconcile_to_themselves(self):
        a = Ensemble(build(as_residues(UBIQUITIN[:20])), label="a")
        b = Ensemble(build(as_residues(UBIQUITIN[:20]), seed=1), label="b")
        corr = reconcile(a, b)
        assert corr.is_identical
        assert corr.n_aligned == 20
        assert corr.identity == 1.0

    def test_point_mutant_is_named_as_a_paper_would(self):
        wt_seq = UBIQUITIN[:20]
        mutant_seq = wt_seq[:11] + "A" + wt_seq[12:]
        wt = Ensemble(build(as_residues(wt_seq)), label="wild type")
        mut = Ensemble(build(as_residues(mutant_seq), seed=1), label="mutant")
        corr = reconcile(wt, mut)
        assert len(corr.substitutions) == 1
        assert str(corr.substitutions[0]) == f"{wt_seq[11]}12A"
        assert not corr.is_identical

    def test_truncated_construct_leaves_unmatched_residues(self):
        full = Ensemble(build(as_residues(UBIQUITIN[:24])), label="full")
        short = Ensemble(build(as_residues(UBIQUITIN[4:24]), seed=1), label="short")
        corr = reconcile(full, short)
        assert corr.n_aligned == 20
        assert corr.unmatched_a.size == 4
        assert corr.unmatched_b.size == 0
        np.testing.assert_array_equal(corr.unmatched_a, [0, 1, 2, 3])

    def test_unrelated_sequences_are_refused_on_coverage(self):
        """Identity alone does not catch this.

        With free end gaps these two align on 2 columns at 50% identity --
        clearing any identity floor while covering a twentieth of the molecule.
        Coverage is the guard that works.
        """
        a = Ensemble(build(as_residues("ACDEFGHIKLMNPQRSTVWY" * 2)), label="a")
        b = Ensemble(build(as_residues("WWWWWWWWWWPPPPPPPPPP" * 2), seed=1), label="b")
        with pytest.raises(ValueError, match="describes a fragment"):
            reconcile(a, b)

    def test_a_domain_inside_its_parent_is_fully_covered(self):
        """Coverage is measured against the shorter sequence, so a domain
        contained in a larger protein passes without an override -- every
        residue of it found a counterpart. It is unrelated sequences, which
        cover neither, that the floor is for."""
        parent = Ensemble(build(as_residues(UBIQUITIN[:40])), label="parent")
        domain = Ensemble(build(as_residues(UBIQUITIN[:12]), seed=1), label="domain")
        corr = reconcile(parent, domain)
        assert corr.n_aligned == 12
        assert corr.coverage == 1.0
        assert corr.unmatched_a.size == 28

    def test_differing_chain_counts_are_refused(self):
        # A monomer against a dimer of the same sequence. Concatenating the
        # chains would let the aligner slide one chain against the other,
        # which is cheap in score and nonsense as a map.
        monomer_traj = build(as_residues("ACDEFGHIKL"))
        dimer_top = md.Topology()
        for _ in range(2):
            chain = dimer_top.add_chain()
            for name in as_residues("ACDEFGHIKL"):
                residue = dimer_top.add_residue(name, chain)
                atoms = ["N", "CA", "C", "O"] + ([] if name == "GLY" else ["CB"])
                for atom in atoms:
                    dimer_top.add_atom(atom, md.element.carbon, residue)
        doubled = np.concatenate([monomer_traj.xyz, monomer_traj.xyz + 3.0], axis=1)
        dimer = Ensemble(md.Trajectory(doubled, dimer_top), label="dimer")
        monomer = Ensemble(monomer_traj, label="monomer")
        with pytest.raises(ValueError, match="protein chain"):
            reconcile(monomer, dimer)

    def test_reordered_named_chains_are_paired_by_identity(self):
        ordered = Ensemble(
            build_chains((("A", "ACDE"), ("B", "HIKL"))), label="ordered"
        )
        swapped = Ensemble(
            build_chains((("B", "HIKL"), ("A", "ACDE"))), label="swapped"
        )
        corr = reconcile(ordered, swapped)
        mapping = corr.residue_map()
        assert mapping[0] == 4  # chain A follows chain B in the second topology
        assert mapping[4] == 0
        assert corr.identity == 1.0


class TestFeatureColumns:
    @pytest.mark.parametrize("measure", ["cbcn", "cacn", "caba", "cata", "sasa"])
    def test_map_width_matches_the_matrix(self, measure, tmp_path):
        """The map has to stay in step with the functions it describes."""
        traj = build(as_residues("ACDEFGHIKLM"), n_frames=5)
        path, top_path = tmp_path / "t.dcd", tmp_path / "t.pdb"
        traj.save_dcd(str(path))
        traj[0].save_pdb(str(top_path))
        matrix = compute_ensemble_representation([str(path)], str(top_path), measure)[0]
        assert len(feature_residues(traj.topology, measure)) == matrix.shape[1]

    def test_glycine_mutation_shifts_cbcn_columns(self):
        """The case a column-for-column comparison gets wrong.

        Glycine has no C-beta, so mutating a residue to glycine removes one
        cbcn column and renumbers every column after it. Comparing column k to
        column k would compare different residues from the mutation onward and
        report a difference along the whole C-terminal half.
        """
        wt_seq = "ACDEFHIKLM"
        mut_seq = "ACDEGHIKLM"  # F5G
        wt = Ensemble(build(as_residues(wt_seq)), label="wt")
        mut = Ensemble(build(as_residues(mut_seq), seed=1), label="F5G")

        wt_cols = feature_residues(wt.topology, "cbcn")
        mut_cols = feature_residues(mut.topology, "cbcn")
        assert len(wt_cols) == 10 and len(mut_cols) == 9  # the missing C-beta

        corr = reconcile(wt, mut)
        take_wt, take_mut = corr.columns_for("cbcn", wt.topology, mut.topology)

        assert len(take_wt) == len(take_mut) == 9
        # Every paired column must describe corresponding residues.
        mapping = corr.residue_map()
        for i, j in zip(take_wt, take_mut):
            assert mapping[wt_cols[i][0]] == mut_cols[j][0]
        # And the glycine column of the wild type is dropped, not misaligned.
        assert 4 not in [wt_cols[i][0] for i in take_wt]

    def test_deletion_drops_only_the_broken_angular_windows(self):
        """A virtual bond angle needs three *consecutive* residues.

        Where a deletion removes residue i+1, the residues either side still
        correspond individually, but the angle between them is not the same
        quantity and the column must be dropped.
        """
        full_seq = "ACDEFHIKLMNPQR"
        short_seq = full_seq[:6] + full_seq[7:]  # delete residue index 6
        full = Ensemble(build(as_residues(full_seq)), label="full")
        short = Ensemble(build(as_residues(short_seq), seed=1), label="del")

        corr = reconcile(full, short)
        take_full, take_short = corr.columns_for("caba", full.topology, short.topology)

        n_windows_full = len(feature_residues(full.topology, "caba"))
        assert len(take_full) == len(take_short)
        # The three windows spanning the deleted residue cannot survive.
        assert len(take_full) == n_windows_full - 3

    def test_columns_line_up_for_a_real_comparison(self):
        """End to end: reconcile, slice, and the matrices are comparable."""
        wt_seq = "ACDEFHIKLMNPQR"
        mut_seq = "ACDEGHIKLMNPQR"  # F5G, removing a C-beta
        wt = Ensemble(build(as_residues(wt_seq), n_frames=40))
        mut = Ensemble(build(as_residues(mut_seq), n_frames=40, seed=2))

        rep_wt = compute_cbcn(wt.trajectory)
        rep_mut = compute_cbcn(mut.trajectory)
        assert rep_wt.shape[1] != rep_mut.shape[1]  # 2.1 would refuse here

        corr = reconcile(wt, mut)
        take_wt, take_mut = corr.columns_for("cbcn", wt.topology, mut.topology)
        aligned_wt = rep_wt[:, take_wt]
        aligned_mut = rep_mut[:, take_mut]
        assert aligned_wt.shape[1] == aligned_mut.shape[1] > 0


class TestStudyAcrossMolecules:
    """`Prothon.from_ensembles` end to end, where the molecules differ."""

    def _study(self, tmp_path, wt_seq, mut_seq, compact_from=8, n=300):
        from prothon import Prothon

        wt = Ensemble(build(as_residues(wt_seq), n_frames=n, seed=1), label="wild type")
        mut = Ensemble(
            build(as_residues(mut_seq), n_frames=n, seed=2, compact_from=compact_from),
            label="mutant",
        )
        return Prothon.from_ensembles(
            [wt, mut], output_dir=str(tmp_path), random_state=0
        )

    def test_a_glycine_mutant_can_be_compared_at_all(self, tmp_path):
        # F5G removes a C-beta, so the two representations have different
        # widths and the file-based route refuses outright.
        study = self._study(tmp_path, "ACDEFHIKLMNPQR", "ACDEGHIKLMNPQR")
        results = study.compare_ensembles(order_parameters="cbcn", s_num=2)
        assert len(results["cbcn"]) == 1
        assert results["cbcn"][0].resolved

    def test_features_are_labelled_by_reference_numbering(self, tmp_path):
        """The mistake this exists to prevent.

        After reconciliation the columns are a subset of the reference's.
        Numbering them 1..n would put the label of one residue under the value
        of another, and the figure would look entirely reasonable.
        """
        study = self._study(tmp_path, "ACDEFHIKLMNPQR", "ACDEGHIKLMNPQR")
        result = study.compare_ensembles(order_parameters="cbcn", s_num=2)["cbcn"][0]

        index = result.feature_index
        assert index is not None
        assert len(index) == len(result.local_dissimilarity)
        # Residue 5 is the glycine: it has no C-beta in the mutant, so it
        # cannot appear, and everything after it keeps its own number.
        assert 5 not in index
        assert list(index) == [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        assert result.feature_labels.tolist() == [
            "1", "2", "3", "4", "6", "7", "8", "9", "10", "11",
            "12", "13", "14",
        ]

    def test_identical_molecules_need_no_reconciliation(self, tmp_path):
        study = self._study(tmp_path, "ACDEFHIKLMNPQR", "ACDEFHIKLMNPQR")
        assert study.shares_topology
        result = study.compare_ensembles(order_parameters="cbcn", s_num=2)["cbcn"][0]
        # Identity must remain explicit: CBCN's columns are not necessarily
        # residues 1..n because glycines do not contribute a C-beta.
        np.testing.assert_array_equal(
            result.feature_index, np.arange(1, len(result.local_dissimilarity) + 1)
        )
        assert len(result.feature_labels) == len(result.local_dissimilarity)

    def test_identical_agv_fast_path_keeps_the_glycine_gap(self, tmp_path):
        study = self._study(tmp_path, "AGV", "AGV", n=20)
        reference = np.zeros((20, 2))
        other = np.zeros((20, 2))
        _, _, index, labels = study._align_columns(
            reference, other, 0, 1, "cbcn"
        )
        np.testing.assert_array_equal(index, [1, 3])
        assert labels.tolist() == ["1", "3"]

    def test_equal_count_mutant_does_not_take_the_fast_path(self, tmp_path):
        from prothon import Prothon

        wild_type = Ensemble(build(as_residues("ACDE"), n_frames=4), label="wt")
        mutant = Ensemble(build(as_residues("ACNE"), n_frames=4), label="D3N")
        assert wild_type.trajectory.n_atoms == mutant.trajectory.n_atoms

        study = Prothon.from_ensembles([wild_type, mutant])
        assert not study.shares_topology
        reference = np.zeros((4, 4))
        other = np.zeros((4, 4))
        _, _, index, labels = study._align_columns(
            reference, other, 0, 1, "cacn"
        )

        assert (0, 1) in study.correspondences
        np.testing.assert_array_equal(index, [1, 2, 3, 4])
        assert labels.tolist() == ["1", "2", "3", "4"]

    def test_equal_count_different_molecules_are_reconciled_and_refused(self):
        from prothon import Prothon

        a = Ensemble(build(as_residues("AAAAAAAAAA"), n_frames=2), label="alanine")
        b = Ensemble(build(as_residues("VVVVVVVVVV"), n_frames=2), label="valine")
        assert a.trajectory.n_atoms == b.trajectory.n_atoms
        study = Prothon.from_ensembles([a, b])
        matrix = np.zeros((2, 10))
        with pytest.raises(ValueError):
            study._align_columns(matrix, matrix.copy(), 0, 1, "cacn")

    def test_manifest_records_the_correspondence(self, tmp_path):
        study = self._study(tmp_path, "ACDEFHIKLMNPQR", "ACDEGHIKLMNPQR")
        study.compare_ensembles(order_parameters="cbcn", s_num=2)
        manifest = json.loads((tmp_path / "cbcn_output" / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["ensembles"][0]["label"] == "wild type"
        corr = manifest["correspondences"][0]
        assert corr["substitutions"] == ["F5G"]
        assert corr["n_aligned"] == 14
        assert corr["coverage"] == 1.0
        assert corr["alignment"][0]["reference"] == "ACDEFHIKLMNPQR"
        assert manifest["results"][0]["feature_index"] == [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14
        ]
        assert manifest["results"][0]["feature_labels"][4] == "6"

    def test_a_truncated_construct_is_compared_on_what_it_shares(self, tmp_path):
        from prothon import Prothon

        full = Ensemble(build(as_residues("ACDEFHIKLMNPQR"), n_frames=300, seed=1),
                        label="full")
        short = Ensemble(build(as_residues("EFHIKLMNPQR"), n_frames=300, seed=2),
                         label="construct")
        study = Prothon.from_ensembles([full, short], output_dir=str(tmp_path),
                                       random_state=0)
        result = study.compare_ensembles(order_parameters="cacn", s_num=2)["cacn"][0]
        # The construct starts at the full protein's residue 4.
        assert result.feature_index[0] == 4
        assert result.feature_index[-1] == 14
        assert result.feature_labels[0] == "4"
        assert result.feature_labels[-1] == "14"

    def test_weights_reach_the_estimator(self, tmp_path):
        """A study over a weighted ensemble must give a different answer from
        the same frames unweighted, or the weights are decoration."""
        from prothon import Prothon

        frames = build(as_residues("ACDEFHIK"), n_frames=400, seed=1)
        other = build(as_residues("ACDEFHIK"), n_frames=400, seed=2, compact_from=4)
        # Concentrate on the second half, which is where the two differ.
        w = np.concatenate([np.full(200, 0.1 / 200), np.full(200, 0.9 / 200)])

        plain = Prothon.from_ensembles(
            [Ensemble(frames, label="a"), Ensemble(other, label="b")],
            output_dir=str(tmp_path / "plain"), random_state=0,
        ).compare_ensembles(order_parameters="cacn", s_num=2)["cacn"][0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weighted = Prothon.from_ensembles(
                [Ensemble(frames, label="a"),
                 Ensemble(other, label="b", weights=w)],
                output_dir=str(tmp_path / "weighted"), random_state=0,
            ).compare_ensembles(order_parameters="cacn", s_num=2)["cacn"][0]

        assert weighted.global_dissimilarity != plain.global_dissimilarity

    def test_a_weighted_ensemble_against_an_unweighted_one_warns(self, tmp_path):
        from prothon import Prothon

        a = Ensemble(build(as_residues("ACDEF"), n_frames=200), label="deposited",
                     weights=np.linspace(1, 2, 200))
        b = Ensemble(build(as_residues("ACDEF"), n_frames=200, seed=1), label="md")
        study = Prothon.from_ensembles([a, b], output_dir=str(tmp_path),
                                       random_state=0)
        with pytest.warns(UserWarning, match="treated as uniform"):
            study.compare_ensembles(order_parameters="cacn", s_num=2)

    def test_effective_sample_size_is_recorded(self, tmp_path):
        from prothon import Prothon

        # One conformer carrying half the probability: 200 frames, worth ~4.
        w = np.full(200, 0.5 / 199)
        w[0] = 0.5
        a = Ensemble(build(as_residues("ACDEF"), n_frames=200), label="a")
        b = Ensemble(build(as_residues("ACDEF"), n_frames=200, seed=1), label="b",
                     weights=w)
        study = Prothon.from_ensembles([a, b], output_dir=str(tmp_path),
                                       random_state=0)
        with pytest.warns(UserWarning, match="treated as uniform"):
            with pytest.raises(ValueError, match="independent conformations"):
                study.compare_ensembles(order_parameters="cacn", s_num=2)

    def test_fewer_than_two_ensembles_is_refused(self):
        from prothon import Prothon

        one = Ensemble(build(as_residues("ACDEF")), label="only")
        with pytest.raises(ValueError, match="at least two"):
            Prothon.from_ensembles([one])

    def test_paths_are_resolved_rather_than_refused(self, tmp_path):
        """`from_ensembles` used to take only Ensemble objects and reject
        paths. There is now one way in, and it takes either -- so a path is
        loaded rather than refused, and a path that does not exist says so."""
        from prothon import Prothon

        with pytest.raises(FileNotFoundError, match="No such source"):
            Prothon.from_ensembles(["a.dcd", "b.dcd"])

        traj = build(as_residues("ACDEF"), n_frames=20)
        traj.save_dcd(str(tmp_path / "x.dcd"))
        traj[0].save_pdb(str(tmp_path / "t.pdb"))
        study = Prothon(
            ensembles=[str(tmp_path / "x.dcd"), str(tmp_path / "x.dcd")],
            topology=str(tmp_path / "t.pdb"),
        )
        assert len(study.ensembles) == 2
