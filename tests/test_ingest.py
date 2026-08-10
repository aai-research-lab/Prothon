"""Tests for the ingest layer.

The interesting cases are the ones where a naive column-for-column comparison
would give a plausible, wrong answer: a mutation to or from glycine, which
removes a C-beta and shifts every ``cbcn`` column after it, and a deletion,
which breaks the windows the angular measures are defined on.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from prothon.core.representation import compute_cbcn, compute_ensemble_representation
from prothon.ingest import (
    Ensemble,
    align,
    feature_residues,
    reconcile,
    sequence_of,
)
from prothon.ingest.sequence import THREE_TO_ONE, residue_letter

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

    def test_from_pdb_models_refuses_a_mismatched_model(self, tmp_path):
        build(as_residues("ACDEF"), n_frames=1)[0].save_pdb(str(tmp_path / "a.pdb"))
        build(as_residues("ACDE"), n_frames=1)[0].save_pdb(str(tmp_path / "b.pdb"))
        with pytest.raises(ValueError, match="same molecule"):
            Ensemble.from_pdb_models(str(tmp_path))

    def test_missing_file_is_named(self):
        with pytest.raises(FileNotFoundError, match="nowhere.xtc"):
            Ensemble.from_trajectory("nowhere.xtc", "nowhere.pdb")


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
