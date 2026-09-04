"""Tests for comparing one chain of a complex.

The failure to guard against is silence: MDTraj's ``chainid`` selector takes an
integer, and a chain letter passed to it matches nothing and returns an empty
selection rather than failing. A user who types ``chains="A"`` and gets an
empty ensemble back has been told nothing.
"""

from __future__ import annotations

import json

import mdtraj as md
import numpy as np
import pytest
from test_ingest import as_residues

from prothon import Prothon
from prothon.cli import main
from prothon.ingest import Ensemble


def complex_traj(n_frames=200, seed=0, chains=(("A", "ACDEFHIKLMNP"), ("B", "ACDEFHIKLMNQ"))):
    """A two-chain system whose labelled protomers are homologous."""
    rng = np.random.default_rng(seed)
    top = md.Topology()
    for chain_id, seq in chains:
        chain = top.add_chain()
        chain.chain_id = chain_id
        for name in as_residues(seq):
            residue = top.add_residue(name, chain)
            for atom in ("N", "CA", "C", "O", "CB"):
                top.add_atom(atom, md.element.carbon, residue)

    xyz = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
    for frame in range(n_frames):
        for atom in top.atoms:
            xyz[frame, atom.index] = [
                atom.residue.index * 0.38 + atom.residue.chain.index * 3.0, 0.0, 0.0
            ] + rng.normal(0, 0.2, 3)
    return md.Trajectory(xyz, top)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("chains")
    complex_traj(300, 1).save_dcd(str(d / "a.dcd"))
    complex_traj(300, 2).save_dcd(str(d / "b.dcd"))
    complex_traj(1, 1)[0].save_pdb(str(d / "top.pdb"))
    return d


def load(files, name="a.dcd"):
    return Ensemble.from_trajectory(str(files / name), str(files / "top.pdb"))


class TestSelecting:
    def test_a_letter_selects_the_chain(self, files):
        whole = load(files)
        assert whole.trajectory.topology.n_residues == 24
        assert whole.select_chains("A").trajectory.topology.n_residues == 12

    def test_an_index_works_too(self, files):
        assert load(files).select_chains(1).trajectory.topology.n_residues == 12

    def test_several_chains_may_be_kept(self, files):
        for spec in ("A,B", ["A", "B"], [0, 1]):
            assert load(files).select_chains(spec).trajectory.topology.n_residues == 24

    def test_the_label_records_what_was_kept(self, files):
        assert "chain" in load(files).select_chains("B").label

    def test_the_provenance_records_it(self, files):
        assert load(files).select_chains("B").provenance["chains"] == [1]

    def test_weights_survive_the_selection(self, files):
        whole = load(files)
        whole.weights = whole._validate_weights(
            np.linspace(1, 2, whole.n_frames), whole.n_frames
        )
        assert load(files).select_chains("A").n_frames == whole.n_frames
        picked = whole.select_chains("A")
        np.testing.assert_allclose(picked.weights, whole.weights)


class TestRefusals:
    def test_an_unknown_letter_is_refused_not_silently_empty(self, files):
        """`topology.select("chainid A")` returns zero atoms without
        complaining, so a naive implementation hands back an empty ensemble."""
        with pytest.raises(ValueError, match="no chain 'Z'"):
            load(files).select_chains("Z")

    def test_the_message_names_the_chains_that_are_there(self, files):
        with pytest.raises(ValueError, match="Chains present: A, B"):
            load(files).select_chains("Z")

    def test_an_index_out_of_range_is_refused(self, files):
        with pytest.raises(ValueError, match="out of range"):
            load(files).select_chains(7)

    def test_an_empty_selection_is_refused(self, files):
        with pytest.raises(ValueError, match="no chains selected"):
            load(files).select_chains("")

    def test_a_mismatched_count_is_refused(self, files):
        from prothon.ingest.sources import resolve_all

        with pytest.raises(ValueError, match="chain selections for"):
            resolve_all(
                [str(files / "a.dcd"), str(files / "b.dcd")],
                str(files / "top.pdb"),
                chains=["A", "B", "A"],
            )


class TestThroughAStudy:
    def test_one_chain_shared_by_every_ensemble(self, files):
        study = Prothon(
            ensembles=[str(files / "a.dcd"), str(files / "b.dcd")],
            topology=str(files / "top.pdb"), chains="A", random_state=0,
        )
        assert all(e.trajectory.topology.n_residues == 12 for e in study.ensembles)

    def test_a_different_chain_from_each(self, files):
        """A protomer of one complex against a protomer of another. The two
        are different molecules, and the sequence alignment reconciles them."""
        study = Prothon(
            ensembles=[str(files / "a.dcd"), str(files / "b.dcd")],
            topology=str(files / "top.pdb"), chains=["A", "B"], random_state=0,
        )
        assert study.ensembles[0].provenance["chains"] == [0]
        assert study.ensembles[1].provenance["chains"] == [1]
        assert not study.shares_topology

    def test_a_comparison_runs_on_one_chain(self, files):
        study = Prothon(
            ensembles=[str(files / "a.dcd"), str(files / "b.dcd")],
            topology=str(files / "top.pdb"), chains="A", random_state=0,
        )
        result = study.compare("cacn", s_num=2)["cacn"][0]
        assert result.local_dissimilarity.size == 12

    def test_the_whole_complex_gives_more_columns(self, files):
        whole = Prothon(
            ensembles=[str(files / "a.dcd"), str(files / "b.dcd")],
            topology=str(files / "top.pdb"), random_state=0,
        ).compare("cacn", s_num=2)["cacn"][0]
        assert whole.local_dissimilarity.size == 24


class TestThroughTheCommandLine:
    def test_a_chain_flag(self, files, capsys):
        code = main([
            "compare", "-e", str(files / "a.dcd"), str(files / "b.dcd"),
            "-t", str(files / "top.pdb"), "--chains", "A",
            "-p", "cacn", "-s", "0", "--s-num", "2",
        ])
        assert code == 0
        assert "CACN" in capsys.readouterr().out

    def test_one_chain_per_ensemble(self, files, capsys):
        code = main([
            "compare", "-e", str(files / "a.dcd"), str(files / "b.dcd"),
            "-t", str(files / "top.pdb"), "--chains", "A", "B",
            "-p", "cacn", "-s", "0", "--s-num", "2",
        ])
        assert code == 0

    def test_an_unknown_chain_exits_two(self, files, capsys):
        code = main([
            "compare", "-e", str(files / "a.dcd"), str(files / "b.dcd"),
            "-t", str(files / "top.pdb"), "--chains", "Z", "-p", "cacn",
        ])
        assert code == 2
        assert "no chain" in capsys.readouterr().err

    def test_it_reaches_a_study_file(self, files, tmp_path):
        from prothon.config import Study

        path = Study(
            ensembles=[
                {"ensemble": str(files / "a.dcd"),
                 "topology": str(files / "top.pdb"), "chains": "A"},
                {"ensemble": str(files / "b.dcd"),
                 "topology": str(files / "top.pdb"), "chains": "B"},
            ],
            settings={"order_parameters": "cacn", "random_state": 0, "s_num": 2},
        ).save(tmp_path / "s.yml")
        loaded = Study.from_file(path).resolve()
        assert [e.provenance["chains"] for e in loaded] == [[0], [1]]

    def test_configured_chains_and_weight_files_reach_the_json_result(
        self, files, tmp_path, capsys
    ):
        """One real loader path exercises config, chains, weights and CLI."""
        from prothon.config import Study

        weight_a = tmp_path / "a.weights"
        weight_b = tmp_path / "b.weights"
        np.savetxt(weight_a, np.linspace(1.0, 2.0, 300))
        np.savetxt(weight_b, np.linspace(2.0, 1.0, 300) ** 2)
        output = tmp_path / "result"
        path = Study(
            ensembles=[
                {
                    "ensemble": str(files / "a.dcd"),
                    "topology": str(files / "top.pdb"),
                    "chains": "A",
                    "weights": str(weight_a),
                    "label": "chain A",
                },
                {
                    "ensemble": str(files / "b.dcd"),
                    "topology": str(files / "top.pdb"),
                    "chains": "B",
                    "weights": str(weight_b),
                    "label": "chain B",
                },
            ],
            settings={
                "order_parameters": "cacn",
                "random_state": 0,
                "sample_size": 300,
                "n_permutations": 10,
                "s_num": 2,
                "no_block_permutation": True,
            },
            output_dir=str(output),
        ).save(tmp_path / "weighted-chains.yml")

        assert main(["compare", "--config", str(path), "--json"]) == 0
        result = json.loads(capsys.readouterr().out)["cacn"][0]
        assert result["n_frames"] == [300, 300]
        assert len(result["feature_index"]) == 12
        assert all(10.0 < count < 300.0 for count in result["effective_samples"])

        manifest = json.loads(
            (output / "cacn_output" / "manifest.json").read_text(encoding="utf-8")
        )
        assert [item["chains"] for item in manifest["study"]["ensembles"]] == [
            "A",
            "B",
        ]
        assert all(item["weighted"] for item in manifest["ensembles"])
