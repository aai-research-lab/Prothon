"""Tests for reading a study from a file.

The thing that makes a configuration worth having is not that it holds the
flags. It is that it holds what a flag cannot: a topology and a label and a
weight vector per ensemble. And the thing that makes it trustworthy is that it
refuses a key it does not recognise, because a silently ignored setting is a
file that lies.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_ingest import as_residues, build

from prothon.cli import main
from prothon.config import Study, load_study, resolve_ensembles

SEQ = "ACDEFHIKLMNP"


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cfg")
    build(as_residues(SEQ), n_frames=300, seed=1).save_dcd(str(d / "a.dcd"))
    build(as_residues(SEQ), n_frames=300, seed=2, compact_from=7).save_dcd(str(d / "b.dcd"))
    build(as_residues(SEQ), n_frames=1)[0].save_pdb(str(d / "top.pdb"))
    models = d / "models"
    models.mkdir()
    traj = build(as_residues(SEQ), n_frames=40, seed=3)
    for i in range(40):
        traj[i].save_pdb(str(models / f"s{i:02d}.pdb"))
    return d


def write(path, text):
    path.write_text(text)
    return str(path)


class TestReading:
    def test_a_minimal_study(self, files, tmp_path):
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - source: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
  - source: {files / 'b.dcd'}
    topology: {files / 'top.pdb'}
""")
        study = load_study(path)
        assert len(study.ensembles) == 2
        assert study.reference_index() == 0

    def test_a_bare_source_is_allowed(self, files, tmp_path):
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - {files / 'models'}
  - {files / 'models'}
""")
        assert len(load_study(path).ensembles) == 2

    def test_each_ensemble_may_have_its_own_topology(self, files, tmp_path):
        """The thing a single `--topology` flag cannot express, and the main
        reason to write a study down rather than type it."""
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - source: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    label: simulation
  - source: {files / 'models'}
    label: generated
""")
        loaded = resolve_ensembles(load_study(path))
        assert [e.label for e in loaded] == ["simulation", "generated"]
        assert loaded[0].n_frames == 300
        assert loaded[1].n_frames == 40

    def test_weights_may_come_from_a_file(self, files, tmp_path):
        weights = tmp_path / "w.txt"
        np.savetxt(weights, np.linspace(1, 2, 300))
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - source: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    weights: {weights}
  - source: {files / 'b.dcd'}
    topology: {files / 'top.pdb'}
""")
        loaded = resolve_ensembles(load_study(path))
        assert loaded[0].weights is not None
        assert loaded[0].weights.sum() == pytest.approx(1.0)
        assert loaded[1].weights is None

    def test_a_stride_is_honoured(self, files, tmp_path):
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - source: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    stride: 10
  - source: {files / 'b.dcd'}
    topology: {files / 'top.pdb'}
""")
        loaded = resolve_ensembles(load_study(path))
        assert loaded[0].n_frames == 30

    def test_the_reference_may_be_a_label(self, files, tmp_path):
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - source: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    label: wild type
  - source: {files / 'b.dcd'}
    topology: {files / 'top.pdb'}
    label: mutant
reference: mutant
""")
        assert load_study(path).reference_index() == 1


class TestRefusals:
    """A configuration that ignores what it does not understand is a file that
    lies. Each of these names what was wrong and what was expected."""

    def test_an_unknown_top_level_key(self, tmp_path):
        path = write(tmp_path / "s.yml", "ensembls:\n  - a\n  - b\n")
        with pytest.raises(ValueError, match="Did you mean ensembles"):
            load_study(path)

    def test_an_unknown_setting(self, files, tmp_path):
        """The case that matters most: a misspelled `random_state` would leave
        the study unseeded and say nothing."""
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - {files / 'models'}
  - {files / 'models'}
compare:
  random_seed: 0
""")
        with pytest.raises(ValueError, match="Did you mean random_state"):
            load_study(path)

    def test_an_unknown_ensemble_key(self, files, tmp_path):
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - source: {files / 'a.dcd'}
    topolgy: {files / 'top.pdb'}
  - {files / 'models'}
""")
        with pytest.raises(ValueError, match="Did you mean topology"):
            load_study(path)

    def test_a_missing_source(self, tmp_path):
        path = write(tmp_path / "s.yml", "ensembles:\n  - source: a\n  - topology: b\n")
        with pytest.raises(ValueError, match="has no 'source'"):
            load_study(path)

    def test_fewer_than_two_ensembles(self, tmp_path):
        path = write(tmp_path / "s.yml", "ensembles:\n  - source: a\n")
        with pytest.raises(ValueError, match="at least two"):
            load_study(path)

    def test_no_ensembles_at_all(self, tmp_path):
        path = write(tmp_path / "s.yml", "compare:\n  metric: jsd\n")
        with pytest.raises(ValueError, match="no ensembles"):
            load_study(path)

    def test_a_reference_label_that_matches_nothing(self, files, tmp_path):
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - {files / 'models'}
  - {files / 'models'}
reference: nowhere
""")
        with pytest.raises(ValueError, match="is not one of the ensembles"):
            load_study(path).reference_index()

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="configuration file"):
            load_study(tmp_path / "absent.yml")

    def test_something_that_is_not_a_mapping(self, tmp_path):
        path = write(tmp_path / "s.yml", "- a\n- b\n")
        with pytest.raises(ValueError, match="mapping of settings"):
            load_study(path)


class TestThroughTheCommandLine:
    def test_a_study_runs(self, files, tmp_path, capsys):
        path = write(tmp_path / "s.yml", f"""
description: a test study
ensembles:
  - source: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    label: wild type
  - source: {files / 'b.dcd'}
    topology: {files / 'top.pdb'}
    label: mutant
reference: wild type
compare:
  order_parameters: cbcn
  random_state: 0
  s_num: 2
output_dir: {tmp_path / 'out'}
""")
        assert main(["compare", "--config", path]) == 0
        assert "CBCN" in capsys.readouterr().out

    def test_a_flag_overrides_the_file(self, files, tmp_path, capsys):
        """So a study can be re-run with one thing changed — a different seed,
        a different output directory — without editing it."""
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - source: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
  - source: {files / 'b.dcd'}
    topology: {files / 'top.pdb'}
compare:
  order_parameters: cbcn
  random_state: 0
  s_num: 2
output_dir: {tmp_path / 'from_file'}
""")
        assert main([
            "compare", "--config", path,
            "--order-parameters", "cacn", "-o", str(tmp_path / "from_flag"),
        ]) == 0
        assert "CACN" in capsys.readouterr().out
        assert (tmp_path / "from_flag" / "cacn_output").exists()
        assert not (tmp_path / "from_file").exists()

    def test_the_manifest_records_the_study(self, files, tmp_path):
        """A result found later should carry the question it answered, not
        only the answer."""
        path = write(tmp_path / "s.yml", f"""
description: recorded
ensembles:
  - source: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    label: wild type
  - source: {files / 'b.dcd'}
    topology: {files / 'top.pdb'}
    label: mutant
compare:
  order_parameters: cbcn
  random_state: 0
  s_num: 2
output_dir: {tmp_path / 'out'}
""")
        main(["compare", "--config", path])
        manifest = json.loads(
            (tmp_path / "out" / "cbcn_output" / "manifest.json").read_text()
        )
        study = manifest["study"]
        assert study["description"] == "recorded"
        assert [e["label"] for e in study["ensembles"]] == ["wild type", "mutant"]
        assert study["settings"]["random_state"] == 0

    def test_neither_ensembles_nor_config_is_refused(self, capsys):
        assert main(["compare", "-p", "cbcn"]) == 2
        assert "--ensembles" in capsys.readouterr().err

    def test_a_bad_config_exits_two_not_a_traceback(self, tmp_path, capsys):
        path = write(tmp_path / "s.yml", "ensembls:\n  - a\n")
        assert main(["compare", "--config", path]) == 2
        assert "Did you mean ensembles" in capsys.readouterr().err


class TestStudyObject:
    def test_labels_fall_back_to_the_source(self):
        study = Study(ensembles=[{"source": "a/b/wt.dcd"}, {"source": "mut.dcd"}])
        assert study.labels == ["wt.dcd", "mut.dcd"]

    def test_an_index_reference_works(self):
        study = Study(ensembles=[{"source": "a"}, {"source": "b"}], reference=1)
        assert study.reference_index() == 1

    def test_an_out_of_range_index_is_refused(self):
        study = Study(ensembles=[{"source": "a"}, {"source": "b"}], reference=5)
        with pytest.raises(ValueError, match="out of range"):
            study.reference_index()

    def test_it_serialises(self):
        study = Study(ensembles=[{"source": "a"}, {"source": "b"}], description="x")
        payload = json.loads(json.dumps(study.to_dict()))
        assert payload["description"] == "x"
