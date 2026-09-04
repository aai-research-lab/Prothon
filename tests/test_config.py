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
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestReading:
    def test_a_minimal_study(self, files, tmp_path):
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - ensemble: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
  - ensemble: {files / 'b.dcd'}
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
  - ensemble: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    label: simulation
  - ensemble: {files / 'models'}
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
  - ensemble: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    weights: {weights}
  - ensemble: {files / 'b.dcd'}
    topology: {files / 'top.pdb'}
""")
        loaded = resolve_ensembles(load_study(path))
        assert loaded[0].weights is not None
        assert loaded[0].weights.sum() == pytest.approx(1.0)
        assert loaded[1].weights is None

    def test_a_stride_is_honoured(self, files, tmp_path):
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - ensemble: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    stride: 10
  - ensemble: {files / 'b.dcd'}
    topology: {files / 'top.pdb'}
""")
        loaded = resolve_ensembles(load_study(path))
        assert loaded[0].n_frames == 30

    def test_the_reference_may_be_a_label(self, files, tmp_path):
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - ensemble: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    label: wild type
  - ensemble: {files / 'b.dcd'}
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
  - ensemble: {files / 'a.dcd'}
    topolgy: {files / 'top.pdb'}
  - {files / 'models'}
""")
        with pytest.raises(ValueError, match="Did you mean topology"):
            load_study(path)

    def test_a_missing_ensemble_key(self, tmp_path):
        path = write(tmp_path / "s.yml", "ensembles:\n  - ensemble: a\n  - topology: b\n")
        with pytest.raises(ValueError, match="has no 'ensemble' key"):
            load_study(path)

    def test_the_older_key_still_works(self, files, tmp_path):
        """`source` was the first name for it. Files written against that
        keep working."""
        path = write(tmp_path / "s.yml", f"""
ensembles:
  - source: {files / 'models'}
  - source: {files / 'models'}
""")
        assert len(load_study(path).resolve()) == 2

    def test_fewer_than_two_ensembles(self, tmp_path):
        path = write(tmp_path / "s.yml", "ensembles:\n  - ensemble: a\n")
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
  - ensemble: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    label: wild type
  - ensemble: {files / 'b.dcd'}
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
  - ensemble: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
  - ensemble: {files / 'b.dcd'}
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
  - ensemble: {files / 'a.dcd'}
    topology: {files / 'top.pdb'}
    label: wild type
  - ensemble: {files / 'b.dcd'}
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
            (tmp_path / "out" / "cbcn_output" / "manifest.json").read_text(encoding="utf-8")
        )
        study = manifest["study"]
        assert study["description"] == "recorded"
        assert [e["label"] for e in study["ensembles"]] == ["wild type", "mutant"]
        assert study["compare"]["random_state"] == 0
        # The manifest records where the study came from; a written file does
        # not, because that would point at a different file.
        assert study["path"].endswith("s.yml")

    def test_neither_ensembles_nor_config_is_refused(self, capsys):
        assert main(["compare", "-p", "cbcn"]) == 2
        assert "--ensembles" in capsys.readouterr().err

    def test_a_bad_config_exits_two_not_a_traceback(self, tmp_path, capsys):
        path = write(tmp_path / "s.yml", "ensembls:\n  - a\n")
        assert main(["compare", "--config", path]) == 2
        assert "Did you mean ensembles" in capsys.readouterr().err


class TestOneObjectForEveryInterface:
    """The point of the redesign: flags, a file and Python all build the same
    object, so a setting reachable from one cannot be missing from another."""

    def test_flags_become_a_study(self, files):
        import argparse

        from prothon.config.schema import parameters_for

        args = argparse.Namespace(
            ensembles=[str(files / "a.dcd"), str(files / "b.dcd")],
            topology=str(files / "top.pdb"),
            reference=0, output_dir=None, config=None, save_config=None,
            **{p.name: p.default for p in parameters_for("compare")
               if p.name not in {"ensembles", "topology", "reference",
                                 "output_dir", "config", "save_config"}},
        )
        args.random_state = 0
        args.s_num = 2
        study = Study.from_arguments(args)

        assert len(study.ensembles) == 2
        assert study.ensembles[0]["topology"] == str(files / "top.pdb")
        assert study.settings["random_state"] == 0

    def test_a_flag_not_given_is_not_written_down(self, files):
        """The parser preserves absence instead of inventing false flags."""
        from prothon.cli import build_parser

        args = build_parser().parse_args(
            ["compare", "--ensembles", str(files / "a.dcd")]
        )
        settings = Study.from_arguments(args).settings
        assert settings == {}

    @pytest.mark.parametrize(
        ("flag", "text", "setting", "configured", "expected"),
        [
            ("--order-parameters", "cbcn", "order_parameters", "sasa", "cbcn"),
            ("--metric", "jsd", "metric", "ks", "jsd"),
            ("--n-permutations", "100", "n_permutations", 23, 100),
            ("--n-jobs", "1", "n_jobs", 4, 1),
            ("--sample-size", "1000", "sample_size", 41, 1000),
            ("--s-num", "5", "s_num", 2, 5),
            ("--x-num", "100", "x_num", 31, 100),
            ("--alpha", "0.05", "alpha", 0.01, 0.05),
            ("--report", "summary", "report", "table", "summary"),
            ("--dimred", "none", "dimred", "pca", "none"),
        ],
    )
    def test_an_explicit_schema_default_overrides_the_file(
        self, flag, text, setting, configured, expected
    ):
        from prothon.cli import build_parser

        args = build_parser().parse_args(
            ["compare", "--config", "study.yml", flag, text]
        )
        study = Study(
            ensembles=[{"ensemble": "a"}, {"ensemble": "b"}],
            settings={setting: configured},
        )

        assert study.merged_with(args).settings[setting] == expected

    @pytest.mark.parametrize(
        ("configured", "flag", "present", "absent"),
        [
            ({"no_block_permutation": True}, "--block-permutation",
             "block_permutation", "no_block_permutation"),
            ({"block_permutation": True}, "--no-block-permutation",
             "no_block_permutation", "block_permutation"),
        ],
    )
    def test_an_explicit_block_choice_replaces_its_opposite(
        self, configured, flag, present, absent
    ):
        from prothon.cli import build_parser

        args = build_parser().parse_args(
            ["compare", "--config", "study.yml", flag]
        )
        study = Study(
            ensembles=[{"ensemble": "a"}, {"ensemble": "b"}],
            settings=configured,
        ).merged_with(args)

        assert study.settings[present] is True
        assert absent not in study.settings

    def test_an_explicit_legacy_flag_overrides_false(self):
        from prothon.cli import build_parser

        args = build_parser().parse_args(
            ["compare", "--config", "study.yml", "--legacy-statistics"]
        )
        study = Study(
            ensembles=[{"ensemble": "a"}, {"ensemble": "b"}],
            settings={"legacy_statistics": False},
        )

        assert study.merged_with(args).settings["legacy_statistics"] is True

    def test_an_explicit_default_reference_overrides_the_file(self):
        from prothon.cli import build_parser

        args = build_parser().parse_args(
            ["compare", "--config", "study.yml", "--reference", "0"]
        )
        study = Study(
            ensembles=[{"ensemble": "a"}, {"ensemble": "b"}],
            reference=1,
        )

        assert study.merged_with(args).reference == "0"

    def test_a_study_round_trips_through_a_file(self, files, tmp_path):
        original = Study(
            ensembles=[
                {"ensemble": str(files / "a.dcd"),
                 "topology": str(files / "top.pdb"), "label": "wild type"},
                {"ensemble": str(files / "b.dcd"),
                 "topology": str(files / "top.pdb"), "label": "mutant"},
            ],
            reference="mutant",
            settings={"order_parameters": "cbcn", "random_state": 0, "s_num": 2},
            description="round trip",
        )
        path = original.save(tmp_path / "written.yml")
        again = Study.from_file(path)

        assert again.labels == original.labels
        assert again.reference_index() == original.reference_index()
        assert again.settings == original.settings
        assert again.description == original.description

    def test_a_written_file_does_not_point_at_its_source(self, files, tmp_path):
        """How a study was reached is not part of what it says. A rewritten
        file that recorded `path` or `config` would point at a different
        file."""
        path = write(tmp_path / "in.yml", f"""
ensembles:
  - {files / 'models'}
  - {files / 'models'}
""")
        out = Study.from_file(path).save(tmp_path / "out.yml")
        text = (tmp_path / "out.yml").read_text(encoding="utf-8")
        assert "path:" not in text
        assert "config:" not in text
        assert Study.from_file(out).labels

    def test_the_three_paths_agree(self, files, tmp_path):
        """A file and a directly built study give the same answer, because
        they are the same object by the time anything runs."""
        common = dict(
            ensembles=[
                {"ensemble": str(files / "a.dcd"),
                 "topology": str(files / "top.pdb")},
                {"ensemble": str(files / "b.dcd"),
                 "topology": str(files / "top.pdb")},
            ],
            settings={"order_parameters": "cbcn", "random_state": 0, "s_num": 2},
        )
        from_python = Study(**common).run().summary()
        path = Study(**common).save(tmp_path / "s.yml")
        from_file = Study.from_file(path).run().summary()
        assert from_python == from_file

    @pytest.mark.parametrize(
        ("setting", "value", "argument", "expected"),
        [
            ("order_parameters", ["cbcn", "cacn"], "order_parameters", "cbcn,cacn"),
            ("metric", "ks", "metric", "ks"),
            ("random_state", 17, "random_state", 17),
            ("n_permutations", 23, "n_permutations", 23),
            ("n_jobs", 3, "n_jobs", 3),
            ("sample_size", 41, "sample_size", 41),
            ("s_num", 7, "s_num", 7),
            ("x_num", 31, "x_num", 31),
            ("alpha", 0.01, "alpha", 0.01),
            ("block_permutation", True, "block_permutation", True),
            ("no_block_permutation", True, "block_permutation", False),
            ("legacy_statistics", True, "legacy", True),
        ],
    )
    def test_every_computation_setting_reaches_the_table_benchmark(
        self, setting, value, argument, expected
    ):
        study = Study(
            ensembles=[{"ensemble": "a"}, {"ensemble": "b"}],
            settings={setting: value},
            output_dir="results",
        )
        arguments = study.benchmark_arguments()

        assert arguments[argument] == expected
        assert arguments["output_dir"] == "results"
        if setting == "s_num":
            assert arguments["floor_repeats"] == expected

    def test_table_mode_refuses_a_projection_instead_of_ignoring_it(self):
        study = Study(
            ensembles=[{"ensemble": "a"}, {"ensemble": "b"}],
            settings={"dimred": "pca"},
        )
        with pytest.raises(ValueError, match="not part of the table report"):
            study.benchmark_arguments()

    def test_saving_from_the_command_line(self, files, tmp_path, capsys):
        """A command line typed once becomes a study that can be committed."""
        out = tmp_path / "typed.yml"
        assert main([
            "compare", "-e", str(files / "a.dcd"), str(files / "b.dcd"),
            "-t", str(files / "top.pdb"), "-p", "cbcn", "-s", "0",
            "--s-num", "2", "--save-config", str(out),
        ]) == 0
        assert out.exists()

        written = Study.from_file(out)
        assert len(written.ensembles) == 2
        assert written.settings["random_state"] == 0
        assert "block_permutation" not in written.settings


class TestStudyObject:
    def test_labels_fall_back_to_the_source(self):
        study = Study(ensembles=[{"ensemble": "a/b/wt.dcd"}, {"ensemble": "mut.dcd"}])
        assert study.labels == ["wt.dcd", "mut.dcd"]

    def test_an_index_reference_works(self):
        study = Study(ensembles=[{"ensemble": "a"}, {"ensemble": "b"}], reference=1)
        assert study.reference_index() == 1

    def test_an_out_of_range_index_is_refused(self):
        study = Study(ensembles=[{"ensemble": "a"}, {"ensemble": "b"}], reference=5)
        with pytest.raises(ValueError, match="out of range"):
            study.reference_index()

    def test_it_serialises(self):
        study = Study(ensembles=[{"ensemble": "a"}, {"ensemble": "b"}], description="x")
        payload = json.loads(json.dumps(study.to_dict()))
        assert payload["description"] == "x"
