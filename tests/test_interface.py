"""Tests for the interface: the schema, the sources, and the command line.

The thing under test is consistency. A flag and a keyword argument are two
renderings of the same idea, and when each is written by hand they drift --
this project's did, with `--seed` on one side and `random_state` on the other.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_ingest import as_residues, build

from prothon import Prothon
from prothon.cli import build_parser, main
from prothon.config.schema import COMMANDS, PARAMETERS, parameters_for
from prothon.ingest import Ensemble
from prothon.ingest.sources import describe_source, resolve, resolve_all

SEQ = "ACDEFHIKLMNP"


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("iface")
    build(as_residues(SEQ), n_frames=300, seed=1).save_dcd(str(directory / "a.dcd"))
    build(as_residues(SEQ), n_frames=300, seed=2, compact_from=7).save_dcd(
        str(directory / "b.dcd")
    )
    build(as_residues(SEQ), n_frames=1)[0].save_pdb(str(directory / "top.pdb"))
    models = directory / "models"
    models.mkdir()
    traj = build(as_residues(SEQ), n_frames=30, seed=3)
    for i in range(30):
        traj[i].save_pdb(str(models / f"s{i:02d}.pdb"))
    return directory


class TestSchema:
    def test_every_flag_has_a_python_name(self):
        """The flag is derived from the name, so they cannot disagree."""
        for spec in PARAMETERS:
            assert spec.flag == "--" + spec.name.replace("_", "-")

    def test_short_flags_are_unique_within_a_command(self):
        for command in COMMANDS:
            shorts = [p.short for p in parameters_for(command.name) if p.short]
            assert len(shorts) == len(set(shorts)), f"{command.name} reuses a flag"

    def test_no_parameter_shadows_help(self):
        assert all(p.short != "-h" for p in PARAMETERS)

    def test_every_command_has_parameters_or_is_info(self):
        for command in COMMANDS:
            assert parameters_for(command.name) or command.name == "info"

    def test_the_parser_is_generated_from_the_schema(self):
        """Not written alongside it. A flag absent from the parser means a
        parameter absent from the schema."""
        parser = build_parser()
        actions = {a.dest for a in parser._subparsers._group_actions[0].choices[
            "compare"]._actions}
        for spec in parameters_for("compare"):
            assert spec.name in actions, f"--{spec.name} is declared but not offered"

    def test_the_cli_and_the_api_agree_on_names(self):
        """The mismatch this replaced: --seed against random_state."""
        import inspect

        api = set(inspect.signature(Prothon.__init__).parameters)
        api |= set(inspect.signature(Prothon.compare_ensembles).parameters)
        for name in ("ensembles", "topology", "random_state", "output_dir",
                     "order_parameters"):
            assert any(p.name == name for p in PARAMETERS)
            assert name in api, f"{name} is a flag but not a keyword argument"


class TestTheReadmeDescribesThisVersion:
    """A README is the first thing a reader sees and the last thing anybody
    edits. Both headline commands in this one were still the 2.x form three
    releases after it changed, because an edit had targeted text that had
    already moved and silently did nothing."""

    @staticmethod
    def _readme():
        import pathlib

        for parent in pathlib.Path(__file__).resolve().parents:
            candidate = parent / "README.md"
            if candidate.exists():
                return candidate.read_text()
        pytest.skip("README.md not found beside the tests")

    def test_no_command_uses_the_superseded_flags(self):
        text = self._readme()
        for flag in ("-traj", "-top ", "--seed", "--measures"):
            assert flag not in text, f"the README still uses {flag}"

    def test_every_flag_shown_exists(self):
        import re

        from prothon.cli import build_parser

        known = {"--help", "--version"}
        for parser in build_parser()._subparsers._group_actions[0].choices.values():
            known |= {o for a in parser._actions for o in a.option_strings}

        shown = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", self._readme()))
        unknown = sorted(f for f in shown if f not in known)
        assert not unknown, f"the README shows flags that do not exist: {unknown}"

    def test_it_mentions_what_the_tool_can_do(self):
        """Not a style rule: a capability absent from the README is one nobody
        finds."""
        text = self._readme()
        for probe in ("--config", "--order-parameters", "--report", "PED000"):
            assert probe in text, f"the README does not mention {probe}"


class TestDocumentedCodeBlocks:
    """A fenced block labelled `json` is lexed as JSON when the docs are built,
    and a block that does not parse fails the build -- after the tests pass, on
    a clean checkout, which is the worst place to find out.

    An incremental Sphinx build does not re-read an unchanged page, so a local
    build can report success on a file it never looked at. This checks the
    content directly, on the machine where it was written.
    """

    @staticmethod
    def _blocks():
        import pathlib
        import re

        for parent in pathlib.Path(__file__).resolve().parents:
            if (parent / "docs").is_dir():
                root = parent
                break
        else:
            pytest.skip("docs/ not found beside the tests")

        files = sorted((root / "docs").glob("*.md")) + [root / "README.md"]
        for path in files:
            if not path.exists():
                continue
            text = path.read_text()
            for match in re.finditer(r"```(\w+)\n(.*?)```", text, re.S):
                yield (
                    path.name,
                    text[: match.start()].count("\n") + 1,
                    match.group(1),
                    match.group(2),
                )

    def test_every_json_block_is_json(self):
        import json

        for name, line, lang, body in self._blocks():
            if lang != "json":
                continue
            try:
                json.loads(body)
            except ValueError as error:
                pytest.fail(f"{name}:{line} is labelled json and is not: {error}")

    def test_every_yaml_block_is_yaml(self):
        yaml = pytest.importorskip("yaml")

        for name, line, lang, body in self._blocks():
            if lang not in {"yaml", "yml"}:
                continue
            try:
                yaml.safe_load(body)
            except yaml.YAMLError as error:
                pytest.fail(f"{name}:{line} is labelled yaml and is not: {error}")


class TestSources:
    def test_a_trajectory_needs_a_topology(self, files):
        with pytest.raises(ValueError, match="needs a topology"):
            resolve(str(files / "a.dcd"))

    def test_a_trajectory_with_a_topology(self, files):
        assert resolve(str(files / "a.dcd"), str(files / "top.pdb")).n_frames == 300

    def test_a_directory_of_structures(self, files):
        assert resolve(str(files / "models")).n_frames == 30

    def test_a_glob(self, files):
        assert resolve(str(files / "models" / "*.pdb")).n_frames == 30

    def test_an_ensemble_passes_through(self, files):
        ensemble = resolve(str(files / "models"))
        assert resolve(ensemble) is ensemble

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("PED00024", "a PED accession"),
            ("PED00001:e002", "a PED accession"),
            ("md.xtc", "a trajectory"),
            ("entry.pdb", "a structure file"),
        ],
    )
    def test_sources_are_recognised_by_inspection(self, source, expected):
        assert describe_source(source) == expected

    def test_a_missing_source_says_what_was_expected(self):
        with pytest.raises(FileNotFoundError, match="PED00024"):
            resolve("nowhere.xtc")

    def test_comma_separated_and_separate_are_the_same_request(self, files):
        top = str(files / "top.pdb")
        a, b = str(files / "a.dcd"), str(files / "b.dcd")
        assert len(resolve_all(f"{a},{b}", top)) == 2
        assert len(resolve_all([a, b], top)) == 2

    def test_sources_and_loaded_ensembles_can_be_mixed(self, files):
        loaded = resolve(str(files / "models"))
        resolved = resolve_all([str(files / "a.dcd"), loaded], str(files / "top.pdb"))
        assert len(resolved) == 2
        assert resolved[1] is loaded


class TestConstructor:
    def test_one_constructor_takes_sources(self, files):
        study = Prothon(
            ensembles=[str(files / "a.dcd"), str(files / "b.dcd")],
            topology=str(files / "top.pdb"),
            random_state=0,
        )
        assert len(study.ensembles) == 2
        assert study.shares_topology

    def test_it_also_takes_loaded_ensembles(self, files):
        a = Ensemble.from_trajectory(str(files / "a.dcd"), str(files / "top.pdb"))
        b = Ensemble.from_trajectory(str(files / "b.dcd"), str(files / "top.pdb"))
        assert len(Prothon(ensembles=[a, b]).ensembles) == 2

    def test_the_old_name_warns(self, files):
        with pytest.warns(DeprecationWarning, match="ensembles="):
            Prothon(
                traj_files=[str(files / "a.dcd"), str(files / "b.dcd")],
                topology=str(files / "top.pdb"),
            )

    def test_from_ensembles_still_works(self, files):
        a = Ensemble.from_trajectory(str(files / "a.dcd"), str(files / "top.pdb"))
        b = Ensemble.from_trajectory(str(files / "b.dcd"), str(files / "top.pdb"))
        assert len(Prothon.from_ensembles([a, b]).ensembles) == 2

    def test_both_names_at_once_is_refused(self):
        with pytest.raises(TypeError, match="not both"):
            Prothon(ensembles=["a"], traj_files=["b"])

    def test_one_ensemble_is_refused(self, files):
        with pytest.raises(ValueError, match="at least two"):
            Prothon(ensembles=[str(files / "a.dcd")], topology=str(files / "top.pdb"))

    def test_ensembles_of_different_sizes_are_noticed(self, files):
        """A directory of 30 structures against a 300-frame trajectory is a
        legitimate comparison, and not one that shares a topology by accident."""
        study = Prothon(
            ensembles=[str(files / "a.dcd"), str(files / "models")],
            topology=str(files / "top.pdb"),
        )
        assert study.shares_topology      # same molecule, different sampling


class TestCommandLine:
    def test_compare_with_long_flags(self, files, capsys):
        code = main([
            "compare",
            "--ensembles", str(files / "a.dcd"), str(files / "b.dcd"),
            "--topology", str(files / "top.pdb"),
            "--order-parameters", "cbcn", "--random-state", "0", "--s-num", "2",
        ])
        assert code == 0
        assert "floor" in capsys.readouterr().out

    def test_compare_with_short_flags(self, files, capsys):
        code = main([
            "compare", "-e", f"{files / 'a.dcd'},{files / 'b.dcd'}",
            "-t", str(files / "top.pdb"), "-p", "cbcn", "-s", "0",
        ])
        assert code == 0
        assert "CBCN" in capsys.readouterr().out

    def test_a_reference_may_be_a_source_rather_than_an_index(self, files, capsys):
        code = main([
            "compare", "-e", str(files / "a.dcd"), str(files / "b.dcd"),
            "-r", str(files / "models"), "-t", str(files / "top.pdb"),
            "-p", "cacn", "-s", "0", "--s-num", "2",
        ])
        assert code == 0
        # Three ensembles now: the reference plus the two compared.
        assert capsys.readouterr().out.count("ensemble ") >= 2

    def test_a_reference_may_still_be_an_index(self, files, capsys):
        assert main([
            "compare", "-e", str(files / "a.dcd"), str(files / "b.dcd"),
            "-r", "1", "-t", str(files / "top.pdb"), "-p", "cacn",
            "-s", "0", "--s-num", "2",
        ]) == 0

    def test_the_table_view_is_the_same_command(self, files, capsys):
        """A benchmark is a comparison against a reference, presented
        differently. It does not need a command of its own, and having one
        invited the two to drift apart."""
        code = main([
            "compare", "-e", str(files / "models"), str(files / "b.dcd"),
            "-r", str(files / "a.dcd"), "-t", str(files / "top.pdb"),
            "-p", "cbcn", "-s", "0", "--report", "table",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "margin" in out and "precision" in out

    def test_the_two_views_run_the_same_comparison(self, files, capsys):
        """The table is a view, not a different calculation. The dissimilarity
        one reports is the dissimilarity the other reports."""
        import re

        common = [
            "-e", str(files / "b.dcd"), "-r", str(files / "a.dcd"),
            "-t", str(files / "top.pdb"), "-p", "cbcn", "-s", "0", "--s-num", "2",
        ]
        main(["compare", *common])
        summary = capsys.readouterr().out
        main(["compare", *common, "--report", "table"])
        table = capsys.readouterr().out

        from_summary = float(re.search(r"d = ([\d.]+)", summary).group(1))
        from_table = float(re.search(r"\| ([\d.]+) \| [\d.]+ \| \+", table).group(1))
        assert from_summary == pytest.approx(from_table, abs=0.02)

    def test_the_table_needs_something_to_rank(self, files, capsys):
        code = main([
            "compare", "-e", str(files / "a.dcd"), "-r", "0",
            "-t", str(files / "top.pdb"), "--report", "table",
        ])
        assert code == 2

    def test_validate(self, files, tmp_path, capsys):
        import mdtraj as md

        from prothon.validate import radius_of_gyration

        rg = radius_of_gyration(md.load(str(files / "a.dcd"), top=str(files / "top.pdb")))
        measurements = tmp_path / "rg.txt"
        np.savetxt(measurements, [[float(rg.mean()), 0.05]])
        code = main([
            "validate", "-e", str(files / "a.dcd"), "-t", str(files / "top.pdb"),
            "--observable", "rg", "--experimental", str(measurements), "-s", "0",
        ])
        assert code == 0
        assert "chi2_red" in capsys.readouterr().out

    def test_a_closed_pipe_is_not_an_error(self, monkeypatch, capsys):
        """`prothon info | head` closes the pipe mid-write. Every Unix tool has
        to survive that quietly; a traceback would be the only thing a reader
        sees of an otherwise successful run."""
        import builtins

        printed = {"n": 0}
        real_print = builtins.print

        def failing_print(*args, **kwargs):
            printed["n"] += 1
            if printed["n"] > 3:
                raise BrokenPipeError(32, "Broken pipe")
            real_print(*args, **kwargs)

        monkeypatch.setattr(builtins, "print", failing_print)
        # No os.dup2 stub: under pytest's capture, stdout has no real
        # descriptor, and the handler has to survive that too rather than
        # raising a second exception while cleaning up after the first.
        assert main(["info"]) == 0

    def test_info(self, capsys):
        assert main(["info"]) == 0
        out = capsys.readouterr().out
        assert "compare" in out and "cbcn" in out and "PED" in out

    def test_the_2x_command_line_still_works(self, files, capsys):
        with pytest.warns(DeprecationWarning, match="--random-state"):
            code = main([
                "-traj", f"{files / 'a.dcd'},{files / 'b.dcd'}",
                "-top", str(files / "top.pdb"), "-m", "cbcn", "--seed", "0",
            ])
        assert code == 0
        assert "CBCN" in capsys.readouterr().out

    def test_a_trajectory_without_a_topology_is_refused(self, files, capsys):
        code = main(["compare", "-e", str(files / "a.dcd"), str(files / "b.dcd")])
        assert code == 2
        assert "needs a topology" in capsys.readouterr().err

    def test_no_user_visible_name_still_says_measure(self):
        """The rename has to reach the JSON and the help text, not only the
        Python names. A `measure` key in a manifest beside an
        `--order-parameters` flag is the drift this was meant to end."""
        import dataclasses

        from prothon.batch.benchmark import BenchmarkRow
        from prothon.config.schema import COMMANDS, PARAMETERS
        from prothon.core.dissimilarity import ComparisonResult
        from prothon.core.precision_recall import PrecisionRecall

        for cls in (ComparisonResult, BenchmarkRow, PrecisionRecall):
            names = {f.name for f in dataclasses.fields(cls)}
            assert "measure" not in names, f"{cls.__name__} still has a measure field"

        text = " ".join(
            [p.help for p in PARAMETERS] + [c.help for c in COMMANDS]
        ).lower()
        assert "measures" not in text, "help text still says measures"

    def test_the_2x_keyword_warns(self, files):
        study = Prothon(
            ensembles=[str(files / "a.dcd"), str(files / "b.dcd")],
            topology=str(files / "top.pdb"), random_state=0,
        )
        with pytest.warns(DeprecationWarning, match="order_parameters"):
            study.compare_ensembles(measures="cbcn", s_num=2)

    def test_there_is_no_benchmark_command(self):
        """It was a second name for `compare --reference`, and two commands
        for one operation is how they come to differ."""
        from prothon.config.schema import COMMANDS

        assert "benchmark" not in {c.name for c in COMMANDS}

    def test_validate_without_uncertainties_is_refused(self, files, tmp_path, capsys):
        measurements = tmp_path / "one.txt"
        np.savetxt(measurements, [1.23])
        code = main([
            "validate", "-e", str(files / "a.dcd"), "-t", str(files / "top.pdb"),
            "--experimental", str(measurements),
        ])
        assert code == 2
        assert "arbitrary units" in capsys.readouterr().err

    def test_json_output_is_parseable(self, files, capsys):
        import json

        main([
            "compare", "-e", str(files / "a.dcd"), str(files / "b.dcd"),
            "-t", str(files / "top.pdb"), "-p", "cbcn", "-s", "0",
            "--s-num", "2", "--json",
        ])
        payload = json.loads(capsys.readouterr().out)
        assert payload["cbcn"][0]["noise_floor"] >= 0
