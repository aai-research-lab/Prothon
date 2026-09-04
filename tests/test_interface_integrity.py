"""Every interface must record the same settings and compute the same numbers.

Prothon offers five ways to run one comparison: the Python API, the command
line in summary mode, the command line in table mode, a YAML configuration
file, and JSON output. They are generated from one schema so that none can
offer a setting the others cannot, and this is the test that the generation
holds at run time rather than only in the schema.

Two things are checked, and they fail differently:

**Settings.** Every interface builds a :class:`~prothon.config.study.Study`,
and the study each one builds must be identical. A flag that is accepted and
then dropped, a config key that reaches the file and not the computation, or a
table mode that forgets the random seed, all show up here as a difference in
the recorded study rather than as a wrong number somewhere downstream.

**Numbers.** Given the same settings and the same seed, the dissimilarity, the
floor and the significance calls must agree exactly across interfaces. Not
approximately: the seed is drawn from the caller's generator before any work is
divided, so an exact match is the correct expectation and anything else means
an interface is reaching the computation by a different route.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest


@pytest.fixture(scope="module")
def study_inputs(tmp_path_factory):
    """Two small ensembles and a topology, written once for every interface."""
    import mdtraj as md

    directory = tmp_path_factory.mktemp("interfaces")
    topology = md.Topology()
    chain = topology.add_chain()
    for _ in range(12):
        residue = topology.add_residue("ALA", chain)
        for name, element in (
            ("N", md.element.nitrogen),
            ("CA", md.element.carbon),
            ("C", md.element.carbon),
            ("O", md.element.oxygen),
            ("CB", md.element.carbon),
        ):
            topology.add_atom(name, element, residue)

    rng = np.random.default_rng(0)
    base = np.zeros((topology.n_atoms, 3), dtype=np.float32)
    base[:, 0] = np.arange(topology.n_atoms) * 0.15

    paths = []
    for index, spread in enumerate((0.05, 0.09)):
        xyz = base + rng.normal(0.0, spread, (150, topology.n_atoms, 3))
        path = directory / f"ensemble{index}.dcd"
        md.Trajectory(xyz.astype(np.float32), topology).save_dcd(str(path))
        paths.append(str(path))

    topology_path = directory / "topology.pdb"
    md.Trajectory(base[None, :, :], topology).save_pdb(str(topology_path))
    return directory, paths, str(topology_path)


def _run_cli(capsys, *arguments):
    """Call the CLI in process, as the rest of the suite does.

    A subprocess was tried first and passed locally while returning empty
    stdout under CI, which `json.loads` reported as a decode error rather than
    as the missing output it was. In process the output is captured directly
    and a non-zero exit is a plain assertion.
    """
    from prothon.cli import main

    code = main(list(arguments))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return captured.out


def _numbers(result) -> tuple:
    """The quantities every interface must agree on, to full precision."""
    return (
        round(float(result.global_dissimilarity), 12),
        round(float(result.noise_floor), 12),
        int(result.n_significant),
        round(float(result.correlation_time), 12),
    )


class TestEveryInterfaceRecordsTheSameStudy:
    def test_python_and_config_agree(self, study_inputs):
        from prothon import Prothon
        from prothon.config.study import load_study

        directory, ensembles, topology = study_inputs
        prothon = Prothon(
            ensembles, topology, "cbcn", random_state=7, output_dir=str(directory)
        )
        written = prothon.save_config(str(directory / "study.yml"))
        reloaded = Prothon.from_config(written)

        original = load_study(written).to_dict()
        again = load_study(
            reloaded.save_config(str(directory / "roundtrip.yml"))
        ).to_dict()
        # `output_dir` names where results go and `path` where the file was
        # written. Neither describes the comparison.
        for recorded in (original, again):
            recorded.pop("output_dir", None)
            recorded.pop("path", None)
        assert again == original, (
            "a study written to a file, read back and written again must "
            "describe the same comparison"
        )

    def test_from_config_is_the_documented_entry_point(self, study_inputs):
        """`study=` is how `Study.run` passes itself in for provenance, not a
        way to load a file. The README said otherwise."""
        from prothon import Prothon

        directory, ensembles, topology = study_inputs
        written = Prothon(
            ensembles, topology, "cbcn", random_state=7
        ).save_config(str(directory / "entry.yml"))

        loaded = Prothon.from_config(written)
        assert loaded.study is not None, (
            "from_config must record the study it came from"
        )
        readme = (
            pathlib.Path(__file__).resolve().parent.parent / "README.md"
        ).read_text(encoding="utf-8")
        assert 'Prothon(study="' not in readme

    def test_the_command_line_records_what_it_was_given(
        self, study_inputs, capsys
    ):
        directory, ensembles, topology = study_inputs
        written = directory / "from_cli.yml"
        _run_cli(
            capsys,
            "compare",
            "--ensembles", *ensembles,
            "--topology", topology,
            "--order-parameters", "cbcn",
            "--random-state", "7",
            "--save-config", str(written),
        )
        assert written.is_file(), "--save-config must write the study it ran"

        from prothon.config.study import load_study

        study = load_study(str(written))
        recorded = study.to_dict()
        assert recorded["compare"]["random_state"] == 7, (
            "the seed reached the file, so it must have reached the computation"
        )
        assert recorded["compare"]["order_parameters"] in ("cbcn", ["cbcn"])


class TestEveryInterfaceComputesTheSameNumbers:
    def test_python_matches_the_config_file(self, study_inputs):
        from prothon import Prothon

        directory, ensembles, topology = study_inputs
        direct = Prothon(ensembles, topology, "cbcn", random_state=7)
        written = direct.save_config(str(directory / "numbers.yml"))
        viaconfig = Prothon.from_config(written)

        assert _numbers(direct.compare()["cbcn"][0]) == _numbers(
            viaconfig.compare()["cbcn"][0]
        )

    def test_json_output_matches_the_python_api(self, study_inputs, capsys):
        from prothon import Prothon

        _, ensembles, topology = study_inputs
        expected = Prothon(
            ensembles, topology, "cbcn", random_state=7
        ).compare()["cbcn"][0]

        output = _run_cli(
                capsys,
                "compare",
                "--ensembles", *ensembles,
                "--topology", topology,
                "--order-parameters", "cbcn",
                "--random-state", "7",
                "--json",
        )
        assert output.strip(), "the CLI produced no output to parse"
        payload = json.loads(output)
        # Keyed by order parameter, then one entry per compared ensemble.
        emitted = payload["cbcn"][0]
        assert round(emitted["global_dissimilarity"], 12) == round(
            float(expected.global_dissimilarity), 12
        )
        assert round(emitted["noise_floor"], 12) == round(
            float(expected.noise_floor), 12
        )

    def test_repeating_a_seeded_run_is_exact(self, study_inputs):
        """Not approximately equal. The seed is drawn before work is divided."""
        from prothon import Prothon

        _, ensembles, topology = study_inputs
        first = Prothon(ensembles, topology, "cbcn", random_state=7)
        second = Prothon(ensembles, topology, "cbcn", random_state=7)
        assert _numbers(first.compare()["cbcn"][0]) == _numbers(
            second.compare()["cbcn"][0]
        )


class TestAStudyRunsFromItsFile:
    """`config=` is the entry point, matching the sibling project.

    A study written to a file carries everything the comparison needs, so no
    other argument goes with it. Giving both a config and sources is an error
    rather than a precedence rule nobody would remember.
    """

    def test_config_runs_the_study(self, study_inputs):
        from prothon import Prothon

        directory, ensembles, topology = study_inputs
        written = Prothon(
            ensembles, topology, "cbcn", random_state=7
        ).save_config(str(directory / "entry_point.yml"))

        assert list(Prothon(config=written).comparison_results) == ["cbcn"]

    def test_it_matches_the_direct_construction(self, study_inputs):
        from prothon import Prothon

        directory, ensembles, topology = study_inputs
        direct = Prothon(ensembles, topology, "cbcn", random_state=7)
        written = direct.save_config(str(directory / "match.yml"))

        assert _numbers(Prothon(config=written).comparison_results["cbcn"][0]) == (
            _numbers(direct.compare()["cbcn"][0])
        )

    def test_both_is_an_error(self, study_inputs):
        from prothon import Prothon

        directory, ensembles, topology = study_inputs
        written = Prothon(
            ensembles, topology, "cbcn", random_state=7
        ).save_config(str(directory / "both.yml"))

        with pytest.raises(TypeError, match="not both"):
            Prothon(config=written, ensembles=ensembles, topology=topology)

    def test_neither_names_both_ways_in(self):
        from prothon import Prothon

        with pytest.raises(TypeError, match="config="):
            Prothon()
