"""Tests for the benchmark harness.

The case it exists for is the last class: two models that a table of raw
dissimilarities would rank the wrong way round, because one of them sampled
less.
"""

from __future__ import annotations

import json

import mdtraj as md
import numpy as np
import pytest
from test_ingest import as_residues, build

from prothon import benchmark
from prothon.ingest import Ensemble

SEQ = "ACDEFHIKLMNP"


def ensemble(n, seed, label, compact=None):
    return Ensemble(
        build(as_residues(SEQ), n_frames=n, seed=seed, compact_from=compact),
        label=label,
        provenance={"kind": "synthetic", "sampling_kind": "iid"},
    )


@pytest.fixture(scope="module")
def reference():
    return ensemble(3000, 1, "MD")


class TestTheTable:
    def test_one_row_per_model(self, reference):
        models = [ensemble(800, s, f"model-{s}") for s in (2, 3, 4)]
        result = benchmark(reference, models, order_parameters="cbcn", random_state=0)
        assert len(result.rows) == 3
        assert {r.model for r in result.rows} == {"model-2", "model-3", "model-4"}

    def test_several_measures_give_several_blocks(self, reference):
        result = benchmark(
            reference, [ensemble(800, 2, "m")], order_parameters="cbcn,cacn", random_state=0
        )
        assert len(result.rows) == 2
        assert {r.order_parameter for r in result.rows} == {"cbcn", "cacn"}
        assert result.table("cbcn").count("\n") >= 2

    def test_a_matching_model_is_not_resolvable(self, reference):
        result = benchmark(
            reference, [ensemble(1000, 2, "match")], order_parameters="cbcn", random_state=0
        )
        row = result.rows[0]
        assert not row.resolved
        assert "indistinguishable" in row.verdict

    def test_a_collapsed_model_is_caught(self, reference):
        result = benchmark(
            reference, [ensemble(1000, 3, "collapsed", compact=6)],
            order_parameters="cbcn", random_state=0,
        )
        row = result.rows[0]
        assert row.resolved
        assert row.missed or row.invented

    def test_results_serialise(self, reference, tmp_path):
        benchmark(
            reference, [ensemble(800, 2, "m")], order_parameters="cbcn",
            random_state=0, output_dir=str(tmp_path),
        )
        assert (tmp_path / "benchmark.md").exists()
        payload = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
        assert payload["reference"] == "MD"
        assert payload["rows"][0]["model"] == "m"
        assert "margin" in payload["rows"][0]
        assert payload["settings"]["n_permutations"] == 100
        assert payload["rows"][0]["analysis_settings"] == payload["settings"]
        assert payload["rows"][0]["feature_index"] == list(range(1, 13))
        assert payload["rows"][0]["feature_labels"] == [
            str(index) for index in range(1, 13)
        ]

    def test_the_methods_report_records_every_computation_setting(self, reference):
        result = benchmark(
            reference,
            [ensemble(800, 2, "m")],
            order_parameters="cbcn",
            random_state=7,
            metric="ks",
            sample_size=40,
            n_permutations=9,
            s_num=3,
            floor_repeats=4,
            x_num=21,
            alpha=0.01,
            n_jobs=1,
        )

        methods = result.methods()
        for text in (
            "order_parameters=['cbcn']", "metric='ks'", "sample_size=40",
            "n_permutations=9", "s_num=3", "coverage_floor_repeats=4",
            "x_num=21", "alpha=0.01", "n_jobs=1", "random_state=7",
        ):
            assert text in methods

    def test_no_models_is_refused(self, reference):
        with pytest.raises(ValueError, match="No models"):
            benchmark(reference, [], order_parameters="cbcn")


class TestSamplingIsPartOfTheResult:
    """The reason the table reports a margin rather than a distance."""

    def test_the_floor_rises_as_a_model_samples_less(self, reference):
        """A small sample cannot resemble anything closely, so the smallest
        difference that could be resolved from it is larger."""
        models = [ensemble(n, 9, f"n{n}") for n in (60, 300, 1500)]
        result = benchmark(reference, models, order_parameters="cbcn", random_state=0)
        floors = {r.n_model: r.noise_floor for r in result.rows if not r.refused}
        assert floors[60] > floors[1500], (
            "a thinly sampled model must carry a higher resolution limit"
        )

    def test_ranking_on_distance_alone_would_be_wrong(self, reference):
        """Two models equally wrong, one sampled thinly.

        Their raw dissimilarities come out close together, but the thin one
        carries a much higher floor — so a table ranked on distance flatters
        it, and a table ranked on the margin above the floor does not.
        """
        thick = ensemble(1200, 3, "thick", compact=6)
        thin = ensemble(40, 4, "thin", compact=6)
        result = benchmark(reference, [thick, thin], order_parameters="cbcn", random_state=0)
        rows = {r.model: r for r in result.rows}

        assert rows["thin"].noise_floor > 1.5 * rows["thick"].noise_floor
        # The margin is what corrects for it.
        assert rows["thick"].margin > rows["thin"].margin

    def test_the_table_is_ordered_by_margin(self, reference):
        models = [
            ensemble(1200, 3, "collapsed", compact=6),
            ensemble(1000, 2, "match"),
        ]
        table = benchmark(
            reference, models, order_parameters="cbcn", random_state=0
        ).table()
        assert table.index("collapsed") < table.index("match")

    def test_a_refusal_is_a_row_not_an_exception(self, reference):
        """An ensemble too small to support a comparison produces a row saying
        so. A benchmark that raises on one model loses the others; a benchmark
        that invents a number for it is worse."""
        tiny = Ensemble(
            build(as_residues(SEQ), n_frames=6, seed=7), label="tiny"
        )
        result = benchmark(
            reference, [ensemble(800, 2, "fine"), tiny],
            order_parameters="cbcn", random_state=0,
        )
        rows = {r.model: r for r in result.rows}
        assert rows["tiny"].refused
        assert rows["tiny"].dissimilarity is None
        assert rows["tiny"].verdict == "refused"
        assert rows["tiny"].metadata["prothon_version"]
        assert rows["tiny"].metadata["analysis_settings"]
        assert rows["tiny"].metadata["effective_samples"] is None
        assert rows["tiny"].metadata["refusal_details"]["reason"]
        assert rows["fine"].dissimilarity is not None   # the others survive

    def test_a_refused_row_renders(self, reference):
        tiny = Ensemble(build(as_residues(SEQ), n_frames=6, seed=7), label="tiny")
        table = benchmark(
            reference, [tiny], order_parameters="cbcn", random_state=0
        ).table()
        assert "tiny" in table and "—" in table


class TestAsymmetry:
    def test_reference_and_model_are_not_interchangeable(self, reference):
        """Precision and recall swap when the roles do, so the reference has
        to be the ensemble being matched."""
        model = ensemble(1000, 3, "collapsed", compact=6)
        forward = benchmark(reference, [model], order_parameters="cbcn", random_state=0).rows[0]
        backward = benchmark(model, [reference], order_parameters="cbcn", random_state=0).rows[0]
        assert forward.recall == pytest.approx(backward.precision, abs=0.05)


class TestSamplingProvenanceAndWeights:
    def test_a_trajectory_reference_and_iid_model_keep_separate_plans(self, reference):
        trajectory_reference = Ensemble(
            reference.trajectory,
            label="MD trajectory",
            provenance={
                "kind": "trajectory",
                "sampling_kind": "trajectory",
                "correlation_time_frames": 10.0,
                "stride": 5,
            },
        )
        model = ensemble(800, 20, "generated")
        row = benchmark(
            trajectory_reference,
            [model],
            order_parameters="cbcn",
            random_state=0,
            x_num=30,
            n_permutations=10,
        ).rows[0]

        provenance = row.metadata["sampling_provenance"]
        assert provenance["reference"]["sampling_kind"] == "trajectory"
        assert provenance["reference"]["time_stride"] == 5
        assert provenance["model"]["sampling_kind"] == "iid"
        assert [plan["sampling_kind"] for plan in row.metadata["sampling_plans"]] == [
            "trajectory",
            "iid",
        ]
        assert row.metadata["permutation_blocks"]["common_block_length"] == 20
        assert row.metadata["coverage_floor"]["block_length"] == [20, 1]
        assert row.metadata["effective_samples"][0] < row.n_reference
        assert row.metadata["effective_samples"][1] == pytest.approx(row.n_model)
        assert row.metadata["permutation_blocks"]["effective_units"][1] < row.n_model

    def test_reweighting_changes_the_rank(self, reference):
        matching = ensemble(600, 21, "matching")
        collapsed = ensemble(600, 22, "collapsed", compact=6)
        mixed = md.join([matching.trajectory, collapsed.trajectory])
        weights = np.r_[np.full(600, 0.99 / 600), np.full(600, 0.01 / 600)]
        models = [
            Ensemble(
                mixed,
                label="unweighted",
                provenance={"kind": "synthetic", "sampling_kind": "iid"},
            ),
            Ensemble(
                mixed,
                label="reweighted",
                weights=weights,
                provenance={"kind": "synthetic", "sampling_kind": "iid"},
            ),
        ]

        result = benchmark(
            reference,
            models,
            order_parameters="cbcn",
            random_state=0,
            x_num=30,
            n_permutations=10,
        )
        rows = {row.model: row for row in result.rows}

        assert rows["reweighted"].margin < rows["unweighted"].margin
        assert result.table().index("unweighted") < result.table().index("reweighted")
        assert rows["reweighted"].metadata["weighted"] == [False, True]
        assert rows["reweighted"].metadata["frame_weight_effective_samples"][1] < 700
        assert rows["reweighted"].metadata["coverage_floor"]["effective_units"][1] < 700

    def test_one_boolean_cannot_override_two_sampling_provenances(self, reference):
        with pytest.raises(ValueError, match="cannot describe a benchmark"):
            benchmark(
                reference,
                [ensemble(100, 23, "generated")],
                block_permutation=False,
            )
