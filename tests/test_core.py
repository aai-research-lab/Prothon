"""Tests for the study object, the figures it writes, and the command line."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from prothon import Prothon
from prothon.cli import main
from prothon.plot.figures import (
    get_ensemble_colors,
    get_method_output_dir,
    replot_global_dissimilarity,
    replot_local_dissimilarity,
)


@pytest.fixture
def study(ensemble_files, topology_file, tmp_path):
    return Prothon(
        ensemble_files, topology_file, output_dir=str(tmp_path), random_state=0
    )


class TestConstruction:
    def test_accepts_a_comma_separated_string(self, ensemble_files, topology_file):
        study = Prothon(",".join(ensemble_files), topology_file)
        assert len(study.traj_files) == 3

    def test_a_single_ensemble_is_refused(self, ensemble_files, topology_file):
        # Comparing one ensemble with itself is not a comparison, and the 2.0
        # code accepted it and produced an empty result set.
        with pytest.raises(ValueError, match="at least two ensembles"):
            Prothon(ensemble_files[:1], topology_file)

    def test_missing_trajectory_is_named(self, topology_file):
        with pytest.raises(FileNotFoundError, match="absent.dcd"):
            Prothon(["absent.dcd", "gone.dcd"], topology_file)

    def test_missing_topology_is_named(self, ensemble_files):
        with pytest.raises(FileNotFoundError, match="Topology"):
            Prothon(ensemble_files, "no-such-top.pdb")


class TestCompareEnsembles:
    def test_returns_one_result_per_non_reference_ensemble(self, study):
        results = study.compare_ensembles(order_parameters="cbcn", s_num=2)
        assert set(results) == {"cbcn"}
        assert [r.ensemble_index for r in results["cbcn"]] == [1, 2]

    def test_finds_the_difference_it_was_given(self, study):
        # Ensemble 2 is compacted from residue 7; ensemble 1 differs from the
        # reference only by noise.
        results = study.compare_ensembles(order_parameters="cbcn", s_num=2)
        similar, different = results["cbcn"]
        assert different.global_dissimilarity > similar.global_dissimilarity

    def test_localises_the_difference_to_the_right_residues(self, study):
        results = study.compare_ensembles(order_parameters="cbcn", s_num=2)
        local = results["cbcn"][1].local_dissimilarity
        assert local[7:].mean() > local[:5].mean()

    def test_writes_the_expected_artifacts(self, study, tmp_path):
        study.compare_ensembles(order_parameters="cbcn", s_num=2)
        out = tmp_path / "cbcn_output"
        for name in (
            "ensemble_0_matrix.csv",
            "ensemble_0_matrix.png",
            "cbcn_global_dissimilarity_bar.png",
            "cbcn_global_dissimilarity_line.png",
            "cbcn_combined_local_dissimilarity.png",
            "manifest.json",
        ):
            assert (out / name).exists(), name

    def test_manifest_records_the_parameters(self, study, tmp_path):
        study.compare_ensembles(order_parameters="cbcn", s_num=2, alpha=0.01)
        manifest = json.loads((tmp_path / "cbcn_output" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["parameters"]["alpha"] == 0.01
        assert manifest["parameters"]["random_state"] == 0
        assert manifest["order_parameter"] == "cbcn"
        assert len(manifest["results"]) == 2
        assert "prothon_version" in manifest

    def test_multiple_measures_in_one_run(self, study):
        results = study.compare_ensembles(order_parameters="cbcn,cacn", s_num=2)
        assert set(results) == {"cbcn", "cacn"}

    def test_circular_measure_is_routed_as_circular(self, study):
        results = study.compare_ensembles(order_parameters="cata", s_num=2)
        assert results["cata"][0].raw_local_dissimilarity.size > 0

    def test_out_of_range_reference_is_refused(self, study):
        with pytest.raises(ValueError, match="out of range"):
            study.compare_ensembles(order_parameters="cbcn", ref=9)

    def test_unknown_projection_is_refused(self, study):
        with pytest.raises(ValueError, match="Unknown dimensionality"):
            study.compare_ensembles(order_parameters="cbcn", dimred="umap", s_num=2)

    def test_projection_runs_when_asked(self, study, tmp_path):
        study.compare_ensembles(order_parameters="cbcn", dimred="pca", s_num=2)
        assert (tmp_path / "cbcn_output" / "dim_reduction_pca.png").exists()
        assert set(study.get_dimred_results("cbcn")) == {"pca"}

    def test_mds_is_refused_above_the_frame_limit(self, study, monkeypatch):
        # Refusing beats an out-of-memory failure an hour in; and the refusal
        # must not lose the comparison that already succeeded.
        import prothon.plot.figures as plotting

        monkeypatch.setattr(plotting, "MDS_FRAME_LIMIT", 10)
        results = study.compare_ensembles(order_parameters="cbcn", dimred="mds", s_num=2)
        assert results["cbcn"]
        assert study.get_dimred_results("cbcn") == {}


class TestAccessorsAndSummary:
    def test_accessors_return_none_before_a_run(self, study):
        assert study.get_comparison_results("cbcn") is None
        assert study.get_representation_data("cbcn") is None

    def test_summary_mentions_the_floor(self, study):
        study.compare_ensembles(order_parameters="cbcn", s_num=2)
        text = study.summary()
        assert "floor" in text and "CBCN" in text

    def test_summary_before_running(self, study):
        assert "No comparisons" in study.summary()


class TestReplotting:
    def test_replot_honours_styling(self, study):
        # Version 2.0 accepted these and discarded them.
        study.compare_ensembles(order_parameters="cbcn", s_num=2)
        figure = study.replot_global_dissimilarity(
            "cbcn", plot_type="bar", xlabel="Q", ylabel="D", title="Custom"
        )
        axes = figure.axes[0]
        assert axes.get_xlabel() == "Q"
        assert axes.get_ylabel() == "D"
        assert axes.get_title() == "Custom"

    def test_replot_does_not_overwrite_the_saved_figure(self, study, tmp_path):
        study.compare_ensembles(order_parameters="cbcn", s_num=2)
        saved = tmp_path / "cbcn_output" / "cbcn_global_dissimilarity_bar.png"
        before = saved.stat().st_mtime_ns
        study.replot_global_dissimilarity("cbcn", plot_type="bar", color="k")
        assert saved.stat().st_mtime_ns == before

    def test_replot_local_accepts_raw(self, study):
        study.compare_ensembles(order_parameters="cbcn", s_num=2)
        figure = study.replot_local_dissimilarity("cbcn", 1, raw=True, color="r")
        assert figure.axes[0].get_ylabel() == "Local dissimilarity"
        np.testing.assert_array_equal(
            figure.axes[0].lines[0].get_xdata(),
            study.comparison_results["cbcn"][0].feature_index,
        )

    def test_replot_before_running_is_refused(self, study):
        with pytest.raises(ValueError, match="Run compare_ensembles first"):
            study.replot_global_dissimilarity("cbcn")

    def test_unknown_ensemble_index_lists_the_options(self, study):
        study.compare_ensembles(order_parameters="cbcn", s_num=2)
        with pytest.raises(ValueError, match="Available: 1, 2"):
            study.replot_local_dissimilarity("cbcn", 7)

    def test_standalone_replot_helpers(self):
        results = [
            {"ensemble_index": 1, "global_dissimilarity": 0.3, "noise_floor": 0.1},
            {"ensemble_index": 2, "global_dissimilarity": 0.5, "noise_floor": 0.1},
        ]
        assert replot_global_dissimilarity("cbcn", results).axes[0].get_xticks().size
        assert replot_local_dissimilarity("cbcn", np.linspace(0, 1, 30), 1) is not None

    def test_replot_uses_chain_aware_feature_identity(self):
        figure = replot_local_dissimilarity(
            "cacn",
            np.array([0.1, 0.2, 0.3]),
            1,
            feature_index=np.array([1, 2, 3]),
            feature_labels=np.array(["A:1", "A:2", "B:1"]),
        )
        assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == [
            "A:1", "A:2", "B:1"
        ]


class TestPlottingHelpers:
    def test_palette_is_stable_and_long_enough(self):
        assert get_ensemble_colors(3) == ["red", "gold", "darkgreen"]
        assert get_ensemble_colors(20) == get_ensemble_colors(20)
        assert len(get_ensemble_colors(20)) == 20

    def test_output_directory_is_created(self, tmp_path):
        path = get_method_output_dir(str(tmp_path), "cbcn")
        assert os.path.isdir(path) and path.endswith("cbcn_output")


class TestCli:
    def test_info_exits_cleanly(self, capsys):
        assert main(["--info"]) == 0
        assert "cbcn" in capsys.readouterr().out

    def test_no_arguments_prints_help_rather_than_failing(self, capsys):
        """A bare invocation should show what the tool can do. It used to
        error on missing required flags, which told a new reader nothing."""
        assert main([]) == 0
        assert "compare" in capsys.readouterr().out

    def test_an_unrecognised_argument_exits_two(self, capsys):
        assert main(["-traj", "a.dcd", "--nonsense"]) == 2
        assert "unrecognised" in capsys.readouterr().err

    def test_end_to_end_run(self, ensemble_files, topology_file, tmp_path, capsys):
        code = main([
            "-traj", ",".join(ensemble_files),
            "-top", topology_file,
            "-m", "cbcn",
            "-o", str(tmp_path),
            "--s-num", "2",
            "--seed", "0",
        ])
        assert code == 0
        assert "floor" in capsys.readouterr().out
        assert (tmp_path / "cbcn_output" / "manifest.json").exists()

    def test_json_output_is_parseable(self, ensemble_files, topology_file, tmp_path, capsys):
        main([
            "-traj", ",".join(ensemble_files), "-top", topology_file,
            "-m", "cbcn", "-o", str(tmp_path), "--s-num", "2", "--json",
        ])
        payload = json.loads(capsys.readouterr().out)
        assert payload["cbcn"][0]["noise_floor"] >= 0

    def test_a_bad_order_parameter_returns_two_not_a_traceback(
        self, ensemble_files, topology_file, tmp_path, capsys
    ):
        code = main([
            "-traj", ",".join(ensemble_files), "-top", topology_file,
            "-m", "wrong", "-o", str(tmp_path),
        ])
        assert code == 2
        assert "Unknown order parameter" in capsys.readouterr().err


class TestBackwardCompatibility:
    def test_old_import_path_still_works(self):
        with pytest.warns(DeprecationWarning, match="lowercase"):
            import importlib

            import Prothon as legacy

            importlib.reload(legacy)
        assert legacy.Prothon is Prothon


class TestPathCasing:
    def test_no_two_tracked_paths_differ_only_by_case(self):
        """macOS and Windows treat `prothon` and `Prothon` as one path.

        The 2.1 series briefly shipped `src/Prothon/__init__.py` beside
        `src/prothon/__init__.py`. On a case-insensitive filesystem git wrote
        one over the other, the package directory arrived named `Prothon`, and
        `import prothon` failed on every Mac. Linux CI passed throughout.
        """
        import subprocess
        from collections import defaultdict

        root = Path(__file__).resolve().parent.parent
        listing = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True
        )
        if listing.returncode != 0:  # not a git checkout (installed sdist)
            pytest.skip("not a git working tree")

        seen = defaultdict(list)
        for path in listing.stdout.split():
            seen[path.lower()].append(path)

        collisions = {k: v for k, v in seen.items() if len(v) > 1}
        assert not collisions, f"paths differing only by case: {collisions}"


class TestGlobalDissimilarityIsNotFiltered:
    """The magnitude of a difference must not depend on the significance filter.

    Through 2.1 ``global_dissimilarity`` was a mean over the *masked*
    per-feature values, so it read as exactly zero whenever nothing survived --
    including when the sampling was too poor to run a test at all and every
    p-value was withheld. ``resolved`` then compared that zero against a noise
    floor which is itself an unmasked mean over every feature, so the three
    ubiquitin ensembles furthest from the reference reported as unresolvable.
    """

    def _result(
        self,
        *,
        withheld,
        raw=0.40,
        floor=0.035,
        threshold=None,
        assessable=True,
        n=70,
    ):
        import numpy as np

        from prothon.compare.dissimilarity import ComparisonResult

        return ComparisonResult(
            ensemble_index=1,
            reference_index=0,
            global_dissimilarity=float(raw),
            masked_global_dissimilarity=0.0,
            local_dissimilarity=np.zeros(n),
            raw_local_dissimilarity=np.full(n, raw),
            p_values=np.ones(n),
            significant=np.zeros(n, dtype=bool),
            noise_floor=float(floor),
            n_frames=(5000, 5000),
            noise_floor_threshold=threshold,
            noise_floor_assessable=assessable,
            p_values_withheld=withheld,
        )

    def test_a_withheld_p_value_does_not_make_a_large_difference_unresolvable(self):
        result = self._result(withheld=True)
        assert result.resolved

    def test_nothing_significant_does_not_make_a_large_difference_unresolvable(self):
        result = self._result(withheld=False)
        assert result.resolved

    def test_the_two_means_are_compared_against_the_same_thing(self):
        """The floor is an unmasked mean, so `resolved` must use one too."""
        result = self._result(withheld=True)
        assert result.masked_global_dissimilarity == 0.0
        assert result.global_dissimilarity > result.masked_global_dissimilarity
        assert result.resolved is (result.global_dissimilarity > result.noise_floor)

    def test_a_difference_below_the_floor_is_still_unresolved(self):
        result = self._result(withheld=False, raw=0.02, floor=0.035)
        assert not result.resolved

    def test_the_mean_floor_is_not_the_decision_threshold(self):
        result = self._result(
            withheld=False, raw=0.04, floor=0.035, threshold=0.05
        )
        assert result.global_dissimilarity > result.noise_floor
        assert not result.resolved

    def test_too_few_floor_units_withhold_the_verdict(self):
        result = self._result(withheld=True, assessable=False)
        assert result.resolved is None
