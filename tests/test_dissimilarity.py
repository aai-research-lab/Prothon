"""Tests for density estimation, Jensen-Shannon distance and the statistics.

Several of these exist because version 2.0 got the case wrong, and are named so
that a regression says which behaviour came back.
"""

from __future__ import annotations

import numpy as np
import pytest

from prothon.compare.dissimilarity import (
    benjamini_hochberg,
    dissimilarity,
    estimate_pdf,
    jsd_local,
    random_sample,
)


class TestEstimatePdf:
    def test_returns_grid_and_density_of_requested_length(self):
        rng = np.random.default_rng(0)
        grid, pdf = estimate_pdf(rng.normal(size=300), -4, 4, 64)
        assert grid.shape == pdf.shape == (64,)

    def test_density_integrates_to_about_one(self):
        rng = np.random.default_rng(0)
        grid, pdf = estimate_pdf(rng.normal(size=2000), -6, 6, 400)
        assert np.trapezoid(pdf, grid) == pytest.approx(1.0, abs=0.02)

    def test_constant_column_does_not_raise(self):
        # A buried residue with zero SASA in every frame. Version 2.0 handed
        # this to gaussian_kde and died on a singular covariance matrix.
        grid, pdf = estimate_pdf(np.full(200, 3.0), 0, 10, 50)
        assert np.isfinite(pdf).all()
        assert pdf.sum() > 0

    def test_circular_grid_always_spans_a_full_turn(self):
        rng = np.random.default_rng(0)
        values = rng.uniform(-0.2, 0.2, 200)
        grid, _ = estimate_pdf(values, 0.0, 0.1, 90, circular=True)
        assert grid[0] == pytest.approx(-np.pi)
        assert grid[-1] == pytest.approx(np.pi)

    def test_circular_density_is_continuous_across_the_wraparound(self):
        # A population centred on pi straddles both ends of the grid. On a
        # linear kernel the estimate falls to nothing in the middle of the
        # population; on a circular one the two ends agree.
        rng = np.random.default_rng(3)
        angles = np.mod(rng.normal(np.pi, 0.3, 800) + np.pi, 2 * np.pi) - np.pi
        _, pdf = estimate_pdf(angles, -np.pi, np.pi, 180, circular=True)
        assert pdf[0] == pytest.approx(pdf[-1], rel=0.05)
        assert pdf[0] > pdf[len(pdf) // 2]

    def test_circular_density_integrates_to_about_one(self):
        rng = np.random.default_rng(5)
        angles = rng.vonmises(0.5, 4.0, 1500)
        grid, pdf = estimate_pdf(angles, -np.pi, np.pi, 400, circular=True)
        assert np.trapezoid(pdf, grid) == pytest.approx(1.0, abs=0.03)

    def test_empty_sample_is_refused(self):
        with pytest.raises(ValueError, match="empty sample"):
            estimate_pdf(np.array([]), 0, 1, 10)


class TestJsdLocal:
    def test_identical_samples_give_near_zero(self, identical_matrices):
        a, b = identical_matrices
        assert jsd_local(a, b, -4, 4, 80).max() < 0.2

    def test_disjoint_samples_approach_one(self, shifted_matrices):
        a, b = shifted_matrices
        assert jsd_local(a, b, -4, 10, 120).min() > 0.9

    def test_bounded_in_unit_interval(self, shifted_matrices):
        a, b = shifted_matrices
        values = jsd_local(a, b, -4, 10, 120)
        assert values.min() >= 0.0 and values.max() <= 1.0

    def test_symmetric(self, shifted_matrices):
        a, b = shifted_matrices
        np.testing.assert_allclose(
            jsd_local(a, b, -4, 10, 60), jsd_local(b, a, -4, 10, 60), atol=1e-12
        )

    def test_mismatched_feature_counts_are_refused(self):
        with pytest.raises(ValueError, match="different feature counts"):
            jsd_local(np.zeros((10, 4)), np.zeros((10, 5)), 0, 1, 10)


class TestBenjaminiHochberg:
    def test_leaves_a_single_p_value_alone(self):
        np.testing.assert_allclose(benjamini_hochberg(np.array([0.03])), [0.03])

    def test_is_monotone_and_bounded(self):
        rng = np.random.default_rng(0)
        adjusted = benjamini_hochberg(rng.uniform(size=200))
        assert adjusted.min() >= 0.0 and adjusted.max() <= 1.0
        raw = rng.uniform(size=50)
        adj = benjamini_hochberg(raw)
        assert (adj >= raw - 1e-12).all()  # adjustment never lowers a p-value

    def test_rejects_fewer_than_the_uncorrected_test(self):
        # 100 null features at alpha=0.05 yield about five false positives
        # uncorrected. Version 2.0 had no correction at all.
        rng = np.random.default_rng(1)
        raw = rng.uniform(size=100)
        assert (benjamini_hochberg(raw) < 0.05).sum() <= (raw < 0.05).sum()

    def test_empty_input(self):
        assert benjamini_hochberg(np.array([])).size == 0


class TestRandomSample:
    def test_shape_and_reproducibility(self):
        data = np.arange(200).reshape(50, 4).astype(float)
        first = random_sample(data, 30, np.random.default_rng(0))
        second = random_sample(data, 30, np.random.default_rng(0))
        assert first.shape == (30, 4)
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_differ(self):
        data = np.arange(200).reshape(50, 4).astype(float)
        assert not np.array_equal(
            random_sample(data, 30, np.random.default_rng(0)),
            random_sample(data, 30, np.random.default_rng(1)),
        )


class TestParallelReproducesSerial:
    """The reason seeds are drawn up front rather than inside the workers.

    A generator does not survive being sent to a worker process, and letting
    each worker seed itself would make the result depend on how many cores
    happened to be free. Drawing the seeds from the caller's generator before
    the work is divided makes a parallel run and a serial run identical.
    """

    @staticmethod
    def pair(seed=0, n=600, k=6):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, k)), rng.normal(0.4, 1, (n, k))

    def test_the_result_does_not_depend_on_the_worker_count(self):
        a, b = self.pair()
        common = dict(
            x_min=-4, x_max=4, x_num=60, s_num=4, n_permutations=20,
            sample_size=600, random_state=0, block_permutation=False,
        )
        serial = dissimilarity(a, b, n_jobs=1, **common)
        parallel = dissimilarity(a, b, n_jobs=2, **common)

        np.testing.assert_allclose(
            serial.raw_local_dissimilarity, parallel.raw_local_dissimilarity
        )
        np.testing.assert_allclose(serial.p_values, parallel.p_values)
        assert serial.noise_floor == pytest.approx(parallel.noise_floor)

    def test_it_holds_for_the_blocked_null_too(self):
        a, b = self.pair(seed=1)
        common = dict(
            x_min=-4, x_max=4, x_num=60, s_num=4, n_permutations=20,
            sample_size=600, random_state=3, block_permutation=True,
        )
        serial = dissimilarity(a, b, n_jobs=1, **common)
        parallel = dissimilarity(a, b, n_jobs=2, **common)
        np.testing.assert_allclose(serial.p_values, parallel.p_values)


class TestDissimilarity:
    def test_shifted_ensembles_are_resolved(self, shifted_matrices):
        a, b = shifted_matrices
        result = dissimilarity(a, b, -4, 10, x_num=60, s_num=3, random_state=0)
        assert result.global_dissimilarity > result.noise_floor
        assert result.resolved
        assert result.n_significant == 6

    def test_retains_power_on_a_subtle_difference(self):
        # Half a standard deviation apart -- detectable, but not the trivial
        # case. A test made conservative enough to pass the null case is only
        # useful if it still finds real differences.
        rng = np.random.default_rng(21)
        a = rng.normal(0.0, 1.0, (600, 6))
        b = rng.normal(0.5, 1.0, (600, 6))
        result = dissimilarity(a, b, -4, 6, x_num=60, s_num=3, random_state=0)
        assert result.n_significant == 6
        assert result.resolved

    def test_false_positive_rate_is_controlled(self):
        """The test that version 2.0 failed outright.

        Two independent samples of the same distribution must mostly not be
        called different. The 2.0 bootstrap null flagged 100% of features here;
        anything near that is the old behaviour returning.
        """
        flagged = total = 0
        for seed in range(12):
            rng = np.random.default_rng(1000 + seed)
            a = rng.normal(size=(400, 8))
            b = rng.normal(size=(400, 8))
            result = dissimilarity(a, b, -4, 4, x_num=50, s_num=3, random_state=seed)
            flagged += result.n_significant
            total += 8
        assert flagged / total < 0.10

    def test_legacy_null_is_known_to_be_anticonservative(self):
        """Pin the 2.0 behaviour so nobody restores it by accident.

        Bootstrap resamples of one ensemble share most of their frames, so the
        within-ensemble null is about half the true sampling variability and
        everything clears it. Kept reachable for regenerating old figures, and
        documented as wrong.
        """
        rng = np.random.default_rng(2024)
        a = rng.normal(size=(400, 6))
        b = rng.normal(size=(400, 6))
        legacy = dissimilarity(a, b, -4, 4, x_num=50, s_num=3, random_state=0, legacy=True)
        current = dissimilarity(a, b, -4, 4, x_num=50, s_num=3, random_state=0)
        assert legacy.n_significant == 6
        assert current.n_significant < legacy.n_significant

    def test_noise_floor_matches_independent_sampling(self):
        """The floor should sit near the distance between two independent
        samples of the same distribution, not half of it."""
        rng = np.random.default_rng(9)
        a = rng.normal(size=(600, 4))
        b = rng.normal(size=(600, 4))
        result = dissimilarity(a, b, -4, 4, x_num=60, s_num=4, random_state=0)
        independent = jsd_local(a[:300], b[:300], -4, 4, 60).mean()
        assert 0.4 * independent < result.noise_floor < 2.5 * independent

    def test_noise_floor_is_reported_and_positive(self, identical_matrices):
        a, b = identical_matrices
        result = dissimilarity(a, b, -4, 4, x_num=60, s_num=3, random_state=0)
        assert result.noise_floor > 0

    def test_raw_values_survive_masking(self, identical_matrices):
        # The masked curve is zero where nothing reached significance; the raw
        # curve still shows how close each residue came.
        a, b = identical_matrices
        result = dissimilarity(a, b, -4, 4, x_num=60, s_num=3, random_state=0)
        assert (result.raw_local_dissimilarity > 0).any()

    def test_per_feature_significance_is_not_all_or_nothing(self):
        # Version 2.0's `local[p >= 0.05] = 0` used one scalar p-value as a
        # mask over every residue, so either all of them survived or none did.
        rng = np.random.default_rng(4)
        reference = rng.normal(0, 1, (400, 6))
        other = reference.copy()
        other[:, :3] = rng.normal(6, 1, (400, 3))
        result = dissimilarity(reference, other, -4, 10, x_num=60, s_num=3, random_state=0)
        assert 0 < result.n_significant < 6
        assert result.significant[:3].all()
        assert not result.significant[3:].any()

    def test_seeded_runs_are_identical(self, shifted_matrices):
        a, b = shifted_matrices
        kwargs = dict(x_num=50, s_num=3, random_state=42)
        first = dissimilarity(a, b, -4, 10, **kwargs)
        second = dissimilarity(a, b, -4, 10, **kwargs)
        assert first.global_dissimilarity == second.global_dissimilarity
        np.testing.assert_array_equal(first.p_values, second.p_values)

    def test_legacy_mode_broadcasts_one_p_value(self, shifted_matrices):
        a, b = shifted_matrices
        result = dissimilarity(
            a, b, -4, 10, x_num=50, s_num=3, random_state=0, legacy=True
        )
        assert len(set(result.p_values.tolist())) == 1

    def test_mismatched_features_are_refused(self):
        with pytest.raises(ValueError, match="Feature counts differ"):
            dissimilarity(np.zeros((60, 4)), np.zeros((60, 5)), 0, 1)

    def test_small_ensembles_warn(self):
        rng = np.random.default_rng(0)
        small = rng.normal(size=(12, 3))
        with pytest.warns(UserWarning, match="12 independent conformations"):
            dissimilarity(small, small, -4, 4, x_num=30, s_num=2, random_state=0)

    def test_result_supports_dictionary_access(self, shifted_matrices):
        # Version 2.0 returned dicts and user code indexes them.
        a, b = shifted_matrices
        result = dissimilarity(a, b, -4, 10, x_num=50, s_num=3, random_state=0)
        assert result["global_dissimilarity"] == result.global_dissimilarity
        assert result["ensemble_index"] == 0
        assert result.get("nonexistent", "fallback") == "fallback"
        with pytest.raises(KeyError):
            result["nonexistent"]

    def test_result_serialises_to_json_types(self, shifted_matrices):
        import json

        a, b = shifted_matrices
        result = dissimilarity(a, b, -4, 10, x_num=50, s_num=3, random_state=0)
        assert json.loads(json.dumps(result.to_dict()))["order_parameter"] == ""
