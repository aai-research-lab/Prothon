"""Tests for correlation-time estimation and block permutation.

The measurement these exist to protect: with frames correlated in time, the
frame-permutation null called 99% of features different when nothing differed.
"""

from __future__ import annotations

import numpy as np
import pytest

from prothon.compare.dissimilarity import _structured_subsample, dissimilarity
from prothon.sampling.correlation import (
    MINIMUM_BLOCKS,
    block_labels,
    correlation_profile,
    correlation_time,
    effective_frames,
    plan_blocks,
)
from prothon.sampling.floor import split_half_floor
from prothon.sampling.null import permutation_null


def ou(n_frames, n_features, tau, rng, mean=0.0):
    """An Ornstein-Uhlenbeck series: correlated frames, known correlation time,
    and a stationary distribution that does not depend on tau."""
    phi = np.exp(-1.0 / tau)
    series = np.empty((n_frames, n_features))
    series[0] = rng.normal(size=n_features)
    noise = rng.normal(size=(n_frames, n_features)) * np.sqrt(1 - phi**2)
    for t in range(1, n_frames):
        series[t] = phi * series[t - 1] + noise[t]
    return series + mean


class TestTrajectorySubsampling:
    def test_a_trajectory_is_reduced_to_one_contiguous_window(self):
        data = np.arange(5000, dtype=float)[:, None]
        sample = _structured_subsample(
            data, None, 1000, np.random.default_rng(0), "trajectory", None
        )
        assert sample.matrix.shape == (1000, 1)
        np.testing.assert_array_equal(np.diff(sample.matrix[:, 0]), np.ones(999))
        assert sample.strategy == "contiguous window"

    def test_weights_stay_attached_to_the_contiguous_frames(self):
        data = np.arange(100, dtype=float)[:, None]
        weights = np.arange(1, 101, dtype=float)
        sample = _structured_subsample(
            data, weights, 20, np.random.default_rng(3), "trajectory", None
        )
        indices = sample.matrix[:, 0].astype(int)
        expected = weights[indices] / weights[indices].sum()
        np.testing.assert_allclose(sample.weights, expected)


class TestCorrelationTime:
    @pytest.mark.parametrize("tau", [1, 2, 5, 10, 20, 50])
    def test_recovers_a_known_correlation_time(self, tau):
        """For AR(1), tau_int = (1 + phi) / (1 - phi) exactly.

        The summary across features is the upper quartile rather than a high
        quantile: per-feature estimates are noisy, and a 90th percentile over
        a few hundred of them chases the worst estimate rather than the
        slowest residue — on this data it lands 30-150% above the truth and
        grows with the number of features.
        """
        rng = np.random.default_rng(tau)
        series = ou(20000, 4, tau, rng)
        phi = np.exp(-1.0 / tau)
        expected = (1 + phi) / (1 - phi)
        assert correlation_time(series) == pytest.approx(expected, rel=0.3)

    def test_independent_frames_give_one(self):
        rng = np.random.default_rng(0)
        assert correlation_time(rng.normal(size=(5000, 6))) == pytest.approx(
            1.0, abs=0.5
        )

    def test_a_constant_feature_does_not_break_it(self):
        rng = np.random.default_rng(1)
        series = rng.normal(size=(1000, 3))
        series[:, 1] = 4.0
        assert np.isfinite(correlation_time(series))

    def test_a_short_series_reports_no_correlation(self):
        rng = np.random.default_rng(2)
        assert correlation_time(rng.normal(size=(8, 3))) == 1.0

    def test_shuffling_the_frames_destroys_the_estimate(self):
        """The estimate is a property of the frame *order*. A shuffled matrix
        has no correlation time, and the docstring warns that this silently
        disables the correction -- so the behaviour is pinned."""
        rng = np.random.default_rng(3)
        series = ou(4000, 4, 20.0, rng)
        assert correlation_time(series) > 10
        shuffled = series[rng.permutation(series.shape[0])]
        assert correlation_time(shuffled) < 3

    def test_the_reproduced_iid_slow_mixture_defeats_q75(self):
        """The audit reproduction: q75 ~1.05, slow median ~71.96."""
        rng = np.random.default_rng(27)
        iid = rng.normal(size=(4000, 80))
        slow = ou(4000, 20, 45.0, rng)

        profile = correlation_profile(np.column_stack((iid, slow)))
        slow_median = float(np.median(profile.feature_times[-20:]))

        assert profile.quantile_tau == pytest.approx(1.05, abs=0.1)
        assert slow_median == pytest.approx(71.96, abs=1.0)
        assert profile.tau == slow_median
        assert profile.summary == "coherent slow-feature median"
        assert set(profile.slow_features) == set(range(80, 100))

    def test_one_slow_loop_cannot_hide_below_the_upper_quartile(self):
        """Eight slow columns are invisible to q75 among 100 columns.

        They are nevertheless a coherent region and need blocks long enough
        for their own dynamics. The slow-group median protects them without
        using the noisiest individual estimate.
        """
        rng = np.random.default_rng(77)
        fast = ou(3000, 92, 2.0, rng)
        slow = ou(3000, 8, 20.0, rng)

        profile = correlation_profile(np.column_stack((fast, slow)))

        assert profile.quantile_tau < 8.0, "q75 reproduces the old failure"
        assert profile.summary == "coherent slow-feature median"
        assert profile.tau > 20.0
        assert set(profile.slow_features) == set(range(92, 100))

    def test_separate_slow_groups_are_combined_conservatively(self):
        rng = np.random.default_rng(80)
        matrix = np.column_stack((
            ou(2500, 30, 2.0, rng),
            ou(2500, 4, 20.0, rng),
            ou(2500, 30, 2.0, rng),
            ou(2500, 4, 20.0, rng),
            ou(2500, 32, 2.0, rng),
        ))

        profile = correlation_profile(matrix)

        expected = set(range(30, 34)) | set(range(64, 68))
        assert set(profile.slow_features) == expected
        assert profile.tau > 20.0

    def test_circular_slow_features_are_estimated_across_the_branch_cut(self):
        rng = np.random.default_rng(90)
        iid_angles = rng.uniform(-np.pi, np.pi, size=(3000, 92))
        slow_angles = np.pi + 0.5 * ou(3000, 8, 20.0, rng)
        slow_angles = (slow_angles + np.pi) % (2.0 * np.pi) - np.pi
        matrix = np.column_stack((iid_angles, slow_angles))

        linear = correlation_profile(matrix)
        circular = correlation_profile(matrix, circular=True)

        assert set(circular.slow_features) == set(range(92, 100))
        assert circular.tau > 30.0
        assert circular.tau > 1.5 * linear.tau

    def test_one_noisy_column_does_not_set_the_global_block_plan(self):
        rng = np.random.default_rng(78)
        fast = ou(3000, 99, 2.0, rng)
        lone_slow = ou(3000, 1, 20.0, rng)

        profile = correlation_profile(np.column_stack((fast, lone_slow)))

        assert profile.summary == "upper quartile"
        assert profile.slow_features == ()
        assert profile.tau == profile.quantile_tau

    def test_a_homogeneous_system_keeps_the_stable_summary(self):
        profile = correlation_profile(
            ou(3000, 100, 20.0, np.random.default_rng(4))
        )
        phi = np.exp(-1.0 / 20.0)
        expected = (1 + phi) / (1 - phi)

        assert profile.summary == "upper quartile"
        assert profile.tau == pytest.approx(expected, rel=0.3)
        assert profile.tau < np.quantile(profile.feature_times, 0.95)

    def test_constants_are_removed_before_feature_subsampling(self):
        rng = np.random.default_rng(79)
        constants = np.ones((2000, 240))
        dynamic = ou(2000, 10, 2.0, rng)
        dynamic[:, 0] = np.nan

        profile = correlation_profile(
            np.column_stack((constants, dynamic)), max_features=5
        )

        assert profile.n_assessable_features == 9
        assert profile.n_sampled_features == 5
        assert all(column >= 241 for column in profile.sampled_features)

    def test_profile_parameters_are_validated(self):
        matrix = np.arange(100.0)[:, None]
        with pytest.raises(ValueError, match="max_features"):
            correlation_profile(matrix, max_features=0)
        with pytest.raises(ValueError, match="quantile"):
            correlation_profile(matrix, quantile=1.1)


class TestMixedRateCalibration:
    def test_the_slow_region_is_protected_without_withholding_the_null(self):
        """A mixed fast/slow correlated null remains an actual test.

        Both ensembles have 18 fast and two slow columns and are drawn from
        the same stationary distribution. The old q75 plan used the fast time;
        the profile uses the two-column slow group, leaves enough complete
        blocks to test, and makes no false call for this fixed calibration.
        """

        def mixed(seed, slow_shift=0.0):
            rng = np.random.default_rng(seed)
            return np.column_stack((
                ou(2000, 18, 2.0, rng),
                ou(2000, 2, 20.0, rng, mean=slow_shift),
            ))

        with pytest.warns(UserWarning, match="still rising"):
            result = dissimilarity(
                mixed(100), mixed(200), -5.0, 5.0,
                x_num=20, s_num=5, n_permutations=40,
                sample_size=2000, random_state=0,
            )

        assert result.correlation_time > 20.0
        assert not result.correlation_time_converged
        assert result.n_blocks == 33
        assert result.n_blocks >= MINIMUM_BLOCKS
        assert not result.p_values_withheld
        assert result.n_significant == 0
        profiles = result.metadata["correlation_profiles"]
        assert profiles["reference"]["assessable_feature_columns"] == list(range(20))
        assert profiles["reference"]["slow_feature_columns"] == [18, 19]
        assert profiles["comparison"]["slow_feature_columns"] == [18, 19]

        with pytest.warns(UserWarning, match="still rising"):
            shifted = dissimilarity(
                mixed(100), mixed(200, slow_shift=1.0), -6.0, 6.0,
                x_num=20, s_num=5, n_permutations=40,
                sample_size=2000, random_state=0,
            )
        np.testing.assert_array_equal(
            shifted.significant,
            np.array([False] * 18 + [True, True]),
        )


class TestBlockPlanning:
    def test_effective_frames_divides_by_the_correlation_time(self):
        assert effective_frames(10000, 20.0) == pytest.approx(500.0)
        assert effective_frames(10000, 1.0) == pytest.approx(10000.0)

    def test_uncorrelated_data_uses_single_frame_blocks(self):
        length, count = plan_blocks(1000, 1.0)
        assert length == 1 and count == 1000

    def test_blocks_are_a_few_correlation_times_long(self):
        length, count = plan_blocks(2000, 20.0)
        assert 20 <= length <= 60
        assert count >= MINIMUM_BLOCKS

    def test_a_block_is_never_shortened_to_manufacture_blocks(self):
        """The one thing the planner must not do.

        A block shorter than the correlation time does not contain the
        correlation, so the null it builds is the frame-permutation null under
        a block-shaped name — and the block count that was supposed to reveal
        the problem is the count that was forced to look healthy. The planner
        returns few blocks and the caller refuses.
        """
        length, count = plan_blocks(500, 400.0)
        assert length >= 400, "the block must still hold a correlation time"
        assert count < MINIMUM_BLOCKS, "and the shortfall must be visible"

    def test_labels_cover_every_frame_once(self):
        labels = block_labels(1000, 30)
        assert labels.size == 1000
        assert labels.min() == 0
        assert np.all(np.diff(labels) >= 0)

    def test_a_trailing_stub_is_merged_backwards(self):
        """A final block of two frames would have unrepresentative correlation
        structure, so it joins the block before it."""
        labels = block_labels(100, 30)
        _, counts = np.unique(labels, return_counts=True)
        assert counts.min() >= 30


class TestWholeBlockPermutation:
    @staticmethod
    def _labelled_ensembles():
        # block_labels(10, 4) produces blocks of 4 and 6 frames. Globally
        # unique labels let the statistic detect whether either block was
        # divided between the relabelled ensembles.
        labels_a = np.repeat([0.0, 1.0], [4, 6])
        labels_b = np.repeat([2.0, 3.0], [4, 6])
        return (
            np.column_stack([labels_a, labels_a]),
            np.column_stack([labels_b, labels_b]),
        )

    @staticmethod
    def _split_count_and_left_size(left, right, _weights_a, _weights_b):
        split = np.intersect1d(left[:, 0], right[:, 0]).size
        return np.array([split, left.shape[0]], dtype=float)

    def test_a_relabelled_block_is_never_split_between_ensembles(self):
        reference, other = self._labelled_ensembles()
        null = permutation_null(
            1,
            self._split_count_and_left_size,
            reference,
            other,
            n_permutations=200,
            rng=np.random.default_rng(0),
            block_length=4,
        )

        np.testing.assert_array_equal(null[:, 0], np.zeros(200))
        # Two whole blocks are assigned to each side. Their unequal lengths
        # mean the frame count may vary; exact frame counts would require a cut.
        assert set(null[:, 1]) == {8.0, 10.0, 12.0}

    def test_parallel_and_serial_block_relabellings_agree(self):
        reference, other = self._labelled_ensembles()
        arguments = (
            self._split_count_and_left_size,
            reference,
            other,
            40,
        )
        serial = permutation_null(
            1, *arguments, rng=np.random.default_rng(7), block_length=4
        )
        parallel = permutation_null(
            2, *arguments, rng=np.random.default_rng(7), block_length=4
        )
        np.testing.assert_array_equal(serial, parallel)

    def test_asymmetric_units_are_never_split(self):
        labels_ref = np.repeat([0.0, 1.0], [4, 6])
        labels_other = np.repeat([2.0, 3.0, 4.0], [3, 3, 4])
        reference = np.column_stack([labels_ref, labels_ref])
        other = np.column_stack([labels_other, labels_other])
        units_ref = [np.arange(4), np.arange(4, 10)]
        units_other = [np.arange(3), np.arange(3, 6), np.arange(6, 10)]

        null = permutation_null(
            1,
            self._split_count_and_left_size,
            reference,
            other,
            n_permutations=100,
            rng=np.random.default_rng(4),
            units_a=units_ref,
            units_b=units_other,
        )

        np.testing.assert_array_equal(null[:, 0], np.zeros(100))
        assert np.unique(null[:, 1]).size > 1

    def test_sparse_unit_weights_never_make_an_empty_candidate(self):
        reference, other = self._labelled_ensembles()
        units = [np.arange(4), np.arange(4, 10)]
        weights_ref = np.r_[np.full(4, 0.25), np.zeros(6)]
        weights_other = np.r_[np.full(4, 0.25), np.zeros(6)]

        def weighted_difference(left, right, weights_left, weights_right):
            difference = np.average(left[:, 0], weights=weights_left) - np.average(
                right[:, 0], weights=weights_right
            )
            return np.array([difference, difference])

        null = permutation_null(
            1,
            weighted_difference,
            reference,
            other,
            n_permutations=100,
            rng=np.random.default_rng(6),
            weights_a=weights_ref,
            weights_b=weights_other,
            units_a=units,
            units_b=units,
        )

        assert np.isfinite(null).all()


class TestPerEnsembleSampling:
    def test_a_trajectory_and_iid_model_keep_different_native_plans(self):
        rng = np.random.default_rng(123)
        reference = ou(900, 4, 10.0, rng)
        model = rng.normal(size=(900, 4))

        result = dissimilarity(
            reference,
            model,
            -5,
            5,
            x_num=25,
            s_num=3,
            n_permutations=20,
            sample_size=600,
            random_state=0,
            sampling_kind_ref="trajectory",
            sampling_kind="iid",
            correlation_time_frames_ref=10.0,
        )

        plans = result.metadata["sampling_plans"]
        assert [plan["sampling_kind"] for plan in plans] == ["trajectory", "iid"]
        assert [plan["native_block_length"] for plan in plans] == [20, 1]
        assert result.metadata["sampling_strategies"] == [
            "contiguous window",
            "uniform without replacement",
        ]
        # The null uses equal-sized exchange units without pretending the IID
        # model itself has temporal autocorrelation. Its native floor remains
        # frame based.
        assert result.metadata["permutation_units"] == [30, 30]
        assert result.metadata["noise_floor_quantile"] == 0.95

    def test_the_old_false_boolean_cannot_override_explicit_provenance(self):
        rng = np.random.default_rng(124)
        a, b = rng.normal(size=(100, 2)), rng.normal(size=(100, 2))
        with pytest.raises(ValueError, match="cannot be combined"):
            dissimilarity(
                a,
                b,
                -4,
                4,
                block_permutation=False,
                sampling_kind_ref="trajectory",
                sampling_kind="iid",
            )


class TestCorrelationAwareFloor:
    @staticmethod
    def _split_count_and_left_size(left, right, _weights_a, _weights_b):
        split = np.intersect1d(left[:, 0], right[:, 0]).size
        return np.array([split, left.shape[0]], dtype=float)

    def test_a_temporal_block_is_never_split_between_halves(self):
        labels = np.repeat([0.0, 1.0], [4, 6])
        ensemble = np.column_stack([labels, labels])
        floor = split_half_floor(
            1,
            self._split_count_and_left_size,
            (ensemble,),
            repeats=100,
            rng=np.random.default_rng(0),
            block_lengths=4,
        )

        np.testing.assert_array_equal(floor[:, 0], np.zeros(100))
        assert set(floor[:, 1]) == {4.0, 6.0}

    def test_complete_replicas_are_the_exchangeable_units(self):
        labels = np.repeat([0.0, 1.0, 2.0, 3.0], [3, 5, 4, 6])
        ensemble = np.column_stack([labels, labels])
        floor = split_half_floor(
            1,
            self._split_count_and_left_size,
            (ensemble,),
            repeats=100,
            rng=np.random.default_rng(1),
            replica_labels=labels,
        )

        np.testing.assert_array_equal(floor[:, 0], np.zeros(100))
        assert np.unique(floor[:, 1]).size > 1

    def test_random_rows_understate_a_correlated_floor(self):
        ensemble = ou(2000, 8, 20.0, np.random.default_rng(12))

        def mean_difference(left, right, _weights_a, _weights_b):
            return np.abs(left.mean(axis=0) - right.mean(axis=0))

        iid = split_half_floor(
            1,
            mean_difference,
            (ensemble,),
            repeats=100,
            rng=np.random.default_rng(2),
        )
        blocked = split_half_floor(
            1,
            mean_difference,
            (ensemble,),
            repeats=100,
            rng=np.random.default_rng(2),
            block_lengths=40,
        )
        assert blocked.mean() > 3.0 * iid.mean()

    def test_dissimilarity_routes_its_block_plan_into_the_floor(self):
        rng = np.random.default_rng(88)
        reference = ou(1000, 6, 20.0, rng)
        other = ou(1000, 6, 20.0, rng)
        common = dict(
            x_num=30,
            s_num=5,
            n_permutations=10,
            sample_size=1000,
            random_state=5,
        )
        iid = dissimilarity(
            reference, other, -5, 5, block_permutation=False, **common
        )
        blocked = dissimilarity(
            reference,
            other,
            -5,
            5,
            correlation_time_frames=20.0,
            **common,
        )

        assert blocked.noise_floor > 1.5 * iid.noise_floor
        assert blocked.noise_floor_assessable


class TestTheFalsePositiveRateIsFixed:
    """The measurement this whole module exists for."""

    @staticmethod
    def _rate(tau, block, seeds=8, frames=2000):
        """Returns (rate, withheld). The second value matters: a rate of zero
        because no p-value was reported is not a rate of zero."""
        hits = total = withheld = 0
        for seed in range(seeds):
            rng = np.random.default_rng(400 + seed * 13 + int(tau))
            a, b = ou(frames, 6, tau, rng), ou(frames, 6, tau, rng)
            result = dissimilarity(
                a, b, -5, 5, x_num=50, s_num=2, sample_size=frames,
                n_permutations=100, random_state=seed, block_permutation=block,
            )
            hits += result.n_significant
            total += 6
            withheld += int(result.p_values_withheld)
        return hits / total, withheld

    @pytest.mark.parametrize("tau", [5.0, 20.0])
    def test_block_permutation_restores_calibration(self, tau):
        """Frame permutation calls almost everything different; block
        permutation sits near the nominal 5%."""
        frame_rate, _ = self._rate(tau, block=False)
        block_rate, withheld = self._rate(tau, block=None)
        assert frame_rate > 0.5, "the failure this fixes should be reproduced"
        # A rate of zero because the p-value was withheld would pass a bare
        # threshold while proving nothing, so the absence of withholding is
        # asserted separately.
        assert withheld == 0, "2000 frames at this correlation time should be testable"
        assert block_rate < 0.15

    def test_default_subsampling_preserves_the_block_null(self):
        """The default thousand-frame path must preserve temporal order.

        The original calibration tests set ``sample_size=frames`` and never
        entered the branch used by a trajectory longer than 1000 frames. The
        old branch shuffled the selected rows and restored the 94% false-call
        failure that block permutation was meant to remove.
        """
        hits = total = withheld = 0
        for seed in range(8):
            rng = np.random.default_rng(900 + seed)
            a = ou(2000, 6, 20.0, rng)
            b = ou(2000, 6, 20.0, rng)
            result = dissimilarity(
                a, b, -5, 5, x_num=25, s_num=2,
                n_permutations=50, random_state=seed,
                correlation_time_frames=20.0,
            )
            hits += result.n_significant
            total += 6
            withheld += int(result.p_values_withheld)

        assert withheld == 0
        assert hits / total < 0.15

    def test_it_still_finds_a_real_difference(self):
        """A test made conservative enough to pass the null case is only
        useful if it still has power."""
        hits = total = 0
        for seed in range(8):
            rng = np.random.default_rng(50 + seed)
            a = ou(2000, 6, 20.0, rng)
            b = ou(2000, 6, 20.0, rng, mean=0.8)
            result = dissimilarity(
                a, b, -5, 6, x_num=50, s_num=2, sample_size=2000,
                n_permutations=100, random_state=seed,
            )
            hits += result.n_significant
            total += 6
        assert hits / total > 0.7

    def test_uncorrelated_data_is_unaffected(self):
        """Blocking must not cost anything where there is no correlation."""
        rng = np.random.default_rng(9)
        a, b = rng.normal(size=(1000, 5)), rng.normal(2.0, 1.0, (1000, 5))
        with_blocks = dissimilarity(
            a, b, -4, 6, x_num=50, s_num=2, random_state=0
        )
        without = dissimilarity(
            a, b, -4, 6, x_num=50, s_num=2, random_state=0, block_permutation=False
        )
        assert with_blocks.n_significant == without.n_significant
        assert with_blocks.correlation_time == pytest.approx(1.0, abs=0.5)


class TestRefusal:
    def test_block_count_is_recomputed_after_default_subsampling(self):
        rng = np.random.default_rng(14)
        a = rng.normal(size=(5000, 2))
        b = rng.normal(size=(5000, 2))
        with pytest.warns(
            UserWarning, match="3 independent blocks in 1000 sampled frames"
        ):
            result = dissimilarity(
                a, b, -5, 5, x_num=20, s_num=2, n_permutations=5,
                random_state=0, correlation_time_frames=150.0,
            )
        assert result.n_blocks == 3
        assert result.p_values_withheld
        assert not result.noise_floor_assessable
        assert result.resolved is None
        assert result.metadata["sampled_frames"] == [1000, 1000]
        assert result.metadata["sampling_strategy"] == "contiguous window"

    def test_a_short_trajectory_of_a_slow_system_withholds_the_p_value(self):
        """300 frames of a system with a correlation time of 120 frames.

        The estimate saturates here — it returns about 33, not 120, because
        the autocorrelation sum runs out of data before it runs out of
        correlation. What makes that safe is that the block length is never
        shortened to manufacture blocks: even the saturated estimate leaves
        only four, and four is refused.
        """
        rng = np.random.default_rng(10)
        a, b = ou(300, 5, 120.0, rng), ou(300, 5, 120.0, rng)
        with pytest.warns(UserWarning, match="independent blocks"):
            result = dissimilarity(
                a, b, -5, 5, x_num=50, s_num=2, sample_size=300,
                n_permutations=100, random_state=0,
            )
        assert not result.p_values_reported
        assert result.n_significant == 0
        assert result.noise_floor > 0          # the floor survives

    def test_the_result_says_why(self):
        rng = np.random.default_rng(11)
        a, b = ou(300, 5, 120.0, rng), ou(300, 5, 120.0, rng)
        with pytest.warns(UserWarning):
            result = dissimilarity(
                a, b, -5, 5, x_num=50, s_num=2, sample_size=300,
                n_permutations=100, random_state=0,
            )
        # The estimate saturates well below the true 120 frames, and there
        # are too few blocks either way.
        assert 20 < result.correlation_time < 120
        assert result.n_blocks < MINIMUM_BLOCKS

    def test_a_supplied_correlation_time_is_used(self):
        rng = np.random.default_rng(12)
        a, b = rng.normal(size=(2000, 4)), rng.normal(size=(2000, 4))
        result = dissimilarity(
            a, b, -4, 4, x_num=40, s_num=2, sample_size=2000,
            n_permutations=50, random_state=0, correlation_time_frames=25.0,
        )
        assert result.correlation_time == 25.0
        assert result.metadata["block_length"] > 1


    def test_too_few_blocks_is_caught_separately(self):
        """The other way to have too little: a long enough trajectory in
        correlation times, but chopped into too few blocks to permute."""
        rng = np.random.default_rng(13)
        a, b = ou(200, 4, 3.0, rng), ou(200, 4, 3.0, rng)
        result = dissimilarity(
            a, b, -5, 5, x_num=40, s_num=2, sample_size=200,
            n_permutations=50, random_state=0, correlation_time_frames=30.0,
        )
        assert not result.p_values_reported


class TestSaturationIsCaughtByAPlateauNotARatio:
    """A ratio test puts the saturated estimate in its own denominator.

    The correlation-time estimator saturates on a short series: the sum runs
    out of data before it runs out of correlation, and returns a plausible
    number in the right units that is smaller than the truth.

    The obvious guard is to require `n / tau_hat` above some threshold. It does
    not work, and it fails in the worst direction. The worse the saturation,
    the smaller `tau_hat`, the *larger* the ratio, and the healthier the series
    looks. On prefixes of a ubiquitin trajectory whose settled value is 45
    frames, a 250-frame prefix returned 5 and the ratio read 50 -- above any
    sane threshold -- while the true ratio was 5.6.

    The plateau test asks the only question the data can answer: estimate it
    again on less data and see whether the answer moved. It fits a slope of
    log tau against log n across four prefixes rather than taking a ratio
    between two, because two points cannot tell a dip from a plateau -- on
    prefixes of the real trajectory the sequence ran 5, 17, 21, 19, 30, 45 and
    the single dip from 21 to 19 made a two-point ratio report a plateau in the
    middle of a clear climb.
    """

    @staticmethod
    def _ou(n, tau_exp, seed=0, features=40):
        import numpy as np

        rng = np.random.default_rng(seed)
        phi = np.exp(-1.0 / tau_exp)
        x = np.zeros((n, features))
        for i in range(1, n):
            x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * rng.normal(size=features)
        return x

    def test_a_short_series_is_flagged(self):
        from prothon.sampling.correlation import correlation_time_estimate

        estimate = correlation_time_estimate(self._ou(500, 45.0))
        assert not estimate.converged
        assert estimate.slope > 0.15

    def test_a_long_series_is_not(self):
        from prothon.sampling.correlation import correlation_time_estimate

        estimate = correlation_time_estimate(self._ou(20000, 45.0))
        assert estimate.converged
        assert estimate.slope <= 0.15

    def test_the_estimate_is_a_lower_bound_when_flagged(self):
        """Not merely uncertain: wrong in a known direction."""
        import numpy as np

        from prothon.sampling.correlation import correlation_time_estimate

        short = correlation_time_estimate(self._ou(500, 45.0))
        long = correlation_time_estimate(self._ou(20000, 45.0))
        assert short.tau < long.tau
        # And the settled value is the integrated time of the process, which
        # for AR(1) is about twice the relaxation time it was built from.
        phi = np.exp(-1.0 / 45.0)
        assert long.tau > 0.5 * (1 + phi) / (1 - phi)

    def test_a_series_too_short_to_halve_claims_nothing(self):
        """The one answer that is certainly unearned is `converged`."""
        from prothon.sampling.correlation import correlation_time_estimate

        estimate = correlation_time_estimate(self._ou(20, 5.0))
        assert not estimate.converged

    def test_the_prefix_estimates_are_reported(self):
        """A verdict the caller cannot check is a verdict taken on trust."""
        from prothon.sampling.correlation import correlation_time_estimate

        estimate = correlation_time_estimate(self._ou(2000, 45.0))
        assert len(estimate.prefix_taus) >= 3
        assert 2000 in estimate.prefix_taus

    def test_the_flag_reaches_the_comparison_result(self):
        import warnings

        from prothon.compare.dissimilarity import dissimilarity

        a, b = self._ou(400, 45.0, seed=0), self._ou(400, 45.0, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = dissimilarity(
                a, b, -6.0, 6.0, s_num=3, n_permutations=20,
                sample_size=400, random_state=0,
            )
        assert not result.correlation_time_converged
        assert result.to_dict()["correlation_time_converged"] is False


    def test_an_uncorrelated_series_settles_at_one(self):
        """The other end, where the answer is known exactly."""
        import numpy as np

        from prothon.sampling.correlation import correlation_time_estimate

        rng = np.random.default_rng(0)
        estimate = correlation_time_estimate(rng.normal(size=(4000, 40)))
        assert estimate.converged
        assert estimate.tau < 1.5

    def test_a_dip_in_one_prefix_does_not_read_as_a_plateau(self):
        """The failure that retired the two-point ratio.

        A sequence that climbs overall but dips once must still be flagged.
        Reproduced from the real trajectory: 5, 17, 21, 19, 30, 45, where the
        21-to-19 step made a ratio of the last two points report 0.90.
        """
        import numpy as np

        from prothon.sampling.correlation import PLATEAU_SLOPE

        lengths = np.array([250, 500, 1000, 2000, 2500, 5000], dtype=float)
        taus = np.array([5, 17, 21, 19, 30, 45], dtype=float)

        two_point = taus[3] / taus[2]
        assert two_point < 1.0, "the dip is what made the old test pass"

        slope = float(np.polyfit(np.log(lengths), np.log(taus), 1)[0])
        assert slope > PLATEAU_SLOPE, "the slope must still catch it"


class TestTheWarningFiresWhereItMatters:
    """A warning nobody reads is worse than no warning.

    The convergence check trips easily when the correlation time hovers near
    one: the estimate is noisy, and the log-slope of a noisy near-constant is
    meaningless. Left unguarded it fired on every small test fixture, which is
    how a warning becomes something to filter out.
    """

    @staticmethod
    def _ou(n, tau_exp, seed=0, features=30):
        import numpy as np

        rng = np.random.default_rng(seed)
        phi = np.exp(-1.0 / tau_exp)
        x = np.zeros((n, features))
        for i in range(1, n):
            x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * rng.normal(size=features)
        return x

    def _warnings_from(self, a, b, n):
        import warnings

        from prothon.compare.dissimilarity import dissimilarity

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dissimilarity(
                a, b, -6.0, 6.0, s_num=3, n_permutations=10,
                sample_size=n, random_state=0,
            )
        return [str(w.message) for w in caught]

    def test_silent_when_the_correction_is_immaterial(self):
        """An essentially uncorrelated series says nothing about convergence."""
        import numpy as np

        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(300, 20)), rng.normal(size=(300, 20))
        messages = self._warnings_from(a, b, 300)
        assert not any("correlation time" in m for m in messages)

    def test_fires_when_the_correlation_is_real(self):
        a, b = self._ou(2000, 45.0, seed=0), self._ou(2000, 45.0, seed=1)
        messages = self._warnings_from(a, b, 2000)
        assert any("still rising" in m for m in messages)

    def test_unverified_is_not_the_same_claim_as_rising(self):
        """Too short to fit a trend is not the same as a trend that was found."""
        import numpy as np

        from prothon.sampling.correlation import correlation_time_estimate

        estimate = correlation_time_estimate(self._ou(200, 45.0))
        assert not estimate.converged
        assert not np.isfinite(estimate.slope), (
            "an unfittable trend must be distinguishable from a rising one"
        )
