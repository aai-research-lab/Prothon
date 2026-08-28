"""Tests for correlation-time estimation and block permutation.

The measurement these exist to protect: with frames correlated in time, the
frame-permutation null called 99% of features different when nothing differed.
"""

from __future__ import annotations

import numpy as np
import pytest

from prothon.core.correlation import (
    MINIMUM_BLOCKS,
    block_labels,
    correlation_time,
    effective_frames,
    plan_blocks,
)
from prothon.core.dissimilarity import dissimilarity


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
