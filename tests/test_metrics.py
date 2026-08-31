"""Tests for the per-feature metrics.

The two that matter are the circular ones. A linear metric applied to a torsion
gives a plausible number that is wrong by a large factor, and nothing about the
output says so -- the same failure the linear kernel density estimate had in
version 2.0.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from prothon.compare.dissimilarity import dissimilarity, jsd_local
from prothon.compare.distance import (
    METRICS,
    describe_metric,
    feature_distance,
    resolve_metric,
)


def rotate(values, shift):
    """Move a circular sample round by `shift`, staying in [-pi, pi)."""
    return np.mod(values + shift + np.pi, 2 * np.pi) - np.pi


class TestRegistry:
    def test_every_metric_is_callable(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=500), rng.normal(1, 1, 500)
        for name in METRICS:
            value = feature_distance(a, b, name, -4, 5, 80)
            assert np.isfinite(value) and value >= 0

    def test_bounded_metrics_stay_in_the_unit_interval(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(0, 1, 800), rng.normal(20, 1, 800)
        for name, spec in METRICS.items():
            if spec.bounded:
                assert feature_distance(a, b, name, -5, 25, 120) <= 1.0

    def test_unknown_metric_suggests_a_neighbour(self):
        with pytest.raises(ValueError, match="Did you mean jsd"):
            resolve_metric("jsdd")

    def test_description_states_the_scale(self):
        assert "bounded" in describe_metric("jsd")
        assert "feature units" in describe_metric("wasserstein")


class TestWasserstein:
    def test_reports_in_the_features_own_units(self):
        """The reason to offer it. 'This residue gains 1.4 contacts' is a
        sentence about the protein; a Jensen-Shannon distance is not."""
        rng = np.random.default_rng(2)
        for separation in (1.4, 3.0, 6.5):
            a = rng.normal(0.0, 1.0, 6000)
            b = rng.normal(separation, 1.0, 6000)
            assert feature_distance(a, b, "wasserstein") == pytest.approx(
                separation, abs=0.08
            )

    def test_identical_samples_give_about_zero(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=4000), rng.normal(size=4000)
        assert feature_distance(a, b, "wasserstein") < 0.05

    def test_circular_measures_the_short_way_round(self):
        """A linear Wasserstein reports 4.4 radians where the true circular
        separation is 0.28 -- a factor of twenty-one, reported without
        complaint. This is the metric-layer version of the bug that the von
        Mises kernel fixed for density estimation."""
        rng = np.random.default_rng(4)
        a = rng.vonmises(3.0, 60, 4000)
        b = rng.vonmises(-3.0, 60, 4000)
        true_separation = 2 * np.pi - 6.0

        circular = feature_distance(a, b, "wasserstein", circular=True)
        linear = feature_distance(a, b, "wasserstein", circular=False)

        assert circular == pytest.approx(true_separation, abs=0.05)
        assert linear > 10 * circular

    def test_circular_is_invariant_to_rotation(self):
        rng = np.random.default_rng(5)
        a = rng.vonmises(0.5, 20, 3000)
        b = rng.vonmises(-1.0, 20, 3000)
        values = []
        for shift in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            values.append(
                feature_distance(
                    rotate(a, shift), rotate(b, shift), "wasserstein", circular=True
                )
            )
        assert max(values) - min(values) < 1e-9


class TestTheDistanceIsAlwaysDefined:
    """Both estimators floor their density, so no input produces an infinite
    Kullback-Leibler term and no fallback value has to be invented.

    The first attempt at this returned 1.0 whenever the distance came out
    non-finite. That is right for two genuinely disjoint distributions and
    wrong for a near-degenerate feature whose kernel has collapsed, and the two
    are indistinguishable at the point of decision. It raised the measured
    noise floor by two thirds on solvent accessibility, where buried residues
    are nearly constant, and took the significant-residue count to zero.
    """

    def test_disjoint_distributions_reach_one_by_arithmetic(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.05, (3000, 1))
        b = rng.normal(5, 0.05, (3000, 1))
        assert jsd_local(a, b, -1, 6, 100)[0] == pytest.approx(1.0, abs=1e-6)

    def test_two_halves_of_one_ensemble_stay_near_zero(self):
        """The case the fallback broke: a floor that fires here inflates the
        resolution limit and hides every real difference behind it."""
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (6000, 1))
        assert jsd_local(x[:3000], x[3000:], -4, 4, 100)[0] < 0.06

    def test_a_nearly_constant_feature_is_not_maximally_different(self):
        """Buried solvent accessibility is nearly constant, and a collapsed
        kernel must not read as two unrelated distributions."""
        rng = np.random.default_rng(2)
        a = rng.normal(0.001, 0.0005, (3000, 1))
        b = rng.normal(0.001, 0.0005, (3000, 1))
        assert jsd_local(a, b, a.min(), a.max(), 100)[0] < 0.2

    def test_no_density_contains_an_exact_zero(self):
        from prothon.compare.dissimilarity import estimate_pdf

        rng = np.random.default_rng(3)
        for circular in (False, True):
            _, density = estimate_pdf(
                rng.normal(0.9, 0.15, 3000), -np.pi, np.pi, 100, circular
            )
            assert (density > 0).all(), f"exact zero with circular={circular}"


class TestAConcentratedCircularFeatureIsNotReportedAsIdentical:
    """A tight torsion drove the von Mises kernel to underflow, which made the
    Jensen-Shannon distance infinite, which the caller reported as zero.

    The chain matters more than any link in it: a numerical underflow became a
    maximal distance became no distance, and the output was 0.0000 --- a value
    that looks like a measurement and reads as agreement.
    """

    @staticmethod
    def concentrated(seed, centre, spread=0.15, n=4000):
        return np.random.default_rng(seed).normal(centre, spread, (n, 1))

    def test_a_real_difference_is_not_reported_as_zero(self):
        a = self.concentrated(0, 0.9)
        b = self.concentrated(1, 1.0, 0.20)
        assert jsd_local(a, b, -np.pi, np.pi, 100, True)[0] > 0.1

    def test_circular_and_linear_agree_away_from_the_wrap(self):
        """A distribution nowhere near the wrap should give the same answer
        either way. Before the fix the circular value was 0.29 low."""
        a = self.concentrated(2, 0.9)
        b = self.concentrated(3, 1.0, 0.20)
        low, high = min(a.min(), b.min()), max(a.max(), b.max())
        linear = jsd_local(a, b, low, high, 100, False)[0]
        circular = jsd_local(a, b, -np.pi, np.pi, 100, True)[0]
        assert circular == pytest.approx(linear, abs=0.02)

    def test_identical_input_still_gives_zero(self):
        a = self.concentrated(4, 0.5)
        assert jsd_local(a, a.copy(), -np.pi, np.pi, 100, True)[0] == pytest.approx(0.0)

    def test_disjoint_support_gives_the_maximum_not_the_minimum(self):
        """Two torsions with no overlap are as different as two distributions
        can be. The answer is 1, not 0."""
        a = self.concentrated(5, -2.5, 0.05)
        b = self.concentrated(6, 2.5, 0.05)
        assert jsd_local(a, b, -np.pi, np.pi, 100, True)[0] > 0.99

    def test_a_density_never_contains_an_exact_zero(self):
        from prothon.compare.dissimilarity import estimate_pdf

        _, density = estimate_pdf(
            self.concentrated(7, 0.9).ravel(), -np.pi, np.pi, 100, True
        )
        assert (density > 0).all()
        assert np.isfinite(density).all()


class TestSupremum:
    def test_kuiper_is_rotation_invariant_and_ks_is_not(self):
        """KS asks for the largest gap between two cumulative distributions.
        On a circle that depends on where the circle was cut, which is not a
        property of the data: over 24 rotations of one interleaved pair, KS
        ranges from 0.25 to 0.50. Kuiper's statistic does not move."""
        rng = np.random.default_rng(6)
        a = np.concatenate([rng.vonmises(-1.6, 40, 2000), rng.vonmises(1.6, 40, 2000)])
        b = np.concatenate([rng.vonmises(0.0, 40, 2000), rng.vonmises(3.1, 40, 2000)])

        ks, kuiper = [], []
        for shift in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            x, y = rotate(a, shift), rotate(b, shift)
            ks.append(feature_distance(x, y, "ks", circular=False))
            kuiper.append(feature_distance(x, y, "ks", circular=True))

        assert max(kuiper) - min(kuiper) < 1e-9
        assert max(ks) - min(ks) > 0.1


class TestWeights:
    @pytest.mark.parametrize("metric", sorted(METRICS))
    def test_weights_change_the_answer(self, metric):
        rng = np.random.default_rng(7)
        x = np.concatenate([rng.normal(0, 0.5, 2000), rng.normal(4, 0.5, 2000)])
        y = rng.normal(0, 0.5, 4000)
        w = np.concatenate([np.full(2000, 0.9 / 2000), np.full(2000, 0.1 / 2000)])

        plain = feature_distance(x, y, metric, -3, 7, 120)
        weighted = feature_distance(x, y, metric, -3, 7, 120, weights_x=w)
        # Weighting away the mode that y lacks must bring them closer.
        assert weighted < plain

    def test_uniform_weights_match_no_weights(self):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=1500), rng.normal(0.7, 1, 1500)
        w = np.full(1500, 1.0 / 1500)
        for metric in METRICS:
            assert feature_distance(x, y, metric, -4, 5, 100) == pytest.approx(
                feature_distance(x, y, metric, -4, 5, 100, weights_x=w, weights_y=w),
                abs=1e-6,
            )


class TestThroughTheStatistics:
    def test_the_metric_reaches_the_estimator(self):
        """The test that was missing, and the bug it would have caught.

        `dissimilarity` accepted `metric=` and recorded it in the metadata
        while calling the estimator without it, so every comparison was
        Jensen-Shannon whatever was asked for. Every test passed: they checked
        the label rather than the number, and a null calibration run cannot
        notice, because all three metrics are correctly calibrated and so give
        the same rate. It showed up as three metrics agreeing to five decimal
        places over eight thousand features, which no two estimators do.
        """
        rng = np.random.default_rng(31)
        a = rng.normal(size=(400, 6))
        b = rng.normal(0.8, 1.0, (400, 6))

        raw = {}
        for metric in sorted(METRICS):
            raw[metric] = dissimilarity(
                a, b, -4, 5, x_num=60, s_num=2, metric=metric, random_state=3
            ).raw_local_dissimilarity

        for left, right in itertools.combinations(sorted(METRICS), 2):
            assert not np.allclose(raw[left], raw[right]), (
                f"{left} and {right} produced identical statistics; the metric "
                f"is not reaching the estimator"
            )

    def test_wasserstein_through_dissimilarity_is_in_feature_units(self):
        """A second, independent check on the same thing: the statistic that
        comes back must be the quantity the metric promises, not merely
        different from the others."""
        rng = np.random.default_rng(32)
        a = rng.normal(0.0, 1.0, (3000, 3))
        b = rng.normal(2.0, 1.0, (3000, 3))
        result = dissimilarity(
            a, b, -5, 7, x_num=80, s_num=2, metric="wasserstein",
            sample_size=3000, random_state=4,
        )
        # Two Gaussians of equal width two apart: W1 is the separation.
        assert result.raw_local_dissimilarity.mean() == pytest.approx(2.0, abs=0.1)

    @pytest.mark.parametrize("metric", sorted(METRICS))
    def test_every_metric_gets_its_own_noise_floor(self, metric):
        """A Wasserstein comparison needs a Wasserstein floor. Borrowing one
        from another metric would compare a distance in radians against a
        threshold in bounded units."""
        rng = np.random.default_rng(9)
        a = rng.normal(0, 1, (400, 5))
        b = rng.normal(2, 1, (400, 5))
        result = dissimilarity(
            a, b, -4, 6, x_num=50, s_num=3, metric=metric, random_state=0
        )
        assert result.noise_floor > 0
        assert result.resolved
        assert result.metadata["metric"] == metric

    @pytest.mark.parametrize("metric", sorted(METRICS))
    def test_false_positives_stay_controlled_under_every_metric(self, metric):
        """The calibration check, repeated for each metric. A new distance
        function that quietly broke the permutation null would show up here."""
        flagged = total = 0
        for seed in range(8):
            rng = np.random.default_rng(300 + seed)
            a, b = rng.normal(size=(300, 6)), rng.normal(size=(300, 6))
            result = dissimilarity(
                a, b, -4, 4, x_num=40, s_num=2, metric=metric, random_state=seed
            )
            flagged += result.n_significant
            total += 6
        assert flagged / total < 0.15

    def test_jsd_remains_the_default(self):
        rng = np.random.default_rng(10)
        a, b = rng.normal(size=(300, 4)), rng.normal(1.5, 1, (300, 4))
        np.testing.assert_allclose(
            jsd_local(a, b, -4, 6, 60),
            jsd_local(a, b, -4, 6, 60, metric="jsd"),
        )

    def test_unknown_metric_fails_before_the_permutation_loop(self):
        rng = np.random.default_rng(11)
        a = rng.normal(size=(200, 3))
        with pytest.raises(ValueError, match="Unknown metric"):
            dissimilarity(a, a, -4, 4, x_num=30, s_num=2, metric="nope")
