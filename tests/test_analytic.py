"""Estimators checked against closed forms, not against themselves.

Most of the test suite asserts that a function behaves sensibly: bounded where
it should be bounded, larger for ensembles that differ more, calibrated under a
null. That catches a broken implementation but not a subtly wrong one, because
the expectations are written by the same person who wrote the code.

Where a quantity has a known value, that is a stronger test. This file collects
the cases where one exists.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import vonmises

from prothon.compare.dissimilarity import (
    benjamini_hochberg,
    effective_sample_size,
    estimate_pdf,
)
from prothon.compare.distance import feature_distance
from prothon.compare.joint import maximum_mean_discrepancy


class TestDensityEstimation:
    def test_gaussian_kde_recovers_a_normal_density(self):
        rng = np.random.default_rng(0)
        grid, estimated = estimate_pdf(rng.normal(0, 1, 20000), -5, 5, 400)
        exact = np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi)
        # Integrated absolute error, which is twice the total variation
        # distance between the estimate and the truth.
        error = np.trapezoid(np.abs(estimated - exact), grid)
        assert error < 0.05

    def test_von_mises_kernel_recovers_a_von_mises_density(self):
        """The circular estimator against the distribution it is named for."""
        kappa, loc = 4.0, 0.7
        rng = np.random.default_rng(1)
        sample = rng.vonmises(loc, kappa, 20000)
        grid, estimated = estimate_pdf(sample, 0, 0, 400, circular=True)
        exact = vonmises.pdf(grid, kappa, loc=loc)
        error = np.trapezoid(np.abs(estimated - exact), grid)
        assert error < 0.08

    def test_von_mises_kernel_recovers_a_density_at_the_wraparound(self):
        """The same test with the population centred on the cut, which is where
        a linear estimator fails."""
        kappa, loc = 6.0, np.pi
        rng = np.random.default_rng(2)
        sample = rng.vonmises(loc, kappa, 20000)
        grid, estimated = estimate_pdf(sample, 0, 0, 400, circular=True)
        exact = vonmises.pdf(grid, kappa, loc=loc)
        assert np.trapezoid(np.abs(estimated - exact), grid) < 0.08

    @pytest.mark.parametrize("circular", [False, True])
    def test_densities_integrate_to_one(self, circular):
        rng = np.random.default_rng(3)
        sample = rng.vonmises(0.2, 3, 5000) if circular else rng.normal(size=5000)
        grid, density = estimate_pdf(sample, -6, 6, 500, circular=circular)
        assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=0.02)


class TestWassersteinClosedForms:
    def test_between_normals_of_equal_variance(self):
        """W1 between N(a, s^2) and N(b, s^2) is exactly |a - b|."""
        rng = np.random.default_rng(4)
        for separation in (0.5, 2.0, 5.0):
            a = rng.normal(0.0, 1.0, 20000)
            b = rng.normal(separation, 1.0, 20000)
            assert feature_distance(a, b, "wasserstein") == pytest.approx(
                separation, abs=0.03
            )

    def test_between_uniforms_of_equal_width(self):
        """A shift of a uniform distribution moves every quantile by the shift,
        so W1 is the shift."""
        rng = np.random.default_rng(5)
        for shift in (0.25, 1.0):
            a = rng.uniform(0, 1, 20000)
            b = rng.uniform(shift, 1 + shift, 20000)
            assert feature_distance(a, b, "wasserstein") == pytest.approx(
                shift, abs=0.02
            )

    def test_between_nested_uniforms(self):
        """W1 between U(0,1) and U(0,2) is the mean of |F^-1 - G^-1| over
        quantiles: the integral of |t - 2t| dt from 0 to 1, which is 1/2."""
        rng = np.random.default_rng(6)
        a, b = rng.uniform(0, 1, 40000), rng.uniform(0, 2, 40000)
        assert feature_distance(a, b, "wasserstein") == pytest.approx(0.5, abs=0.02)

    def test_circular_distance_is_the_short_way_round(self):
        """Two concentrated populations separated by an angle: the circular
        distance must be the angle itself, not the long way round."""
        rng = np.random.default_rng(7)
        for centre in (2.6, 3.0):
            separation = 2 * np.pi - 2 * centre
            a = rng.vonmises(centre, 400, 20000)
            b = rng.vonmises(-centre, 400, 20000)
            assert feature_distance(
                a, b, "wasserstein", circular=True
            ) == pytest.approx(separation, abs=0.02)


class TestSupremumClosedForms:
    def test_ks_between_shifted_uniforms(self):
        """For U(0,1) against U(s, 1+s) with 0 < s < 1, the largest gap between
        the two cumulative distributions is exactly s."""
        rng = np.random.default_rng(8)
        for shift in (0.2, 0.5):
            a = rng.uniform(0, 1, 40000)
            b = rng.uniform(shift, 1 + shift, 40000)
            assert feature_distance(a, b, "ks") == pytest.approx(shift, abs=0.01)

    def test_ks_is_zero_for_the_same_distribution(self):
        rng = np.random.default_rng(9)
        a, b = rng.uniform(size=40000), rng.uniform(size=40000)
        assert feature_distance(a, b, "ks") < 0.02


class TestMaximumMeanDiscrepancy:
    @staticmethod
    def _analytic(delta, sigma, gamma, d):
        """MMD^2 between two isotropic Gaussians under a Gaussian kernel.

        For X ~ N(0, s^2 I) and Y ~ N(delta e_1, s^2 I) with kernel
        exp(-|x-y|^2 / 2g^2), each expectation of the kernel is available in
        closed form and the discrepancy reduces to

            2 c (1 - exp(-delta^2 / (2 (g^2 + 2 s^2)))),
            c = (g^2 / (g^2 + 2 s^2))^(d/2).
        """
        c = (gamma**2 / (gamma**2 + 2 * sigma**2)) ** (d / 2)
        return 2 * c * (1 - np.exp(-(delta**2) / (2 * (gamma**2 + 2 * sigma**2))))

    @pytest.mark.parametrize(
        "dim,delta,gamma", [(2, 1.0, 2.0), (2, 2.0, 2.0), (4, 1.0, 3.0), (8, 1.5, 4.0)]
    )
    def test_matches_the_closed_form_for_gaussians(self, dim, delta, gamma):
        rng = np.random.default_rng(10)
        a = rng.normal(0.0, 1.0, (4000, dim))
        b = rng.normal(0.0, 1.0, (4000, dim))
        b[:, 0] += delta
        measured = maximum_mean_discrepancy(
            a, b, bandwidth=gamma, standardise=False,
            n_permutations=1, sample_size=4000, random_state=0,
        ).statistic
        expected = self._analytic(delta, 1.0, gamma, dim)
        assert measured == pytest.approx(expected, rel=0.10)

    def test_is_near_zero_for_identical_distributions(self):
        rng = np.random.default_rng(11)
        a, b = rng.normal(size=(3000, 4)), rng.normal(size=(3000, 4))
        measured = maximum_mean_discrepancy(
            a, b, bandwidth=2.0, standardise=False,
            n_permutations=1, sample_size=3000, random_state=0,
        ).statistic
        assert abs(measured) < 0.005

    def test_memory_does_not_scale_with_the_number_of_residues(self):
        """The kernel is built from a matrix product rather than a broadcast.

        The obvious expression allocates an (n, n, d) array before reducing it:
        at the default thousand conformations a side that is 2.4 GB for a
        76-residue protein and 9.6 GB for a 300-residue one, so the process is
        killed rather than slowed -- and only ever on real proteins, since a
        fixture with a dozen residues never approaches it.
        """
        import tracemalloc

        peaks = []
        for n_features in (20, 300):
            rng = np.random.default_rng(12)
            a = rng.normal(size=(400, n_features))
            b = rng.normal(0.4, 1.0, (400, n_features))
            tracemalloc.start()
            maximum_mean_discrepancy(a, b, n_permutations=5, random_state=0)
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()

        # Fifteen times the features must not cost anything like fifteen times
        # the memory; the kernel matrix is the same size either way.
        assert peaks[1] < 3 * peaks[0]


class TestExactQuantities:
    def test_effective_sample_size_of_equal_weights_is_the_count(self):
        for n in (10, 1000):
            assert effective_sample_size(np.full(n, 1.0 / n)) == pytest.approx(n)

    def test_effective_sample_size_of_a_known_case(self):
        """One conformer at half the probability and the rest sharing the other
        half gives 1 / (1/4 + 1/(4(n-1))), which tends to 4 for large n."""
        n = 1000
        w = np.full(n, 0.5 / (n - 1))
        w[0] = 0.5
        expected = 1.0 / (0.25 + 0.25 / (n - 1))
        assert effective_sample_size(w) == pytest.approx(expected, rel=1e-9)

    def test_benjamini_hochberg_matches_scipy(self):
        scipy_fdr = pytest.importorskip(
            "scipy.stats", reason="needs scipy"
        ).false_discovery_control
        rng = np.random.default_rng(13)
        for size in (10, 250):
            p = rng.uniform(size=size)
            np.testing.assert_allclose(
                benjamini_hochberg(p), scipy_fdr(p, method="bh"), rtol=1e-12
            )

    def test_benjamini_hochberg_on_a_worked_example(self):
        """Four p-values, adjusted by hand: p_(i) * n / i, made monotone from
        the largest downwards."""
        p = np.array([0.01, 0.02, 0.03, 0.04])
        np.testing.assert_allclose(
            benjamini_hochberg(p), [0.04, 0.04, 0.04, 0.04], rtol=1e-12
        )
