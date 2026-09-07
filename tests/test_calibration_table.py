"""The calibrated threshold, and how far it does not travel.

The grid delivers its rate on the cells it was measured at. Twenty per cent
away in sample size it does not. Both halves are held here, because a
correction trusted outside its evidence produces numbers that look calibrated
and are not, which is the failure the whole exercise exists to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestTheLookup:
    def test_a_measured_cell_returns_its_measured_threshold(self):
        from prothon.sampling.calibration import CALIBRATION, calibrated_threshold

        for (sample_size, tau), expected in CALIBRATION.items():
            threshold, basis = calibrated_threshold(tau, sample_size=sample_size)
            assert threshold == pytest.approx(expected)
            assert basis == "measured"

    def test_far_from_every_cell_it_is_conservative_and_says_so(self):
        from prothon.sampling.calibration import (
            MOST_CONSERVATIVE,
            calibrated_threshold,
        )

        for tau, sample_size in ((10.0, 100_000), (300.0, 1000), (0.01, 1000)):
            threshold, basis = calibrated_threshold(tau, sample_size=sample_size)
            assert basis == "conservative"
            assert threshold == pytest.approx(MOST_CONSERVATIVE)

    def test_without_a_sample_size_it_will_not_guess(self):
        """Guessing one is how the single-axis version under-corrected."""
        from prothon.sampling.calibration import (
            MOST_CONSERVATIVE,
            calibrated_threshold,
        )

        threshold, basis = calibrated_threshold(10.0)
        assert basis == "conservative"
        assert threshold == pytest.approx(MOST_CONSERVATIVE)

    def test_a_different_alpha_is_refused_rather_than_scaled(self):
        from prothon.sampling.calibration import calibrated_threshold

        threshold, basis = calibrated_threshold(10.0, alpha=0.01, sample_size=1000)
        assert threshold == pytest.approx(0.01)
        assert basis == "uncorrected"

    def test_the_conservative_threshold_is_the_strictest_measured(self):
        """Erring towards calling less is the safe direction off-grid."""
        from prothon.sampling.calibration import CALIBRATION, MOST_CONSERVATIVE

        assert MOST_CONSERVATIVE == min(CALIBRATION.values())


class TestTheGridIsHonestAboutItsCoverage:
    def test_degenerate_cells_are_absent_not_filled(self):
        """Three cells had under half their studies testable.

        The survivors there are the studies whose correlation time happened to
        be estimated low, which is not a random sample of the null. Fitting a
        threshold to them gave 0.0050 and a corrected rate of 0.0%: a line so
        strict it never fires.
        """
        from prothon.sampling.calibration import CALIBRATION

        for absent in ((500, 25.0), (500, 50.0), (1000, 50.0)):
            assert absent not in CALIBRATION

    def test_every_threshold_is_stricter_than_the_nominal_alpha(self):
        """The p-values run hot, so every correction can only tighten."""
        from prothon.sampling.calibration import (
            CALIBRATED_ALPHA,
            CALIBRATION,
        )

        assert all(t < CALIBRATED_ALPHA for t in CALIBRATION.values())

    def test_the_module_states_that_it_does_not_transfer(self):
        """The negative result belongs where the threshold is supplied."""
        from prothon.sampling import calibration

        text = (calibration.__doc__ or "").lower()
        assert "does not transfer" in text
        assert "11.0%" in text, "the off-grid measurement must be stated"

    def test_calibration_is_off_by_default(self):
        import inspect

        from prothon.compare.dissimilarity import dissimilarity

        assert (
            inspect.signature(dissimilarity).parameters["calibrated"].default is False
        ), "switching this on silently would change what every result means"


class TestTheFlagTightensAndNeverLoosens:
    def test_a_calibrated_run_calls_no_more_than_a_nominal_one(self):
        import warnings

        from prothon.compare.dissimilarity import dissimilarity

        rng = np.random.default_rng(0)
        phi = np.exp(-1.0 / 10.0)

        def draw():
            x = np.zeros((800, 6))
            for t in range(1, 800):
                x[t] = phi * x[t - 1] + np.sqrt(1 - phi**2) * rng.normal(size=6)
            return x

        a, b = draw(), draw() + 0.6
        span = max(abs(a).max(), abs(b).max()) * 1.1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            common = dict(
                s_num=2, x_num=60, sample_size=400, n_permutations=100,
                alpha=0.05, random_state=0,
            )
            loose = dissimilarity(a, b, -span, span, calibrated=False, **common)
            strict = dissimilarity(a, b, -span, span, calibrated=True, **common)
        assert int(strict.n_significant) <= int(loose.n_significant)
