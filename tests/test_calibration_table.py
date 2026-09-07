"""The calibrated threshold, and the limit of what it is calibrated for.

The table delivers five per cent at the size it was measured at and
under-corrects below it. These tests hold both halves, because a correction
that is trusted outside its evidence is worse than none: the number looks
calibrated and is not.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestTheLookup:
    def test_a_measured_point_returns_its_measured_threshold(self):
        from prothon.sampling.calibration import CALIBRATION, calibrated_threshold

        for tau, expected in CALIBRATION.items():
            threshold, basis = calibrated_threshold(tau)
            assert threshold == pytest.approx(expected)
            assert basis == "measured"

    def test_between_two_points_it_interpolates(self):
        from prothon.sampling.calibration import calibrated_threshold

        threshold, basis = calibrated_threshold(7.5)
        assert basis == "measured"
        # Halfway between the thresholds at tau = 5 and tau = 10.
        assert threshold == pytest.approx(0.0175)

    def test_beyond_the_range_it_holds_and_says_so(self):
        """Holding the endpoint, rather than extending a trend nobody measured."""
        from prothon.sampling.calibration import CALIBRATION, calibrated_threshold

        low, high = min(CALIBRATION), max(CALIBRATION)
        for tau, edge in ((low / 10, low), (high * 10, high)):
            threshold, basis = calibrated_threshold(tau)
            assert basis == "extrapolated"
            assert threshold == pytest.approx(CALIBRATION[edge])

    def test_a_different_alpha_is_not_corrected(self):
        """Scaling a measured correction to an alpha it was not measured at
        would be inventing a number."""
        from prothon.sampling.calibration import calibrated_threshold

        threshold, basis = calibrated_threshold(10.0, alpha=0.01)
        assert threshold == pytest.approx(0.01)
        assert basis == "uncorrected"


class TestTheTableRecordsWhatItWasMeasuredAt:
    """The correction depends on the sample size and the table does not index it.

    Applied at twelve hundred frames subsampled to six hundred, where the table
    was measured at four thousand subsampled to two thousand, the study rate
    goes from 16.9% to 13.6% rather than to 5%. That is a real limit and it is
    invisible from the returned number, so the size is recorded beside the
    table and `calibrated=True` stays off by default until the table covers
    both dimensions.
    """

    def test_the_measurement_size_is_recorded(self):
        from prothon.sampling.calibration import CALIBRATED_AT

        assert CALIBRATED_AT["frames"] == 4000
        assert CALIBRATED_AT["sample_size"] == 2000

    def test_the_limitation_is_documented_where_it_is_used(self):
        from prothon.sampling import calibration

        text = calibration.__doc__ or ""
        assert "sample size" in text, (
            "the sample-size dependence must be stated in the module that "
            "supplies the threshold, not only in the roadmap"
        )

    def test_calibration_is_off_by_default(self):
        import inspect

        from prothon.compare.dissimilarity import dissimilarity

        assert inspect.signature(dissimilarity).parameters["calibrated"].default is (
            False
        ), "switching this on silently would change what every result means"


class TestTheFlagChangesTheThreshold:
    def test_a_calibrated_run_uses_a_stricter_line(self):
        """At tau = 10 the measured threshold is 0.02, below the nominal 0.05,
        so a calibrated run can only call fewer features, never more."""
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
            loose = dissimilarity(
                a, b, -span, span, s_num=2, x_num=60, sample_size=400,
                n_permutations=100, alpha=0.05, random_state=0, calibrated=False,
            )
            strict = dissimilarity(
                a, b, -span, span, s_num=2, x_num=60, sample_size=400,
                n_permutations=100, alpha=0.05, random_state=0, calibrated=True,
            )
        assert int(strict.n_significant) <= int(loose.n_significant)
