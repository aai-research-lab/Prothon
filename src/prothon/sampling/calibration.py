"""The threshold that delivers the error rate the caller asked for.

A permutation p-value from this software runs hot: asked for five per cent, it
calls something in ten to sixteen per cent of null studies, across a fiftyfold
range of correlation times. Six mechanisms were examined and swept and none
accounts for it. Section 0b of the roadmap lists them so they are not retried.

What follows is the other thing an instrument maker does. The error was
measured against a standard and is corrected for. The standard is exact: two
ensembles drawn from one distribution differ in no way, so every call on them
is wrong by construction and the correct rate is whatever was asked for.

The measurement is `scripts/calibrate_threshold.py`, at two thousand null
studies per correlation time, with the threshold fitted on one set and verified
on a disjoint second set:

    tau    uncorrected    corrected, on nulls it had not seen
      1        10.3%                4.3%
      5        13.3%                3.8%
     10        13.1%                5.6%
     25        12.9%                5.1%
     50        15.9%                5.3%

Every row lands within about one point of five per cent, and the standard error
at two thousand replicates is about half a point.

**What this is and is not.** It is a calibration: the rate is delivered because
it was measured and corrected, not because the p-values were shown to be
correct. They are not; the excess is real and its cause is unresolved. Any
paper using this should say so, and this docstring exists so that nobody has to
reconstruct it later.

**Where it does not apply, and this is the important part.** The table was
measured at one sample size: four thousand frames subsampled to two thousand.
It is indexed by correlation time alone, and that is not enough. Applied to a
comparison of twelve hundred frames subsampled to six hundred, the same
thresholds take the study rate from 16.9% to 13.6% rather than to 5%.

The correction therefore depends on the sample size as well as the correlation
time, and a table over one of them under-corrects on anything smaller than what
it was measured on. **Silently**, which is the worst property a correction can
have: the number looks calibrated and is not.

Until the table is measured over both, `calibrated=True` is off by default and
documented as provisional. A two-dimensional table is the obvious fix and it is
a longer measurement, not a harder one.

Outside the measured range of correlation times the correction is an
extrapolation, and this module refuses to extrapolate far: beyond the endpoints
it holds the nearest measured value and reports the basis, so a comparison
where nothing was measured is labelled rather than silently corrected.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CALIBRATED_AT",
    "CALIBRATION",
    "CALIBRATED_ALPHA",
    "calibrated_threshold",
]

#: The alpha the table was measured for. A different alpha needs a different
#: table, which is why this is checked rather than scaled: the relationship
#: between the nominal and the delivered rate is not known to be linear, and
#: assuming it is would be inventing numbers.
CALIBRATED_ALPHA = 0.05

#: Frames per ensemble, and the subsample, the table was measured at. Recorded
#: because the correction depends on both and the table indexes only tau: away
#: from this size it under-corrects.
CALIBRATED_AT = {"frames": 4000, "sample_size": 2000}

#: Correlation time to the threshold that delivers `CALIBRATED_ALPHA` *at
#: `CALIBRATED_AT`*. Measured, not chosen. See `docs/thresholds.md` for the run
#: that produced it and the rate each row achieved on nulls it had not seen.
CALIBRATION: dict[float, float] = {
    1.0: 0.0250,
    5.0: 0.0150,
    10.0: 0.0200,
    25.0: 0.0200,
    50.0: 0.0150,
}


def calibrated_threshold(
    tau: float, alpha: float = CALIBRATED_ALPHA
) -> tuple[float, str]:
    """The threshold to use, and how it was arrived at.

    Returns ``(threshold, basis)``. ``basis`` is one of:

    ``"measured"``
        The correlation time falls inside the calibrated range, so the
        threshold is interpolated between two measured points.
    ``"extrapolated"``
        Outside the range. The nearest measured threshold is held, rather than
        extended along a trend that was never measured. Reported so a caller
        can see that this comparison sits outside the evidence.
    ``"uncorrected"``
        A different alpha from the one the table was measured at. The nominal
        threshold is returned unchanged, because scaling a measured correction
        to an alpha it was not measured at would be inventing a number.

    Linear interpolation between neighbouring points, because the measured
    thresholds vary smoothly and five points do not justify anything more
    elaborate.
    """
    if not np.isclose(alpha, CALIBRATED_ALPHA):
        return float(alpha), "uncorrected"

    taus = np.array(sorted(CALIBRATION))
    thresholds = np.array([CALIBRATION[t] for t in taus])

    if tau <= taus[0]:
        return float(thresholds[0]), (
            "measured" if np.isclose(tau, taus[0]) else "extrapolated"
        )
    if tau >= taus[-1]:
        return float(thresholds[-1]), (
            "measured" if np.isclose(tau, taus[-1]) else "extrapolated"
        )
    return float(np.interp(tau, taus, thresholds)), "measured"
