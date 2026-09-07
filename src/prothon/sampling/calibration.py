"""The threshold that delivers the error rate the caller asked for.

A permutation p-value from this software runs hot. Asked for five per cent, it
calls something in ten to sixteen per cent of null studies, across an eightfold
range of sample sizes and a fiftyfold range of correlation times. Six
mechanisms were examined and swept and none accounts for it; section 0b of the
roadmap lists them so they are not retried.

What follows is the other thing an instrument maker does. The error was
measured against a standard and is corrected for. The standard is exact: two
ensembles drawn from one distribution differ in no way, so every call on them
is wrong by construction and the correct rate is whatever was asked for.

**The measurement.** `scripts/calibrate_threshold.py`, two thousand null
studies per cell, threshold fitted on one set and verified on a disjoint
second. Twenty cells attempted, seventeen usable, and in every usable one the
corrected rate on unseen nulls falls between 3.2% and 5.9%. `docs/thresholds.md`
carries the run.

**Three cells are refused rather than filled.** Where fewer than half the null
studies have enough blocks to report a p-value at all, the survivors are those
whose correlation time happened to be estimated low, which is not a random
sample of the null. Fitting a threshold to them gave 0.0050 and a corrected
rate of 0.0%: a line so strict it never fires. Those cells are absent here.

The boundary is the useful part. **The correction holds wherever the data
supports a test at all, and where it does not the software already refuses**
for its own reasons. The two boundaries coincide, which is what makes this a
calibration rather than a patch over a hole.

**Why two dimensions.** An earlier table was measured at one sample size and
indexed by correlation time alone. Applied at twelve hundred frames subsampled
to six hundred it took the study rate from 16.9% to 13.6% rather than to 5%,
because it supplied 0.0200 where 0.0150 was needed.

**It does not transfer as far as a lookup table needs to, and that is the
finding.** Each cell delivers its rate on its own verification set. Twenty per
cent away in sample size it does not: at a subsample of six hundred against a
cell measured at five hundred, both at a correlation time of ten, the same
threshold gives 11.0% where the cell reported 5.1%. Raising the permutation
count to match the measurement made it worse, not better, so that is not the
difference either.

    configuration                       study rate
    uncorrected                             16.9%
    single-axis table (tau only)            13.6%
    this table, 150 permutations             9.3%
    this table, 200 permutations as measured 11.0%

Two dimensions beat one and neither reaches five per cent off-grid. A
correction that sensitive to configuration is not a lookup table, and shipping
it as one would produce numbers that look calibrated and are not -- the failure
this whole exercise exists to avoid.

**So `calibrated=True` stays off, and is documented as provisional.** It is
useful for reproducing the measurement and for a comparison that sits on a
measured cell. It is not yet a general correction, and the roadmap says what
would make it one.

**What this is and is not.** It is a calibration: where it works, the rate is
delivered because it was measured and corrected, not because the p-values were
shown to be correct. They are not, and the cause is unresolved. Any paper using
this should say so, and this docstring exists so nobody has to reconstruct it
later.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CALIBRATED_ALPHA",
    "CALIBRATION",
    "MOST_CONSERVATIVE",
    "calibrated_threshold",
]

#: The alpha the table was measured for. A different alpha needs a different
#: table, which is why it is checked rather than scaled: the relationship
#: between the nominal and the delivered rate is not known to be linear, and
#: assuming it is would be inventing numbers.
CALIBRATED_ALPHA = 0.05

#: ``(sample_size, tau)`` to the threshold that delivers `CALIBRATED_ALPHA`.
#: Measured, not chosen. Absent cells are absent because fewer than half the
#: studies there could be tested at all; they are not gaps to interpolate over.
CALIBRATION: dict[tuple[int, float], float] = {
    (500, 1.0): 0.0250, (500, 5.0): 0.0150, (500, 10.0): 0.0150,
    (1000, 1.0): 0.0250, (1000, 5.0): 0.0150, (1000, 10.0): 0.0150,
    (1000, 25.0): 0.0100,
    (2000, 1.0): 0.0250, (2000, 5.0): 0.0150, (2000, 10.0): 0.0200,
    (2000, 25.0): 0.0200, (2000, 50.0): 0.0150,
    (4000, 1.0): 0.0250, (4000, 5.0): 0.0199, (4000, 10.0): 0.0150,
    (4000, 25.0): 0.0150, (4000, 50.0): 0.0150,
}

#: The strictest threshold measured. Used where a lookup falls outside the
#: grid: calling less is the safe direction to err in, and the basis returned
#: says the cell was not measured.
MOST_CONSERVATIVE = min(CALIBRATION.values())

#: How far from a measured cell is too far, as a squared distance in log space.
#: About a factor of two on one axis; beyond that the nearest cell is not
#: evidence about the cell being asked for.
_FAR = float(np.log(2.0) ** 2)


def _separation(cell, sample_size, tau):
    """Distance in log space.

    Both axes span an order of magnitude, so a linear distance would let the
    sample size decide every lookup and the correlation time none of them.
    """
    cell_n, cell_tau = cell
    return (
        np.log(max(cell_n, 1) / max(sample_size, 1)) ** 2
        + np.log(cell_tau / max(tau, 1e-6)) ** 2
    )


def calibrated_threshold(
    tau: float,
    alpha: float = CALIBRATED_ALPHA,
    sample_size: int | None = None,
) -> tuple[float, str]:
    """The threshold to use, and how it was arrived at.

    Returns ``(threshold, basis)``, where ``basis`` is one of:

    ``"measured"``
        A cell was measured near this combination of sample size and
        correlation time, and its threshold is used.
    ``"conservative"``
        Nothing was measured near it, so the strictest threshold in the grid is
        used. The basis says the cell was not measured, so a caller can tell
        the difference between evidence and caution.
    ``"uncorrected"``
        A different alpha from the one the table was measured at. The nominal
        threshold is returned unchanged, because scaling a measured correction
        to an alpha it was not measured at would be inventing a number.

    Nearest neighbour rather than interpolation. The measured thresholds sit in
    a narrow band, 0.010 to 0.025, with no systematic trend across sample size;
    a surface fitted through seventeen points would be fitting quantile noise
    and presenting it as structure.
    """
    if not np.isclose(alpha, CALIBRATED_ALPHA):
        return float(alpha), "uncorrected"

    if sample_size is None:
        # The table cannot be indexed without a sample size, and guessing one
        # is exactly how the previous single-axis version under-corrected.
        return float(MOST_CONSERVATIVE), "conservative"

    nearest = min(CALIBRATION, key=lambda c: _separation(c, sample_size, tau))
    if _separation(nearest, sample_size, tau) > _FAR:
        return float(MOST_CONSERVATIVE), "conservative"
    return float(CALIBRATION[nearest]), "measured"
