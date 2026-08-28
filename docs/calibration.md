# Calibration

A significance test promises that when two ensembles are drawn from the same
distribution it will call them different only about as often as the threshold
allows. That is a promise about a rate, and a rate has to be measured.

Everything below is a **null** measurement: both ensembles come from the same
distribution, so every rejection is a false positive by construction.
Reproduce with `scripts/calibration.py`.

## Independent frames

With exchangeable frames the test is calibrated. Measured over 1000 replicates
per setting, 8 features each, Jensen–Shannon distance:

| α | permutations | features called different | 95% CI | studies with ≥1 rejection |
|---|---|---|---|---|
| 0.01 | 100 | 0.19% | 0.11–0.31% | 1.4% |
| 0.01 | 200 | 0.15% | 0.09–0.26% | 1.1% |
| 0.05 | 100 | 0.84% | 0.66–1.06% | 6.1% |
| 0.05 | 200 | 0.76% | 0.59–0.98% | 5.6% |
| 0.10 | 100 | 1.66% | 1.40–1.97% | 10.6% |
| 0.10 | 200 | 1.54% | 1.29–1.83% | 9.9% |

**Read the last column, not the third.** Benjamini–Hochberg controls the false
discovery rate, which is the expected proportion of false positives *among the
rejections* — not the per-feature rate. Under the complete null, where nothing
differs anywhere, controlling the false discovery rate is equivalent to
controlling the probability of making any rejection at all. That probability
is what the last column measures, and it tracks α: 1.4% at 0.01, 6.1% at 0.05,
10.6% at 0.10.

The per-feature rate is well below α by construction and should not be compared
to it.

More permutations shift the rate slightly downward, as the finer p-value
resolution stops marginal features from crossing the threshold.

:::{note}
The measurement above was made with the Jensen–Shannon distance. An equivalent
run for the Wasserstein and Kolmogorov–Smirnov metrics is pending: the
comparison pipeline was, until recently, silently computing Jensen–Shannon
whatever metric was requested, so the numbers previously obtained for the other
two describe the default rather than themselves.
:::

## Frames correlated in time

**This is the important table, and it is not comfortable reading.**

The permutation null assumes frames are exchangeable. Frames from a molecular
dynamics trajectory are correlated in time, so they are not. The measurement
below uses an Ornstein–Uhlenbeck process — a coordinate relaxing in a harmonic
well, which is the simplest thing that behaves like a trajectory — where the
correlation time is set and the number of independent conformations follows
from it as $n(1-\phi)/(1+\phi)$ with $\phi = e^{-1/\tau}$.

The stationary distribution does not depend on $\tau$, so both ensembles are
identical in distribution at every row.

| correlation time τ (frames) | independent conformations in 2000 | features called different | 95% CI |
|---|---|---|---|
| 1 | 924 | 5.5% | 5.0–6.0% |
| 2 | 490 | 23.7% | 22.8–24.6% |
| 5 | 199 | 72.1% | 71.1–73.1% |
| 10 | 100 | 93.7% | 93.2–94.2% |
| 20 | 50 | **99.0%** | 98.8–99.2% |
| 50 | 20 | **99.9%** | 99.8–100.0% |

Nominal rate: 5%. Measured over 1000 replicates per row.

The first row is the control: with frames essentially independent the rate is
5.5%, and its interval brackets the nominal level. Everything below it is the
cost of correlation alone.

### What this means for a result

**The p-values from a single continuous trajectory are not trustworthy.** At a
correlation time of twenty frames — unremarkable for a loop, and short for a
domain motion — essentially every residue is called different when nothing
differs. The failure is not subtle and it is not conservative.

The cause is exchangeability, not the estimator. Permuting correlated frames
produces two groups that each contain whole runs of consecutive conformations,
so the relabelled groups differ from one another more than independent samples
would, and the null is too narrow. The same argument applies to any
per-feature test that treats frames as independent draws, including a
Kolmogorov–Smirnov or a bootstrap test.

**What to do instead, today.**

*Subsample to the correlation time.* Take every $\tau$-th frame, or estimate
the statistical inefficiency and subsample by it. The ensemble is smaller and
the p-values mean what they say. The first row above is what that looks like.

*Use independent replicates.* Compare replicate against replicate to obtain a
floor that includes run-to-run variation, then judge a condition-to-condition
difference against that floor. This is the strongest option available and does
not depend on estimating a correlation time.

*Read the floor, not the p-value.* The split-half noise floor is measured
rather than assumed. It is degraded by correlation too — halves of a correlated
trajectory resemble each other more than independent samples would — but it
degrades far more gracefully than the null does.

**What is planned.** A block permutation, which relabels contiguous blocks of
length ~τ instead of individual frames and so preserves the correlation
structure under the null. The table above is the argument for prioritising it,
and the measurement against which it will be judged.

## Correlated features

Residues in a protein do not move independently. Benjamini–Hochberg controls
the false discovery rate under positive dependence, which is a theorem about
the procedure; this measures the whole pipeline, with a correlation decaying
exponentially along the chain, 20 features, 1000 replicates.

| correlation between neighbouring features | features called different | 95% CI |
|---|---|---|
| 0.0 | 0.38% | 0.30–0.48% |
| 0.5 | 0.62% | 0.52–0.74% |
| 0.9 | 0.62% | 0.52–0.74% |

Correlation between features raises the rate slightly and then stops mattering:
0.5 and 0.9 are indistinguishable. Both remain far below the nominal 5%. This
is the failure mode that does **not** occur — unlike correlation between
frames, which is catastrophic.

## Reproducing

```bash
python scripts/calibration.py --quick                 # minutes
python scripts/calibration.py --replicates 1000       # the published numbers
python scripts/calibration.py --study correlation     # the table above
```

Replicates are independent, so the script parallelises across cores. A
thousand replicates per setting takes roughly an hour on a workstation.
