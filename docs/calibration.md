# Calibration

A significance test promises that when two ensembles are drawn from the same
distribution it will call them different only about as often as the threshold
allows. That is a promise about a rate, and a rate has to be measured.

Everything below is a **null** measurement: both ensembles come from the same
distribution, so every rejection is a false positive by construction.
Reproduce with `scripts/calibration.py`.

## Independent frames

With exchangeable frames the test is calibrated. Across metrics, thresholds
and permutation counts the false-positive rate sits at or below the nominal
level, and the split-half floor is not cleared.

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
| 1 | 924 | 6.5% | 4.6–9.0% |
| 2 | 490 | 23.1% | 18.8–28.1% |
| 5 | 199 | 70.6% | 65.4–75.3% |
| 20 | 50 | **98.8%** | 97.3–99.4% |

Nominal rate: 5%.

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
the procedure; the script measures the whole pipeline, with an exponentially
decaying correlation along the chain.

## Reproducing

```bash
python scripts/calibration.py --quick                 # minutes
python scripts/calibration.py --replicates 1000       # the published numbers
python scripts/calibration.py --study correlation     # the table above
```

Replicates are independent, so the script parallelises across cores. A
thousand replicates per setting takes roughly an hour on a workstation.
