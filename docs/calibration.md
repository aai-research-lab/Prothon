# Calibration

A significance test promises that when two ensembles are drawn from the same
distribution it will call them different only about as often as the threshold
allows. That is a promise about a rate, and a rate has to be measured.

Everything below is a **null** measurement: both ensembles come from the same
distribution, so every rejection is a false positive by construction.
Reproduce with `scripts/calibration.py`.

## Independent frames

With exchangeable frames the test is calibrated. Measured over 1000 replicates
per setting, 8 features each:

| metric | α | permutations | features called different | 95% CI | studies with ≥1 rejection |
|---|---|---|---|---|---|
| `jsd` | 0.01 | 100 | 0.19% | 0.11–0.31% | 1.4% |
| `jsd` | 0.01 | 200 | 0.15% | 0.09–0.26% | 1.1% |
| `jsd` | 0.05 | 100 | 0.84% | 0.66–1.06% | 6.1% |
| `jsd` | 0.05 | 200 | 0.76% | 0.59–0.98% | 5.6% |
| `jsd` | 0.10 | 100 | 1.66% | 1.40–1.97% | 10.6% |
| `jsd` | 0.10 | 200 | 1.54% | 1.29–1.83% | 9.9% |
| `wasserstein` | 0.01 | 100 | 0.24% | 0.15–0.37% | 1.9% |
| `wasserstein` | 0.01 | 200 | 0.15% | 0.09–0.26% | 1.2% |
| `wasserstein` | 0.05 | 100 | 1.00% | 0.80–1.24% | 7.0% |
| `wasserstein` | 0.05 | 200 | 0.83% | 0.65–1.05% | 6.0% |
| `wasserstein` | 0.10 | 100 | 2.01% | 1.73–2.34% | 11.9% |
| `wasserstein` | 0.10 | 200 | 1.84% | 1.57–2.16% | 11.4% |
| `ks` | 0.01 | 100 | 0.24% | 0.15–0.37% | 1.9% |
| `ks` | 0.01 | 200 | 0.20% | 0.12–0.32% | 1.6% |
| `ks` | 0.05 | 100 | 0.88% | 0.69–1.10% | 6.2% |
| `ks` | 0.05 | 200 | 0.81% | 0.64–1.03% | 5.8% |
| `ks` | 0.10 | 100 | 1.90% | 1.62–2.22% | 12.2% |
| `ks` | 0.10 | 200 | 1.66% | 1.40–1.97% | 10.9% |

### Read the last column, not the third

Benjamini–Hochberg controls the false discovery rate: the expected proportion
of false positives *among the rejections*, not the per-feature rate. Under the
complete null, where nothing differs anywhere, controlling the false discovery
rate is the same as controlling the probability of making any rejection at all.
That probability is the last column, and it is what should be compared to α.

The per-feature rate is far below α by construction and comparing it to α says
nothing.

### More permutations are worth buying

The study rate runs consistently above α, and doubling the permutations moves
it toward α: averaged over all nine metric-threshold combinations, the ratio of
observed rate to α is **1.39 at 100 permutations and 1.18 at 200**.

That is the discreteness of a permutation p-value. With ``n`` relabellings a
p-value is a multiple of ``1/(n+1)``, and the pooling across features softens
that without removing it, so a threshold falls between attainable values and
lands slightly on the permissive side. More permutations make the grid finer.

**The default of 100 permutations is therefore mildly anticonservative** —
about 6% instead of 5% — which is worth knowing and is not the same order of
problem as anything else on this page. For a result going into a paper, raise
`n_permutations` to 200 or beyond; the cost is linear.

All three metrics behave the same way, which is the expected result: the
permutation null makes no assumption about the statistic, so any of them is
calibrated, and the residual differences between them are within the intervals.

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

### The fix

Relabelling contiguous blocks of length ~2τ rather than individual frames
restores calibration across the whole range. Both nulls, 1000 replicates each,
on the same data:

| correlation time τ | independent conformations in 2000 | frame permutation | block permutation |
|---|---|---|---|
| 1 | 924 | 5.45% | 1.66% |
| 2 | 490 | 23.67% | 2.04% |
| 5 | 199 | 72.11% | 2.30% |
| 10 | 100 | 93.73% | 2.17% |
| 20 | 50 | 99.01% | 2.24% |
| 50 | 20 | 99.92% | 2.31% |

Nominal 5%. The block rate is flat at about 2% from τ = 1 to τ = 50 — it does
not degrade as the correlation lengthens, which is the property that matters:
a user does not have to know their correlation time for the test to be honest.

It is slightly conservative, as a block permutation with a finite number of
blocks will be, and conservative in the direction that costs power rather than
credibility. Power is not the price: a real 0.8σ shift at τ = 20 is still found
at 98% of features.

Reproduce with `python scripts/calibration.py --study correlation`, which runs
both nulls on the same data.

### A slow minority of features

An upper quartile has a structural blind spot: fewer than 25% slow features
cannot change it, however slow they are. The audit reproduction uses 4000
frames and 100 features: 80 IID normals and 20 AR(1) features with relaxation
time 45. All have the same stationary distribution.

| summary on the mixed system | estimated time (frames) |
|---|---:|
| q75 alone | 1.07 (the audited ~1.05 failure) |
| median of the 20 slow features | 71.90 (the audited ~71.96 value) |
| selected correlation plan | 71.90 |

The slow group is recovered as columns 80–99 (zero-based). Further fixtures
cover one contiguous eight-feature slow loop, two separated four-feature slow
groups, and a slow angular loop crossing the -pi/+pi branch cut. On a
homogeneous 100-feature system at relaxation time 20, the same rule retains q75
at 46.27 frames; q95 is 73.61 frames on that fixture. Thus the correction
protects coherent minorities without paying the growing worst-estimate penalty
on a homogeneous protein.

The end-to-end null fixture has 18 fast and two slow features in each of two
2000-frame ensembles. The selected plan leaves 33 complete blocks, warns that
the correlation estimate is still rising, does not withhold the p-value, and
makes no false call. Shifting only the two slow features by one stationary
standard deviation detects exactly those two, so conservatism has not replaced
power. Reproduce the cases with
`pytest tests/test_correlation.py -k "slow or homogeneous or mixed or circular"`.

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

**If the p-value is withheld.**

*Subsample to the correlation time.* Take every $\tau$-th frame, or estimate
the statistical inefficiency and subsample by it. The ensemble is smaller and
the p-values mean what they say. The first row above is what that looks like.

*Use independent replicates.* Compare replicate against replicate to obtain a
floor that includes run-to-run variation, then judge a condition-to-condition
difference against that floor. This is the strongest option available and does
not depend on estimating a correlation time.

*Read the floor, not only the p-value.* The split-half noise floor is measured
rather than assumed. Its halves are assembled from complete temporal blocks,
or complete independent replicas when those labels are available. Randomly
interleaving correlated frames made both halves share the same slow excursions
and understated the uncertainty. The reported mean is descriptive; the 95th
percentile of the measured distribution is the decision threshold.

**Where blocking cannot help.** With too few blocks, or a trajectory shorter
than about twenty correlation times, there is no p-value to be had at any block
size. Prothon says so rather than printing one.

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

## Whole-ensemble MMD

The MMD gate was declared before running the corrected implementation: a
stationary correlated null may reject 0–3 of 20 pairs, and a
0.5-standard-deviation shift must be detected in at least 8 of 10 pairs. Each trajectory has
500 frames and four features from an AR(1) process with relaxation time 10. The
test is supplied the corresponding 20-frame integrated correlation time, uses
40-frame blocks, 99 relabellings and `alpha=0.05`.

| calculation | calls | predeclared requirement | Wilson 95% CI | result |
|---|---:|---:|---:|---|
| corrected block null | 1/20 | 0–3/20 | 0.9–23.6% | pass |
| deliberately misdeclared IID-row null | 8/8 | diagnostic reproduction | 67.6–100% | audited failure reproduced |
| shifted block alternative | 10/10 | at least 8/10 | 72.2–100% | pass |

The intervals are deliberately shown even though this is a compact CI gate,
not a high-precision estimate of the error rate. The important comparisons are
with the bounds fixed before the run, not with selected seeds after it.

The complete corrected-null p-values were:

```text
0.53, 0.24, 0.58, 0.35, 0.91, 0.89, 0.85, 0.10, 0.44, 0.47,
0.17, 0.81, 0.35, 0.32, 0.08, 0.59, 0.36, 0.79, 0.01, 0.18
```

The complete shifted-alternative p-values were:

```text
0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.03
```

The same deterministic run exercises the remaining sampling designs under
their null:

| design | p-value | units per side | effective units per side |
|---|---:|---:|---:|
| unequal lengths, 500 versus 700 frames | 0.11 | 12, 17 | 11.79, 16.78 |
| unequal probability weights, 320 versus 470 IID rows | 0.92 | 320, 470 | 261.32, 289.23 |
| trajectory versus IID, 500 versus 650 frames | 0.88 | 12, 16 | 11.79, 15.94 |
| eight independent replicas per side | 0.73 | 8, 8 | 8.00, 8.00 |
| circular trajectories crossing -pi/+pi | 0.15 | 12, 12 | 11.79, 11.79 |

A final 120-frame case supplies only three 40-frame blocks per side. Its MMD²
is 0.1492, but its p-value and verdict are withheld as required. Reproduce the
complete record, including every seed, statistic, null moment, software version
and gate, with:

```bash
python scripts/joint_calibration.py --out joint-calibration.json
```

## Whole-ensemble classifier

The C2ST gate uses the same 20 correlated null pairs and a separately declared
0.7-standard-deviation power fixture. It compares the replaced implementation
directly: shuffled row folds let neighbouring frames from the same slow
excursion appear in both training and test data, then an independent-row normal
tail treats all 1000 predictions as independent. The corrected calculation
keeps every 40-frame block in one fold and obtains evidence from whole-label
permutations on a separate, label-blind held-out unit set.

| calculation | calls | predeclared requirement | Wilson 95% CI | result |
|---|---:|---:|---:|---|
| grouped block null | 0/20 | 0–3/20 | 0.0–16.1% | pass |
| replaced shuffled-row/normal null | 10/10 | diagnostic reproduction | 72.2–100% | audited failure reproduced |
| shifted grouped alternative | 9/10 | at least 8/10 | 59.6–98.2% | pass |

The old null AUCs range from 0.683 to 0.806 even though both inputs have the
same stationary distribution. With complete blocks, the AUCs range from 0.382
to 0.678 and average 0.501. The p-value does not reuse these grouped-CV
predictions: it comes from the untouched unit set, while AUC remains the effect
size.

The complete grouped-null p-values were:

```text
0.29, 0.58, 0.22, 0.68, 0.73, 0.46, 0.79, 0.54, 0.67, 0.71,
0.44, 0.20, 0.52, 0.76, 0.13, 0.25, 0.95, 0.73, 0.12, 0.18
```

The complete shifted-alternative p-values were:

```text
0.01, 0.03, 0.07, 0.01, 0.02, 0.04, 0.03, 0.02, 0.01, 0.03
```

The fixed null designs also pass: unequal lengths (`p=0.06`), unequal weights
(`p=0.23`), trajectory versus IID (`p=0.09`), eight independent replicas per
side (`p=0.55`), and circular trajectories (`p=0.08`). A 120-frame case has
only three blocks per side; its AUC of 0.620 is retained, but three possible
held-out labels give resolution only to `1/3`, so the p-value is withheld.

`scripts/joint_calibration.py` emits every C2ST AUC, accuracy, p-value, null
moment, fold/split/forest/permutation seed, unit count, effective count, and
class/weight balance alongside the MMD record. The JSON therefore contains the
complete run rather than only these aggregate gates.

## Reproducing

```bash
python scripts/calibration.py --quick                 # minutes
python scripts/calibration.py --replicates 1000       # the published numbers
python scripts/calibration.py --study correlation     # the table above
python scripts/joint_calibration.py                    # deterministic joint gates
```

`scripts/calibration.py` exits nonzero when a predeclared corrected-null gate
fails. Its JSON output records raw rejection/support counts, seed ranges,
Wilson intervals, the git commit and software versions. Rows that deliberately
force frame permutation on correlated data remain in the record as negative
controls, but are excluded from the release verdict. The scheduled scientific
calibration workflow retains this file and the joint MMD/C2ST record whether a
gate passes or fails.

Replicates are independent, so the script parallelises across cores. A
thousand replicates per setting takes roughly an hour on a workstation.
