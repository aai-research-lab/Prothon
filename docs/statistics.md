# The statistics

This is the page to read before quoting a number from Prothon.

## Probability weights

Every public weighted API uses the same contract. Supply exactly one numeric
weight per frame. Values must be finite and non-negative, at least one must be
positive, and their sum must be finite and positive. The scale is immaterial:
valid weights are normalised internally, so `2, 3, 5` and `0.2, 0.3, 0.5`
describe the same ensemble. Zero-weight frames are allowed; negative values,
NaNs, infinities, an all-zero vector, and a length mismatch are refused before
an effective sample size or statistic is computed.

## The floor comes first

Two independent halves of a *single* ensemble are not at distance zero from
each other. A finite sample never reproduces a continuous distribution exactly,
so any distance measured between two samples includes a contribution that has
nothing to do with the systems being compared.

Prothon measures that contribution directly. For explicitly independent
structures it splits **each** ensemble into random disjoint halves. For a
trajectory it assigns complete temporal blocks to the halves, so a slow
excursion is never interleaved into both sides. Complete independent replicas
can be supplied as still stronger units.

The mean of these measurements remains the descriptive **noise floor**. A mean
is not a decision threshold: `ComparisonResult.resolved` clears only the 95th
percentile of the measured floor distribution. The mean, threshold and full
distribution are all recorded.

The same engine supplies the per-feature precision/recall floor and the
experimental-agreement floor. Their APIs default to trajectory sampling because
a bare array carries no provenance. Independently generated structures must opt
into `sampling_kind="iid"` (and `sampling_kind_ref="iid"` for a
precision/recall reference); a supplied correlation time greater than one frame
alongside an IID claim is rejected as contradictory.

Halves of one ensemble are used because they are the only pair of samples
guaranteed to come from the same distribution without assuming what that
distribution is. A bootstrap assumes the sample is the population; a parametric
reference assumes a shape. Two halves differ only by sampling, by
construction.

Reporting a difference without the resolution limit beside it invites a reader
to interpret a number the sampling cannot support. The floor is cheap to
measure and there is no good reason to omit it.

### The floor is conservative, by a quarter to a half

Halves have half the frames, so the floor measures the resolution limit at
*n*/2 while the study has *n*. Measured on samples from one distribution with
the Jensen–Shannon distance:

| frames in each sample | distance between two of them |
|---|---|
| 250 | 0.079 |
| 500 | 0.063 |
| 1000 | 0.050 |

A 1000-frame ensemble reports a floor of about 0.063 where the limit at 1000
frames is 0.050 — roughly **1.25× too high**.

The error is in the safe direction: a difference called resolvable is
resolvable, and some real differences near the limit are called unresolvable
instead. Read a result within about a quarter of its floor as borderline
rather than settled.

#### Why the factor is not corrected for

The obvious fix is to divide by √2, on the grounds that the distance between
two samples of one distribution goes as *n*^(−1/2). It was measured, and it
does not.

| metric | Gaussian | bimodal | skewed | uniform |
|---|---|---|---|---|
| `jsd` | −0.351 | −0.390 | −0.465 | −0.428 |
| `wasserstein` | −0.495 | −0.563 | −0.597 | −0.592 |
| `ks` | −0.492 | −0.530 | −0.560 | −0.546 |

Wasserstein and Kolmogorov–Smirnov sit near −0.5. **Jensen–Shannon, the
default, does not** — it decays more slowly, because it is estimated from a
kernel density whose bandwidth also depends on the sample size, so its floor
carries a smoothing bias that fades more slowly than the sampling error. The
exponent moves with the shape of the distribution as well.

A single √2 would therefore be about right for two of the three metrics and too
large for the third, pushing its floor *below* the true limit. A floor that
understates the resolution limit is exactly the failure this measurement exists
to prevent.

Measuring the slope rather than assuming it does not rescue it either.
Extrapolating from splits at *n*/4 and *n*/2 gives a two-point slope that is
itself noisy, and on a skewed distribution the extrapolated Jensen–Shannon
floor came out 29% below the truth — again in the unsafe direction.

So the floor stays as the split-half distribution: conservative, in the
direction that costs sensitivity rather than credibility, and documented
rather than adjusted. Its mean describes the sampling contribution and its
95th percentile controls the resolved/unresolved verdict.
Reproduce with `python scripts/floor_scaling.py`.

## The null distribution

For each feature, Prothon asks whether the two ensembles' distributions differ
by more than chance. The null is built by **permutation**: pool the frames of
both ensembles, relabel them at random into two groups of the original sizes,
and measure the distance. Repeat. Under the hypothesis that both ensembles
sample the same distribution the labels carry no information, so every
relabelling is as likely as the one observed, and the resulting distances are
the exact null distribution of the statistic — with no assumption about its
shape (Good 2005).

### Why not a bootstrap of each ensemble against itself

A natural-looking alternative is to resample each ensemble twice and use the
distance between those resamples as the null. It does not work, and the failure
is large enough to be worth stating explicitly.

Two resamples of $n$ frames drawn with replacement from the same $n$ frames
share about 63% of their points. They therefore resemble each other far more
closely than two *independent* samples of that size do. Measured on a 400-frame
Gaussian ensemble:

| quantity | value |
|---|---|
| bootstrap of the ensemble against itself | 0.046 |
| two independent samples of the same distribution | 0.097 |
| observed between-ensemble distance | 0.090 |

A null that tight is cleared by any honest between-ensemble distance. Over 40
replicates in which both ensembles were drawn from an **identical**
distribution, such a null calls 100% of features different; the permutation
null sits at 1.2%.

`legacy=True`, or `--legacy-statistics`, selects the bootstrap null for anyone
regenerating a figure published under it. It is documented as unsound.

### Resolution of the p-values

A per-feature p-value from $n$ relabellings cannot fall below $1/(n+1)$, and
after correcting across a few hundred residues nothing at that resolution
survives. Prothon standardises each feature by its own null mean and spread,
then pools the standardised null values across features, giving a resolution of
about $1/(n_{\text{perm}} \times n_{\text{features}})$.

The assumption is that the standardised null is comparable across features. For
distances computed from equal sample sizes on a shared grid that holds; it
would not hold if features had wildly different sample sizes.

The p-values remain discrete, and the consequence is measurable: at the default
100 permutations the test rejects about 1.4 times as often as the threshold
allows, falling to about 1.2 times at 200 — roughly 6% instead of 5%. Raise
`n_permutations` for a result going into a paper. See the
[calibration page](calibration.md).

### The whole-ensemble MMD null

MMD uses the same exchangeability rule, but makes one joint decision rather
than one decision per feature. Its null therefore relabels the independent
units from both inputs: complete temporal blocks for a trajectory, complete
replicas when labels are supplied, and individual rows only when an ensemble
is explicitly declared IID. The larger native unit length controls a mixed
design, so IID rows are grouped conservatively rather than exchanged one at a
time against whole trajectory blocks.

Weights are properties of conformations, not labels. Prothon first puts the
two inputs on a common mean-one mass scale, then moves group labels while each
weight stays with its observation. The positive and negative sides of the MMD
quadratic form are renormalised to unit mass after every admissible whole-unit
assignment. This matters for both unequal probability weights and unequal
ensemble lengths: directly permuting the old signed vector detached weights,
while pooling separately normalised `1/m` and `1/n` masses would reveal which
input an observation came from whenever `m != n`.

Four units on each side give 35 distinct balanced MMD values after equivalent
complementary labels are identified; three give only ten and cannot resolve a
5% threshold. Prothon also computes Kish's effective count from the probability
mass collected by each block or replica. If either the actual or effective
count is below four, or the distinct assignments cannot resolve `alpha`, MMD²
remains available but `p_value` and `distinguishable` are `None`. The requested
number of Monte Carlo relabellings must also provide resolution finer than
`alpha`; repeatedly drawing a small assignment set cannot manufacture finer
evidence than the exact design contains.

## Multiple testing

A 300-residue protein tested at $\alpha = 0.05$ produces fifteen false
positives by construction. Per-feature p-values are corrected with the
Benjamini–Hochberg procedure, which controls the false discovery rate — the
expected proportion of false positives among the residues declared different
(Benjamini and Hochberg 1995).

## How much sampling is enough

The question is not how many frames an ensemble has but how many *independent*
conformations it is worth. For a weighted ensemble the two differ sharply. The
effective sample size (Kish 1965),

$$n_{\text{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2},$$

gives the number that matters: a thousand frames in which one conformer carries
half the probability is worth **four** independent samples. For a trajectory,
weights are first summed within each temporal block (or complete replica) and
Kish's count is applied to those independent-unit masses. This combines weight
concentration and time correlation instead of reporting either correction in
isolation. Sizing a noise floor by the frame count would produce error bars for
an ensemble nobody sampled.

Prothon warns below 50 effective samples and refuses below 10.

## Density estimation

Distances that need a density (Jensen–Shannon, and the supports behind
precision and recall) use kernel density estimation with Silverman's bandwidth
rule (Silverman 1986). Circular features use a von Mises kernel with Taylor's
plug-in concentration (Taylor 2008) on a grid spanning a full turn.

The bandwidth and the grid are choices, and they bias the estimate without
appearing in the number that comes out. Where that matters, `--metric
wasserstein` computes the distance from the samples directly and needs neither.

A feature that never varies — a buried residue with zero solvent exposure in
every frame — has no density in the usual sense and is handled explicitly
rather than failing.

## Time correlation and exchangeability

**Frames from a single continuous trajectory are correlated in time**, and a
null that relabels individual frames assumes they are not. The consequence was
measured rather than left as a caveat, and it is large:

| correlation time (frames) | independent conformations in 2000 | features called different |
|---|---|---|
| 1 | 924 | 5.5% |
| 5 | 199 | 72.1% |
| 20 | 50 | **99.0%** |
| 50 | 20 | **99.9%** |

Nominal rate 5%; both ensembles drawn from the same distribution at every row.
The full measurement is on the [calibration page](calibration.md).

**Prothon therefore relabels blocks, not frames.** The correlation time is
estimated from the data by an integrated autocorrelation with Sokal's window,
blocks are made a couple of correlation times long, and whole blocks are
relabelled — so the null is built from data that still looks like a trajectory.
On the same data as the table above:

| correlation time τ | frame permutation | block permutation |
|---|---|---|
| 1 | 5.5% | 1.7% |
| 5 | 72.1% | 2.3% |
| 20 | 99.0% | 2.2% |
| 50 | 99.9% | 2.3% |

Nominal 5%, 1000 replicates each. The block rate is flat across the range
rather than degrading with τ, so the test does not require a user to know
their correlation time in order to be honest.

The cross-feature summary starts at q75, but q75 cannot see a slow region that
occupies less than one quarter of a protein. Prothon therefore checks the
per-feature times for a *coherent slow group*: at least two columns must be
separated from the median by both a factor of three and four robust standard
deviations on the log scale, and their median must exceed q75 threefold. When
those conditions hold, the slow-group median sets the block length. This is
deliberately not q90, q95 or the maximum: the upper tail of a homogeneous
system grows with the number of features because the estimator is noisy, so a
fixed high quantile would make larger proteins look slower for statistical
rather than physical reasons.

Constant and non-finite columns are unassessable and are removed before the
200-feature limit is applied. Result metadata records the summary selected,
the assessable and sampled feature counts, and the zero-based columns in any
slow group. The scalar correlation time is therefore traceable to the features
that caused it.

For a circular representation, the same procedure estimates each feature on
its sine and cosine coordinates and retains the slower component. This makes
-pi and +pi adjacent, as they are physically, without unwrapping a trajectory
and inventing a history of complete rotations. Dissimilarity and
precision/recall pass their declared circularity into the sampling plan.

`MINIMUM_BLOCKS` remains eight. That threshold comes from the 35 distinct
balanced partitions of eight units, not from the feature aggregation rule, so
changing the correlation summary does not justify tuning it. The mixed null
and shifted alternative in the calibration suite check both sides: the new
plan retains enough blocks to issue a verdict, makes no null call in its fixed
fixture, and still detects the two deliberately shifted slow features.

This happens automatically. `block_permutation=False` disables it for an
ensemble whose frames genuinely are independent — generated structures, or an
already-subsampled trajectory — where blocking costs resolution for nothing.

**Rows must be in the order the frames were generated.** A shuffled or
concatenated matrix has no correlation time to estimate, and the correction
silently finds none.

### When there is not enough to test with

Two situations leave too little independent information for any p-value.
Prothon retains the measured floor values as descriptive diagnostics, but
withholds both the p-value and the resolved/unresolved floor verdict rather
than printing decisions the sampling cannot support:

A permutation p-value over six blocks cannot resolve a 5% threshold, let alone
survive correction across a few hundred residues. Prothon therefore refuses
below eight.

What makes that check trustworthy is that **the block length is never shortened
to manufacture blocks**. A block shorter than the correlation time does not
contain the correlation, so the null it builds is the frame-permutation null
under a block-shaped name — and the block count that was meant to reveal the
problem is the count that was forced to look healthy. A short trajectory of a
slow system therefore reports few blocks and is refused, even where the
correlation time itself has saturated: 300 frames of a system whose estimator
saturates at 33 leaves four blocks.

`ComparisonResult.p_values_withheld` is set and the summary says
why. Sample for longer, or compare independent replicates — which gives a floor
including run-to-run variation and needs no correlation time at all.

## References

Benjamini and Hochberg 1995; Good 2005; Kish 1965; Silverman 1986; Taylor 2008.
Full citations on the [references page](references.md).
