# The statistics

This is the page to read before quoting a number from Prothon.

## The floor comes first

Two independent halves of a *single* ensemble are not at distance zero from
each other. A finite sample never reproduces a continuous distribution exactly,
so any distance measured between two samples includes a contribution that has
nothing to do with the systems being compared.

Prothon measures that contribution directly. It splits the reference in half at
random, compares the halves, repeats, and reports the mean as the **noise
floor**. A global dissimilarity below its floor is reported as *not resolvable
at this sampling*, and `ComparisonResult.resolved` is `False`.

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.0000 (floor 0.1422) — not resolvable at this sampling
  ensemble 2: d = 0.5859 (floor 0.1426) — 9/12 residues differ
```

Reporting a difference without the resolution limit beside it invites a reader
to interpret a number the sampling cannot support. The floor is cheap to
measure and there is no good reason to omit it.

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
half the probability is worth **four** independent samples. Sizing a noise floor
by the frame count instead would produce error bars for an ensemble nobody
sampled.

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

## What is not corrected for

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

| correlation time (frames) | frame permutation | block permutation |
|---|---|---|
| 1 | 3.1% | 0.0% |
| 5 | 64.1% | 1.6% |
| 20 | 100.0% | 1.6% |

This happens automatically. `block_permutation=False` disables it for an
ensemble whose frames genuinely are independent — generated structures, or an
already-subsampled trajectory — where blocking costs resolution for nothing.

**Rows must be in the order the frames were generated.** A shuffled or
concatenated matrix has no correlation time to estimate, and the correction
silently finds none.

### When there is not enough to test with

Two situations leave too little independent information for any p-value, and
Prothon reports the measured floor and withholds the p-value rather than
printing one it cannot support:

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
