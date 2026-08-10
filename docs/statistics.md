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

No other ensemble-comparison package reports this, and it is the single most
useful thing here. Differences below the resolution limit are routinely
published.

## The null distribution

For each feature, Prothon asks whether the two ensembles' distributions differ
by more than chance. The null is built by **permutation**: pool the frames of
both ensembles, relabel them at random into two groups of the original sizes,
and measure the distance. Repeat. That is the exact distribution of the
statistic when the ensembles are the same, and it assumes nothing about the
shape of anything.

### Why version 2.0 was replaced

Version 2.0 built its null differently: it drew two bootstrap resamples from
the *same* ensemble and measured the distance between them.

Two resamples of $n$ frames drawn with replacement from the same $n$ frames
share about 63% of their points. They therefore resemble each other far more
closely than two independent samples of that size do. Measured on a 400-frame
Gaussian ensemble:

| quantity | value |
|---|---|
| bootstrap null used by 2.0 | 0.046 |
| two independent samples, same distribution | 0.097 |
| observed between-ensemble distance | 0.090 |

The null is too tight by roughly a factor of two, so any honest
between-ensemble distance clears it. Over 40 replicates in which both ensembles
were drawn from an **identical** distribution:

| | features called different | studies with ≥1 false positive |
|---|---|---|
| 2.0 bootstrap null | **100%** | **100%** |
| 2.1 permutation null | 1.2% | 7.5% |

The old test cannot return a negative.

`legacy=True`, or `--legacy-statistics`, reproduces version 2.0 exactly for
regenerating published figures. It is documented as unsound.

### Resolution of the p-values

A per-feature p-value from $n$ relabellings cannot fall below $1/(n+1)$, and
after correcting across a few hundred residues nothing at that resolution
survives. Prothon standardises each feature by its own null mean and spread,
then pools the standardised null values across features, which gives a
resolution of about $1/(n_{\text{perm}} \times n_{\text{features}})$.

The assumption is that the standardised null is comparable across features.
For distances computed from equal sample sizes on a shared grid that holds; it
would not hold if features had wildly different sample sizes.

## Multiple testing

A 300-residue protein tested at $\alpha = 0.05$ produces fifteen false
positives by construction. Per-feature p-values are corrected with the
Benjamini–Hochberg procedure, controlling the false discovery rate.

Version 2.0 did something different and worse: it computed a single pooled
p-value and wrote `local_diss[p_value >= 0.05] = 0.0`. With a scalar
`p_value`, NumPy reads that as a mask over the whole array — so one test
decided the fate of every residue together.

## How much sampling is enough

The question is not how many frames an ensemble has but how many *independent*
conformations it is worth. For a weighted ensemble these differ sharply. Kish's
effective sample size,

$$n_{\text{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2},$$

gives the number that matters: a thousand frames in which one conformer carries
half the probability is worth **four** independent samples.

Prothon warns below 50 effective samples and refuses below 10. Sizing a noise
floor by the frame count instead would produce error bars for an ensemble
nobody sampled.

:::{note}
This is the same failure as the bootstrap null above, arrived at from a
different direction: a quantity that looks like a sample size, is smaller than
it appears, and makes everything downstream look more certain than it is. It is
worth asking of any count that appears in a denominator.
:::

## What Prothon does not yet correct for

**Frames from a single continuous trajectory are correlated in time.** The
permutation null assumes frames are exchangeable. They are not, within one MD
trajectory, so an ensemble holds fewer independent conformations than it has
frames and the p-values remain somewhat optimistic. A block permutation over
the correlation time is planned.

Until then: the split-half noise floor is *measured* rather than assumed, and
is the more trustworthy of the two guides. If a difference clears the floor
comfortably, the conclusion is robust; if it clears only the p-value threshold,
be careful.

**Independent replicate trajectories are the fix available today.** Comparing
replicate against replicate gives a floor that includes run-to-run variation,
which is the honest reference for judging a condition-to-condition difference.
