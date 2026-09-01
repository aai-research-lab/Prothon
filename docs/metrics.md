# Metrics

Two different questions, and Prothon answers both.

## Per-residue distances

These compare each feature's distribution separately and give one value per
residue. Choose with `--metric` or `metric=`.

| name | what it is | scale |
|---|---|---|
| `jsd` | Jensen–Shannon distance between estimated densities | bounded [0, 1] |
| `wasserstein` | average distance the probability mass must move | the feature's own units |
| `ks` | Kolmogorov–Smirnov statistic (Kuiper's, on circular features) | bounded [0, 1] |

The permutation null, the false-discovery correction and the noise floor are
all computed under whichever metric is chosen. A Wasserstein comparison gets a
Wasserstein floor, not a threshold borrowed from a different scale.

### Jensen–Shannon (default)

The square root of the Jensen–Shannon divergence (Lin 1991), which is a true
metric on probability distributions (Endres and Schindelin 2003). Bounded in
[0, 1] with base-2 logarithms, so a residue's value means the same thing on a
contact number as on a torsion and values are comparable across proteins. This
is the distance the method was introduced with (Aina, Hsueh and Plotkin 2023).

It is estimated from kernel densities, so it inherits a grid and a bandwidth;
both are choices, both bias the estimate, and neither is visible in the number.

### Wasserstein-1

The optimal-transport distance under a linear cost (Villani 2009): the average
distance the probability mass must move to turn one distribution into the
other. It needs no grid and no bandwidth — it is computed from the samples
directly — and reports in the feature's own units. Verified against known
separations: two Gaussians 1.4 apart give 1.390, and 3.0 apart give 3.031.

*"This residue gains 1.4 contacts"* is a sentence about the protein. *"This
residue has a Jensen–Shannon distance of 0.31"* is a sentence about the
comparison. The cost is that it is unbounded and comparable to nothing else,
which is why it is offered rather than made the default.

On circular features Prothon uses the circular optimal-transport distance,
which minimises the same integral over a constant offset (Delon, Salomon and
Sobolevski 2010). A linear treatment of two tight torsion populations either
side of the wraparound reports 4.43 radians where the true separation is 0.28.

### Kolmogorov–Smirnov

The largest gap between the two empirical cumulative distributions — a
distribution-free statistic that needs no density and no tuning, which makes it
a useful check on a result obtained from a smoothed estimate.

On circular features Prothon substitutes **Kuiper's statistic** (Kuiper 1960),
the largest gap upward plus the largest gap downward. The reason is that the KS
statistic on a circle depends on where the circle was cut rather than on the
data: over 24 rotations of one interleaved pair it ranges from 0.25 to 0.50,
while Kuiper's does not move.

## Whole-ensemble comparison

Per-residue metrics cannot see a difference that lives in the relationship
*between* features. Two loops that visit the same positions as the wild type,
but no longer at the same time, give an identical profile at every residue and
are a different ensemble.

```python
prothon.distinguishability("cbcn", method="c2st")
```

```
C2ST: distinguishable (p < 1e-06), AUC = 0.941
  driven mostly by residues 34, 35, 41, 42
```

**Maximum mean discrepancy** (`method="mmd"`) embeds each conformation in a
reproducing kernel Hilbert space and measures the distance between the mean
embeddings (Gretton et al. 2012). With a characteristic kernel it is zero only
when the distributions are equal, so it detects any difference given enough
samples. Prothon uses a Gaussian kernel with the median heuristic bandwidth and
a permutation null. It gives a calibrated p-value and no indication of where
the difference is.

**The classifier two-sample test** (`method="c2st"`) trains a classifier to
tell the ensembles apart and asks whether it beats chance (Lopez-Paz and Oquab
2017). Prothon uses a random forest scored out of fold. The area under the
curve is a bounded, immediately readable effect size, and the classifier
reports which residues it used. A forest rather than a linear model, because
two ensembles differing in *spread* rather than in mean — a rigid loop against
a mobile one — are not linearly separable.

:::{warning}
The classifier's p-value comes from an asymptotic normal null, and its far tail
is where that approximation is worst; the cross-validation folds also share
training data, so the predictions are not quite independent. Prothon reports
anything below `1e-6` as a bound. **Quote the AUC, not the p-value.**
:::

## Coverage and fidelity

A dissimilarity says two ensembles differ. It does not say how, and the two
ways of differing need opposite work: a model that never opens a cryptic pocket
and one that opens pockets no physics produces are both wrong and score alike
on any symmetric distance.

```python
prothon.coverage_and_fidelity("cbcn")
```

```
precision 0.977 (floor 0.975), recall 0.787 (floor 0.976)
  misses conformations at 4 residue(s): 5, 6, 7, 8
```

The decomposition follows Sajjadi et al. (2018) and Kynkäänniemi et al. (2019),
applied here to each local order parameter separately so that the answer names
residues rather than scoring a whole ensemble.

**Recall** asks how much of the reference's support the compared ensemble
reaches. Low recall is a missed state — mode collapse, a loop that never
unfolds.

**Precision** asks how much of what it emits lands where the reference has
support. Low precision is an invented state.

The support is a highest-density region, so the null value is exact: two
ensembles from the same distribution score the coverage level itself (0.95 by
default). Both quantities carry a floor measured by splitting the reference in
half — and that floor is **per residue**, because a rigid residue and a mobile
one are not equally easy to cover. One averaged threshold flags about half the
unchanged residues in a protein by construction.

The reference and the compared ensemble are not interchangeable: precision and
recall swap when they are.

## References

Aina, Hsueh and Plotkin 2023; Delon, Salomon and Sobolevski 2010; Endres and
Schindelin 2003; Gretton et al. 2012; Kuiper 1960; Kynkäänniemi et al. 2019;
Lin 1991; Lopez-Paz and Oquab 2017; Sajjadi et al. 2018; Villani 2009. Full
citations on the [references page](references.md).
