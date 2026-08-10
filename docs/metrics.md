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

Bounded, so a residue's value means the same thing on a contact number as on a
torsion, and values are comparable across proteins. It is estimated from
kernel densities, so it inherits a grid and a bandwidth; both are choices, both
bias the estimate, and neither is visible in the number.

### Wasserstein-1

Needs no grid and no bandwidth — it is computed from the samples directly — and
reports in the feature's own units. Verified against known separations: two
Gaussians 1.4 apart give 1.390, and 3.0 apart give 3.031.

*"This residue gains 1.4 contacts"* is a sentence about the protein. *"This
residue has a Jensen–Shannon distance of 0.31"* is a sentence about the
comparison. The cost is that it is unbounded and comparable to nothing else,
which is why it is offered rather than made the default.

On circular features Prothon uses the circular optimal-transport distance
(Delon et al. 2010). A linear Wasserstein between two tight torsion populations
either side of the wraparound reports 4.43 radians where the truth is 0.28.

### Kolmogorov–Smirnov

Offered because PENSA reports it, so a claim that one method finds something
another misses can be checked on the same statistic.

On circular features Prothon substitutes **Kuiper's statistic**. The KS
statistic is the largest gap between two cumulative distributions, which on a
circle depends on where the circle was cut rather than on the data: over 24
rotations of one interleaved pair, KS ranges from 0.25 to 0.50. Kuiper's does
not move.

## Whole-ensemble comparison

*(In development — on `main`, not in 2.1.0.)*

Per-residue metrics cannot see a difference that lives in the relationship
*between* features. Two loops that visit the same positions as the wild type,
but no longer at the same time, give an identical profile at every residue and
are a different ensemble.

```python
study.distinguishability(measure="cbcn", method="c2st")
```

```
C2ST: distinguishable (p < 1e-06), AUC = 0.941
  driven mostly by residues 34, 35, 41, 42
```

**Maximum mean discrepancy** (`method="mmd"`) is a kernel two-sample test with
a permutation null. It gives a calibrated p-value and no indication of where
the difference is.

**The classifier two-sample test** (`method="c2st"`) trains a random forest to
tell the ensembles apart, scored out of fold. The area under the curve is a
bounded, immediately readable effect size, and the classifier reports which
residues it used. A forest rather than a linear model, because two ensembles
differing in *spread* rather than mean — a rigid loop against a mobile one —
are not linearly separable.

:::{warning}
The classifier's p-value comes from an asymptotic normal null, and its far tail
is where that approximation is worst; the cross-validation folds also share
training data, so the predictions are not quite independent. Prothon reports
anything below `1e-6` as a bound. **Quote the AUC, not the p-value.**
:::

## Coverage and fidelity

*(In development.)*

A dissimilarity says two ensembles differ. It does not say how, and the two
ways of differing need opposite work: a model that never opens a cryptic pocket
and one that opens pockets no physics produces are both wrong and score alike
on any symmetric distance.

```python
study.coverage_and_fidelity(measure="cbcn")
```

```
precision 0.977 (floor 0.975), recall 0.787 (floor 0.976)
  misses conformations at 4 residue(s): 5, 6, 7, 8
```

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
