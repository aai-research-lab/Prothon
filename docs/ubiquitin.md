# Ubiquitin, re-run

Every ensemble against Q99. `published` is the statistical treatment of the 2023 paper; `current` is the default now.

## CBCN

| against Q99 | raw d | floor | resolved | published | current | τ | blocks | independent |
|---|---|---|---|---|---|---|---|---|
| Q95 | 0.0618 | 0.0358 | yes | 70/70 | 0/70 | 45 | 54 | 110 |
| Q90 | 0.1706 | 0.0361 | yes | 70/70 | 11/70 | 269 | 9 | 19 |
| Q85 | 0.2700 | 0.0354 | yes | 70/70 | withheld | 356 | 7 | 14 |
| Q80 | 0.3439 | 0.0355 | yes | 70/70 | withheld | 381 | 6 | 13 |
| Q75 | 0.3971 | 0.0349 | yes | 70/70 | withheld | 434 | 5 | 12 |

Largest change in any per-residue distance: **0.00e+00** — unchanged, as it should be: the Jensen–Shannon calculation was always correct and only the significance filter moved.

Residues called different: **350/350** (100%) under the published treatment, across all 5 comparisons and 70 features.

Under the current treatment, **11/140** (8%) of the features that could be tested at all, the remaining 3 comparisons having been withheld.

3 of 5 report no p-value at all: the correlation time leaves too few independent blocks to build a permutation null from. That is a statement about the sampling, not about the ensembles.

## CACN

| against Q99 | raw d | floor | resolved | published | current | τ | blocks | independent |
|---|---|---|---|---|---|---|---|---|
| Q95 | 0.0760 | 0.0360 | yes | 76/76 | 3/76 | 65 | 38 | 77 |
| Q90 | 0.1818 | 0.0354 | yes | 76/76 | 8/76 | 307 | 8 | 16 |
| Q85 | 0.2990 | 0.0350 | yes | 76/76 | withheld | 379 | 6 | 13 |
| Q80 | 0.3788 | 0.0344 | yes | 76/76 | withheld | 466 | 5 | 11 |
| Q75 | 0.3942 | 0.0349 | yes | 76/76 | withheld | 459 | 5 | 11 |

Largest change in any per-residue distance: **0.00e+00** — unchanged, as it should be: the Jensen–Shannon calculation was always correct and only the significance filter moved.

Residues called different: **380/380** (100%) under the published treatment, across all 5 comparisons and 76 features.

Under the current treatment, **11/152** (7%) of the features that could be tested at all, the remaining 3 comparisons having been withheld.

3 of 5 report no p-value at all: the correlation time leaves too few independent blocks to build a permutation null from. That is a statement about the sampling, not about the ensembles.

## CABA

| against Q99 | raw d | floor | resolved | published | current | τ | blocks | independent |
|---|---|---|---|---|---|---|---|---|
| Q95 | 0.0605 | 0.0361 | yes | 74/74 | 13/74 | 28 | 87 | 178 |
| Q90 | 0.1132 | 0.0358 | yes | 74/74 | 8/74 | 240 | 10 | 21 |
| Q85 | 0.1529 | 0.0360 | yes | 74/74 | 13/74 | 244 | 10 | 20 |
| Q80 | 0.1877 | 0.0358 | yes | 74/74 | 32/74 | 205 | 12 | 24 |
| Q75 | 0.2074 | 0.0357 | yes | 74/74 | 38/74 | 235 | 10 | 21 |

Largest change in any per-residue distance: **0.00e+00** — unchanged, as it should be: the Jensen–Shannon calculation was always correct and only the significance filter moved.

Residues called different: **370/370** (100%) under the published treatment, across all 5 comparisons and 74 features.

Under the current treatment, **104/370** (28%) of the features that could be tested at all.

## CATA

| against Q99 | raw d | floor | resolved | published | current | τ | blocks | independent |
|---|---|---|---|---|---|---|---|---|
| Q95 | 0.0832 | 0.0379 | yes | 73/73 | 4/73 | 42 | 59 | 119 |
| Q90 | 0.1607 | 0.0376 | yes | 73/73 | 10/73 | 220 | 11 | 23 |
| Q85 | 0.2325 | 0.0377 | yes | 73/73 | 32/73 | 197 | 12 | 25 |
| Q80 | 0.2745 | 0.0385 | yes | 73/73 | 43/73 | 210 | 11 | 24 |
| Q75 | 0.3108 | 0.0378 | yes | 73/73 | 48/73 | 234 | 10 | 21 |

Largest change in any per-residue distance: **3.13e-01** — expected here, and a finding in its own right. This is a circular feature, and the published treatment estimated its density on a linear grid: a torsion whose values straddle the wrap at ±π appears as two separated modes, and two ensembles sitting on opposite sides of the wrap appear to share no support at all. The magnitudes change for those residues, not only the significance calls.

Residues called different: **365/365** (100%) under the published treatment, across all 5 comparisons and 73 features.

Under the current treatment, **137/365** (38%) of the features that could be tested at all.

## SASA

| against Q99 | raw d | floor | resolved | published | current | τ | blocks | independent |
|---|---|---|---|---|---|---|---|---|
| Q95 | 0.0820 | 0.0330 | yes | 76/76 | 4/76 | 77 | 32 | 65 |
| Q90 | 0.1656 | 0.0332 | yes | 76/76 | 20/76 | 212 | 11 | 24 |
| Q85 | 0.2463 | 0.0335 | yes | 76/76 | 29/76 | 261 | 9 | 19 |
| Q80 | 0.2879 | 0.0339 | yes | 76/76 | withheld | 337 | 7 | 15 |
| Q75 | 0.3443 | 0.0332 | yes | 76/76 | withheld | 386 | 6 | 13 |

Largest change in any per-residue distance: **0.00e+00** — unchanged, as it should be: the Jensen–Shannon calculation was always correct and only the significance filter moved.

Residues called different: **380/380** (100%) under the published treatment, across all 5 comparisons and 76 features.

Under the current treatment, **53/228** (23%) of the features that could be tested at all, the remaining 2 comparisons having been withheld.

2 of 5 report no p-value at all: the correlation time leaves too few independent blocks to build a permutation null from. That is a statement about the sampling, not about the ensembles.

## Across order parameters

`published` divides by every comparison, as that treatment reported a p-value for all of them. `current` divides by the comparisons a test could be run on, which is not the same denominator -- a withheld p-value is not a residue that was tested and found not to differ.

| order parameter | features | τ range | published | current | withheld |
|---|---|---|---|---|---|
| `cbcn` | 70 | 45–434 | 100% | 8% | 3/5 |
| `cacn` | 76 | 65–466 | 100% | 7% | 3/5 |
| `caba` | 74 | 28–244 | 100% | 28% | 0/5 |
| `cata` | 73 | 42–234 | 100% | 38% | 0/5 |
| `sasa` | 76 | 77–386 | 100% | 23% | 2/5 |

**1845 per-residue tests** in total across 5 order parameters.
