# Performance

What a run costs, measured rather than asserted. Every point runs in its own
process and reports its own peak resident memory.

Measured on a Linux workstation, Python 3.11.

## Against chain length (2000 conformations)

| residues | cbcn (s) | cata (s) | sasa (s) | compare (s) | peak (GB) |
|---|---|---|---|---|---|
| 25 | 0.03 | 0.01 | 0.16 | 3.60 | 0.16 |
| 50 | 0.07 | 0.01 | 0.34 | 7.59 | 0.22 |
| 100 | 0.26 | 0.01 | 0.74 | 17.17 | 0.41 |
| 200 | 1.09 | 0.02 | 1.64 | 40.61 | 0.53 |
| 400 | 4.38 | 0.03 | 3.70 | 91.42 | 0.56 |

Slope of log time against log chain length (1 is linear, 2 is quadratic),
fitted only through points slower than 0.1 s:

- `cbcn`: 2.03 (from 3 points)
- `cata`: too fast to fit at these sizes
- `sasa`: 1.14 (from 5 points)
- `compare`: 1.18 (from 5 points)

## Against ensemble size (100 residues)

| conformations | cbcn (s) | cata (s) | sasa (s) | compare (s) | peak (GB) |
|---|---|---|---|---|---|
| 500 | 0.07 | 0.01 | 0.19 | 8.83 | 0.21 |
| 2000 | 0.26 | 0.01 | 0.75 | 17.24 | 0.41 |
| 10000 | 1.29 | 0.04 | 3.69 | 18.66 | 0.65 |
| 50000 | 6.85 | 0.29 | 18.34 | 24.62 | 1.25 |

Slope of log time against log ensemble size, fitted only through points slower
than 0.1 s:

- `cbcn`: 1.01 (from 3 points)
- `cata`: too fast to fit at these sizes
- `sasa`: 0.99 (from 4 points)
- `compare`: 0.20 (from 4 points)

## What the numbers mean

**Representation is linear in the number of conformations.** The measured
slopes are 1.01 for `cbcn` and 0.99 for `sasa`. That is the claim the method
rests on, and it is measured rather than asserted.

**Representation is quadratic in chain length**, and it has to be: the number
of residue pairs is. `cbcn` measures 2.03. The vectorised implementation
reduces the constant, not the exponent. What keeps a large protein tractable is
that the exponent applies to residues, which number in the hundreds, and not to
conformations, which number in the tens of thousands.

`sasa` measures 1.14 because Shrake–Rupley is linear in atoms with a
neighbour-list cutoff, rather than over all pairs.

**A comparison is nearly independent of ensemble size.** The measured slope is
0.20. Fifty thousand conformations cost 24.6 s against 8.8 s for five hundred:
a hundredfold larger ensemble for under three times the time. Ensembles larger
than `sample_size` (1000 by default) are subsampled without replacement before
the permutation null, so what grows with ensemble size is the representation,
not the statistics.

Raising `sample_size` buys a tighter noise floor at linear cost, and is the
first parameter to raise for a result going into a paper.

**A comparison dominates the total.** Above about fifty residues the
permutation null costs more than building the representation — 91 s against 4 s
at 400 residues. It grows linearly in chain length, because there is one
density per residue, and linearly in `n_permutations`.

**Peak memory is set by the trajectory, not by the analysis.** Fifty thousand
conformations of a hundred residues peak at 1.25 GB, most of which is the
trajectory itself. Pair distances are computed in blocks sized from a memory
budget rather than a fixed pair count, so the analysis adds roughly 130 MB
whether the trajectory has eight thousand frames or two hundred thousand.

## What this costs for a real study

Two 500-residue proteins, 50,000 conformations each, one measure: a few minutes
of representation and a few minutes of comparison, inside 2 GB. A study of a
dozen ensembles across three measures is an afternoon on one core, and the
ensembles are independent, so it parallelises trivially.

## Reading these numbers

They were measured on one machine and are indicative, not a specification. The
same script on an Apple Silicon laptop runs roughly twice as fast throughout,
with the same slopes. What transfers between machines is the shape — linear in
conformations, quadratic in residues, nearly flat in ensemble size for a
comparison — and not the seconds.

Timings below a tenth of a second are excluded from the fits, because they
measure process start-up rather than the work. That is why `cata` reports no
slope: virtual torsions over a hundred residues take a hundredth of a second,
and nothing in this grid can measure their scaling.

## Regression budgets

The automated quick envelope gates properties that transfer between machines,
not absolute seconds. Every measurement must finish. The fitted ensemble-size
exponents for `cbcn`, local comparison, block-aware MMD and grouped C2ST must
be at most 1.30. Their chain-length ceilings are 2.50 for the pairwise contact
representation and 1.60 for each comparison method. Peak resident memory at
the historical 8,000-frame regression point must remain at or below 1.0 GB.
The full envelope also limits the `sasa` ensemble-size exponent to 1.30.

These deliberately loose ceilings catch the old `cbcn` regression (1.38 and
1.24 GB at 8,000 frames) without treating runner speed as a scientific
constant. Measurements, fits, versions, commit, platform and every gate result
are retained in JSON even when the workflow fails.

## Reproducing this table

```bash
python scripts/scale_envelope.py \
  --out docs/performance.md \
  --json performance-evidence.json
python scripts/scale_envelope.py --full     # the grid above
```

Each point runs in its own process and reports its own peak resident memory.
The trajectories are synthetic globules rather than real proteins, so that
chain length and ensemble size vary independently; the contact distribution is
chosen so a realistic fraction of pairs falls inside the cutoff.
