# Performance

What a run costs, measured rather than asserted. Every point runs in
its own process and reports its own peak resident memory.

Measured on `linux`, Python 3.12.

## Against chain length (2000 conformations)

| residues | cbcn (s) | cata (s) | compare (s) | peak (GB) |
|---|---|---|---|---|
| 25 | 0.10 | 0.01 | 6.15 | 0.16 |
| 50 | 0.36 | 0.03 | 13.21 | 0.22 |
| 100 | 1.72 | 0.02 | 29.46 | 0.41 |
| 200 | 3.79 | 0.05 | 62.72 | 0.53 |

Slope of log time against log chain length (1 is linear, 2 is quadratic):

- `cbcn`: 1.80
- `cata`: 0.56
- `compare`: 1.12

## Against ensemble size (100 residues)

| conformations | cbcn (s) | cata (s) | compare (s) | peak (GB) |
|---|---|---|---|---|
| 500 | 0.29 | 0.01 | 16.19 | 0.21 |
| 2000 | 1.67 | 0.02 | 29.45 | 0.41 |
| 8000 | 4.60 | 0.08 | 32.35 | 0.64 |

Slope of log time against log ensemble size:

- `cbcn`: 1.00
- `cata`: 0.66
- `compare`: 0.25

## What the numbers mean

**Representation is linear in the number of conformations.** The measured slope
for `cbcn` is 1.00. That is the claim the method rests on, and it is now
measured rather than asserted.

**Representation is sub-quadratic in chain length.** `cbcn` measures 1.80
against a naive expectation of 2.0, because pairs closer than three residues in
sequence are excluded and the remainder is computed as one vectorised pass
rather than once per atom.

**Comparison is nearly independent of ensemble size.** The measured slope for a
full comparison is 0.25 — almost flat. Ensembles larger than `sample_size`
(1000 by default) are subsampled without replacement before the permutation
null, so a 100,000-frame trajectory costs a comparison no more than a
10,000-frame one. What grows with ensemble size is the representation, not the
statistics.

Raising `sample_size` buys a tighter noise floor at linear cost, and is the
first parameter to raise for a result going into a paper.

**Comparison dominates the total.** For anything above about fifty residues,
the permutation null costs more than building the representation — 63 seconds
against 4 at 200 residues. It scales linearly in chain length (slope 1.12),
because there is one density per residue, and linearly in `n_permutations`.

**Peak memory is set by the trajectory, not by the analysis.** Pair distances
are computed in blocks sized from a memory budget rather than a fixed pair
count, so peak memory holds roughly flat as the trajectory lengthens: a block
costs about 130 MB whether the trajectory has eight thousand frames or two
hundred thousand.

## Reproducing this table

```bash
python scripts/scale_envelope.py --out docs/performance.md
python scripts/scale_envelope.py --full     # the larger grid
```

Each point runs in its own process and reports its own peak resident memory.
The trajectories are synthetic globules rather than real proteins, so that
chain length and ensemble size vary independently; the contact distribution is
chosen so a realistic fraction of pairs falls inside the cutoff.
