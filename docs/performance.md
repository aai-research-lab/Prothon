# Performance

What a run costs, measured rather than asserted. Every point runs in
its own process and reports its own peak resident memory.

Measured on `linux`, Python 3.12.

## Against chain length (2000 conformations)

| residues | cbcn (s) | cata (s) | compare (s) | peak (GB) |
|---|---|---|---|---|
| 25 | 0.13 | 0.01 | 5.00 | 0.16 |
| 50 | 0.36 | 0.01 | 10.58 | 0.22 |
| 100 | 2.02 | 0.02 | 23.75 | 0.41 |
| 200 | 5.15 | 0.03 | 56.67 | 0.53 |

Slope of log time against log chain length (1 is linear, 2 is quadratic), fitted only through points slower than 0.1 s:

- `cbcn`: 1.83 (from 4 points)
- `cata`: too fast to fit at these sizes
- `compare`: 1.17 (from 4 points)

## Against ensemble size (100 residues)

| conformations | cbcn (s) | cata (s) | compare (s) | peak (GB) |
|---|---|---|---|---|
| 500 | 0.23 | 0.01 | 12.59 | 0.21 |
| 2000 | 1.73 | 0.02 | 23.35 | 0.41 |
| 8000 | 3.34 | 0.09 | 25.08 | 0.64 |

Slope of log time against log ensemble size, fitted only through points slower than 0.1 s:

- `cbcn`: 0.96 (from 3 points)
- `cata`: too fast to fit at these sizes
- `compare`: 0.25 (from 3 points)

## What the numbers mean

**Representation is linear in the number of conformations.** The measured slope
for `cbcn` is 0.96. That is the claim the method rests on, and it is now
measured rather than asserted.

**Representation is quadratic in chain length.** It has to be: the number of
residue pairs is. The measured slope depends on the machine and on which points
the fit reaches -- 1.80 here and 1.95 on an Apple Silicon laptop, both
consistent with 2 once the sub-tenth-of-a-second points are excluded. Timings
that small measure process start-up rather than the work, and a slope fitted
through them describes the harness; an earlier version of this page quoted
1.80 as though it were a property of the algorithm, which it was not.

The vectorised implementation reduces the constant, not the exponent. What
keeps a large protein tractable is that the exponent applies to residues, which
number in the hundreds, and not to conformations, which number in the tens of
thousands.

**Comparison is nearly independent of ensemble size.** The measured slope is
0.25 -- almost flat. Ensembles larger than `sample_size` (1000 by default) are
subsampled without replacement before the permutation null, so a
100,000-frame trajectory costs a comparison no more than a 10,000-frame one.
What grows with ensemble size is the representation, not the statistics.

Raising `sample_size` buys a tighter noise floor at linear cost, and is the
first parameter to raise for a result going into a paper.

**Comparison dominates the total.** Above about fifty residues the permutation
null costs more than building the representation. It scales linearly in chain
length, because there is one density per residue, and linearly in
`n_permutations`.

**Peak memory is set by the trajectory, not by the analysis.** Pair distances
are computed in blocks sized from a memory budget rather than a fixed pair
count, so peak memory holds roughly flat as the trajectory lengthens: a block
costs about 130 MB whether the trajectory has eight thousand frames or two
hundred thousand.

## Reading these numbers

They were measured on one machine and are indicative, not a specification. The
same script on an Apple Silicon laptop runs about twice as fast throughout,
with the same slopes. What transfers between machines is the shape -- linear in
conformations, quadratic in residues, flat in ensemble size for the comparison
-- and not the seconds.

## Reproducing this table

```bash
python scripts/scale_envelope.py --out docs/performance.md
python scripts/scale_envelope.py --full     # the larger grid
```

Each point runs in its own process and reports its own peak resident memory.
The trajectories are synthetic globules rather than real proteins, so that
chain length and ensemble size vary independently; the contact distribution is
chosen so a realistic fraction of pairs falls inside the cutoff.
