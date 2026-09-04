# Correction — significance testing on trajectories, releases up to 2.3.2

**Dated 2026-09-04. Affects every release up to and including 2.3.2. Fixed in
2.4.0.**

If you compared molecular dynamics trajectories with Prothon 2.3.2 or earlier
and relied on the per-residue significance calls, the noise floor, precision and
recall, MMD, C2ST, or the benchmark ranking, please read this. The
dissimilarity values themselves are unaffected.

## What was wrong

Prothon subsamples before building the permutation null, and the default sample
size is 1000 frames. Any trajectory longer than that was subsampled first, and
the subsample was drawn in **random order**.

The block permutation then relabelled contiguous runs of that shuffled array —
runs of frames that were no longer adjacent in time. Blocks are the unit of
exchangeability precisely because neighbouring frames are correlated. A block
assembled from frames scattered across a trajectory carries no such
correlation, so the block null degenerated into a frame null and inherited a
frame null's error rate.

Measured on two independent stationary trajectories drawn from the **same**
distribution, where every rejection is by construction a false positive:

| | features called different |
|---|---|
| default path, 2000 frames subsampled to 1000 | **113/120 (94.2%)** |
| all 2000 frames, kept in time order | 2/120 (1.7%) |

The published figure of 1.7–2.3% was the second row. It was correct for the
configuration it described and that configuration is not the default.

## Why the tests did not catch it

Every study in `scripts/calibration.py` set the sample size equal to the frame
count, so the subsampling branch never ran. The validation measured a path
users do not take, and the number it produced was published in the README, the
documentation, the release notes and a manuscript as the software's headline
guarantee.

`scripts/calibration.py --study default_path` now varies the sample size below
the frame count. It is the measurement that would have caught this.

## What changes

Affected, on trajectories longer than the sample size:

- per-residue significance calls and the count of differing residues
- `resolved`, `within_floor` and the noise floor
- precision and recall flags
- MMD and C2ST verdicts
- benchmark ranking, which depends on floors

Not affected:

- dissimilarity values themselves, per residue and global
- density estimation and the distances
- anything computed with `sample_size` greater than or equal to the frame count

The last point is why the ubiquitin analysis distributed with this project is
unaffected: `scripts/ubiquitin_rerun.py` sets the sample size to the full frame
count.

## What to do

1. **Upgrade to 2.4.0 or later.**
2. **Re-run any comparison of trajectories longer than your sample size** where
   a per-residue call, a floor or a ranking mattered. Dissimilarity magnitudes
   do not need re-running.
3. If you reported a count of significantly different residues from an affected
   run, re-derive it before citing it.

To check whether a specific past result was affected, compare the trajectory
length against the `sample_size` used. Equal or smaller means the subsampling
branch never ran and the result stands.

## Provenance

Found by an independent source and executable audit of commit `a76e35c`. The
audit reproduced the failure deterministically before any code was changed, and
those reproductions are committed as tests rather than described.
