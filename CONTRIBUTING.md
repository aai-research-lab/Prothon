# Contributing to Prothon

Contributions are welcome — bug reports, new order parameters, better
estimators, documentation.

## Getting set up

```bash
git clone https://github.com/aai-research-lab/Prothon.git
cd Prothon
pip install -e ".[dev]"
pytest
```

## What a change needs

**A test.** Especially for a bug fix: the test should fail before the change
and pass after it, and be named so that a future regression says which
behaviour came back.

**A changelog entry**, under `[Unreleased]` in `CHANGELOG.md`. If a change
alters numerical results, say so plainly and say why the new numbers are the
right ones.

**Comments that explain why, not what.** The code says what it does. What a
reader cannot recover is why the bandwidth is chosen that way, or why a null
distribution is built from permutations rather than bootstrap resamples. That
reasoning belongs in the source.

## Adding an order parameter

1. Write `compute_<name>(traj)` in `core/representation.py`, returning an
   `(n_frames, n_features)` array.
2. Register it in `MEASURES`, declaring `circular` and `per_residue`. Getting
   `circular` wrong is silent and wrong, not loud and wrong: a linear kernel on
   circular data produces plausible numbers that understate dissimilarity at
   the wraparound.
3. Add it to `_COMPUTE`. A test checks the two stay in step.
4. Test the shape, the value range, and one case where you know the answer.

## Statistics

Changes to `core/dissimilarity.py` need a calibration check as well as a unit
test: a null case where both ensembles are drawn from the same distribution,
asserting that the false-positive rate stays near the nominal level. There is
one in `tests/test_dissimilarity.py` to copy. This is not optional — the bug
that motivated the 2.1 release passed every unit test in 2.0 and failed exactly
this check.

## Editing by text match

Several changes in this project's history have been applied by matching a
string and replacing it. When the string has moved, the match fails, the
replacement writes nothing, and the operation reports success. It has happened
to the README three times and to a test file once, and in every case the loss
was silent.

If you edit that way, assert the anchor exists before writing and assert the
result contains what you added. The test count is the cheapest check available:
if a commit claims to add tests, `pytest --collect-only` should report more
than it did before.

## Scripts

Everything in `scripts/` carries a `#!/usr/bin/env python3` line and is
executable, so it runs either way:

```bash
scripts/calibration.py --help
python scripts/calibration.py --help
```

## Before a release

Two things in `scripts/` are measurements rather than tests, and both have
found defects that the test suite could not:

```bash
python scripts/scale_envelope.py --full     # time and memory across sizes
python scripts/calibration.py --replicates 1000
python scripts/floor_scaling.py             # how the floor depends on n
```

The scale envelope found a memory bug that needed 8,000 frames to appear; the
calibration harness found a metric that was never reaching the estimator, which
showed up only as three metrics agreeing to five decimal places over eight
thousand features. Neither is reachable from a suite that runs on fourteen
residues and a few hundred frames, and neither is expensive enough to justify
skipping.

## Style

`ruff check src tests`. Line length 100.
