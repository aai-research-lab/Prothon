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

A `v*` tag invokes the same reusable quality workflow as branches and pull
requests. The publisher cannot build or obtain its PyPI OIDC token unless the
complete Python 3.9–3.13 matrix on Linux, macOS and Windows, the documentation
build, and the real-structure tests all pass at that exact tagged commit.
The wheel and source distribution are then built once and retained as one
artifact. Separate jobs install each into a clean virtual environment without
checking out the source tree, run `pip check`, import the installed package and
exercise the installed CLI. PyPI receives those same files only after both
artifact jobs pass.

All third-party actions are pinned to immutable commit SHAs. The trailing
version comment is documentary, not executable. Dependabot proposes grouped
updates each week; review the upstream release notes and retain a full 40-digit
SHA when accepting one. Workflow tokens are read-only unless a job declares a
narrower exception (the publisher receives only `id-token: write`).

The reusable workflow also creates a `security-evidence` artifact. It contains
the resolved runtime environment, scanner versions, a tracked-source secret
scan, and a CycloneDX JSON SBOM with vulnerability data. A possible secret, any
known dependency vulnerability, or an incomplete dependency audit blocks the
quality workflow and therefore blocks publication. Do not weaken or suppress a
finding to make a release pass: resolve it, or document and review a narrowly
identified exception in a separate pull request.

To reproduce the two scans locally after installing `requirements/security.txt`:

```bash
detect-secrets scan --no-verify
pip-audit --strict --progress-spinner off
```

After PyPI publishes a release, update the conda-forge feedstock to the exact
sdist version and SHA-256, then copy that reviewed recipe back to
`recipes/prothon/recipe.yaml`. The scheduled and post-publish conda-sync
workflow compares the entire reference recipe with the live feedstock and
checks its version and digest against PyPI:

```bash
python scripts/check_conda_sync.py
```

The check is necessarily post-publish because the final PyPI sdist hash does
not exist beforehand. A failure means the conda release is incomplete; it is
not a reason to disable the monitor.

Two things in `scripts/` are measurements rather than tests, and both have
found defects that the test suite could not:

```bash
python scripts/scale_envelope.py --full     # time and memory across sizes
python scripts/calibration.py --replicates 1000
python scripts/floor_scaling.py             # how the floor depends on n
```

The monthly `Scientific calibration evidence` workflow runs the correlated
local null over 250 replicates per predeclared setting and runs the complete
MMD/C2ST null-and-power harness. Both commands return nonzero when a gate fails,
and the workflow retains their JSON records even on failure. The local record
includes raw counts, seed ranges, Wilson intervals, the commit and dependency
versions; explicitly forced IID permutation of correlated data is labelled as
a negative control and cannot make the corrected gate pass.

The scale envelope found a memory bug that needed 8,000 frames to appear; the
calibration harness found a metric that was never reaching the estimator, which
showed up only as three metrics agreeing to five decimal places over eight
thousand features. Neither is reachable from a suite that runs on fourteen
residues and a few hundred frames, and neither is expensive enough to justify
skipping.

## Style

`ruff check src tests`. Line length 100.
