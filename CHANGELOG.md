# Changelog

All notable changes to Prothon are recorded here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — ingest and reconciliation (towards 3.0)

- **`prothon.ingest.Ensemble`**: a set of conformations with per-frame weights,
  provenance and a quality record. Loads from a trajectory, a set of
  trajectories joined as one ensemble, a multi-model PDB, or a directory of
  single-model PDBs — how generative models and structure predictors emit
  ensembles. Weights are validated and normalised; negative weights are refused
  rather than normalised, since they usually mean log-weights that were never
  exponentiated.
- **`prothon.ingest.reconcile`**: the residue correspondence between two
  ensembles that are *not the same molecule*. Comparison across differing
  sequences — a mutant against its wild type, a construct against a longer one,
  coarse-grained against all-atom — is possible because the representation is
  local and needs no common coordinate frame. Methods built on superposition
  cannot ask these questions at all.
- **`Correspondence.columns_for`**: representation columns are derived from the
  residue map rather than assumed. A column is not always a residue. Glycine
  has no C-beta, so a mutation to glycine removes one `cbcn` column and
  renumbers every column after it; comparing column *k* to column *k* would
  compare different residues from the mutation onward and report a difference
  along the whole C-terminal half. `caba` and `cata` are windows of three and
  four consecutive alpha carbons, so a column survives only where the whole
  window has a counterpart *and* those counterparts are still consecutive.
- **Sequence extraction that survives a force field.** `mdtraj` returns `None`
  for AMBER's `HIE`, `HIP` and `CYX`, so a sequence read from an
  AMBER-prepared topology comes back with holes and every alignment built on it
  is wrong. Protonation- and force-field-specific names are resolved
  explicitly, and a residue carrying N, CA and C is treated as part of the
  chain whatever it is called — which catches modified residues that would
  otherwise be dropped, shifting every index after them.
- **Needleman–Wunsch alignment with Gotoh affine gaps and BLOSUM62**, written
  in-package rather than adding a dependency. End gaps are free by default: the
  common case is two constructs differing by a terminal overhang, and charging
  for it pushes the aligner to absorb the overhang into interior gaps.

- **`Prothon.from_ensembles`**: a study over `Ensemble` objects rather than
  filenames. Where the molecules differ, the residue correspondence is worked
  out and each representation reduced to the columns that describe the same
  positions — so a mutant, a truncated construct or a coarse-grained model can
  be compared against a reference. A study built from filenames shares one
  topology and is unchanged.
- **Per-residue results are indexed by the reference ensemble's numbering.**
  After reconciliation the surviving columns are a subset, and numbering them
  1..n would put the label of one residue under the value of another — a
  figure that looks entirely reasonable and is wrong. `ComparisonResult`
  carries `feature_index`, and the figures use it.
- The manifest records what was reconciled: aligned count, identity, coverage,
  substitutions named as a paper would name them, unmatched residues on each
  side, and the alignment itself.

### Refusals

- **Coverage, not just identity.** Free end gaps make the aligner behave
  locally when sequences are unrelated: two unrelated 40-residue sequences
  align on *two* columns at 50% identity, clearing any identity floor while
  covering a twentieth of the molecule. Reconciliation therefore requires half
  the shorter sequence to align, and refuses below it.
- Sequences below 25% identity — the twilight zone, where an alignment stops
  being evidence that positions correspond.
- Ensembles whose protein chain counts differ. Concatenating the chains of a
  complex lets the aligner slide one chain against another, which is cheap in
  score and nonsense as a map.
- Conformations with differing atom counts within one ensemble.
- A measure whose windows no difference between the molecules leaves intact —
  `caba` and `cata` span three and four consecutive alpha carbons, and a
  deletion can break every one of them. The error says which per-residue
  measure to reach for instead.

### Added — weighted ensembles

- **Per-frame weights reach the estimator.** Gaussian and von Mises kernels
  both take them, so a deposited ensemble's conformer probabilities and a
  reweighted simulation's frame weights are used rather than recorded and
  ignored. The permutation null carries each frame's weight with the frame,
  because a conformation and its probability are one observation; the
  split-half floor renormalises within each half.
- **Effective sample size (Kish).** A thousand frames in which one conformer
  carries half the probability is worth four independent samples. Sizing a
  noise floor by the frame count instead produces error bars for an ensemble
  nobody sampled — the same failure as the bootstrap null this release
  replaced: a quantity that looks like a sample size, is smaller than it
  appears, and makes everything downstream look more certain than it is.
  Reported on every result and recorded in the manifest.
- Comparisons are **refused below ten effective samples** and warned below
  fifty, on the effective count rather than the frame count.
- Comparing a weighted ensemble against an unweighted one warns. Treating the
  second as uniform is the only thing available, but it is an assumption about
  that ensemble rather than a fact about it, and it is the usual shape of a
  deposited ensemble compared against a simulation.

## [2.1.0] — 2026-08-10

A correctness and packaging release. The public API is unchanged: code written
against 2.0 runs without modification. **Numerical results will differ**, and
the reasons are set out below — the significance test in 2.0 was not sound.

### Fixed — the significance test

Version 2.0 built its null distribution by drawing two bootstrap resamples from
the *same* ensemble and measuring the Jensen–Shannon distance between them.
Two resamples of *n* frames drawn with replacement from the same *n* frames
share roughly 63% of their points, so they resemble each other far more closely
than two independent samples of the same size. The null was therefore too tight
by about a factor of two, and any honest between-ensemble distance cleared it.

Measured on a 400-frame Gaussian ensemble:

| quantity | value |
|---|---|
| bootstrap null used by 2.0 | 0.046 |
| two independent samples, same distribution | 0.097 |
| observed between-ensemble distance | 0.090 |

The consequence, over 40 null replicates in which both ensembles were drawn
from an identical distribution:

| | features called different | studies with ≥1 false positive |
|---|---|---|
| 2.0 bootstrap null | **100%** | **100%** |
| 2.1 permutation null | 1.2% | 7.5% |

Version 2.0 reported two independent samples of the same distribution as
differing significantly at every residue.

The replacement is a permutation test: the frames of both ensembles are pooled
and relabelled at random into two groups of the original sizes, which gives the
exact distribution of the statistic under the hypothesis that the ensembles are
the same. Per-feature values are standardised and pooled before correction, so
100 relabellings give p-value resolution fine enough to survive a
false-discovery-rate correction over several hundred residues.

`legacy=True` (CLI: `--legacy-statistics`) reproduces 2.0's behaviour exactly,
for regenerating published figures. It is documented as unsound.

### Fixed — everything else

- **Per-residue significance.** 2.0 computed one pooled p-value and wrote
  `local_diss[p_value >= 0.05] = 0.0`. With a scalar `p_value`, NumPy reads
  that as a mask over the whole array, so a single test decided the fate of
  every residue at once. Each feature is now tested separately, and the
  resulting p-values are Benjamini–Hochberg corrected — a 300-residue protein
  tested at α = 0.05 yields fifteen false positives by construction.
- **Circular densities for torsions.** `cata` values wrap at ±π. 2.0 estimated
  them with a Gaussian kernel on a linear grid, which splits a population
  straddling the wraparound across both ends and puts a false trough between
  them. Circular measures now use a von Mises kernel with Taylor's plug-in
  bandwidth on a grid spanning a full turn. Each measure declares whether it is
  circular, so the call site cannot forget.
- **`scikit-learn` is now declared as a dependency.** 2.0 imported it for PCA,
  MDS and t-SNE without listing it, so a clean `pip install` failed on any run
  that reached dimensionality reduction — which was the CLI default.
- **Constant features no longer crash the run.** A buried residue with zero
  SASA in every frame gave `gaussian_kde` a singular covariance matrix and took
  down the whole study. Such columns now get a degenerate density.
- **Float32 overflow in the contact sigmoid.** `mdtraj` returns float32
  distances; `exp()` overflows above ~88 in float32 while the clip allowed 700.
  The result was right by accident, via `1/(1+inf) → 0`, and noisy with
  warnings. Distances are promoted to float64 first.
- **Replotting no longer overwrites saved figures**, and the documented
  `xlabel`, `ylabel`, `title` and `color` arguments now take effect. In 2.0
  `replot_global_dissimilarity` accepted them, discarded them, and re-saved
  over the original file.
- **Reproducibility.** Resampling drew from the global NumPy state, so two runs
  of one study gave different p-values with nothing recording why. `Prothon`
  now takes `random_state`.
- **Empty atom selections are named.** A coarse-grained model with no C-beta
  atoms produced an inscrutable NumPy error several frames down the stack.

### Added

- **Noise floor on every result.** The distance between two disjoint halves of
  a single ensemble is the smallest difference the sampling can resolve. It is
  reported alongside every comparison, drawn on every dissimilarity figure, and
  `ComparisonResult.resolved` says plainly whether the measurement clears it.
- **`manifest.json` per measure**, recording inputs, parameters, seed, Prothon
  version and full results — so a run can be reproduced and not merely admired.
- **`ComparisonResult`**, a typed result object that still supports dictionary
  access (`result["global_dissimilarity"]`) so 2.0 code keeps working. Carries
  the unmasked per-residue values, per-residue p-values, the significance mask
  and the noise floor.
- **`Prothon.summary()`** and a readable default CLI output. `--json` restores
  the 2.0 behaviour of dumping everything to stdout.
- `prothon --info`, listing measures and detected backends.
- Backward-compatibility shim: `from Prothon import Prothon` still works,
  with a `DeprecationWarning`. Removed in 3.0. Shipped as a single module
  `src/Prothon.py`, not a package — a directory named `Prothon` beside one
  named `prothon` is a single path on macOS and Windows.
- 81 tests, a GitHub Actions matrix across operating systems and Python
  versions, and `ruff` linting.

### Changed

- **Distribution name is `prothon-ensembles`; the import name and the command
  are both `prothon`.** PyPI's `prothon` was registered in 2020 by an unrelated
  protobuf generator and names there are permanent. conda-forge, where the name
  is free, gets `prothon`. Version 2.0 was never published to either index, so
  nothing that already works breaks.
- Releases publish to PyPI from a tag via trusted publishing (OpenID Connect),
  so no long-lived API token exists to leak.
- **`src/` layout and `pyproject.toml`**, replacing `setup.py`. Versioning via
  `setuptools-scm`.
- **Dimensionality reduction is off by default.** It defaulted to
  `pca,mds,tsne`; MDS builds a dense frame-by-frame distance matrix, so on a
  real trajectory the default turned a short comparison into an out-of-memory
  failure. MDS is now refused above 5,000 frames with a message naming the
  memory it would need, and a refusal no longer discards the comparison that
  already succeeded.
- **Contact numbers are computed once rather than once per atom.** 2.0 rebuilt
  the pair list in Python for every atom and recomputed the same distances.
  Identical output to float32 precision:

  | residues | 2.0 | 2.1 | speedup |
  |---|---|---|---|
  | 50 | 0.21 s | 0.05 s | 4× |
  | 100 | 1.37 s | 0.48 s | 3× |
  | 200 | 10.66 s | 1.10 s | 10× |
  | 300 | 36.03 s | 2.22 s | 16× |

  Pairs are processed in blocks, so a long trajectory of a large protein no
  longer needs the whole distance matrix resident.
- Progress reporting moved from `print` to the `logging` module, so an
  embedding program can silence or redirect it.
- Ensembles with fewer than 50 frames now emit a warning that the noise floor
  understates the true uncertainty.

### Licence

Relicensed from GPL-3.0 to **MIT**, matching the rest of the AAI Research Lab
tooling and removing an adoption barrier: a number of industrial groups have
blanket policies against GPL dependencies, and they are a large part of the
audience for ensemble comparison.

The change is not retroactive. Versions up to and including 2.0.0 were
distributed under GPL-3.0 and copies obtained under it stay governed by it.
Nothing in the dependency stack required copyleft — MDTraj is LGPL, and NumPy,
SciPy, Matplotlib and scikit-learn are BSD.

### Known limitations

- The permutation null assumes frames are exchangeable. Frames from a single
  continuous MD trajectory are correlated in time, so an ensemble holds fewer
  independent conformations than it has frames and the p-values remain somewhat
  optimistic. A block permutation over the correlation time is planned for 3.0.
  The split-half noise floor is measured rather than assumed and is the more
  trustworthy guide.
- Ensembles must share a topology. Comparison across differing sequences —
  wild type against mutant, ortholog against ortholog — is planned for 3.0.

## [2.0.0] — 2025-04-23

- Restructured the single-module version 1 into a package.
- Added dimensionality reduction (PCA, MDS, t-SNE), matrix heatmaps, combined
  local dissimilarity plots and a replotting API.

## [1.0.1] — 2023

- Original release accompanying Aina, Hsueh & Plotkin, *J. Chem. Inf. Model.*
  **2023**, 63 (11), 3453–3461. Preserved under `legacy/`.
