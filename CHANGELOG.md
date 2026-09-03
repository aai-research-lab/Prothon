# Changelog

All notable changes to Prothon are recorded here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — one sampling-aware floor across every analysis

- Precision/recall and experimental validation now use the same split-half
  engine as dissimilarity. Trajectories split complete temporal blocks,
  explicitly IID structures split random frames, and supplied replica labels
  split complete independent replicas.
- Both APIs record the full floor distribution, its decision quantile, the
  correlation plan, block length and independent-unit count. Precision/recall
  calls use a conservative lower-tail threshold; experimental agreement uses
  the upper 95th percentile rather than the mean.
- Fewer than eight independent floor units now withhold missed/invented and
  within-floor verdicts instead of silently classifying under-sampled data.
- Contradictory sampling metadata, such as `sampling_kind="iid"` with a
  correlation time greater than one frame, is rejected.

### Fixed — correlation-aware noise floors

- Split-half floors now assign complete temporal blocks to each half instead
  of randomly interleaving time-correlated frames. Independent replica labels
  are also accepted and make complete replicas the indivisible units.
- The global resolved/unresolved verdict now uses the 95th percentile of the
  measured floor distribution rather than treating its mean as a decision
  threshold. The mean remains available as `noise_floor` for compatibility,
  while the threshold and full distribution are recorded explicitly.
- When fewer than eight independent blocks are available, the floor remains a
  descriptive measurement but no resolved/unresolved verdict is made.

### Fixed — indivisible permutation blocks

- Block permutation now assigns complete blocks to the relabelled ensembles.
  Previously it shuffled complete blocks, concatenated them, and then cut the
  concatenation at the original frame count; whenever an unequal trailing
  block crossed that cut, a supposedly indivisible block was split between
  both ensembles.
- Relabelling preserves the original number of blocks in each ensemble. Frame
  counts may differ when blocks have unequal lengths, which is the valid block
  design and keeps every correlated unit intact.
- Serial and parallel permutation paths now call the same implementation and
  are regression-tested to return identical results from the same seed.

### Fixed — trajectory subsampling and block accounting

- A trajectory longer than the default 1000-frame comparison sample is now
  reduced to a contiguous window whenever block permutation is active. The old
  random row order destroyed temporal correlation immediately before applying
  a correction that depends on that order, restoring an anti-conservative
  frame-permutation null under a different name.
- Block length, independent-block count and the too-few-blocks refusal are now
  computed from the frames that actually enter the test. A 5000-frame
  trajectory sampled to 1000 frames no longer claims blocks from the 4000
  frames that were not tested.
- Comparison metadata records the number of sampled frames and whether all
  frames, a contiguous trajectory window, uniform IID sampling, or the legacy
  bootstrap supplied the calculation.

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

### Added — the metric layer

- **Three per-feature distances**, selected with `metric=` or `--metric`:
  `jsd` (default, bounded), `wasserstein` and `ks`. The permutation null, the
  false-discovery correction and the noise floor are all computed under
  whichever is chosen, so a Wasserstein comparison gets a Wasserstein floor
  rather than a threshold borrowed from another scale.
- **Wasserstein-1 reports in the feature's own units** — contacts, radians,
  square nanometres. "This residue gains 1.4 contacts" is a sentence about the
  protein; a Jensen-Shannon distance is a sentence about the comparison. It
  also needs no grid and no bandwidth, so it carries none of the density
  estimate's bias. The cost is that it is unbounded and not comparable across
  measures.
- **Circular features are measured the short way round.** Two tight torsion
  populations either side of the wraparound are 0.28 radians apart; a linear
  Wasserstein distance reports 4.43, twenty-one times too large, without
  complaint. Circular features use Delon's circular optimal-transport
  distance.
- **Kuiper's statistic replaces Kolmogorov–Smirnov on circular features.** The
  KS statistic is the largest gap between two cumulative distributions, which
  on a circle depends on where the circle was cut rather than on the data:
  over 24 rotations of one interleaved pair, KS ranges from 0.25 to 0.50.
  Kuiper's does not move.
- KS is offered because PENSA reports it, so a claim that one method finds
  something another misses can be checked on the same statistic.

### Added — whole-ensemble comparison

- **`Prothon.distinguishability`**, with two methods that read the *joint*
  distribution rather than each feature separately. The per-residue metrics
  cannot see a difference that lives in the relationship between features: two
  loops that visit the same positions but no longer at the same time give an
  identical profile at every residue and are a different ensemble. On a
  constructed case with identical marginals, the largest per-residue
  Jensen-Shannon distance is 0.06 — correctly, since each feature really does
  have the same distribution — and both methods here find the difference.
- **Maximum mean discrepancy** with a permutation null. The squared MMD is a
  quadratic form in a signed weight vector, so a relabelling is a permutation
  of that vector rather than a rebuild of the kernel: 200 permutations over a
  thousand conformations a side take under a tenth of a second.
- **Classifier two-sample test**, reporting how separable the ensembles are
  and which features the classifier used — the interpretability MMD cannot
  offer. A random forest rather than a linear model, because two ensembles
  differing in *spread* rather than mean are not linearly separable and that
  is a difference anybody would want found. Scored out of fold, since a
  classifier scored on its own training data separates two halves of one
  ensemble perfectly.
- Classifier p-values below 1e-6 are reported as a bound. The asymptotic null
  is a normal approximation whose far tail is not literal, and the folds share
  training data so the predictions are not quite independent. A raw 1e-222 is
  arithmetic, not evidence; the area under the curve is the number to quote.
- Circular features are encoded as (cos, sin) for both methods. This matters
  measurably to MMD's statistic and much less to the classifier, since a tree
  splits at thresholds and can carve the wraparound out — and the permutation
  null absorbs part of the distortion in any case. Stated that way because it
  would be tidier, and wrong, to imply it is as severe as the earlier circular
  problems in this release.

### Fixed

- `pyproject.toml` no longer pins `fallback_version`. A literal there goes
  stale at the next tag and would let an sdist built without SCM metadata
  report a version it is not.

### Added — what a model misses, and what it invents

- **`Prothon.coverage_and_fidelity`** splits a difference into precision and
  recall. A model that never opens a cryptic pocket and one that opens pockets
  no physics produces are both wrong, score alike on any symmetric distance,
  and need opposite work. On a constructed pair the mean Jensen-Shannon
  distances are 0.26 and 0.21 — indistinguishable — while recall collapses for
  the first and precision for the second.
- **Per residue**, which is the part local order parameters make possible. The
  machine-learning formulations return two numbers for a whole sample; here
  the answer is not "recall 0.62" but "covers the fold, misses the 40–55 loop",
  which is a sentence a model developer can act on.
- **The support is a highest-density region**, so the null value is exact:
  under identical distributions both quantities equal the coverage level by
  construction, and a departure is readable against it.
- **A per-feature floor**, measured by splitting the reference in half. Two
  halves of one ensemble do not cover each other perfectly either. The floor
  has to be per feature and not averaged: measured on a test protein the four
  rigid residues floor at 0.956 and the four bimodal ones at 0.997, because a
  smoothed multimodal density has a wide highest-density region and a nominal
  95% level over-covers there. One averaged threshold flags about half the
  unchanged residues by construction — which it did, in the first version of
  this code, on residues that were identical in every ensemble.

### Documentation

- **Read the Docs**, built with Sphinx and MyST from `docs/`: installation, a
  first comparison, the measures, the metrics, the statistics, comparison
  across different molecules, the CLI reference, and an autodoc API page.
- **A statistics page** that states plainly what the software cannot resolve,
  why the null is built by permutation rather than by resampling an ensemble
  against itself, and what is still uncorrected — time correlation within a
  single trajectory.
- **The documentation build runs in CI**, with warnings treated as errors, so a
  dead cross-reference or a page missing from the table of contents fails on a
  pull request rather than after merging.
- Coverage is uploaded to Codecov from the Ubuntu 3.11 job.

### Fixed

- **The Python badge linked to the wrong package.** It pointed at
  `pypi.org/project/prothon`, which is an unrelated protobuf generator
  registered in 2020, not this project. Both the PyPI and Python badges now
  read from `prothon-ensembles` and are generated from the package metadata
  rather than hard-coded, so they cannot drift from `requires-python`.
- The README and documentation describe what Prothon does, with every method
  cited. A `references.md` page carries full citations, and a check confirms
  that every author named in the prose appears there.

### Added — validation

- **Real proteins, in real formats.** A network-marked suite runs every measure,
  metric and method against structures from the MDTraj corpus: a 20-model NMR
  ensemble, three single structures, an RNA hairpin, and 501 frames of dynamics
  written as XTC, DCD and NetCDF. It runs in its own CI job; the ordinary suite
  stays offline and fast.
- **Estimators checked against closed forms** rather than against themselves:
  the Gaussian and von Mises kernels against the densities they estimate,
  Wasserstein-1 against its exact value for shifted normals, shifted uniforms
  and nested uniforms, the Kolmogorov–Smirnov statistic against its exact value
  for shifted uniforms, Benjamini–Hochberg against SciPy's implementation, and
  the maximum mean discrepancy against its closed form for two Gaussians under
  a Gaussian kernel — which it matches to within a few percent.
- `maximum_mean_discrepancy` takes `bandwidth` and `standardise`, so the
  statistic can be checked against a case with a known value instead of only
  against its own null.

### Fixed

- **The kernel matrix no longer allocates a three-dimensional array.**
  Computing pairwise distances as `((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)`
  materialises an `(n, n, d)` intermediate: at the default thousand
  conformations a side that is 2.4 GB for a 76-residue protein and 9.6 GB for a
  300-residue one, so the process was killed rather than slowed. Expanding the
  square instead turns it into one matrix product — 9.6 GB becomes 90 MB. It
  was invisible until the software met a protein, because no test fixture has
  more than fourteen residues.
- A null built from a single relabelling returned a NaN standard deviation from
  a division by zero degrees of freedom. It returns zero.

### Fixed — contact-number memory

- **The pair block is sized from a memory budget, not a fixed pair count.**
  Each block of pair distances costs `pairs x frames`, so a block fixed at 4096
  pairs allocated 30 MB over a thousand frames and 6.5 GB over two hundred
  thousand — the chunking stopped chunking exactly when it began to matter.
  Measured at 8,000 frames the fix takes `cbcn` from 11.07 s to 3.72 s and peak
  memory from 1.24 GB to 0.64 GB, and the measured scaling in ensemble size
  from 1.38 to **1.00**. Peak memory now holds roughly flat as a trajectory
  lengthens.

### Added — a measured performance envelope

- **`scripts/scale_envelope.py`** and a `performance` documentation page giving
  time and peak memory across chain length and ensemble size, with fitted
  scaling exponents. Each point runs in its own process and reports its own
  peak resident memory, because the expensive allocations happen inside
  compiled extensions and a high-water mark in one process would attribute the
  largest allocation to everything after it.
- The page records what the documentation previously only asserted, measured
  to 400 residues and 50,000 conformations: representation is linear in
  conformations (1.01 for cbcn, 0.99 for sasa) and quadratic in chain length
  (2.03, as the pair count requires); a comparison is nearly independent of
  ensemble size (0.20 — a hundredfold larger ensemble for under three times the
  time, because ensembles are subsampled before the permutation null); and the
  comparison dominates the total above about fifty residues.

### Added — a measured calibration, and what it found

- **`scripts/calibration.py`** measures the false-positive rate under the null
  across metrics, thresholds, permutation counts, correlated features, and
  frames correlated in time. Every replicate draws both ensembles from the same
  distribution, so every rejection is a false positive.
- **The p-values are not trustworthy on a single continuous trajectory, and the
  size of the problem is now known.** With frames from an Ornstein–Uhlenbeck
  process at correlation time τ, against a nominal 5%:

  | τ (frames) | independent conformations in 2000 | features called different |
  |---|---|---|
  | 1 | 924 | 5.5% |
  | 2 | 490 | 23.7% |
  | 5 | 199 | 72.1% |
  | 10 | 100 | 93.7% |
  | 20 | 50 | **99.0%** |
  | 50 | 20 | **99.9%** |

  At a correlation time of twenty frames — unremarkable for a loop — essentially
  every residue is called different when nothing differs. The documentation
  previously described this as the p-values being "somewhat optimistic", which
  understated it by a wide margin.
- The statistics page now carries the measurement instead of the caveat, with
  three things a user can do today: subsample to the correlation time, use
  independent replicates, or read the floor rather than the p-value.
- **All three metrics are calibrated on exchangeable frames**, measured over
  1000 replicates each. The rate to compare against α is the probability of any
  rejection, which is what Benjamini–Hochberg controls under the complete null.
- **The default of 100 permutations is mildly anticonservative.** Averaged over
  metrics and thresholds, the observed rate is 1.39 times α at 100
  permutations and 1.18 times at 200 — roughly 6% where 5% is asked for. This
  is the discreteness of a permutation p-value, and the fix is to raise
  `n_permutations`, at linear cost.

### Fixed — the metric was not reaching the estimator

- **`dissimilarity` accepted `metric=`, recorded it, and ignored it.** The two
  calls that compute the observed statistic and its permutation null were made
  without the argument, so every comparison used the Jensen–Shannon default
  whatever was requested. `--metric wasserstein` and `--metric ks` produced
  Jensen–Shannon results.
- Every test passed over it, because they checked that the metadata carried the
  right label rather than that the number was different. Two tests now assert
  the behaviour: that no two metrics produce identical statistics on the same
  data, and that Wasserstein through the full pipeline returns the separation
  between two Gaussians in feature units.
- It surfaced from a calibration run: three metrics agreeing to five decimal
  places across eight thousand features, which no two estimators do.

### Added — block permutation

- **The null relabels contiguous blocks rather than individual frames**, so it
  is built from data that still looks like a trajectory. The correlation time
  is estimated per feature by an integrated autocorrelation with Sokal's
  window, summarised by the upper quartile across features, and blocks are made
  a couple of correlation times long. On the same null data, 1000 replicates
  per setting:

  | correlation time τ | frame permutation | block permutation |
  |---|---|---|
  | 1 | 5.45% | 1.66% |
  | 5 | 72.11% | 2.30% |
  | 20 | 99.01% | 2.24% |
  | 50 | 99.92% | 2.31% |

  Nominal 5%. The block rate is flat across the range rather than degrading
  with τ. Power is not the price: a real 0.8σ shift at τ = 20 is still found at
  98% of features.
- On by default; `block_permutation=False` disables it where frames genuinely
  are independent. `ComparisonResult` now carries `correlation_time`,
  `n_blocks` and `p_values_withheld`.

### Added — the benchmark harness

- **`prothon --benchmark`** and `prothon.benchmark` compare several ensembles
  against one reference on equal terms, reporting for each: the distance, the
  noise floor for *that model's own sample size*, the margin between them,
  precision and recall per residue, and a verdict.
- **The table is ordered by margin, not by distance.** Sample size changes the
  distance: against a 20,000-frame reference, two ensembles from the same
  distribution give a floor of 0.064 at 5,000 conformations and 0.109 at 50,
  and a model with a real half-sigma bias scores 0.216 at 5,000 and 0.129 at
  50. The same model and the same error, a smaller number because it sampled
  less — so a table of raw dissimilarities ranks the thinly sampled model
  first.
- **A model whose sampling cannot support a comparison gets a row saying so**,
  and the other models in the run are unaffected. Raising would lose them;
  inventing a number would be worse.
- Conformations from a generative model are independent draws, so the
  benchmark disables block permutation — there is no time correlation to
  correct for, and blocking would cost resolution for nothing.

### Added — the Protein Ensemble Database

- **`Ensemble.from_ped("PED00024")`**, with `ped_entry`, `ped_ensemble` and
  `ped_ensembles` for looking at an entry before downloading it. Optional
  caching, since entries run to tens of megabytes.
- Three details of the API decide the shape of this, and each would give a
  wrong answer if assumed: the `ensemble-pdb` endpoint returns a gzipped
  **tar**, not a gzipped PDB; the model count in the metadata is not reliable
  for parsing, since the first `MODEL` record shares a line with a tar header
  and a naive count of PED00024 finds 575 where the entry says 576; and an
  entry may hold several separate determinations, which are returned
  separately rather than merged.
- PED publishes no conformer populations, so ensembles loaded from it are
  uniformly weighted. Recorded as a fact about the database rather than left
  as an unexamined default.

### Added — scoring against experiment

- **`prothon.validate`** predicts what an experiment would have measured and
  checks against what it did: radius of gyration, end-to-end distance,
  pairwise and PRE distances, FRET efficiencies, and ³J(HN,HA) couplings by
  the Karplus relation.
- **Every score carries a floor.** A perfect ensemble does not score
  χ²_red = 1: measured on ensembles whose true average *is* the experimental
  value, a perfect ensemble of 20 conformations scores 0.77 and one of 5000
  scores 0.00. The floor comes from scoring one half of the ensemble against
  the other, and `within_floor` says whether the agreement is already inside
  what the sampling permits.
- **Sixth-power averaging where the physics requires it.** A PRE reports
  ⟨r⁻⁶⟩^(−1/6), not ⟨r⟩. On a distribution with a rare compact state — 90% at
  5.0 nm, 10% at 1.5 nm — those are 2.19 nm and 4.64 nm, and the linear average
  misses precisely the state PRE exists to detect. A rigid test case cannot
  catch this: on a narrow distribution the two agree to a hundredth of a
  nanometre.
- **Chemical shifts, SAXS profiles and RDCs are deliberately not computed.**
  Each needs something absent from the coordinates — an empirical predictor
  trained on a database, an explicit solvent layer, or an alignment tensor
  fitted to the data being compared against. `score_observable` accepts
  predictions from any external tool and scores them with the same floor.
- Experimental uncertainties are required rather than optional: a chi-squared
  without them is a sum of squares in arbitrary units.

### Changed — order parameters, not measures

- **`--measures` is `--order-parameters`, short `-p`**, and `MEASURES` is
  `ORDER_PARAMETERS`. "Measure" collided with "metric" — a metric *is* a
  measure of distance — so `--measures cbcn --metric jsd` read as a
  distinction without a difference. "Local order parameter" is the term of the
  paper and of the original code's own docstrings. Four words now separate four
  levels: **order parameter** (the local quantity), **representation** (the
  matrix built from it), **metric** (the distance between distributions of it),
  **observable** (what an experiment measures).
- The rename reaches everything a user sees, not only the Python names: the
  `manifest.json` key is `order_parameter`, `ComparisonResult`,
  `BenchmarkRow` and `PrecisionRecall` carry `order_parameter`, `prothon info`
  says "order parameters", and the documentation page is
  `order_parameters.md`. A test asserts that no user-visible name still says
  "measure", because a `measure` key in a manifest beside an
  `--order-parameters` flag is exactly the drift this was meant to end.
- The 2.x names remain as aliases and warn: `MEASURES`, `Measure`,
  `resolve_measure`, `describe_measure`, `measures=`, `methods=`, `-m`.

### Fixed — compiled readers no longer print over the results

- MDTraj reads several formats through VMD's molfile plugins, which announce
  themselves on file descriptor 1 from C: two lines per trajectory, which
  `contextlib.redirect_stdout` cannot catch because the C code never consults
  `sys.stdout`. On a study of a dozen ensembles the plugin outnumbered the
  result, and under `--json` it made the output unparseable. The descriptor is
  now redirected while a file is read, and restored in a `finally` so an
  exception cannot leave a process writing to `/dev/null`.
- `--verbose` turns the redirect off, so a genuine diagnostic from a reader is
  visible when somebody is looking for one.
- **`prothon info | head` no longer ends in a traceback.** Closing a pipe
  mid-write raised `BrokenPipeError` out of `main`, so the only thing a reader
  saw of an otherwise successful run was a stack trace.

### Documentation

- **An examples page** working through nine things people actually want to do,
  with real output from a run of the code: two conditions on one protein,
  several order parameters at once, a difference that lives between residues,
  missed states against invented ones, molecules that are not the same
  molecule, several models ranked against one reference, scoring against
  experiment, loading from the Protein Ensemble Database, and mixing sources of
  different kinds in one run.
- Every flag, method and keyword shown is checked against the code rather than
  written from memory.

### Added — the study

- **`prothon compare --config study.yml`**. A command line asks a question
  once; a file records one, and can be committed beside the manuscript, diffed
  when it changes, and handed to somebody who has the data but not the terminal
  session.
- **A `Study` is the object every interface builds.** Flags are parsed into
  one, a file is read into one, Python constructs one, and all of them then run
  the same object. A setting reachable from one interface and not another is a
  bug that cannot happen when there is only one place for settings to live.
- **`--save-config` writes the study a command line describes**, so a command
  typed once can be committed rather than reconstructed. Only what was actually
  given is written: a flag left at its default is a flag nobody chose.
- **It expresses three things a flag cannot**: a topology per ensemble —
  `--topology` is one path for every source, which is right for conditions of
  one system and wrong for a mutant or an ortholog — a label per ensemble, and
  per-frame weights from a file. A `stride` too.
- **Every key is checked against the schema.** A configuration that silently
  ignores what it does not recognise is a file that lies: a misspelled
  `random_seed` would leave the study unseeded and say nothing, and the run
  would look fine. Unknown keys are refused with the closest known name
  offered, at the top level, inside `compare`, and inside an ensemble.
- Flags override the file, so a study re-runs with a different seed or output
  directory without being edited.
- Each ensemble names where its conformations are with `ensemble:`. `source:`
  was the first name for it and still works.
- **The manifest records the study**, so a result found later carries the
  question it answered rather than only the answer.
- `pyyaml` is now a declared dependency.

### Fixed — the README described a superseded interface

- Both headline commands still used `-traj`, `-top` and `--seed` three
  releases after those became `--ensembles`, `--topology` and
  `--random-state`, and the flag table listed none of `--config`,
  `--save-config`, `--order-parameters` or `--report`. An earlier edit had
  targeted text that had already moved and silently did nothing.
- A test now reads the README and fails on a flag that does not exist in the
  parser, on any of the superseded names, and on the absence of a capability
  worth finding. A README is the first thing a reader sees and the last thing
  anybody edits.
- A fenced block labelled `json` or `yaml` is now checked to be that. One in
  `config.md` contained an elision and failed the documentation build on a
  clean checkout while a local incremental build reported success — an
  incremental Sphinx build does not re-read an unchanged page, so it can pass
  on a file it never looked at.

### Changed — one import

- **`from prothon import Prothon` is the whole of what a user imports.**
  A workflow needed five: the class, `Ensemble`, the validation functions,
  `Study`, and `benchmark`. A capability behind an import nobody guesses is a
  capability nobody finds.
- Reachable from the class: `Prothon.from_config`, `Prothon.load`,
  `Prothon.order_parameters()`, `Prothon.metrics()`, `Prothon.observables()`.
  Reachable from a study: `compare`, `rank`, `validate`, `save_config`,
  alongside `distinguishability` and `coverage_and_fidelity`. The underlying
  functions remain importable; nothing requires it.
- `--metric` gains the short form `-m`.
- **`topology` takes one path or one per ensemble.** A single topology is right
  when comparing conditions of one system and wrong for everything else — a
  mutant has its own, and so does an ortholog — which made the flag unable to
  express the capability this package was rebuilt around. `None` in the list
  means that source carries its own, as a PED accession and a multi-model PDB
  both do.

### Fixed — text is read as UTF-8

- A test of the documentation read files without naming an encoding, so it
  used the platform default — UTF-8 on Linux and macOS, cp1252 on Windows —
  and failed on the first em-dash. A scan found seven more places with the same
  latent bug, and a test now walks the source tree and fails on any text read
  or write that does not name its encoding.

### Added — global order parameters

- **`rg`, `ree`, `asph` and `nu`** join the five local ones. A radius of
  gyration could be scored against experiment but not compared on, which was
  an asymmetry with no reason behind it.
- These give one column rather than one per residue, so they say *whether* two
  ensembles differ in size or shape rather than *where*. The summary reports
  `differs` instead of counting residues.
- **`asph`** is built from the gyration tensor eigenvalues: 0 for a sphere,
  0.25 for a disc, 1 for a rod, verified against all three.
- **`nu`** is fitted on each conformation rather than once over the ensemble,
  because a comparison needs a distribution. Recovers 0.5 on an ideal random
  walk. The per-frame spread is real — roughly 0.15 at thirty residues — and
  is part of what two ensembles are compared on. A chain too short to fit is
  refused.
- `OrderParameter` gains `scope`, since `per_residue` was answering two
  questions: whether column *i* is residue *i*, and whether there are many
  columns at all. A windowed torsion is the first without being the second.

### Fixed — two claims about what is being measured

- **The per-conformation scaling exponent is not the ensemble Flory
  exponent**, and the documentation said it was. The ensemble fit takes the
  log of the averaged distance; the per-conformation fit averages the log, and
  Jensen's inequality makes the second smaller by about 0.03 on an ideal chain
  — an offset that does not shrink with chain length, so it is not a sampling
  artefact. Comparing two ensembles on `nu` is unaffected, since the same
  estimator is applied to both, but a mean taken from it should not be quoted
  as ν.
- **The noise floor splits every ensemble, not only the reference**, which the
  documentation had wrong in three places. The resolution limit of a
  comparison is set by whichever side is sampled worse.
- **And the floor is conservative by about a quarter.** Halves have half the
  frames, so it measures the limit at *n*/2 while the study has *n*: a
  1000-frame ensemble reports about 0.063 where the limit at 1000 frames is
  0.050. The error is in the safe direction, and correcting it would mean
  assuming the distance scales as *n*^(−1/2) for every metric, which has been
  measured only for the Jensen–Shannon distance on Gaussians. **Now measured
  for all three metrics on four distribution shapes, and the correction is
  refused**: Wasserstein and Kolmogorov–Smirnov scale near *n*^(−0.5), but
  Jensen–Shannon — the default — decays as *n*^(−0.35) to *n*^(−0.47), because
  it is estimated from a kernel density whose bandwidth also depends on sample
  size. A single √2 would push its floor *below* the true limit, which is the
  failure the floor exists to prevent. Extrapolating a measured slope from two
  split sizes fails the same way, landing 29% low on a skewed distribution.
  `scripts/floor_scaling.py` reproduces the measurement.

### Added — one chain of a complex

- **`chains`** on the constructor, `--chains` on the command line, and a
  `chains` key per ensemble in a study file. A chain is named by its PDB
  letter or by index, several may be kept, and one selection covers every
  ensemble or one is given each — so a protomer of one complex can be compared
  against a protomer of another.
- **An unknown chain is refused with the available ones named.** MDTraj's own
  selector takes an integer index, and a chain letter passed to it matches
  nothing and returns an empty selection without complaining, so the failure
  would otherwise be an ensemble with no atoms rather than a message.

### Fixed — a concentrated torsion was reported as no difference at all

- A tightly concentrated circular feature drove the von Mises kernel to a
  concentration of order a thousand, at which it underflows to exact zero a
  few grid points from the peak. An exact zero opposite a positive value makes
  a Kullback–Leibler term infinite, and the Jensen–Shannon distance with it.
- **The caller then reported that infinity as `0.0`.** A pair of torsion
  distributions genuinely 0.29 apart came back as identical, silently, on
  every buried residue of every comparison using `cata`.
- The density now carries a floor 300 orders of magnitude below anything that
  matters, and a non-finite distance is reported as 1.0 — no shared support is
  the *most* two distributions can differ, not the least.
- Found because the ubiquitin re-analysis compares the same quantity under two
  code paths and checks they agree. Only `cata` disagreed, by 0.327.

### Added — `n_jobs`

- The permutation null and the split-half floor are loops over independent
  draws and together are over 90% of the cost of a comparison. Both now run in
  parallel: `n_jobs` on `dissimilarity` and `compare_ensembles`, `--n-jobs` on
  the command line, `-1` for every core.
- **The result does not depend on the worker count.** Seeds are drawn from the
  caller's generator before the work is divided, so a parallel run reproduces a
  serial one exactly; a test asserts it for both the frame and block nulls.
- Work is sent in chunks of permutations rather than one per task, since the
  pooled representation has to reach each worker and sending it a hundred times
  costs more than the permutations do.
- `scripts/parallel_speedup.py` measures the gain, because it depends on the
  machine: each worker imports NumPy and SciPy before doing anything, and on a
  small job that exceeds the work saved.
### Fixed — the density floor belongs on both estimators

- The circular estimator was given a density floor to stop an underflowed
  kernel producing an infinite Kullback–Leibler term; the linear one was not,
  and a non-finite distance there was reported as `1.0`.
- **That was wrong for a near-degenerate feature.** Buried solvent
  accessibility is nearly constant, its kernel collapses, and a value of 1.0
  says "maximally different" about two halves of the same ensemble. It raised
  the measured noise floor on `sasa` from 0.032 to 0.053 and took the count of
  significant residues to zero.
- Both estimators now floor their density, the distance is defined for every
  input, and two genuinely disjoint distributions reach 1.0 by arithmetic
  rather than by fallback. A non-finite result now raises rather than
  returning a number, because neither 0 nor 1 can be told from a measurement
  by the caller.

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
- **A p-value where the sampling cannot support one**: fewer independent
  blocks than a permutation null can be built from. The block length is never
  shortened to manufacture blocks, so a short trajectory of a slow system
  shows up as a shortfall rather than being papered over — 300 frames of a
  system whose correlation time saturates the estimator at 33 leaves four
  blocks, and four is refused. The measured noise floor is reported either way.
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
