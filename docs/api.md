# Python API

## One import

`from prothon import Prothon` is the whole of what a user needs. Everything
below is reachable from the class or from an instance — the registries as
functions on the class, the other ways of starting a study as constructors,
and the analyses as methods. The underlying functions remain importable for
anyone who wants them.

## The shape of it

The sections below follow the order a study runs in, which is not the order the
package is laid out in. Both are given, because the first is what you want when
you are working and the second is what you want when you are reading the source.

| stage | what it does | module |
|---|---|---|
| **Load** | trajectories, directories, PED accessions, multi-model PDB; and the residue correspondence when the molecules differ | `prothon.ingest` |
| **Represent** | conformations to an *M* × *N* matrix of local order parameters | `prothon.core.representation` |
| **Compare** | distances between distributions, per residue and globally | `prothon.core.dissimilarity`, `prothon.core.metrics` |
| **Qualify** | correlation time, blocks, effective sample size — what the comparison is worth | `prothon.core.correlation` |
| **Beyond per-residue** | whole-ensemble tests, precision and recall | `prothon.core.ensemble_metrics`, `prothon.core.precision_recall` |
| **Score** | against experimental observables | `prothon.validate` |
| **Batch** | several ensembles against one reference | `prothon.batch` |
| **Record** | a study as a file, and the schema behind every interface | `prothon.config` |
| **Draw** | figures, each carrying its noise floor | `prothon.core.plotting` |

The **Qualify** row is the one with no counterpart elsewhere. Most tools go from
represent to compare and stop; the question of what the comparison is worth,
given how the data was sampled, is the reason for this one.

## The study object

```{eval-rst}
.. autoclass:: prothon.Prothon
   :members:
   :undoc-members:
   :show-inheritance:
```

## Results

```{eval-rst}
.. autoclass:: prothon.ComparisonResult
   :members:
```

## Representations

```{eval-rst}
.. automodule:: prothon.core.representation
   :members: MEASURES, Measure, compute_ensemble_representation, compute_representation, describe_measure, resolve_measure
```

## Distances and statistics

```{eval-rst}
.. automodule:: prothon.core.dissimilarity
   :members: dissimilarity, jsd_local, estimate_pdf, effective_sample_size, benjamini_hochberg
```

```{eval-rst}
.. automodule:: prothon.core.metrics
   :members: METRICS, Metric, feature_distance, describe_metric, resolve_metric
```

## Correlation and blocking

```{eval-rst}
.. automodule:: prothon.core.correlation
   :members: correlation_time, effective_frames, plan_blocks, block_labels, MINIMUM_BLOCKS
```

## Ingest and reconciliation

```{eval-rst}
.. automodule:: prothon.ingest.ensemble
   :members: Ensemble, EnsembleQuality
```

```{eval-rst}
.. automodule:: prothon.ingest.reconcile
   :members: Correspondence, Substitution, reconcile, feature_residues
```

```{eval-rst}
.. automodule:: prothon.ingest.ped
   :members: ped_entry, ped_ensemble, ped_ensembles
```

```{eval-rst}
.. automodule:: prothon.ingest.sequence
   :members: Alignment, align, sequence_of, chain_sequences, residue_letter
```

## Whole-ensemble comparison

```{eval-rst}
.. automodule:: prothon.core.ensemble_metrics
   :members: EnsembleComparison, distinguishability, maximum_mean_discrepancy, classifier_two_sample
```

```{eval-rst}
.. automodule:: prothon.core.precision_recall
   :members: PrecisionRecall, precision_recall
```

## Benchmarking

```{eval-rst}
.. automodule:: prothon.batch.benchmark
   :members: benchmark, BenchmarkResult, BenchmarkRow
```

## Scoring against experiment

```{eval-rst}
.. automodule:: prothon.validate.observables
   :members: radius_of_gyration, end_to_end, pairwise_distance, pre_distance, fret_efficiency, j_coupling_hn_ha, average_observable, Observable
```

```{eval-rst}
.. automodule:: prothon.validate.score
   :members: score_observable, AgreementResult
```

## The study, and the schema behind every interface

A study built from flags, from a file, or in Python is the same `Study` object,
because all three read one schema. That is why no interface can offer a setting
the others cannot, and why a misspelled key is refused rather than silently
ignored.

```{eval-rst}
.. automodule:: prothon.config.study
   :members: Study, load_study
```

```{eval-rst}
.. automodule:: prothon.config.schema
   :members: Command, Parameter, COMMANDS, PARAMETERS, parameters_for
```

## Plotting

```{eval-rst}
.. automodule:: prothon.core.plotting
   :members: replot_global_dissimilarity, replot_local_dissimilarity, get_ensemble_colors
```
