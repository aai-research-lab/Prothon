# Python API

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

## Plotting

```{eval-rst}
.. automodule:: prothon.core.plotting
   :members: replot_global_dissimilarity, replot_local_dissimilarity, get_ensemble_colors
```
