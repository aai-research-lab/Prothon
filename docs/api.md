# Python API

## One import

`from prothon import Prothon` is the whole of what a user needs.

A study is built with the ensembles, the topology and the order parameters it
is about, so the last of those is named once rather than at every call:

```python
prothon = Prothon(["a.dcd", "b.dcd"], "top.pdb", "cbcn", random_state=0)
prothon.compare()                 # uses cbcn
prothon.compare("cata")           # overrides it for this call only
```

Note that `Prothon.order_parameters()` is the registry of every available
parameter and is unrelated to the constructor argument of the same name. Everything
below is reachable from the class or from an instance — the registries as
functions on the class, the other ways of starting a study as constructors,
and the analyses as methods. The underlying functions remain importable for
anyone who wants them.

## The shape of it

The package is laid out in the order a study runs in, so the module you want is
the stage you are at.

| stage | what it does | module |
|---|---|---|
| **Load** | trajectories, directories, PED accessions, multi-model PDB; and the residue correspondence when the molecules differ | `prothon.ingest` |
| **Represent** | conformations to an *M* × *N* matrix of local order parameters | `prothon.represent` |
| **Compare** | distances between distributions per residue and globally, whole-ensemble tests, precision and recall | `prothon.compare` |
| **Sampling** | correlation time, blocks, effective sample size — what the comparison is worth | `prothon.sampling` |
| **Score** | against experimental observables | `prothon.validate` |
| **Batch** | several ensembles against one reference | `prothon.batch` |
| **Record** | a study as a file, and the schema behind every interface | `prothon.config` |
| **Draw** | figures, each carrying its noise floor | `prothon.plot` |

`prothon.sampling` is the one with no counterpart in comparable tools. Most go
from represent to compare and stop; asking what a comparison is worth, given how
the data was sampled, is the reason for this one, and it is a top-level name
rather than a file inside something else for that reason.

Before 2.3 all of this lived under `prothon.core`, which held eight modules
whose only shared property was not being `ingest`. `prothon.core` remains as a
shim that forwards with a `DeprecationWarning`, and is removed in 3.0.

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
.. automodule:: prothon.represent.order_parameters
   :members: MEASURES, Measure, compute_ensemble_representation, compute_representation, describe_measure, resolve_measure
```

## Distances and statistics

```{eval-rst}
.. automodule:: prothon.compare.dissimilarity
   :members: dissimilarity, jsd_local, estimate_pdf, effective_sample_size, benjamini_hochberg
```

```{eval-rst}
.. automodule:: prothon.compare.distance
   :members: METRICS, Metric, feature_distance, describe_metric, resolve_metric
```

## Correlation and blocking

```{eval-rst}
.. automodule:: prothon.sampling.correlation
   :members: correlation_profile, correlation_time, effective_frames, plan_blocks, block_labels, MINIMUM_BLOCKS
```

## Ingest and reconciliation

```{eval-rst}
.. automodule:: prothon.ingest.ensemble
   :members: Ensemble, EnsembleQuality
```

```{eval-rst}
.. automodule:: prothon.ingest.reconcile
   :members: Correspondence, Substitution, reconcile, feature_residues, feature_identity, residue_identity
```

```{eval-rst}
.. automodule:: prothon.ingest.topology
   :members: TopologyFingerprint, topology_fingerprint, same_topology
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
.. automodule:: prothon.compare.joint
   :members: EnsembleComparison, distinguishability, maximum_mean_discrepancy, classifier_two_sample
```

```{eval-rst}
.. automodule:: prothon.compare.coverage
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
.. automodule:: prothon.plot.figures
   :members: replot_global_dissimilarity, replot_local_dissimilarity, get_ensemble_colors
```
