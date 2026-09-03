# Benchmarking ensembles

A generative model is judged by how closely its conformations resemble a
reference — molecular dynamics, or an experimentally derived ensemble. Prothon
runs the same comparison for every model against the same reference and reports
the same things for each.

```bash
prothon compare --ensembles bioemu/ alphaflow/ bbflow/ --reference md.xtc \
                --topology target.pdb --order-parameters cbcn --report table \
                --output-dir results --random-state 0
```

A benchmark is a comparison against a reference, presented as a ranked table.
It is `compare --report table` rather than a command of its own, because it is
the same calculation.

```
3 comparisons against md
  0 refused for want of sampling
  1 not resolvable above the noise floor

| target | model | n | d | floor | margin | precision | recall | verdict |
|---|---|---|---|---|---|---|---|---|
| target | bbflow | 250 | 0.658 | 0.150 | +0.508 | 0.33 | 0.33 | misses states at 8 residues |
| target | bioemu | 250 | 0.067 | 0.162 | -0.094 | 0.96 | 0.97 | indistinguishable from the reference at this sampling |
```

Each model is a source: a trajectory, a directory of single-model PDBs — which
is how most generative models emit output — a glob, a multi-model PDB, or a PED
accession. Only sources that need a topology use `--topology`, so a deposited
ensemble can sit in the same run as a simulation:

```bash
prothon compare -e bioemu/ PED00024 -r md.xtc -t target.pdb -p cacn -s 0 \
                --report table
```

## Read the margin, not the distance

**Sample size changes the distance.** Measured against a 20,000-frame
reference, two ensembles drawn from the *same* distribution give a noise floor
of 0.064 at 5,000 conformations and 0.109 at 50. A model with a real half-sigma
bias scores 0.216 at 5,000 conformations and 0.129 at 50 — the same model, the
same error, a smaller number because it sampled less.

So a table of raw dissimilarities ranks the thinly sampled model first. Every
row here carries its own 95th-percentile floor threshold, and the **margin** is
the distance above it. The table is ordered by margin, and that is the column
to compare across models. The mean floor remains in machine-readable output as
a descriptive sampling statistic, but does not decide the verdict.

This is not a subtlety that can be waved away by sampling everything equally:
models emit what they emit, and a benchmark that asks each to produce the same
number of conformations is measuring something other than the models.

## Read precision and recall, not just the margin

A single distance says a model is wrong. It does not say how, and the two ways
of being wrong call for opposite work:

- **Low recall** — the model never reaches states the reference visits. A
  cryptic pocket that never opens, a loop that never unfolds.
- **Low precision** — the model puts conformations where the reference has
  none. States no physics produced.

Both are reported per residue, so the answer names positions rather than
scoring a whole ensemble: *misses states at 8 residues* is something a model
developer can act on.

Each carries its own floor, measured by splitting both ensembles into their
own independent units: complete temporal blocks for a trajectory, complete
replicas where those are declared, and individual structures for IID output.
That floor is per residue — a rigid residue and a mobile one are not equally
easy to cover. Missed and invented calls use the lower tail of the stored split
distribution rather than its mean, and are withheld when either side has fewer
than eight independent units.

## Refusals are results

An ensemble worth too few independent conformations to support a comparison
produces a row saying so, rather than a number:

```
| target | tiny-model | 6 | — | — | — | — | — | refused |
```

The other models in the run are unaffected. A benchmark that raises on one
model loses the rest; a benchmark that invents a number for it is worse.

## Each ensemble keeps its own sampling provenance

Samples loaded from separate model PDBs are independent draws; a molecular
dynamics reference is a correlated trajectory. One `block_permutation`
Boolean cannot describe both. The benchmark therefore resolves the provenance
of each ensemble separately and never labels every model IID by assumption:

- `Ensemble.from_trajectory` is a trajectory and has its correlation time
  estimated unless one was supplied in `provenance`.
- `Ensemble.from_pdb_models` is IID.
- `Ensemble.from_files` preserves complete files as independent replica units.
- A directly constructed `Ensemble` defaults conservatively to trajectory
  sampling. Set `provenance={"sampling_kind": "iid"}` only when its rows truly
  are independent.

For a mixed trajectory/IID comparison, the permutation null groups IID rows to
the trajectory's block length so unlike units are not exchanged. This grouping
does not invent autocorrelation in the model: its precision/recall floor still
uses individual IID rows. Probability weights remain attached to their
conformations throughout both calculations. See [the statistics](statistics.md).

## From Python

```python
from prothon import benchmark
from prothon.ingest import Ensemble

reference = Ensemble.from_trajectory("md.xtc", "target.pdb", label="MD")
models = [
    Ensemble.from_pdb_models("bioemu/*.pdb", label="BioEmu"),
    Ensemble.from_pdb_models("alphaflow/*.pdb", label="AlphaFlow"),
]

result = benchmark(reference, models, order_parameters="cbcn,cata", random_state=0)
print(result.table("cbcn"))
result.write("results/")
```

`benchmark.json` carries every number, including the per-residue lists of
missed and invented states. It also records the Prothon version, all analysis
settings, input provenance, weight-only and time-corrected effective sample
sizes, the final block or replica plans, and any refusal reason in every row.

## Across many targets

`benchmark` handles one target. For a set of them, call it per target and
concatenate the rows:

```python
from prothon.batch import BenchmarkResult

rows = []
for name in targets:
    rows += benchmark(
        references[name], models[name], order_parameters="cbcn",
        target=name, random_state=0,
    ).rows

BenchmarkResult(rows, order_parameters=("cbcn",)).write("results/")
```

The comparisons are independent, so this parallelises across targets without
any coordination.
