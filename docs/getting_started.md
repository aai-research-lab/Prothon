# Your first comparison

A comparison needs two or more ensembles. Each source is **one ensemble** —
Prothon never concatenates them, because joining two conditions averages away
the difference the study exists to measure.

```bash
prothon compare --ensembles wild_type.dcd mutant.dcd --topology topology.pdb \
                --order-parameters cbcn --random-state 0
```

or, the same thing more briefly:

```bash
prothon compare -e wild_type.dcd mutant.dcd -t topology.pdb -p cbcn -s 0
```

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.2841 (floor 0.0472) — 34/76 residues differ
```

Three numbers, and none of them means anything without the others.

**`d = 0.2841`** is the global dissimilarity: the mean over residues of the
Jensen–Shannon distance between the two ensembles' distributions of C-beta
contact number (Lin 1991; Aina, Hsueh and Plotkin 2023). It is bounded in
[0, 1].

**`floor 0.0472`** is what two disjoint halves of each ensemble score against
*each other* — the smallest difference this much sampling can resolve. Had `d`
come out at 0.03, the correct reading would be "no difference detectable
here", not "a small difference".

It is conservative by roughly a quarter, because halves have half the frames.
Read a result within about 25% of its floor as borderline. See
[the statistics](statistics.md).

**`34/76 residues differ`** counts residues whose distributions differ by more
than a permutation null allows, after correcting for having asked the question
76 times (Benjamini and Hochberg 1995).

The null relabels contiguous *blocks* of conformations rather than individual
ones, because consecutive frames of a trajectory are nearly the same
conformation and a null that ignores that is far too narrow. The correlation
time is estimated from the data; nothing is required of you. Where a trajectory
holds too few independent blocks to build a null from, Prothon says so and
reports the floor alone:

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.3106 (floor 0.2841) — no p-value: correlation time 340
              frames leaves 5 independent blocks
```

That is a real answer. It says the difference is barely above the resolution
limit and the sampling cannot support a test — which is more useful than a
p-value the data does not justify.

## What it wrote

```text
results/
└── cbcn_output/
    ├── ensemble_0_matrix.csv                    frames × residues
    ├── ensemble_0_matrix.png                    the same as a heatmap
    ├── ensemble_1_matrix.csv
    ├── ensemble_1_matrix.png
    ├── cbcn_ensemble_1_local_dissimilarity.png  per residue, one per comparison
    ├── cbcn_combined_local_dissimilarity.png    every comparison on one figure
    ├── cbcn_global_dissimilarity_bar.png
    ├── cbcn_global_dissimilarity_line.png
    └── manifest.json                            what produced all of it
```

One directory per order parameter, so `-p cbcn,cata` writes `cbcn_output/` and
`cata_output/`.

`manifest.json` holds the inputs, every parameter, the seed, the Prothon
version, and the full numerical results. A run that cannot say what produced it
cannot be repeated, so every run says.

The per-residue figures draw two curves: the masked values, and the raw values
faintly underneath. A residue that fell just short of significance looks
identical to one that did not move at all once masked, and reading a flat zero
as "no change" is the easy mistake.

## From Python

`from prothon import Prothon` is the whole of what you need to import.
Everything else is a method on the study or a function on the class:

```python
from prothon import Prothon

prothon = Prothon(
    ensembles=["wt.dcd", "mut.dcd"], topology="top.pdb",
    order_parameters="cbcn", random_state=0,
)

prothon.compare()                          # per residue, with a floor
prothon.distinguishability()               # differences between residues
prothon.coverage_and_fidelity()            # missed states or invented ones
prothon.rank()                             # ranked against the reference
prothon.validate("rg", [2.71], [0.08])     # against experiment
prothon.save_config()                      # writes prothon.yml
```

The underlying functions are importable — `prothon.validate.score_observable`,
`prothon.ingest.Ensemble` and the rest — but nothing requires it.

## The longer form

```python
from prothon import Prothon

prothon = Prothon(
    ["wild_type.dcd", "mutant.dcd"],
    topology="topology.pdb",
    order_parameters="cbcn",
    output_dir="results",
    random_state=0,
)
results = prothon.compare_ensembles(order_parameters="cbcn,cata", s_num=5)

for comparison in results["cbcn"]:
    print(comparison.ensemble_index, comparison.global_dissimilarity)
    print(comparison.resolved, comparison.noise_floor)
    print(comparison.correlation_time, comparison.n_blocks)
    print(comparison.p_values_withheld)

print(prothon.summary())
```

## Other things to ask

This page covers one comparison. [Examples](examples.md) works through nine
more: several order parameters at once, differences that live between residues,
telling a missed state from an invented one, comparing molecules that are not
the same molecule, ranking several models against one reference, scoring
against experiment, and mixing sources of different kinds in one run.

## Reproducibility

Set `random_state` (or `--seed`). The noise floor and the p-values come from
resampling, so without a seed two runs of one study give different numbers and
nothing records why.

## Choosing the sampling

`s_num` sets the split-half repeats behind the floor, and `n_permutations` the
size of the null. The defaults are adequate for a first look. For a result
going into a paper, raise both — the cost is linear and the floor becomes less
noisy.

An ensemble worth fewer than 50 independent conformations warns; fewer than 10
is refused. See [the statistics](statistics.md#how-much-sampling-is-enough).

## If the frames are already independent

Generated structures, a set of deposited conformers, or a trajectory you have
already subsampled have no correlation to correct for. Blocking them costs
resolution for nothing:

```python
prothon.compare_ensembles(order_parameters="cbcn", block_permutation=False)
```

Prothon detects the absence of correlation and does this for you, so it is
rarely needed — but **the rows must be in the order the frames were
generated**. A shuffled or concatenated matrix has no correlation time to
estimate, and the correction silently finds none.
