# Your first comparison

A comparison needs two or more ensembles and a topology they share. Each
trajectory file is **one ensemble** — Prothon never concatenates them, because
joining two conditions averages away the difference the study exists to
measure.

```bash
prothon -traj wild_type.dcd,mutant.dcd -top topology.pdb -m cbcn --seed 0
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

**`floor 0.0472`** is what two disjoint halves of the reference ensemble score
against *each other*. It is the smallest difference this much sampling can
resolve. Had `d` come out at 0.03, the correct reading would be "no difference
detectable here", not "a small difference".

**`34/76 residues differ`** counts residues whose distributions differ by more
than a permutation null allows, after correcting for having asked the question
76 times (Benjamini and Hochberg 1995).

## What it wrote

```
results/
└── cbcn_output/
    ├── ensemble_0_matrix.csv           the representation, frames × residues
    ├── ensemble_0_matrix.png           the same as a heatmap
    ├── cbcn_ensemble_1_local_dissimilarity.png
    ├── cbcn_combined_local_dissimilarity.png
    ├── cbcn_global_dissimilarity_bar.png
    ├── cbcn_global_dissimilarity_line.png
    └── manifest.json
```

`manifest.json` holds the inputs, every parameter, the seed, the Prothon
version, and the full numerical results. A run that cannot say what produced it
cannot be repeated, so every run says.

The per-residue figures draw two curves: the masked values, and the raw values
faintly underneath. A residue that fell just short of significance looks
identical to one that did not move at all once masked, and reading a flat zero
as "no change" is the easy mistake.

## From Python

```python
from prothon import Prothon

study = Prothon(
    ["wild_type.dcd", "mutant.dcd"],
    topology="topology.pdb",
    output_dir="results",
    random_state=0,
)
results = study.compare_ensembles(methods="cbcn,cata", s_num=5)

for comparison in results["cbcn"]:
    print(comparison.ensemble_index, comparison.global_dissimilarity)
    print(comparison.resolved, comparison.noise_floor)

print(study.summary())
```

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
