# Examples

Ten things people actually want to do. Every output below is real, from a run
of the code as written.

---

## 1. Two conditions, one protein

Wild type against mutant, apo against holo, one force field against another —
the ordinary case, where both ensembles share a topology.

```bash
prothon compare --ensembles wt.dcd mutant.dcd --topology top.pdb \
                --order-parameters cbcn --random-state 0 --output-dir results
```

or, the same thing more briefly:

```bash
prothon compare -e wt.dcd,mutant.dcd -t top.pdb -p cbcn -s 0 -o results
```

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.5625 (floor 0.1209) — 8/14 residues differ
```

The floor is what two halves of the reference score against each other. A
dissimilarity below it would be reported as unresolvable rather than as a small
difference.

From Python, if you want the numbers:

```python
from prothon import Prothon

prothon = Prothon(
    ensembles=["wt.dcd", "mutant.dcd"], topology="top.pdb",
    order_parameters="cbcn", random_state=0,
)
result = prothon.compare()["cbcn"][0]

result.global_dissimilarity     # 0.5625
result.noise_floor              # 0.1209
result.significant              # bool array, one per residue
result.correlation_time         # frames; the null blocks on this
```

Every flag has the same name as its keyword argument, because both are
generated from one schema.

---

## 2. Several order parameters at once

Different order parameters see different things. Contact numbers see tertiary
packing, torsions see local backbone, solvent accessibility sees burial.

```bash
prothon compare -e wt.dcd mutant.dcd -t top.pdb -p cbcn,cata,sasa \
                -s 0 --s-num 10 -o results
```

Each writes its own directory with matrices, figures and a `manifest.json`
recording the parameters that produced them.

Running several is cheap, and disagreement between them is informative: a
change visible in `cata` but not `cbcn` is local backbone rearrangement without
repacking.

---

## 3. A difference that lives between residues

Two loops that visit the same positions but no longer at the same time have an
identical per-residue profile and are a different ensemble. No per-residue
metric can see it.

```python
prothon.distinguishability(order_parameter="cbcn", method="c2st")
```

```
C2ST: distinguishable (p < 1e-06), AUC = 1.000
  driven mostly by residues 8, 10, 9, 14, 12
```

`method="mmd"` runs a kernel two-sample test instead, which gives a calibrated
p-value and no indication of where the difference is. Quote the AUC rather than
the p-value: the classifier's null is asymptotic and its far tail is not
literal.

---

## 4. Did the model miss a state, or invent one?

A single distance says an ensemble is wrong. It does not say how, and the two
ways call for opposite work.

```python
prothon.coverage_and_fidelity(order_parameter="cbcn")
```

```
precision 0.416 (floor 0.961), recall 0.425 (floor 0.964)
  misses conformations at 8 residue(s): 7, 8, 9, 10, 11, 12, 13, 14
  invents conformations at 8 residue(s): 7, 8, 9, 10, 11, 12, 13, 14
```

Low recall is a missed state — a pocket that never opens. Low precision is an
invented one. Both are per residue, so the answer names positions.

---

## 5. A mutant, a truncated construct, a coarse-grained model

Ensembles that are **not the same molecule**. A superposition needs a common
coordinate frame and two different molecules have none; local order parameters
need only a map between residues.

```python
from prothon import Prothon
from prothon.ingest import Ensemble

wt  = Ensemble.from_trajectory("wt.xtc",  "wt.pdb",  label="wild type")
mut = Ensemble.from_trajectory("mut.xtc", "mut.pdb", label="F5G")

prothon = Prothon(ensembles=[wt, mut], random_state=0)
prothon.compare_ensembles(order_parameters="cbcn")
```

```
wild type vs F5G: 12 residues correspond, 91.7% identity, 100.0% coverage
  substitutions: F5G

CBCN (reference: ensemble 0)
  ensemble 1: d = 0.6816 (floor 0.1603) — 9/11 residues differ
```

Note the column count: eleven, not twelve. Glycine has no C-beta, so F5G
removes a `cbcn` column and renumbers everything after it. Columns are derived
from the residue map rather than assumed, and results are indexed by the
reference's numbering.

---

## 6. Several models against one reference

The same comparison, presented as a ranked table. There is no separate
benchmark command, because a benchmark is this view.

```bash
prothon compare -e bioemu/ alphaflow/ bbflow/ -r md.xtc \
                -t target.pdb -p cbcn -s 0 -o results --report table
```

```
| target | model   |   n |     d | floor | margin | precision | recall | verdict |
|--------|---------|-----|-------|-------|--------|-----------|--------|---------|
| target | model-C | 250 | 0.574 | 0.147 | +0.427 |     0.402 |  0.412 | misses states at 7 residues; invents states at 9 residues |
| target | model-B | 250 | 0.573 | 0.147 | +0.426 |     0.407 |  0.409 | misses states at 7 residues; invents states at 7 residues |
| target | model-A | 250 | 0.000 | 0.173 | -0.173 |     0.971 |  0.960 | indistinguishable from the reference at this sampling |
```

**Rank on the margin, not the distance.** A model that samples thinly gets a
higher floor *and* a depressed distance, so a table of raw distances flatters
it. See [benchmarking](benchmark.md).

---

## 7. Against experiment rather than against another ensemble

Comparing two ensembles cannot say which is right.

```bash
prothon validate -e md.xtc -t top.pdb --observable rg --experimental rg.txt
```

```
rg [md]: chi2_red = 0.31 (floor 0.44 ± 0.19) — agrees to within its own sampling
```

A perfect ensemble does not score χ²_red = 1 — it scores whatever its sampling
allows. The floor says what that is. See [validation](validate.md).

From Python, with an observable Prothon does not compute:

```python
from prothon.validate import score_observable

shifts = np.loadtxt("sparta_predictions.txt")     # (n_frames, n_residues)
score_observable(shifts, measured, uncertainty, observable="CA shift")
```

PRE distances need the sixth-power average, which the function does for you:

```python
from prothon.validate import pre_distance

pre_distance(traj, "name CA and resid 42", "name H and resid 76")
```

---

## 8. An ensemble from the Protein Ensemble Database

An accession is a source like any other, so it needs no special handling:

```bash
prothon compare -e PED00024 bioemu_out/ -p cacn -s 0
```

Neither of those needs a topology — a deposited entry and a directory of
structures both carry their own.

From Python, when you want to look before downloading:

```python
from prothon.ingest import ped_entry, ped_ensemble

ped_entry("PED00001")["ensembles"]
# [{'ensemble_id': 'e001', 'models': 11}, {'ensemble_id': 'e002', 'models': 10}, ...]

first = ped_ensemble("PED00024", cache_dir="~/ped")
```

An entry may hold several separate determinations. `ped_ensembles` returns them
all rather than merging them, and `PED00001:e002` names one.

---

## 9. Mixed sources in one comparison

Sources of different kinds go in the same list, because what a source *is* does
not change how it is asked for.

```bash
prothon compare -e md.xtc PED00024 bioemu_out/ -t target.pdb -p cacn -s 0
```

A simulation, a deposited ensemble and a generative model, compared on equal
terms. Only the trajectory uses `--topology`; the other two carry their own.

---

## 10. The study, written down

```bash
prothon compare --config prothon.yml
```

```yaml
description: wild type against the F5G mutant

ensembles:
  - ensemble: wt.xtc
    topology: wt.pdb        # each ensemble may have its own
    label: wild type
  - ensemble: mut.xtc
    topology: mut.pdb
    label: F5G

reference: wild type

compare:
  order_parameters: [cbcn, cata]
  random_state: 0
  n_permutations: 200
```

Something to commit beside the manuscript rather than reconstruct from a shell
history. A flag on the command line overrides the file, so the same study
re-runs with a different seed without being edited.

It works the other way too — a command line typed once becomes a study:

```bash
prothon compare -e wt.xtc mut.xtc -t top.pdb -p cbcn -s 0 --save-config prothon.yml
```

Flags, a file and the Python API all build the same object, so none of them can
offer a setting the others do not. See [the study](config.md).

## Things worth knowing

**Set a seed.** The floor and the p-values come from resampling. Without
`--random-state` two runs of one study give different numbers and nothing
records why.

**Each source is one ensemble.** They are never concatenated — joining two
conditions averages away the difference being measured. Use
`Ensemble.from_files` to join replicates of *one* condition.

**Frames must be in the order they were generated.** The correlation time is
estimated from the frame order, and a shuffled trajectory has none to find, so
the correction silently does nothing.

**Raise `--s-num` and `--n-permutations` for a paper.** The defaults are for a
first look; 100 permutations is about 6% where 5% is asked for. Both cost
linearly.

**Read the floor before the p-value.** The floor is measured. The p-value rests
on assumptions, and [the statistics](statistics.md) says which ones.
