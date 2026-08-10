# Comparing different molecules

*(In development — on `main`, not in 2.1.0.)*

Most interesting comparisons are between ensembles that are not the same
molecule: a wild type against a point mutant, a construct that resolves a loop
against one that does not, a coarse-grained model against an all-atom one, an
ortholog against an ortholog.

Methods built on superposition cannot ask these questions at all, because two
different molecules have no common coordinate frame to superpose into. Local
order parameters need only a map between residues, and a sequence alignment
provides one.

```python
from prothon import Prothon
from prothon.ingest import Ensemble

wt  = Ensemble.from_trajectory("wt.xtc", "wt.pdb", label="wild type")
mut = Ensemble.from_trajectory("mut.xtc", "mut.pdb", label="F5G")

study = Prothon.from_ensembles([wt, mut], random_state=0)
study.compare_ensembles(methods="cbcn")
```

```
wild type vs F5G: 14 residues correspond, 92.9% identity, 100.0% coverage
  substitutions: F5G

CBCN (reference: ensemble 0)
  ensemble 1: d = 0.6336 (floor 0.1369) — 9/13 residues differ
```

## What reconciliation does

The sequences are extracted per chain, aligned with an affine-gap
Needleman–Wunsch under BLOSUM62, and the aligned columns become a residue
correspondence. End gaps are free by default, because the usual case is two
constructs differing by a terminal overhang.

The alignment, the identity, the coverage, the substitutions named as a paper
would name them (`F5G`), and the unmatched residues on each side are all
recorded in `manifest.json`. A per-residue profile across two different
molecules is only interpretable beside the map that produced it.

## Columns are derived, not assumed

A column is not always a residue, and this is where a naive implementation
produces a plausible wrong answer.

**Glycine has no C-beta.** Mutating a residue to glycine removes one `cbcn`
column and renumbers every column after it. Comparing column *k* to column *k*
would compare different residues from the mutation onward and report a
difference along the whole C-terminal half of the protein — a figure that looks
entirely reasonable.

**`caba` and `cata` are windows.** They span three and four consecutive alpha
carbons, so a column survives only where the whole window has a counterpart
*and* those counterparts are still consecutive. A deletion breaks the three
windows spanning it, and those columns are dropped rather than silently
compared across the gap.

Prothon therefore maps residues first and derives columns from that map.

## Residue numbering in figures

After reconciliation the surviving columns are a subset of the reference's.
Numbering them 1..n would put the label of one residue under the value of
another, so every result carries `feature_index` — the position of each feature
on the **reference** ensemble — and the figures are drawn against it.

For the F5G example above the index reads `[1, 2, 3, 4, 6, ..., 14]`, with 5
absent because the mutant's glycine has no C-beta to count contacts for.

## Sources

An `Ensemble` can come from several places:

```python
Ensemble.from_trajectory("md.xtc", "top.pdb", label="MD", stride=10)
Ensemble.from_files(["rep1.xtc", "rep2.xtc"], "top.pdb", label="MD")  # one ensemble
Ensemble.from_pdb_models("bioemu_out/*.pdb", label="BioEmu")
Ensemble.from_pdb_models("nmr_entry.pdb", label="NMR")
```

`from_files` joins replicates of one condition into a single ensemble. Two
different conditions are two ensembles, and joining those would average away
the difference being measured — so it is never done implicitly.

## Weighted ensembles

A deposited ensemble stores a probability per conformer, and a reweighted
simulation produces one per frame.

```python
deposited = Ensemble.from_pdb_models("ped_entry.pdb", weights=probabilities)
```

Weights reach the density estimate, the permutation null and the noise floor.
They also change what the ensemble is worth: see
[effective sample size](statistics.md#how-much-sampling-is-enough).

Comparing a weighted ensemble against an unweighted one warns. Treating the
second as uniform is the only thing available, but it is an assumption about
that ensemble rather than a fact about it — and it is exactly the shape of a
deposited ensemble compared against a simulation.

## What is refused

- Sequences aligning below 25% identity — the twilight zone, where an alignment
  stops being evidence that positions correspond.
- Alignments covering less than half the shorter sequence. Identity alone is
  not enough: with free end gaps, two unrelated 40-residue sequences align on
  *two* columns at 50% identity, clearing any identity floor while covering a
  twentieth of the molecule.
- Ensembles whose protein chain counts differ. Concatenating the chains of a
  complex lets the aligner slide one chain against another, which is cheap in
  score and nonsense as a map.
- A measure whose windows no difference between the molecules leaves intact.
  The error names a per-residue measure to use instead.
