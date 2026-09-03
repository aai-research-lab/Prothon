# Comparing different molecules

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

prothon = Prothon(
    ensembles=["wt.xtc", "mut.xtc"],
    topology=["wt.pdb", "mut.pdb"],     # one each: they are different molecules
    order_parameters="cbcn",
    random_state=0,
)
prothon.compare()
```

or from the command line:

```bash
prothon compare -e wt.xtc mut.xtc -t wt.pdb mut.pdb -p cbcn
```

`None` in the list means that source carries its own topology, which a PED
accession and a multi-model PDB both do:

```python
Prothon(ensembles=["md.xtc", "PED00024"], topology=["top.pdb", None])
```

```
wild type vs F5G: 14 residues correspond, 92.9% identity, 100.0% coverage
  substitutions: F5G

CBCN (reference: ensemble 0)
  ensemble 1: d = 0.6336 (floor 0.1369) — 9/13 residues differ
```

## What reconciliation does

Prothon takes the no-reconciliation fast path only when a deterministic
topology fingerprint matches. That fingerprint includes chain identity and
order, residue identity and order, atom name, element and order, and bond
connectivity. Equal atom counts are not enough: a same-size mutant, an isomer,
or a complex whose chains were reordered enters reconciliation. When both
structures provide the same unique chain IDs, reordered chains are paired by
ID rather than by their position in the files.

The sequences are extracted per chain and aligned globally (Needleman and
Wunsch 1970) with affine gap penalties (Gotoh 1982) under BLOSUM62 (Henikoff
and Henikoff 1992); the aligned columns become the residue correspondence.
Affine penalties matter here: they make one long gap cheaper than several short
ones, which is the difference between recognising a missing loop and scattering
it across the alignment. End gaps are free by default, because the usual case
is two constructs differing by a terminal overhang.

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

Every local result carries two parallel identities. `feature_index` is the
stable, one-based global position of each feature on the **reference**
ensemble. `feature_labels` is for display; in a multichain topology it reads
`A:1`, `A:2`, ..., `B:1` so repeated chain-local residue numbers are
unambiguous. Figures and JSON retain both.

This is explicit even when no reconciliation was needed. CBCN omits glycine,
so an implicit 1..n column index is already wrong for an identical A–G–V pair:
its two feature indices are `[1, 3]`, not `[1, 2]`.

For the F5G example above the index reads `[1, 2, 3, 4, 6, ..., 14]`, with 5
absent because the mutant's glycine has no C-beta to count contacts for.

## Sources

An `Ensemble` can come from several places:

```python
Ensemble.from_trajectory("md.xtc", "top.pdb", label="MD", stride=10)
Ensemble.from_files(["rep1.xtc", "rep2.xtc"], "top.pdb", label="MD")  # one ensemble
Ensemble.from_pdb_models("bioemu_out/*.pdb", label="BioEmu")
Ensemble.from_pdb_models("nmr_entry.pdb", label="NMR")
Ensemble.from_ped("PED00024")                                        # by accession
```

`from_files` joins replicates of one condition into a single ensemble. Two
different conditions are two ensembles, and joining those would average away
the difference being measured — so it is never done implicitly.

## From the Protein Ensemble Database

PED holds ensembles determined from experiment — NMR, SAXS, paramagnetic
relaxation enhancement, often with restrained molecular dynamics. Comparing a
model against one asks a different question from comparing it against a
simulation: not whether it reproduces someone else's force field, but whether
it reproduces what the measurements support.

```python
from prothon.ingest import ped_entry, ped_ensemble, ped_ensembles

alpha_synuclein = ped_ensemble("PED00024")     # 576 conformations, 140 residues
```

**An entry may hold several ensembles.** PED00001 holds `e001`, `e002` and
`e003` — separate determinations of the same protein, not parts of one.
`ped_ensemble` takes one; `ped_ensembles` returns them all, separately, because
merging them would average over exactly the differences the deposition
distinguished. Look before downloading:

```python
entry = ped_entry("PED00001")
[(e["ensemble_id"], e["models"]) for e in entry["ensembles"]]
# [('e001', 11), ('e002', 10), ('e003', 11)]
```

Entries run to tens of megabytes, so pass `cache_dir=` when a benchmark will
load the same one repeatedly.

**Conformers from PED are uniformly weighted.** The database publishes no
populations, so an ensemble loaded from it has uniform weights. That is a fact
about PED rather than an assumption made here.

## One chain of a complex

A complex is often compared one chain at a time: a bound peptide against its
free form, one protomer of a dimer against another, or a domain against the
same domain in a different construct. The rest of the system is a different
molecule, and averaging over it is not what the question asked.

```bash
prothon compare -e bound.xtc free.xtc -t complex.pdb --chains A -p cbcn
```

```python
Prothon(ensembles=["bound.xtc", "free.xtc"], topology="complex.pdb", chains="A")
```

A chain is named by its PDB letter or by index, and several may be kept with
`"A,B"` or `[0, 1]`. One selection covers every ensemble, or give one each:

```python
Prothon(
    ensembles=["dimer.xtc", "other.xtc"],
    topology=["dimer.pdb", "other.pdb"],
    chains=["A", "B"],          # a protomer of each
)
```

Chains selected from different molecules are reconciled by sequence alignment
in the usual way, so a protomer of one complex compares against a protomer of
another even where the sequences differ.

An unknown chain is refused with the available ones named. This matters more
than it sounds: MDTraj's own selector takes an integer index, and a chain
letter passed to it matches nothing and returns an empty selection without
complaining — so the failure would otherwise be an ensemble with no atoms in
it rather than a message.

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

## References

Gotoh 1982; Henikoff and Henikoff 1992; Kish 1965; Needleman and Wunsch 1970.
Full citations on the [references page](references.md).
