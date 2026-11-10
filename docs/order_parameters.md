# The order parameters

An order parameter is a local structural quantity — how many neighbours a
residue has, what angle its backbone makes — evaluated on every conformation.
Computing one over an ensemble gives an `(n_frames, n_features)` matrix: one
row per conformation, one column per residue or per angle. That matrix is the
ensemble's **representation**, and everything downstream works on it rather
than on coordinates, which is why nothing is ever superposed.

Four words appear throughout, and they mean different things:

| word | what it is |
|---|---|
| order parameter | the local quantity — a contact number at one residue |
| representation | the `(frames × features)` matrix computed from one |
| metric | the distance between two distributions of it |
| observable | something an experiment measures |

## Local

One column per residue, or per window of consecutive residues. These say
*where* two ensembles differ. The five below are those introduced with the
method (Aina, Hsueh and Plotkin 2023).

| name | quantity | units | per residue | circular |
|---|---|---|---|---|
| `cbcn` | C-beta contact number | contacts | yes | |
| `cacn` | C-alpha contact number | contacts | yes | |
| `caba` | Virtual Cα–Cα–Cα bond angle | rad | window of 3 | |
| `cata` | Virtual Cα torsion angle | rad | window of 4 | yes |
| `sasa` | Solvent accessible surface area | nm² | yes | |

Five more describe a residue's exposure and side-chain state:

| name | quantity | units | per residue | circular |
|---|---|---|---|---|
| `rsa` | Relative solvent accessibility | | yes | |
| `hseu` | Half-sphere exposure, side-chain half | contacts | yes | |
| `hsed` | Half-sphere exposure, opposite half | contacts | yes | |
| `hsea` | Half-sphere exposure anisotropy | | yes | |
| `chi1` | Side-chain χ1 torsion | rad | yes | yes |

## Global

One column: a single number per conformation, describing the whole molecule.
These say *whether* two ensembles differ in overall size or shape, which is
what a study of a disordered protein usually reports.

| name | quantity | units | reference values |
|---|---|---|---|
| `rg` | Radius of gyration | nm | |
| `ree` | End-to-end distance | nm | |
| `asph` | Asphericity | | 0 sphere · 0.25 disc · 1 rod |
| `nu` | Flory scaling exponent | | 0.33 globule · 0.5 ideal · 0.588 self-avoiding |

```bash
prothon compare -e md.xtc PED00024 -t top.pdb -p rg,nu
```

The summary says `differs` rather than counting residues, since there is one
column and nothing to plot per position.

## Molecular scope and chains

Protein order parameters describe protein by default. `cbcn` and `cacn`
select CA/CB atoms only from protein residues, so a calcium ion named `CA` is
not a residue feature. `rg` uses all protein atoms, while `asph` uses protein
C-alpha atoms. Water, ions, ligands, and membrane molecules do not silently
change either value.

The direct Python functions accept an explicit selection when the intended
object really is a complex:

```python
from prothon.represent.order_parameters import compute_asph, compute_rg

protein_rg = compute_rg(traj)                         # protein only
complex_rg = compute_rg(traj, selection="not water") # deliberate complex
complex_shape = compute_asph(traj, selection="not water and name CA")
```

`ree` and `nu` describe a chain, not a collection of chains. They accept a
chain index or PDB chain ID through `chain=` and refuse a multichain default.
For a complete study, selecting the chain during ingestion remains the clearest
route: `Prothon(..., chains="A")` or `--chains A`.

### `rg` — radius of gyration

The root-mean-square distance of the atoms from their centre of mass, mass
weighted. The single most reported number about a disordered ensemble, and the
one a SAXS experiment measures most directly.

It says how large a conformation is and nothing about its shape: a compact
sphere and an open hairpin can share a radius of gyration.

### `ree` — end-to-end distance

The distance between the first and last alpha carbon. What a FRET experiment
reports, once the dye positions are accounted for.

Together with `rg` it carries shape information that neither has alone. For an
ideal chain ⟨R²ₑₑ⟩/⟨R²_g⟩ = 6; a value below that means the ends are closer
than the chain's overall size implies, which is a long-range contact.

### `asph` — asphericity

From the eigenvalues λ₁ ≤ λ₂ ≤ λ₃ of the gyration tensor:

$$
\Delta = 1 - 3\,\frac{\lambda_1\lambda_2 + \lambda_2\lambda_3 + \lambda_1\lambda_3}
                       {(\lambda_1 + \lambda_2 + \lambda_3)^2}
$$

Zero when the three are equal — a sphere — and one when a single axis carries
everything — a rod. A flat disc gives 0.25.

This is the shape information `rg` discards. Two ensembles with the same
radius of gyration and different asphericity are differently shaped, and a
change in asphericity without a change in `rg` is a rearrangement at constant
size.

### `nu` — Flory scaling exponent

The root-mean-square distance between alpha carbons separated by *s* positions
in sequence goes as *s*<sup>ν</sup>, and ν is the slope on log axes. It is the
standard way of asking what kind of polymer a disordered protein resembles:

| ν | meaning |
|---|---|
| ≈ 0.33 | collapsed globule — a folded protein, or a poor solvent |
| ≈ 0.5 | ideal chain, attraction and excluded volume in balance |
| ≈ 0.588 | self-avoiding walk in good solvent — a denatured chain |

Measured values sit between these. Unfolded proteins in water average about
0.46 and rise toward 0.6 in denaturant (Hofmann et al. 2012).

**Prothon fits ν on each conformation rather than once over the ensemble**,
because a comparison needs a distribution and a single ensemble fit gives one
number. One conformation is enough to fit on: at separation *s* there are
*N* − *s* pairs to average over, and the fit runs across about *N*/2
separations.

:::{important}
**This is not the same quantity as the ensemble Flory exponent, and its mean
should not be quoted as one.**

The ensemble fit takes the logarithm of the averaged distance; the
per-conformation fit averages the logarithm. Those differ, and the second is
always smaller. Measured on an ideal random walk, where ν = 0.5 exactly:

| chain length | ensemble ν | mean per-conformation ν |
|---|---|---|
| 40 | 0.511 | 0.479 |
| 80 | 0.501 | 0.472 |
| 160 | 0.493 | 0.467 |
| 320 | 0.499 | 0.475 |

The offset is about 0.03 and does not shrink with chain length, so it is not a
sampling artefact.

What `nu` gives is the distribution of per-conformation fractal exponents. Two
ensembles compared on it are compared correctly — the same estimator is applied
to both and the offset cancels — but a mean taken from it belongs in a sentence
that says which exponent it is.
:::

The per-conformation value is also genuinely variable, with a spread of roughly
0.15 at thirty residues and 0.10 at a hundred and twenty. That variation is
real rather than fitting noise: a single ensemble holds compact and expanded
conformations with different exponents (Baul and Chakraborty 2024), and it is
that variation two ensembles are compared on.

A chain shorter than about ten residues is refused rather than fitted through
too few separations.

### What comparing them adds

Reporting ⟨Rg⟩ ± SD and comparing the means is the usual practice. Prothon
compares the distributions, and reports the smallest difference the sampling
can resolve beside the result — so two ensembles with the same mean radius of
gyration and different breadth are correctly reported as different, and a
difference smaller than two halves of one ensemble show against each other is
reported as unresolvable.

## Contact numbers

For each pair of C-beta (or C-alpha) atoms separated by more than two residues
in sequence, a smooth cutoff

$$w(d) = \frac{1}{1 + \exp\left[\kappa (d - r_0)\right]}$$

with $\kappa = 50\ \mathrm{nm^{-1}}$ and $r_0 = 1\ \mathrm{nm}$ contributes to
both partners. The result is a differentiable count rather than a step: about 1
well inside the cutoff, about 0 well outside.

Pairs closer than three residues in sequence are always in contact and carry no
information about the fold, so they are excluded.

Sequence separation exists only within one chain. Inter-chain CA/CB pairs are
included as physical contacts, even when the residues happen to be adjacent in
the topology's global numbering. This makes a contact profile sensitive to an
interface without inventing a sequence distance between two molecules.

Glycine has no C-beta, so `cbcn` has one column per *non-glycine* residue. This
matters when comparing ensembles whose sequences differ — see
[comparing different molecules](different_molecules.md).
It also matters for identical sequences: every result stores the actual
one-based residue positions in `feature_index`, so A–G–V is labelled `[1, 3]`
rather than silently renumbered `[1, 2]`.

## Virtual angles

`caba` is the angle over three consecutive alpha carbons; `cata` the torsion
over four. They describe the local backbone geometry without reference to any
frame.

Windows are constructed independently within each protein chain. A window
never crosses a chain boundary, a gap in residue numbering, or a missing
peptide bond when the topology carries bond information. The result therefore
contains only angles that actually belong to one continuous backbone; its
feature labels are generated from exactly the same accepted windows.

`cata` **wraps at ±π**, and that is not a detail. A density estimated on a
linear grid splits a population straddling the wraparound across both ends and
puts a false trough in the middle of it. A linear Wasserstein distance between
two tight torsion populations either side of the cut reports 4.43 radians where
the true separation is 0.28 — a factor of twenty-one, reported without
complaint.

Prothon handles this by having each order parameter *declare* whether it is
circular,
and every estimator read that declaration: a von Mises kernel for densities, a
circular optimal-transport distance for Wasserstein, Kuiper's statistic in
place of Kolmogorov–Smirnov, and a $(\cos, \sin)$ encoding for the
whole-ensemble methods. The alternative is for each call site to remember, and
they will not.

## Solvent accessible surface area

Shrake–Rupley (1973), per residue, in nm², computed through MDTraj (McGibbon
et al. 2015). Useful for detecting changes in burial that
contact numbers can miss, and the order parameter most likely to contain constant
columns — a residue with zero exposure in every frame. Those are handled
explicitly rather than crashing the density estimate.

All atoms remain occluders, so a ligand or binding partner can shield the
protein surface. Only protein residues become output features by default:
waters, ions, and ligands do not appear as extra "residues" in a protein
profile. The direct `compute_sasa(traj, report_selection=...)` function can
deliberately choose another set of residues to report without removing the
rest of the system as occluders.

## Relative solvent accessibility

`rsa` is `sasa` divided by the largest area that residue type can expose,
so a fully exposed glycine and a fully exposed tryptophan both read about 1
instead of 1.0 and 2.9 nm². That makes exposure comparable along a sequence,
which raw `sasa` is not.

The reference maxima are the theoretical values of Tien et al. (2013), from
idealized Gly–X–Gly tripeptides. Their empirical scale and the older scales
of Miller et al. (1987) and Rose et al. (1985) are available through
`compute_rsa(traj, scale=...)`. The numerator is `compute_sasa` itself, so
occluders and residue selection behave exactly as described above.

Three things to know when reading it:

- Values above 1 happen and are not clipped. A residue can protrude further
  than in the reference tripeptide; Tien et al. report that few do.
- Chain-terminal residues carry extra area from the free amino or carboxyl
  group, which no reference accounts for, so they read high. They are kept
  rather than dropped.
- A residue type with no reference (a ligand, a modified residue) is `NaN`,
  with a warning, rather than normalised by a guessed value.

Like `sasa`, `rsa` inherits a small dependence on orientation from the fixed
Shrake–Rupley sphere lattice. On the 2EQQ ensemble, rotating every model
changed `sasa` by up to about 0.014 nm² per residue at MDTraj's default 960
points, falling to 0.001 nm² at 20,000 points; for `rsa` that is of order
0.01. It is far below the differences the comparison reports as significant,
but it means two copies of one structure in different orientations are not
bit-identical.

## Half-sphere exposure

Draw a 1.3 nm sphere around a residue's Cα and cut it in half with the plane
perpendicular to the Cα→Cβ direction. `hseu` counts the other residues' Cα
atoms in the half the side chain points into; `hsed` counts those in the
other half (Hamelryck 2005). `hsea` is $(u - d)/(u + d)$, bounded in
$[-1, 1]$: positive when the neighbourhood is on the side-chain side, which
is what a buried side chain looks like, negative when the side chain points
out of the protein.

Contact numbers say how crowded a residue is. Half-sphere exposure also says
*which way* it faces. A residue can keep the same `cbcn` while its side chain
turns from the core to the solvent, and that turn is exactly the change an
epitope analysis is looking for.

Glycine has no Cβ. Rather than drop it, as `cbcn` does, its direction is
taken from the two flanking Cα atoms, the construction Hamelryck uses for
HSE-α. Only a glycine at a chain end, with no neighbour on one side, has no
direction and reads `NaN`. Biopython builds glycine's pseudo-Cβ differently,
so the two programs agree exactly on every other residue but not on glycine.

There is no sequence-separation exclusion, unlike the contact numbers: the
sphere describes a neighbourhood, and the adjacent residues are part of it.
The cost is a dense pairwise comparison per frame, $O(n^2)$ in residues,
evaluated in frame blocks to bound memory. `hseu`, `hsed` and `hsea` are
computed independently, so requesting all three does the work three times.

## Side-chain torsion χ1

The dihedral N–Cα–Cβ–Xγ, where Xγ is the first heavy atom past Cβ (CG1 for
isoleucine and valine, OG1 for threonine). Eighteen residue types have one;
glycine and alanine do not and get no column. A residue missing one of the
four atoms in a given topology is left out with a warning rather than given
a wrong angle.

χ1 is circular, and it matters more here than for `cata`: the trans rotamer
sits at ±180°, exactly where a linear grid would split it in two. `chi1` is
declared circular, so every estimator uses its circular branch.

The three staggered rotamers sit near −60°, 180° and +60°. Proline is the
exception: its ring holds χ1 within about ±30°, so its column describes ring
pucker rather than a rotamer choice.

## Which to choose

`cbcn` is the default and the order parameter the 2023 paper validated. It is sensitive
to tertiary packing.

`cacn` is the same idea without needing side-chain atoms, so it works on
coarse-grained models and on glycine-rich sequences.

`caba` and `cata` describe backbone geometry and are sensitive to local
secondary structure. Being defined over windows, they are the first to lose columns
when comparing molecules that differ by an insertion or deletion.

`sasa` speaks to burial and exposure directly, which is often what a question
about binding or aggregation is really about.

`rsa`, `hsea` and `chi1` together describe whether a residue is exposed,
which way it faces, and what its side chain is doing: the three properties
an antibody-accessible surface depends on.

Running several is cheap and usually informative: `-p cbcn,cata,sasa`.

## References

Aina, Hsueh and Plotkin 2023; Hamelryck 2005; McGibbon et al. 2015; Miller et
al. 1987; Rose et al. 1985; Shrake and Rupley 1973; Tien et al. 2013.
Full citations on the [references page](references.md).
