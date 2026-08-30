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

## Global

One column: a single number per conformation, describing the whole molecule.
These say *whether* two ensembles differ in overall size or shape, which is
what a paper on a disordered protein usually reports.

| name | quantity | units | typical values |
|---|---|---|---|
| `rg` | Radius of gyration | nm | |
| `ree` | End-to-end distance | nm | |
| `asph` | Asphericity | | 0 sphere, 0.25 disc, 1 rod |
| `nu` | Flory scaling exponent | | 0.33 compact, 0.5 ideal, 0.588 expanded |

```bash
prothon compare -e md.xtc PED00024 -t top.pdb -p rg,nu
```

A global parameter compares the distribution, not the mean. Two ensembles with
the same average radius of gyration and different breadth are different
ensembles, and this reports them as such.

The summary says `differs` rather than counting residues, since there is only
one column and nothing to plot per position.

**`asph`** comes from the gyration tensor eigenvalues, so it describes how the
mass is arranged rather than how much there is: two conformations can share a
radius of gyration and differ completely in shape.

**`nu`** is fitted on each conformation separately — the root-mean-square
distance between residues separated by *s* in sequence goes as *s*<sup>ν</sup>,
and ν is the slope on log axes. Fitting per conformation rather than once over
the ensemble is what makes it comparable: a comparison needs a distribution.
The per-frame value is noisy on a short chain, with a spread of roughly 0.15 at
thirty residues and 0.10 at a hundred and twenty, and that spread is part of
what is being compared. A chain shorter than about ten residues is refused.

## Contact numbers

For each pair of C-beta (or C-alpha) atoms separated by more than two residues
in sequence, a smooth cutoff

$$w(d) = \frac{1}{1 + \exp\left[\kappa (d - r_0)\right]}$$

with $\kappa = 50\ \mathrm{nm^{-1}}$ and $r_0 = 1\ \mathrm{nm}$ contributes to
both partners. The result is a differentiable count rather than a step: about 1
well inside the cutoff, about 0 well outside.

Pairs closer than three residues in sequence are always in contact and carry no
information about the fold, so they are excluded.

Glycine has no C-beta, so `cbcn` has one column per *non-glycine* residue. This
matters when comparing ensembles whose sequences differ — see
[comparing different molecules](different_molecules.md).

## Virtual angles

`caba` is the angle over three consecutive alpha carbons; `cata` the torsion
over four. They describe the local backbone geometry without reference to any
frame.

`cata` **wraps at ±π**, and that is not a detail. A density estimated on a
linear grid splits a population straddling the wraparound across both ends and
puts a false trough in the middle of it. A linear Wasserstein distance between
two tight torsion populations either side of the cut reports 4.43 radians where
the true separation is 0.28 — a factor of twenty-one, reported without
complaint.

Prothon handles this by having each measure *declare* whether it is circular,
and every estimator read that declaration: a von Mises kernel for densities, a
circular optimal-transport distance for Wasserstein, Kuiper's statistic in
place of Kolmogorov–Smirnov, and a $(\cos, \sin)$ encoding for the
whole-ensemble methods. The alternative is for each call site to remember, and
they will not.

## Solvent accessible surface area

Shrake–Rupley (1973), per residue, in nm², computed through MDTraj (McGibbon
et al. 2015). Useful for detecting changes in burial that
contact numbers can miss, and the measure most likely to contain constant
columns — a residue with zero exposure in every frame. Those are handled
explicitly rather than crashing the density estimate.

## Which to choose

`cbcn` is the default and the measure the 2023 paper validated. It is sensitive
to tertiary packing.

`cacn` is the same idea without needing side-chain atoms, so it works on
coarse-grained models and on glycine-rich sequences.

`caba` and `cata` describe backbone geometry and are sensitive to local
secondary structure. Being window measures, they are the first to lose columns
when comparing molecules that differ by an insertion or deletion.

`sasa` speaks to burial and exposure directly, which is often what a question
about binding or aggregation is really about.

Running several is cheap and usually informative: `-p cbcn,cata,sasa`.

## References

Aina, Hsueh and Plotkin 2023; McGibbon et al. 2015; Shrake and Rupley 1973.
Full citations on the [references page](references.md).
