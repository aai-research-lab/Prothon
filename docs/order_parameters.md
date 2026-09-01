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

Running several is cheap and usually informative: `-p cbcn,cata,sasa`.

## References

Aina, Hsueh and Plotkin 2023; McGibbon et al. 2015; Shrake and Rupley 1973.
Full citations on the [references page](references.md).
