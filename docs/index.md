# Prothon

> Efficient comparison of protein conformational ensembles using local order
> parameters.

Prothon represents each ensemble as a vector of probability distributions over
**local order parameters** — contact numbers, virtual bond and torsion angles,
solvent accessibility — and measures the distance between corresponding
distributions. Because the representation is local, no structural superposition
is needed, so the cost is linear in the number of conformations rather than
quadratic.

```bash
pip install prothon-ensembles
prothon compare --ensembles wild_type.dcd mutant.dcd --topology topology.pdb
```

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.2841 (floor 0.0472) — 34/76 residues differ
```

## Read this first

That second number is the point of the software. Two independent halves of a
*single* ensemble are not at distance zero from each other, because a finite
sample never reproduces a continuous distribution exactly. That self-distance
is the smallest difference the sampling can resolve, and Prothon reports it
beside every result. A dissimilarity below its floor is reported as
unresolvable, not as a small difference.

[The statistics](statistics.md) sets out how significance is decided, how
multiplicity is corrected, and what an ensemble's sampling is actually worth.
[Calibration](calibration.md) measures the error rate rather than asserting it,
including the case that matters most: frames correlated in time, where a null
built on individual frames calls 99% of residues different when nothing
differs.

## Contents

```{toctree}
:maxdepth: 2
:caption: Guide

installation
getting_started
examples
config
```

```{toctree}
:maxdepth: 2
:caption: Reference

order_parameters
metrics
statistics
different_molecules
benchmark
validate
cli_reference
api
references
```

```{toctree}
:maxdepth: 2
:caption: Measurements

calibration
circular
convergence
ubiquitin
performance
```

The measurements are not illustrations. Each is produced by a script in
`scripts/`, reports a number rather than a claim, and is the reason some
default in the software is what it is. **Calibration** is the false-positive
rate of the significance test against systems whose correct answer is fixed by
construction. **Circular** is what treating a circular order parameter as
linear costs, which differs by two orders of magnitude between metrics.
**Convergence** asks how long a trajectory must run before a difference of a
given size can be resolved at all. **Ubiquitin** re-analyses the published
dataset the method was introduced on.

## Citation

> Aina, A.; Hsueh, S. C. C.; Plotkin, S. S. PROTHON: A Local Order
> Parameter-Based Method for Efficient Comparison of Protein Ensembles.
> *J. Chem. Inf. Model.* **2023**, *63* (11), 3453–3461.
> [doi:10.1021/acs.jcim.3c00145](https://doi.org/10.1021/acs.jcim.3c00145)
