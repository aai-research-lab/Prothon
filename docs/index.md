# Prothon

> Efficient comparison of protein conformational ensembles using local order
> parameters.

Prothon represents each ensemble as a vector of probability distributions over
**local structural measures** — contact numbers, virtual bond and torsion
angles, solvent accessibility — and measures the distance between corresponding
distributions. Because the representation is local, no structural superposition
is needed, so the cost is linear in the number of conformations rather than
quadratic.

```bash
pip install prothon-ensembles
prothon -traj wild_type.dcd,mutant.dcd -top topology.pdb -m cbcn
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

installation
getting_started
measures
metrics
statistics
calibration
different_molecules
benchmark
cli_reference
performance
api
references
```

## Citation

> Aina, A.; Hsueh, S. C. C.; Plotkin, S. S. PROTHON: A Local Order
> Parameter-Based Method for Efficient Comparison of Protein Ensembles.
> *J. Chem. Inf. Model.* **2023**, *63* (11), 3453–3461.
> [doi:10.1021/acs.jcim.3c00145](https://doi.org/10.1021/acs.jcim.3c00145)
