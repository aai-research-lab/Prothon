<div align="center">

# Prothon

**How different are two protein ensembles — and is the difference real?**

[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.3c00145-blue)](https://doi.org/10.1021/acs.jcim.3c00145)
[![PyPI](https://img.shields.io/pypi/v/prothon-ensembles?label=pypi)](https://pypi.org/project/prothon-ensembles/)
[![Python](https://img.shields.io/pypi/pyversions/prothon-ensembles)](https://pypi.org/project/prothon-ensembles/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Tests](https://github.com/aai-research-lab/Prothon/actions/workflows/tests.yml/badge.svg)](https://github.com/aai-research-lab/Prothon/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/aai-research-lab/Prothon/branch/main/graph/badge.svg)](https://codecov.io/gh/aai-research-lab/Prothon)
[![Docs](https://img.shields.io/readthedocs/prothon?label=docs)](https://prothon.readthedocs.io)

[**Documentation**](https://prothon.readthedocs.io) ·
[**Quick start**](https://prothon.readthedocs.io/en/latest/getting_started.html) ·
[**The statistics**](https://prothon.readthedocs.io/en/latest/statistics.html) ·
[**Cite**](#citation)

</div>

---

```bash
prothon -traj wild_type.dcd,mutant.dcd -top topology.pdb -m cbcn
```

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.2841 (floor 0.0472) — 34/76 residues differ
```

Prothon represents each conformational ensemble as a vector of probability
distributions over **local order parameters** — contact numbers, virtual bond
and torsion angles, solvent accessibility — and measures the distance between
corresponding distributions.

Because the representation is local, no structural superposition is required.
Two consequences follow, and they are the reasons to use it.

**It is linear in ensemble size, not quadratic.** Methods that compare
ensembles through pairwise RMSD must superpose every structure against every
other. Prothon never superposes anything, so ensembles of tens of thousands of
conformations are ordinary rather than prohibitive.

**It can compare ensembles that are not the same molecule.** A superposition
needs a common coordinate frame, and a wild type and its mutant do not have
one. Local order parameters need only a map between residues, which a sequence
alignment provides. *(In development — see below.)*

## It reports what it cannot resolve

Two independent halves of a *single* ensemble have a non-zero distance between
them, because a finite sample never reproduces a continuous distribution
exactly. That self-distance is the resolution limit of the comparison. Prothon
measures it, prints it beside every result, and draws it on every figure. A
difference smaller than the floor is reported as unresolvable rather than as a
small difference.

This matters more than it sounds. The significance test in version 2.0 built
its null from a bootstrap of each ensemble against itself, which is about half
as wide as the true sampling variability — so it called two independent samples
of an *identical* distribution significantly different at **100%** of residues.
Version 2.1 replaced it with a permutation null, which sits at 1.2%. The
[statistics page](https://prothon.readthedocs.io/en/latest/statistics.html)
gives the full account, and `--legacy-statistics` reproduces the old behaviour
for regenerating published figures.

## Install

```bash
pip install prothon-ensembles
prothon --info
```

The distribution is `prothon-ensembles` because PyPI's `prothon` was registered
in 2020 by an unrelated project. **The import name and the command are both
`prothon`.** A conda-forge package named `prothon` is in review.

## Use it

From the command line:

```bash
prothon -traj a.dcd,b.dcd,c.dcd -top top.pdb -m cbcn,cata -o results --seed 0
```

| flag | meaning |
|---|---|
| `-traj` | Trajectory files, one per ensemble, comma-separated. Never concatenated. |
| `-top` | Topology (PDB), shared by all of them. |
| `-m` | Measures: `cbcn`, `cacn`, `caba`, `cata`, `sasa`. |
| `--metric` | Distance: `jsd` (default), `wasserstein`, `ks`. |
| `-r` | Reference ensemble index (default 0). |
| `-o` | Output root. Each measure writes `<measure>_output/`. |
| `-d` | Projections: `pca`, `mds`, `tsne`. Off by default. |
| `--seed` | Set it, and the run is reproducible. |

Or from Python:

```python
from prothon import Prothon

study = Prothon(["wild_type.dcd", "mutant.dcd"], "topology.pdb", random_state=0)
results = study.compare_ensembles(methods="cbcn")

comparison = results["cbcn"][0]
comparison.global_dissimilarity     # 0.2841
comparison.noise_floor              # 0.0472 — the resolution limit
comparison.resolved                 # True: the difference clears the floor
comparison.significant              # bool array, one per residue
comparison.local_dissimilarity      # per residue, zero where not significant
comparison.raw_local_dissimilarity  # per residue, unmasked

print(study.summary())
```

Each measure writes a directory containing the representation matrices as CSV,
heatmaps, global and per-residue figures, and a `manifest.json` recording the
inputs, parameters, seed and version that produced them.

## The measures

| name | quantity | units | circular |
|---|---|---|---|
| `cbcn` | C-beta contact number, smooth cutoff | contacts | |
| `cacn` | C-alpha contact number, smooth cutoff | contacts | |
| `caba` | Virtual Cα–Cα–Cα bond angle | rad | |
| `cata` | Virtual Cα torsion angle | rad | yes |
| `sasa` | Per-residue solvent accessible surface area | nm² | |

Torsions live on a circle, so they are estimated with a von Mises kernel on a
grid spanning a full turn. Each measure declares this and every estimator reads
it — a linear treatment of circular data is wrong by large factors and says
nothing about it.

## In development

Version 2.1.0 is what `pip install` gives you. The following is on `main` and
unreleased; install from git to try it.

```bash
pip install "git+https://github.com/aai-research-lab/Prothon.git"
```

- **Comparison across different molecules.** `prothon.ingest` builds a residue
  correspondence from a sequence alignment, so a mutant, a truncated construct
  or a coarse-grained model can be compared against a reference. Columns are
  derived from the residue map rather than assumed — a mutation to glycine
  removes a C-beta and renumbers every `cbcn` column after it.
- **Weighted ensembles.** Conformer probabilities from a deposited ensemble, or
  frame weights from a reweighted simulation, reach the density estimate. Kish
  effective sample size sizes the noise floor: a thousand frames in which one
  conformer carries half the probability is worth four independent samples.
- **A metric layer.** Wasserstein-1, which reports in the feature's own units
  and needs no grid or bandwidth, and the Kolmogorov–Smirnov statistic
  (Kuiper's, on circular features).
- **Whole-ensemble comparison.** Maximum mean discrepancy and a classifier
  two-sample test, which see differences in the relationship *between* residues
  that no per-residue metric can.
- **Coverage and fidelity.** Precision and recall, per residue, distinguishing
  a model that misses a state from one that invents one — two failures that any
  symmetric distance scores alike and that need opposite work.

## Citation

> Aina, A.; Hsueh, S. C. C.; Plotkin, S. S. PROTHON: A Local Order
> Parameter-Based Method for Efficient Comparison of Protein Ensembles.
> *J. Chem. Inf. Model.* **2023**, *63* (11), 3453–3461.
> DOI: [10.1021/acs.jcim.3c00145](https://doi.org/10.1021/acs.jcim.3c00145)

```bibtex
@article{aina2023prothon,
  author  = {Aina, Adekunle and Hsueh, Shawn C. C. and Plotkin, Steven S.},
  title   = {PROTHON: A Local Order Parameter-Based Method for Efficient
             Comparison of Protein Ensembles},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {63},
  number  = {11},
  pages   = {3453--3461},
  year    = {2023},
  doi     = {10.1021/acs.jcim.3c00145},
}
```

## Upgrading from 2.0

The API is unchanged and existing scripts run without modification, but
**numbers will differ**. See [CHANGELOG.md](CHANGELOG.md) for the full account
and the [statistics page](https://prothon.readthedocs.io/en/latest/statistics.html)
for why. `from Prothon import Prothon` still works and warns; use
`from prothon import Prothon`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The version 1 code accompanying the
2023 paper is preserved unchanged under `legacy/`.

## License

MIT. See [LICENSE](LICENSE).

Prothon was distributed under GPL-3.0 up to and including version 2.0.0. From
2.1.0 the project is MIT-licensed. Copies already obtained under GPL-3.0 remain
governed by that licence — relicensing is not retroactive and takes nothing
away from anyone who has a copy.

---

<div align="center">

Built in the [AAI Research Lab](https://aai-research-lab.github.io) at
California State University Dominguez Hills, on MDTraj, NumPy, SciPy,
scikit-learn and Matplotlib.

</div>
