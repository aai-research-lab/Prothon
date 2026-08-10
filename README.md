<div align="center">

# Prothon

**How different are two protein ensembles — and is the difference real?**

[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.3c00145-blue)](https://doi.org/10.1021/acs.jcim.3c00145)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/prothon/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
and torsion angles, solvent accessibility — and measures the Jensen–Shannon
distance between corresponding distributions. Because the representation is
local, no structural superposition is needed and the cost is linear in the
number of frames rather than quadratic, which is what makes ensembles of tens
of thousands of conformations tractable.

**It reports what it cannot resolve.** Two independent halves of a *single*
ensemble have a non-zero Jensen–Shannon distance, because a finite sample never
reproduces a continuous distribution exactly. That self-distance is the
resolution limit of the comparison, and Prothon measures it, prints it beside
every result, and draws it on every figure. A difference smaller than the floor
is reported as unresolvable rather than as a small difference.

## Install

```bash
conda install -c conda-forge prothon      # preferred
pip install prothon-ensembles             # the distribution name; see below
prothon --info
```

The distribution on PyPI is `prothon-ensembles`, because `prothon` was
registered in 2020 by an unrelated protobuf generator and PyPI names are
permanent. The import name and the command are both `prothon`:

```python
from prothon import Prothon
```

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
comparison.global_dissimilarity   # 0.2841
comparison.noise_floor            # 0.0472 — the resolution limit
comparison.resolved               # True: the difference clears the floor
comparison.significant            # bool array, one per residue
comparison.local_dissimilarity    # per residue, zero where not significant
comparison.raw_local_dissimilarity  # per residue, unmasked

print(study.summary())
```

Each measure writes a directory containing the representation matrices as CSV,
heatmaps, global and per-residue dissimilarity figures, and a `manifest.json`
recording the inputs, parameters, seed and version that produced them.

## The measures

| name | quantity | circular |
|---|---|---|
| `cbcn` | C-beta contact number, smooth cutoff | |
| `cacn` | C-alpha contact number, smooth cutoff | |
| `caba` | Virtual Cα–Cα–Cα bond angle | |
| `cata` | Virtual Cα torsion angle | yes |
| `sasa` | Per-residue solvent accessible surface area | |

Torsions live on a circle, so they are estimated with a von Mises kernel on a
grid spanning a full turn. Each measure declares this, so the call site cannot
forget it.

## Upgrading from 2.0

The API is unchanged and existing scripts run without modification, but
**numbers will differ**, because the significance test in 2.0 was not sound: it
compared each ensemble against a bootstrap of itself, a null about half as wide
as the true sampling variability. Over 40 replicates in which both ensembles
were drawn from an *identical* distribution, 2.0 called 100% of residues
significantly different. The permutation test that replaces it sits at 1.2%.

`--legacy-statistics` reproduces the old behaviour for regenerating published
figures. [CHANGELOG.md](CHANGELOG.md) has the full account.

`from Prothon import Prothon` still works and warns; use
`from prothon import Prothon`.

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The original version 1 code accompanying
the 2023 paper is preserved unchanged under `legacy/`.

## License

MIT. See [LICENSE](LICENSE).

Prothon was distributed under GPL-3.0 up to and including version 2.0.0.
From 2.1.0 the project is MIT-licensed. Copies already obtained under
GPL-3.0 remain governed by that licence — relicensing is not retroactive
and takes nothing away from anyone who has a copy.

---

<div align="center">

Built in the [AAI Research Lab](https://aai-research-lab.github.io) at
California State University Dominguez Hills, on MDTraj, NumPy, SciPy,
scikit-learn and Matplotlib.

</div>
