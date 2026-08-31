<div align="center">

# Prothon

**How different are two protein ensembles — and is the difference real?**

[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.3c00145-blue)](https://doi.org/10.1021/acs.jcim.3c00145)
[![PyPI](https://img.shields.io/pypi/v/prothon-ensembles?label=pypi)](https://pypi.org/project/prothon-ensembles/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/prothon-ensembles/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Tests](https://github.com/aai-research-lab/Prothon/actions/workflows/tests.yml/badge.svg)](https://github.com/aai-research-lab/Prothon/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/aai-research-lab/Prothon/branch/main/graph/badge.svg)](https://codecov.io/gh/aai-research-lab/Prothon)
[![Docs](https://img.shields.io/readthedocs/prothon?label=docs)](https://prothon.readthedocs.io)

[**Documentation**](https://prothon.readthedocs.io) ·
[**Quick start**](https://prothon.readthedocs.io/en/latest/getting_started.html) ·
[**Cite**](#citation)

</div>

---

```bash
pip install prothon-ensembles
prothon compare --ensembles wild_type.dcd mutant.dcd --topology topology.pdb
```

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.2841 (floor 0.0472) — 34/76 residues differ
```

**That second number is why this exists.** Two independent halves of a *single*
ensemble are not at distance zero from each other, because a finite sample never
reproduces a continuous distribution exactly. That self-distance is the smallest
difference your sampling can resolve. Prothon measures it and prints it beside
every result. A difference below the floor is reported as unresolvable, not as a
small difference.

## Why it is fast, and what that buys

Prothon describes each ensemble by the distributions of **local order
parameters** at every residue — contact numbers, backbone angles, solvent
accessibility — and compares those distributions. Nothing is ever superposed.

**Linear in ensemble size, not quadratic.** No all-against-all RMSD, so fifty
thousand conformations is ordinary rather than prohibitive.

**Works across different molecules.** A superposition needs a common coordinate
frame, and a wild type and its mutant do not have one. A residue map from a
sequence alignment is enough.

**Says *where*, not only *how much*.** Per residue, with a p-value.

## It refuses to overstate

Trajectory frames are not independent draws, and a significance test that
pretends otherwise calls almost everything different. Measured on two ensembles
drawn from the **same** distribution, nominal 5%:

| correlation time τ | permuting frames | permuting blocks |
|---|---|---|
| 1 | 5.5% | 1.7% |
| 5 | 72.1% | 2.3% |
| 20 | **99.0%** | 2.2% |
| 50 | **99.9%** | 2.3% |

Prothon estimates the correlation time from your data and permutes contiguous
blocks of it. The block rate is flat across the range, so a valid result does
not depend on you knowing τ in advance.

Where a trajectory holds too few independent blocks to build a null from,
Prothon reports the floor and **withholds the p-value** rather than printing one
the data cannot support.

## Ask it something

```bash
# two conditions, one protein
prothon compare -e wt.dcd mutant.dcd -t top.pdb -p cbcn -s 0

# a simulation, a deposited ensemble and a generative model, on equal terms
prothon compare -e md.xtc PED00024 bioemu/ -t target.pdb

# several models against one reference, ranked on margin above their own floors
prothon compare -e bioemu/ alphaflow/ bbflow/ -r md.xtc -t target.pdb --report table

# or write the study down and commit it beside the manuscript
prothon compare --config study.yml
```

One import, and everything is reachable from it:

```python
from prothon import Prothon

study = Prothon(["wild_type.dcd", "mutant.dcd"], "topology.pdb", random_state=0)

study.compare("cbcn")                 # where do they differ
study.distinguishability()            # differences *between* residues
study.coverage_and_fidelity()         # missed states, or invented ones
study.rank()                          # several against a reference
study.validate("rg", [2.71], [0.08])  # against experiment
study.save_config("study.yml")        # write the study down
```

Flags, a config file and the Python API build the same object, so none of them
can offer a setting the others cannot.

## What else it does

- **Different molecules** — mutant against wild type, or a construct against a
  longer one, through a sequence alignment
- **Weighted ensembles** — conformer probabilities reach the density estimate,
  and Kish's effective sample size sizes the floor
- **Deposited ensembles** — `Ensemble.from_ped("PED00024")`, by accession
- **Whole-ensemble tests** — MMD and a classifier two-sample test, which see
  differences *between* residues that no per-residue metric can
- **Coverage and fidelity** — precision and recall per residue, distinguishing
  an ensemble that misses a state from one that invents one
- **Scoring against experiment** — R<sub>g</sub>, end-to-end distance, PRE,
  FRET and ³J, each beside a floor, because a perfect ensemble of twenty
  conformations scores χ²_red = 0.77 and fitting that to 1.0 is fitting to noise
- **Three distances** — Jensen–Shannon, Wasserstein-1 in the feature's own
  units, and Kolmogorov–Smirnov, with Kuiper's on circular features

Every default above was chosen by a measurement, and the measurements ship with
the software: the false-positive rate against known ground truth, what a linear
treatment of a torsion costs, how long a trajectory must run before a difference
of a given size can be resolved at all, and a re-analysis of the published
dataset the method was introduced on.

## Documentation

All of it, in detail, at **[prothon.readthedocs.io](https://prothon.readthedocs.io)**.

[Quick start](https://prothon.readthedocs.io/en/latest/getting_started.html) ·
[Order parameters](https://prothon.readthedocs.io/en/latest/order_parameters.html) ·
[The statistics](https://prothon.readthedocs.io/en/latest/statistics.html) ·
[Worked examples](https://prothon.readthedocs.io/en/latest/examples.html) ·
[CLI reference](https://prothon.readthedocs.io/en/latest/cli_reference.html) ·
[Python API](https://prothon.readthedocs.io/en/latest/api.html) ·
[Measurements](https://prothon.readthedocs.io/en/latest/calibration.html)

## Install

```bash
pip install prothon-ensembles
prothon info
```

The distribution is `prothon-ensembles` because PyPI's `prothon` was registered
in 2020 by an unrelated project. **The import name and the command are both
`prothon`.**

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

See [CONTRIBUTING.md](CONTRIBUTING.md). The version 1 code accompanying the 2023
paper is preserved unchanged under `legacy/`.

## License

MIT. See [LICENSE](LICENSE).

Earlier releases were distributed under GPL-3.0, and copies obtained under that
licence remain governed by it; relicensing is not retroactive and takes nothing
away from anyone holding one.

---

<div align="center">

Built in the [AAI Research Lab](https://aai-research-lab.github.io) at
California State University Dominguez Hills, on MDTraj, NumPy, SciPy,
scikit-learn and Matplotlib.

</div>
