<div align="center">

# Prothon

**How different are two protein ensembles — and is the difference real?**

[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.3c00145-blue)](https://doi.org/10.1021/acs.jcim.3c00145)
[![PyPI](https://img.shields.io/pypi/v/prothon-ensembles?label=pypi)](https://pypi.org/project/prothon-ensembles/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/prothon?label=conda-forge)](https://anaconda.org/conda-forge/prothon)
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

Two ensembles differ. By how much, and is the difference real?

```bash
prothon compare --ensembles wild_type.dcd mutant.dcd --topology topology.pdb
```

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.2841 (floor 0.0472) — 34/76 residues differ
```

Two numbers, and the second is the one other tools do not give you. Split one
ensemble in half at random and compare the halves: the answer is not zero,
because a finite sample never reproduces a continuous distribution exactly.
That self-distance is the **noise floor** — the smallest difference this much
sampling can resolve.

Here 0.2841 clears a floor of 0.0472 by six times, so the difference is real.
At 0.05 it would not have been, and Prothon would say so rather than report a
small difference. Every result carries its floor, and where the sampling cannot
support a per-residue conclusion, no p-value is printed at all.

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
prothon compare --ensembles wt.dcd mutant.dcd --topology top.pdb \
                --order-parameters cbcn --random-state 0

# a simulation, a deposited ensemble and a generative model, on equal terms
prothon compare --ensembles md.xtc PED00024 bioemu/ --topology target.pdb

# several models against one reference, ranked on margin above their own floors
prothon compare --ensembles bioemu/ alphaflow/ bbflow/ \
                --reference md.xtc --topology target.pdb --report table

# or write the study down and commit it beside the manuscript
prothon compare --config study.yml
```

Every flag has a short form (`-e`, `-t`, `-p`, `-s`) and the same name as its
keyword argument in Python, because both are generated from one schema.

One import, and everything is reachable from it:

```python
from prothon import Prothon

study = Prothon(["wild_type.dcd", "mutant.dcd"], "topology.pdb", "cbcn",
                random_state=0)

study.compare()                       # where do they differ
study.distinguishability()            # differences *between* residues
study.coverage_and_fidelity()         # missed states, or invented ones
study.rank()                          # several against a reference
study.validate("rg", [2.71], [0.08])  # against experiment
study.save_config()                   # writes prothon.yml
```

The order parameter is a property of the study, so it is named once. Any method
takes one to override it for a single call.

Flags, a config file and the Python API build the same object, so none of them
can offer a setting the others cannot.

## What else it does

| | |
|---|---|
| **Different molecules** | Mutant against wild type, or a construct against a longer one, through a sequence alignment |
| **Weighted ensembles** | Conformer probabilities reach the density estimate, and Kish's effective sample size sizes the floor |
| **Deposited ensembles** | `Ensemble.from_ped("PED00024")`, by accession |
| **Whole-ensemble tests** | MMD and a classifier two-sample test, which see differences *between* residues that no per-residue metric can |
| **Coverage and fidelity** | Precision and recall per residue, distinguishing an ensemble that misses a state from one that invents one |
| **Scoring against experiment** | R<sub>g</sub>, end-to-end distance, PRE, FRET and ³J, each beside a floor |
| **Three distances** | Jensen–Shannon, Wasserstein-1 in the feature's own units, and Kolmogorov–Smirnov, with Kuiper's on circular features |
| **Nine order parameters** | Five per residue — contact numbers, backbone angles, solvent accessibility — and four for the whole chain |

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
conda install -c conda-forge prothon
```

The import and the command are both `prothon`. Available on PyPI too — see the
[installation guide](https://prothon.readthedocs.io/en/latest/installation.html).

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

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

Built in the [AAI Research Lab](https://aai-research-lab.github.io) at
California State University Dominguez Hills, on MDTraj, NumPy, SciPy,
scikit-learn and Matplotlib.

</div>
