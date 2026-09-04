<div align="center">

# Prothon

**How different are two protein ensembles — and is the difference real?**

[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.3c00145-blue?labelColor=black)](https://doi.org/10.1021/acs.jcim.3c00145)
[![PyPI](https://img.shields.io/pypi/v/prothon-ensembles?label=pypi&labelColor=black)](https://pypi.org/project/prothon-ensembles/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/prothon?label=conda-forge&color=44A833&labelColor=black)](https://anaconda.org/conda-forge/prothon)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/prothon-ensembles?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/prothon-ensembles)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?labelColor=black)](https://pypi.org/project/prothon-ensembles/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?labelColor=black)](https://opensource.org/licenses/MIT)

[![Tests](https://img.shields.io/github/actions/workflow/status/aai-research-lab/Prothon/tests.yml?branch=main&label=tests&labelColor=black)](https://github.com/aai-research-lab/Prothon/actions/workflows/tests.yml)
[![codecov](https://img.shields.io/codecov/c/github/aai-research-lab/Prothon/main?labelColor=black)](https://codecov.io/gh/aai-research-lab/Prothon)
[![Docs](https://img.shields.io/readthedocs/prothon?label=docs&labelColor=black)](https://prothon.readthedocs.io)
[![conda downloads](https://img.shields.io/conda/dn/conda-forge/prothon?label=conda%20downloads&color=44A833)](https://anaconda.org/conda-forge/prothon)

[**Documentation**](https://prothon.readthedocs.io) ·
[**Quick start**](https://prothon.readthedocs.io/en/latest/getting_started.html) ·
[**Cite**](#citation)

</div>

---

```bash
prothon compare --ensembles wild_type.dcd mutant.dcd --topology topology.pdb
```

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.2841 (floor 0.0472) — 34/76 residues differ
```

Two numbers, and the second is the one that decides whether the first means
anything. Split one ensemble in half at random and compare the halves: the
answer is not zero, because a finite sample never reproduces a continuous
distribution exactly. That self-distance is the **noise floor** — the smallest
difference this much sampling can resolve.

Here 0.2841 clears a floor of 0.0472 by six times. At 0.05 it would not have,
and Prothon would say so rather than report a small difference.

## Install

```bash
conda create -n prothon -c conda-forge prothon
conda activate prothon
```

`prothon info` lists what is installed and what version:

```bash
prothon info
```

For source work, clone the repository. GitHub's **Download ZIP** is an
unversioned audit snapshot; installable, versioned artifacts are the PyPI
wheel/source distribution and the conda-forge package.

## What you can compare

| | |
|---|---|
| **Two conditions, one protein** | Bound against free, folded against unfolded, one force field against another. Per residue, with a p-value, and a floor beside every value. |
| **Two different molecules** | Mutant against wild type, or a construct against a longer one. Superposition needs a common coordinate frame and two different molecules have none; a residue map from a sequence alignment is enough. |
| **A simulation against a deposited ensemble** | PED accessions by name (`PED00024`), multi-model PDB, directories of structures, and every common trajectory format — on equal terms in one command. |
| **Several models against one reference** | `--report table` ranks them by the margin above each model's *own* floor rather than by raw distance, because a thinly sampled ensemble scores a smaller distance for being thinly sampled. |
| **A trajectory against itself** | How much of it is independent, how long it must run before a difference of a given size can be resolved, and whether its correlation time has settled. |
| **An ensemble against experiment** | R<sub>g</sub>, end-to-end distance, PRE, FRET and ³J — each beside a floor, because a perfect ensemble of twenty conformations scores χ²_red = 0.77 and fitting that to 1.0 is fitting to noise. |
| **What no per-residue statistic can see** | Two ensembles can match residue by residue and differ in how residues move *together*. MMD and a classifier two-sample test find that; the classifier names the residues carrying it. |
| **Missed states against invented ones** | Precision and recall per residue. A symmetric distance says two ensembles differ; these say which one is short of a state and which has one too many. |

**It withholds rather than overstates.** Trajectory frames are not independent
draws, and a test that assumes they are calls almost every residue different
when nothing differs. Prothon estimates the correlation time from your data,
keeps the sample in trajectory order, and permutes contiguous blocks of it.
Where a trajectory holds too few independent blocks to build a null from, it
reports the floor and prints no p-value at all.

The false-positive rate is
[measured](https://prothon.readthedocs.io/en/latest/calibration.html), on the
default sampling path, rather than asserted.

## One import, and everything is reachable from it

```python
from prothon import Prothon

prothon = Prothon(["wild_type.dcd", "mutant.dcd"], "topology.pdb", "cbcn",
                  random_state=0)

prothon.compare()                       # where do they differ
prothon.distinguishability()            # differences *between* residues
prothon.coverage_and_fidelity()         # missed states, or invented ones
prothon.rank()                          # several against a reference
prothon.validate("rg", [2.71], [0.08])  # against experiment
prothon.save_config()                   # writes prothon.yml
```

The order parameter is a property of the study, so it is named once. Any method
takes one to override it for a single call.

**Three ways to run the same study**, and none is a subset of another — flags,
file and API are generated from one schema:

| | |
|---|---|
| **The CLI** | `prothon compare --ensembles wt.dcd mut.dcd --topology top.pdb --order-parameters cbcn`, or a flag for any setting. Every long flag has a short form. |
| **A config file** | `prothon compare --config prothon.yml`. A study written down can be committed beside a manuscript, diffed when it changes, and re-run by someone who has the data but not the terminal session. |
| **The Python API** | `Prothon(...)`, or `Prothon.from_config("prothon.yml")`. |

Any command line can be written out as a file with `--save-config`, and any
file can be run from the command line.

## Documentation

**Start here** — [Install](https://prothon.readthedocs.io/en/latest/installation.html) ·
[Your first comparison](https://prothon.readthedocs.io/en/latest/getting_started.html) ·
[Order parameters](https://prothon.readthedocs.io/en/latest/order_parameters.html) ·
[Worked examples](https://prothon.readthedocs.io/en/latest/examples.html)

**Going further** — [The statistics](https://prothon.readthedocs.io/en/latest/statistics.html) ·
[Different molecules](https://prothon.readthedocs.io/en/latest/different_molecules.html) ·
[Distances](https://prothon.readthedocs.io/en/latest/metrics.html) ·
[Scoring against experiment](https://prothon.readthedocs.io/en/latest/validate.html) ·
[Ranking several ensembles](https://prothon.readthedocs.io/en/latest/benchmark.html)

**Measurements** — [Calibration](https://prothon.readthedocs.io/en/latest/calibration.html) ·
[Circular parameters](https://prothon.readthedocs.io/en/latest/circular.html) ·
[Convergence](https://prothon.readthedocs.io/en/latest/convergence.html) ·
[Ubiquitin](https://prothon.readthedocs.io/en/latest/ubiquitin.html) ·
[Performance](https://prothon.readthedocs.io/en/latest/performance.html)

**Reference** — [CLI](https://prothon.readthedocs.io/en/latest/cli_reference.html) ·
[Configuration](https://prothon.readthedocs.io/en/latest/config.html) ·
[Python API](https://prothon.readthedocs.io/en/latest/api.html)

## Citation

> Aina, A.; Hsueh, S. C. C.; Plotkin, S. S. *PROTHON: A Local Order
> Parameter-Based Method for Efficient Comparison of Protein Ensembles.*
> J. Chem. Inf. Model. **2023**, 63, 3453–3461.
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

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

Built in the [AAI Research Lab](https://aai-research-lab.github.io) at
California State University Dominguez Hills, on MDTraj, NumPy, SciPy,
scikit-learn and Matplotlib.

</div>
