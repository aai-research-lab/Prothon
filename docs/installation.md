# Installation

```bash
pip install prothon-ensembles
```

Then check what was found:

```bash
prothon info
```

which lists the order parameters, the metrics, the sources `--ensembles`
accepts, and the version of every backend.

## The name

The distribution on PyPI is **`prothon-ensembles`**. PyPI's `prothon` was
registered in December 2020 by an unrelated project and names there are
permanent.

The import name and the command are both `prothon`:

```python
from prothon import Prothon
```

```bash
prothon --version
```

A conda-forge package named `prothon` is in review. When it lands,
`conda install -c conda-forge prothon` will install the same software under the
name used in the paper.

## Requirements

Python 3.9 or newer. The dependencies are installed automatically:

| package | used for |
|---|---|
| `mdtraj` | trajectory I/O and geometry |
| `numpy`, `scipy` | density estimation and statistics |
| `scikit-learn` | dimensionality reduction, and the classifier two-sample test |
| `matplotlib` | figures |

## From source

For a working copy:

```bash
git clone https://github.com/aai-research-lab/Prothon.git
cd Prothon
pip install -e ".[dev]"
pytest
```
