"""Prothon: efficient comparison of protein conformational ensembles.

An ensemble is represented as a vector of probability distributions over local
order parameters -- contact numbers, virtual bond and torsion angles, solvent
accessibility -- and two ensembles are compared by the Jensen-Shannon distance
between corresponding distributions. Because the representation is local, no
structural superposition is required and the cost is linear in the number of
frames rather than quadratic.

    Aina, A.; Hsueh, S. C. C.; Plotkin, S. S. PROTHON: A Local Order
    Parameter-Based Method for Efficient Comparison of Protein Ensembles.
    J. Chem. Inf. Model. 2023, 63 (11), 3453-3461.

Typical use::

    from prothon import Prothon

    study = Prothon(["wt.dcd", "mutant.dcd"], topology="top.pdb")
    results = study.compare_ensembles(methods="cbcn")

The public surface is deliberately small: :class:`Prothon` for whole studies,
and the functions in :mod:`prothon.core` for anyone assembling their own.
"""

from __future__ import annotations

#: Supported Python range. Declared here and read by pyproject.toml and CI,
#: so the three cannot drift apart.
MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 13)

try:  # pragma: no cover - depends on install method
    from ._version import __version__
except ImportError:  # pragma: no cover - source checkout without setuptools-scm
    __version__ = "2.1.0.dev0"

from .core.dissimilarity import (
    ComparisonResult,
    dissimilarity,
    effective_sample_size,
    estimate_pdf,
    jsd_local,
)
from .core.ensemble_metrics import EnsembleComparison, distinguishability
from .core.metrics import METRICS, describe_metric, feature_distance
from .core.precision_recall import PrecisionRecall, precision_recall
from .core.prothon_core import Prothon
from .core.representation import (
    MEASURES,
    compute_ensemble_representation,
    describe_measure,
)
from .utils import load_trajectories

__all__ = [
    "Prothon",
    "ComparisonResult",
    "EnsembleComparison",
    "PrecisionRecall",
    "precision_recall",
    "distinguishability",
    "MEASURES",
    "METRICS",
    "describe_metric",
    "effective_sample_size",
    "feature_distance",
    "compute_ensemble_representation",
    "describe_measure",
    "dissimilarity",
    "estimate_pdf",
    "jsd_local",
    "load_trajectories",
    "__version__",
]
