"""Core subpackage: representation, dissimilarity, plotting, and the study class.

  - :mod:`representation` -- local order parameters and ensemble matrices
  - :mod:`dissimilarity`  -- density estimation, Jensen-Shannon distance, statistics
  - :mod:`plotting`       -- figures and the files beside them
  - :mod:`prothon_core`   -- the :class:`~prothon.Prothon` study object
"""

from .dissimilarity import ComparisonResult, dissimilarity, estimate_pdf, jsd_local
from .plotting import get_ensemble_colors, get_method_output_dir
from .prothon_core import Prothon
from .representation import MEASURES, compute_ensemble_representation

__all__ = [
    "ComparisonResult",
    "MEASURES",
    "Prothon",
    "compute_ensemble_representation",
    "dissimilarity",
    "estimate_pdf",
    "get_ensemble_colors",
    "get_method_output_dir",
    "jsd_local",
]
