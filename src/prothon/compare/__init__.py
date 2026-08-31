"""How far apart two ensembles are.

Per residue and globally, with three distances to choose between, plus the two
questions a symmetric per-residue distance cannot answer: whether the
difference lies in the relationship *between* residues, and whether an ensemble
is missing states or inventing them.
"""

from .coverage import PrecisionRecall, precision_recall
from .density import estimate_pdf
from .dissimilarity import ComparisonResult, dissimilarity, jsd_local
from .distance import METRICS, feature_distance, resolve_metric

__all__ = [
    "METRICS",
    "ComparisonResult",
    "PrecisionRecall",
    "dissimilarity",
    "estimate_pdf",
    "feature_distance",
    "jsd_local",
    "precision_recall",
    "resolve_metric",
]
