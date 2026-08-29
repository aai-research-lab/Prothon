"""Scoring an ensemble against experiment, rather than against another ensemble.

Comparing two ensembles says how they differ. It cannot say which is right.
This predicts what an experiment would have measured and checks against what it
did.

  - :mod:`observables` -- quantities computed exactly from the coordinates
  - :mod:`score`       -- agreement with measurements, beside what the
    sampling alone contributes

Observables that need an empirical predictor -- chemical shifts, SAXS profiles,
RDCs -- are deliberately not computed here. Compute them with the established
tool and score the numbers through :func:`score.score_observable`, which takes
predictions from anywhere.
"""

from .observables import (
    KARPLUS_VUISTER_BAX,
    NOT_COMPUTED,
    OBSERVABLES,
    Observable,
    average_observable,
    end_to_end,
    fret_efficiency,
    j_coupling_hn_ha,
    pairwise_distance,
    pre_distance,
    radius_of_gyration,
)
from .score import AgreementResult, score_observable

__all__ = [
    "AgreementResult",
    "KARPLUS_VUISTER_BAX",
    "NOT_COMPUTED",
    "OBSERVABLES",
    "Observable",
    "average_observable",
    "end_to_end",
    "fret_efficiency",
    "j_coupling_hn_ha",
    "pairwise_distance",
    "pre_distance",
    "radius_of_gyration",
    "score_observable",
]
