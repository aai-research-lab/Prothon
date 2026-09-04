"""One probability-weight contract for every public analysis entry point."""

from __future__ import annotations

import numpy as np
import pytest

from prothon.compare import (
    dissimilarity,
    estimate_pdf,
    feature_distance,
    jsd_local,
    precision_recall,
)
from prothon.compare.joint import classifier_two_sample, maximum_mean_discrepancy
from prothon.sampling import effective_sample_size, validate_weights
from prothon.validate import average_observable, score_observable


@pytest.mark.parametrize(
    ("weights", "n", "message"),
    [
        ([1.0, 2.0], 3, "2 weights for 3 frames"),
        ([1.0, -0.1, 0.1], 3, "negative"),
        ([1.0, np.nan, 0.1], 3, "non-finite"),
        ([1.0, np.inf, 0.1], 3, "non-finite"),
        ([0.0, 0.0, 0.0], 3, "sum to zero"),
        ([np.finfo(float).max, np.finfo(float).max], 2, "finite.*positive sum"),
    ],
)
def test_invalid_probability_vectors_are_refused(weights, n, message):
    with pytest.raises(ValueError, match=message):
        validate_weights(weights, n)


@pytest.mark.parametrize(
    "weights",
    [
        [2.0, 3.0, 5.0],             # deliberately unnormalised
        [1.0e200, 1.0, 0.0],         # one dominant conformation
        [0.0, 4.0, 0.0],             # zero mass is valid if some mass is positive
    ],
)
def test_valid_probability_vectors_normalise_and_have_a_finite_ess(weights):
    normalised = validate_weights(weights, 3)

    assert np.all(np.isfinite(normalised))
    assert np.all(normalised >= 0.0)
    assert normalised.sum() == pytest.approx(1.0)
    ess = effective_sample_size(weights)
    assert np.isfinite(ess)
    assert 1.0 <= ess <= 3.0


def _public_weight_calls():
    rng = np.random.default_rng(9)
    a = rng.normal(size=(20, 2))
    b = rng.normal(size=(20, 2))
    x, y = a[:, 0], b[:, 0]
    experimental = np.zeros(2)
    uncertainty = np.ones(2)
    return [
        ("density", lambda w: estimate_pdf(x, -3.0, 3.0, 20, weights=w)),
        ("distance", lambda w: feature_distance(x, y, weights_x=w)),
        (
            "local distance",
            lambda w: jsd_local(a, b, -3.0, 3.0, 20, weights1=w),
        ),
        (
            "dissimilarity",
            lambda w: dissimilarity(a, b, -3.0, 3.0, weights_ref=w),
        ),
        (
            "precision/recall",
            lambda w: precision_recall(a, b, weights_ref=w),
        ),
        ("MMD", lambda w: maximum_mean_discrepancy(a, b, weights_a=w)),
        ("C2ST", lambda w: classifier_two_sample(a, b, weights_a=w)),
        ("observable average", lambda w: average_observable(a, weights=w)),
        (
            "observable score",
            lambda w: score_observable(
                a, experimental, uncertainty, weights=w
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("name", "call"),
    _public_weight_calls(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_public_weighted_analysis_uses_the_shared_contract(name, call):
    weights = np.ones(20)
    weights[0] = -1.0

    with pytest.raises(ValueError, match="negative"):
        call(weights)
