"""Tests for the coverage/fidelity split.

The case the module exists for is the first class below: two models wrong in
opposite ways, which any symmetric distance scores about the same and which
need entirely different work.
"""

from __future__ import annotations

import numpy as np
import pytest

from prothon.core.dissimilarity import jsd_local
from prothon.core.precision_recall import precision_recall


def ensemble(n=1200, open_fraction=0.35, extra_state=False, seed=0):
    """Four rigid residues, then four that visit a second state some of the
    time -- a loop that opens. ``extra_state`` adds a third state the
    reference never visits."""
    rng = np.random.default_rng(seed)
    columns = [rng.normal(0, 0.5, n) for _ in range(4)]
    for _ in range(4):
        opens = rng.random(n) < open_fraction
        columns.append(np.where(opens, rng.normal(4, 0.5, n), rng.normal(0, 0.5, n)))
    matrix = np.column_stack(columns)
    if extra_state:
        k = int(0.25 * n)
        matrix[:k, 4:] = rng.normal(-4, 0.5, (k, 4))
    return matrix


class TestTheDistinctionItExistsFor:
    def test_a_symmetric_distance_cannot_tell_them_apart(self):
        reference = ensemble(seed=1)
        collapse = ensemble(open_fraction=0.0, seed=2)
        invention = ensemble(extra_state=True, seed=3)
        a = jsd_local(reference, collapse, -7, 7, 120).mean()
        b = jsd_local(reference, invention, -7, 7, 120).mean()
        assert abs(a - b) < 0.1  # the premise: they look alike

    def test_mode_collapse_shows_as_low_recall(self):
        reference = ensemble(seed=1)
        collapse = ensemble(open_fraction=0.0, seed=2)
        result = precision_recall(reference, collapse, random_state=0)
        assert result.mean_recall < result.mean_floor_recall - 0.1
        assert result.mean_precision > result.mean_floor_precision - 0.05
        assert set(result.missed()) == {5, 6, 7, 8}
        assert result.invented().size == 0

    def test_hallucination_shows_as_low_precision(self):
        reference = ensemble(seed=1)
        invention = ensemble(extra_state=True, seed=3)
        result = precision_recall(reference, invention, random_state=0)
        assert result.mean_precision < result.mean_floor_precision - 0.05
        assert result.mean_recall > result.mean_floor_recall - 0.05
        assert set(result.invented()) == {5, 6, 7, 8}
        assert result.missed().size == 0

    def test_a_good_model_flags_nothing(self):
        reference = ensemble(seed=1)
        good = ensemble(seed=4)
        result = precision_recall(reference, good, random_state=0)
        assert result.missed().size == 0
        assert result.invented().size == 0
        assert "not resolvable" in result.summary()


class TestTheFloor:
    def test_is_per_feature_not_averaged(self):
        """One averaged threshold flags about half the unchanged residues in a
        protein, which is how a plausible wrong answer gets produced.

        The floors differ by feature, and not in the direction one would guess:
        the four rigid residues floor at about 0.956 and the four bimodal ones
        at about 0.997. A smoothed multimodal density has a wide
        highest-density region, so a nominal 95% level over-covers there. That
        is a property of the estimate rather than of the protein, and it is
        precisely why the floor cannot be one number.
        """
        reference = ensemble(seed=1)
        result = precision_recall(reference, ensemble(seed=5), random_state=0)
        assert result.floor_recall.shape == (8,)
        spread = result.floor_recall.max() - result.floor_recall.min()
        assert spread > 0.02, "floors that vary this little would not need to be per-feature"
        assert result.floor_recall[4:].mean() > result.floor_recall[:4].mean()

    def test_identical_distributions_sit_at_the_coverage_level(self):
        """The null value is exact by construction, which is what makes a
        departure from it readable."""
        rng = np.random.default_rng(6)
        a, b = rng.normal(size=(2000, 4)), rng.normal(size=(2000, 4))
        result = precision_recall(a, b, coverage=0.9, random_state=0)
        assert result.mean_precision == pytest.approx(0.9, abs=0.05)
        assert result.mean_recall == pytest.approx(0.9, abs=0.05)

    def test_coverage_level_is_honoured(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=(1500, 3)), rng.normal(size=(1500, 3))
        for level in (0.5, 0.8, 0.95):
            result = precision_recall(a, b, coverage=level, random_state=0)
            assert result.mean_precision == pytest.approx(level, abs=0.07)


class TestAsymmetryAndInputs:
    def test_swapping_the_arguments_swaps_precision_and_recall(self):
        reference = ensemble(seed=1)
        collapse = ensemble(open_fraction=0.0, seed=2)
        forward = precision_recall(reference, collapse, random_state=0)
        backward = precision_recall(collapse, reference, random_state=0)
        assert forward.mean_recall == pytest.approx(backward.mean_precision, abs=0.03)

    def test_weights_are_used(self):
        rng = np.random.default_rng(8)
        reference = rng.normal(0, 1, (1500, 3))
        other = np.vstack([rng.normal(0, 1, (750, 3)), rng.normal(6, 1, (750, 3))])
        w = np.concatenate([np.full(750, 0.99 / 750), np.full(750, 0.01 / 750)])
        plain = precision_recall(reference, other, random_state=0)
        weighted = precision_recall(reference, other, weights=w, random_state=0)
        assert weighted.mean_precision > plain.mean_precision + 0.2

    def test_treating_angles_as_linear_hides_a_real_difference(self):
        """The circular failure here is a false negative, not a false positive.

        Two von Mises populations centred at +3.0 and -3.0 radians are 0.28
        apart on the circle and each has a spread of 0.22, so they are
        genuinely different. Sampled, both wrap past the cut, so as raw numbers
        both look like the same bimodal distribution piled at either end -- and
        a linear treatment reports precision and recall of exactly 1.0 for
        ensembles that do not match.
        """
        rng = np.random.default_rng(9)
        a = rng.vonmises(3.0, 20, (1200, 3))
        b = rng.vonmises(-3.0, 20, (1200, 3))

        circular = precision_recall(a, b, circular=True, random_state=0)
        linear = precision_recall(a, b, circular=False, random_state=0)

        assert circular.missed().size == 3 and circular.invented().size == 3
        assert linear.missed().size == 0 and linear.invented().size == 0
        assert linear.mean_precision > circular.mean_precision + 0.2

    def test_feature_index_labels_the_findings(self):
        reference = ensemble(seed=1)
        collapse = ensemble(open_fraction=0.0, seed=2)
        labels = np.array([10, 11, 12, 13, 40, 41, 42, 43])
        result = precision_recall(
            reference, collapse, feature_index=labels, random_state=0
        )
        assert set(result.missed()) == {40, 41, 42, 43}

    def test_mismatched_features_are_refused(self):
        with pytest.raises(ValueError, match="Feature counts differ"):
            precision_recall(np.zeros((100, 3)), np.zeros((100, 4)))

    def test_a_nonsensical_coverage_is_refused(self):
        a = np.random.default_rng(0).normal(size=(200, 2))
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            precision_recall(a, a, coverage=1.0)

    def test_too_few_effective_conformations_is_refused(self):
        rng = np.random.default_rng(10)
        a = rng.normal(size=(300, 3))
        w = np.full(300, 0.5 / 299)
        w[0] = 0.5
        with pytest.raises(ValueError, match="independent conformations"):
            precision_recall(a, a, weights=w)

    def test_result_serialises(self):
        import json

        reference = ensemble(seed=1)
        result = precision_recall(reference, ensemble(seed=4), random_state=0)
        payload = json.loads(json.dumps(result.to_dict()))
        assert "mean_floor_recall" in payload
        assert isinstance(payload["missed"], list)
