"""Tests for the whole-ensemble comparisons.

The case that justifies the module is the last one: two ensembles with
identical distributions at every feature that are nevertheless different
ensembles, because the features move together in one and not the other. Every
per-residue metric is blind to it by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from prothon.core.dissimilarity import jsd_local
from prothon.core.ensemble_metrics import (
    P_VALUE_FLOOR,
    EnsembleComparison,
    distinguishability,
)


def same(n=600, features=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, features)), rng.normal(size=(n, features))


class TestUnderTheNull:
    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_identical_distributions_are_not_distinguished(self, method):
        """The calibration check, in the form this module needs it."""
        flagged = 0
        for seed in range(8):
            a, b = same(seed=seed)
            result = distinguishability(a, b, method, random_state=seed)
            flagged += int(result.distinguishable)
        assert flagged <= 1

    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_an_ensemble_is_not_distinguishable_from_itself(self, method):
        """Two disjoint halves of one ensemble. A classifier scored on the
        data it was fitted to would separate these perfectly, which is why it
        is scored out of fold."""
        rng = np.random.default_rng(3)
        whole = rng.normal(size=(1200, 5))
        result = distinguishability(whole[:600], whole[600:], method, random_state=0)
        assert not result.distinguishable


class TestPower:
    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_a_shifted_ensemble_is_distinguished(self, method):
        rng = np.random.default_rng(4)
        a = rng.normal(0.0, 1.0, (600, 5))
        b = rng.normal(0.6, 1.0, (600, 5))
        assert distinguishability(a, b, method, random_state=0).distinguishable

    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_a_difference_in_spread_alone_is_distinguished(self, method):
        """Same mean, different flexibility -- a rigid loop against a mobile
        one. Not linearly separable, which is why the classifier is a forest
        and not a logistic regression."""
        rng = np.random.default_rng(5)
        a = rng.normal(0.0, 1.0, (700, 4))
        b = rng.normal(0.0, 2.0, (700, 4))
        assert distinguishability(a, b, method, random_state=0).distinguishable


class TestJointStructure:
    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_finds_a_difference_the_marginals_cannot(self, method):
        """Identical distributions at every feature; different ensembles.

        In one, features 0 and 1 are perfectly correlated; in the other they
        are independent. Every per-residue metric reports nothing, correctly --
        each feature really does have the same distribution. The ensembles are
        still different, and this is the only part of Prothon that can say so.
        """
        rng = np.random.default_rng(6)
        n = 800
        shared = rng.normal(size=n)
        a = np.column_stack(
            [shared, shared, rng.normal(size=n), rng.normal(size=n)]
        )
        b = rng.normal(size=(n, 4))

        assert jsd_local(a, b, -4, 4, 60).max() < 0.15  # the marginals agree
        assert distinguishability(a, b, method, random_state=0).distinguishable


class TestClassifierExtras:
    def test_reports_which_features_carried_the_difference(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(700, 6))
        b = a.copy()
        b = rng.normal(size=(700, 6))
        b[:, 2] += 3.0  # only feature 2 differs
        result = distinguishability(a, b, "c2st", random_state=0)
        leading = result.leading_features(1)
        assert leading[0][0] == 3  # one-based

    def test_feature_index_labels_the_importances(self):
        rng = np.random.default_rng(8)
        a, b = rng.normal(size=(400, 3)), rng.normal(1.5, 1, (400, 3))
        result = distinguishability(
            a, b, "c2st", random_state=0, feature_index=np.array([11, 12, 13])
        )
        assert {i for i, _ in result.leading_features(3)} == {11, 12, 13}

    def test_auc_is_reported_and_bounded(self):
        rng = np.random.default_rng(9)
        a, b = rng.normal(size=(400, 4)), rng.normal(3, 1, (400, 4))
        result = distinguishability(a, b, "c2st", random_state=0)
        assert 0.9 < result.effect <= 1.0

    def test_an_implausible_p_value_is_reported_as_a_bound(self):
        """A raw 1e-200 from a normal approximation's far tail is arithmetic,
        not evidence, and the summary should not print it as though it were."""
        rng = np.random.default_rng(10)
        a, b = rng.normal(size=(600, 4)), rng.normal(4, 1, (600, 4))
        result = distinguishability(a, b, "c2st", random_state=0)
        assert result.p_value < P_VALUE_FLOOR
        assert f"p < {P_VALUE_FLOOR:g}" in result.summary()


class TestMmdSpecifics:
    def test_reports_no_feature_importance(self):
        a, b = same(seed=11)
        assert distinguishability(a, b, "mmd", random_state=0).feature_importance is None
        assert distinguishability(a, b, "mmd", random_state=0).effect is None

    def test_offering_it_a_feature_index_warns(self):
        a, b = same(seed=12)
        with pytest.warns(UserWarning, match="no per-feature contribution"):
            distinguishability(
                a, b, "mmd", random_state=0, feature_index=np.arange(5)
            )

    def test_p_value_can_never_be_zero(self):
        rng = np.random.default_rng(13)
        a, b = rng.normal(size=(300, 4)), rng.normal(6, 1, (300, 4))
        result = distinguishability(a, b, "mmd", n_permutations=50, random_state=0)
        assert result.p_value == pytest.approx(1 / 51)


class TestCircularAndWeights:
    def test_circular_encoding_brings_the_wraparound_closer_for_mmd(self):
        """A Gaussian kernel measures Euclidean distance, so two populations
        either side of the wraparound look far apart as raw numbers and close
        once encoded as (cos, sin). Checked on the statistic, which is what
        the encoding acts on."""
        rng = np.random.default_rng(14)
        a = rng.vonmises(3.10, 80, (600, 3))
        b = rng.vonmises(-3.10, 80, (600, 3))  # 0.08 rad apart on the circle
        encoded = distinguishability(a, b, "mmd", circular=True, random_state=0)
        raw = distinguishability(a, b, "mmd", circular=False, random_state=0)
        assert encoded.statistic < raw.statistic

    def test_the_classifier_is_largely_immune_to_the_wraparound(self):
        """Not every method needs the encoding equally, and saying otherwise
        would be tidier than the truth. A decision tree splits at thresholds
        and can carve the wraparound out with one more split, so the forest
        reaches nearly the same answer either way -- unlike the kernel."""
        rng = np.random.default_rng(21)
        a = rng.vonmises(3.10, 80, (600, 3))
        b = rng.vonmises(-3.10, 80, (600, 3))
        encoded = distinguishability(a, b, "c2st", circular=True, random_state=0)
        raw = distinguishability(a, b, "c2st", circular=False, random_state=0)
        assert abs(encoded.effect - raw.effect) < 0.05

    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_weights_change_the_verdict(self, method):
        rng = np.random.default_rng(15)
        a = rng.normal(0.0, 1.0, (600, 4))
        b = np.vstack([rng.normal(0.0, 1.0, (300, 4)), rng.normal(5.0, 1.0, (300, 4))])
        # Weight the half of b that matches a.
        w = np.concatenate([np.full(300, 0.98 / 300), np.full(300, 0.02 / 300)])
        plain = distinguishability(a, b, method, random_state=0)
        weighted = distinguishability(a, b, method, weights_b=w, random_state=0)
        value = (lambda r: r.effect) if method == "c2st" else (lambda r: r.statistic)
        assert value(weighted) < value(plain)

    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_too_few_effective_samples_is_refused(self, method):
        rng = np.random.default_rng(16)
        a = rng.normal(size=(300, 3))
        b = rng.normal(size=(300, 3))
        w = np.full(300, 0.5 / 299)
        w[0] = 0.5  # worth about four independent conformations
        with pytest.raises(ValueError, match="independent conformations"):
            distinguishability(a, b, method, weights_b=w, random_state=0)


class TestApi:
    def test_unknown_method_is_refused(self):
        a, b = same(seed=17)
        with pytest.raises(ValueError, match="Unknown method"):
            distinguishability(a, b, "magic")

    def test_result_serialises(self):
        import json

        a, b = same(seed=18)
        payload = json.loads(
            json.dumps(distinguishability(a, b, "c2st", random_state=0).to_dict())
        )
        assert payload["method"] == "c2st"
        assert "effective_samples" in payload

    def test_summary_names_the_leading_residues(self):
        rng = np.random.default_rng(19)
        a = rng.normal(size=(500, 5))
        b = a + 0.0
        b = rng.normal(size=(500, 5))
        b[:, 1] += 4.0
        assert "driven mostly by residues" in distinguishability(
            a, b, "c2st", random_state=0
        ).summary()

    def test_is_an_ensemble_comparison(self):
        a, b = same(seed=20)
        assert isinstance(distinguishability(a, b, "mmd", random_state=0),
                          EnsembleComparison)
