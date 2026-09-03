"""Tests for the whole-ensemble comparisons.

The case that justifies the module is the last one: two ensembles with
identical distributions at every feature that are nevertheless different
ensembles, because the features move together in one and not the other. Every
per-residue metric is blind to it by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from prothon.compare.dissimilarity import jsd_local
from prothon.compare.joint import (
    MINIMUM_MMD_UNITS,
    EnsembleComparison,
    _grouped_fold_plan,
    _mmd_signed_weights,
    _mmd_unit_assignment,
    classifier_two_sample,
    distinguishability,
    maximum_mean_discrepancy,
)


def same(n=600, features=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, features)), rng.normal(size=(n, features))


def ou(n, features, relaxation, rng, mean=0.0):
    """Stationary AR(1) trajectory with a known integrated time."""
    phi = np.exp(-1.0 / relaxation)
    values = np.empty((n, features))
    values[0] = rng.normal(size=features)
    noise = rng.normal(size=(n, features)) * np.sqrt(1.0 - phi**2)
    for frame in range(1, n):
        values[frame] = phi * values[frame - 1] + noise[frame]
    return values + mean


class TestUnderTheNull:
    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_identical_distributions_are_not_distinguished(self, method):
        """The calibration check, in the form this module needs it."""
        flagged = 0
        for seed in range(8):
            a, b = same(seed=seed)
            result = distinguishability(
                a,
                b,
                method,
                sampling_kind_a="iid",
                sampling_kind_b="iid",
                random_state=seed,
            )
            flagged += int(result.distinguishable)
        assert flagged <= 1

    @pytest.mark.parametrize("method", ["mmd", "c2st"])
    def test_an_ensemble_is_not_distinguishable_from_itself(self, method):
        """Two disjoint halves of one ensemble. A classifier scored on the
        data it was fitted to would separate these perfectly, which is why it
        is scored out of fold."""
        rng = np.random.default_rng(3)
        whole = rng.normal(size=(1200, 5))
        result = distinguishability(
            whole[:600],
            whole[600:],
            method,
            sampling_kind_a="iid",
            sampling_kind_b="iid",
            random_state=0,
        )
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

    def test_permutation_p_value_has_finite_resolution(self):
        rng = np.random.default_rng(10)
        a, b = rng.normal(size=(600, 4)), rng.normal(4, 1, (600, 4))
        result = distinguishability(
            a,
            b,
            "c2st",
            sampling_kind_a="iid",
            sampling_kind_b="iid",
            random_state=0,
        )
        assert result.p_value == pytest.approx(1 / 201)
        assert result.metadata["p_value_resolution"] == pytest.approx(1 / 201)
        assert "p = 0.00498" in result.summary()


class TestClassifierSamplingUnits:
    def test_a_complete_unit_is_never_split_between_folds(self):
        units_a = [np.arange(start, start + 4) for start in range(0, 24, 4)]
        units_b = [np.arange(start, start + 5) for start in range(24, 54, 5)]
        plan = _grouped_fold_plan(
            units_a,
            units_b,
            54,
            5,
            np.random.default_rng(0),
        )

        for unit in units_a + units_b:
            assert np.unique(plan.frame_folds[unit]).size == 1
        for fold in range(plan.n_folds):
            assert fold in plan.unit_folds_a
            assert fold in plan.unit_folds_b

    def test_complete_replicas_define_the_folds_and_null(self):
        def replicas(seed):
            rng = np.random.default_rng(seed)
            return np.vstack([ou(50, 4, 10.0, rng) for _ in range(8)])

        labels = np.repeat(np.arange(8), 50)
        result = classifier_two_sample(
            replicas(51),
            replicas(52),
            replica_labels_a=labels,
            replica_labels_b=labels,
            n_permutations=99,
            sample_size=500,
            random_state=0,
        )

        assert result.metadata["sampling_units"] == [8, 8]
        assert result.metadata["sampling_unit_sizes"] == [[50] * 8, [50] * 8]
        assert all(
            set(side) == set(range(result.metadata["folds"]))
            for side in result.metadata["cv_unit_folds"]
        )
        assert result.metadata["distinct_labelings"] >= 20
        assert result.p_value is not None

    def test_too_few_held_out_labelings_withholds_only_inference(self):
        a = ou(120, 3, 10.0, np.random.default_rng(53))
        b = ou(120, 3, 10.0, np.random.default_rng(54))
        with pytest.warns(UserWarning, match="p-value.*withheld"):
            result = classifier_two_sample(
                a,
                b,
                correlation_time_frames_a=20.0,
                correlation_time_frames_b=20.0,
                n_permutations=99,
                sample_size=120,
                random_state=0,
            )

        assert result.metadata["sampling_units"] == [3, 3]
        assert result.effect is not None
        assert result.p_value is None
        assert result.distinguishable is None
        assert "AUC" in result.summary()

    def test_metadata_records_folds_seeds_balance_and_provenance(self):
        rng = np.random.default_rng(55)
        a, b = rng.normal(size=(80, 3)), rng.normal(size=(120, 3))
        result = classifier_two_sample(
            a,
            b,
            weights_a=np.linspace(0.2, 2.0, 80),
            weights_b=np.linspace(2.0, 0.2, 120),
            sampling_kind_a="iid",
            sampling_kind_b="iid",
            time_stride_a=2,
            time_stride_b=5,
            n_permutations=39,
            random_state=0,
        )

        assert result.metadata["input_time_stride"] == [2, 5]
        assert len(result.metadata["cv_forest_seeds"]) == result.metadata["folds"]
        assert len(result.metadata["cv_fold_balance"]) == result.metadata["folds"]
        assert result.metadata["inference_train_balance"]["class_units"]
        assert result.metadata["inference_test_balance"]["class_weight_mass"]
        assert result.metadata["weights_attached_to_observations"]
        train_units = {
            tuple(unit) for unit in result.metadata["inference_train_units"]
        }
        test_units = {
            tuple(unit) for unit in result.metadata["inference_test_units"]
        }
        assert train_units.isdisjoint(test_units)
        assert len(train_units | test_units) == sum(result.metadata["sampling_units"])

    def test_trajectory_subsampling_is_contiguous_and_replanned(self):
        a = ou(1200, 3, 10.0, np.random.default_rng(57))
        b = ou(1200, 3, 10.0, np.random.default_rng(58))
        result = classifier_two_sample(
            a,
            b,
            correlation_time_frames_a=20.0,
            correlation_time_frames_b=20.0,
            n_permutations=99,
            random_state=0,
        )

        assert result.n_samples == (1000, 1000)
        assert result.metadata["sampling_units"] == [25, 25]
        for selection in result.metadata["sample_selection"]:
            assert selection["sampling_strategy"] == "contiguous window"
            assert selection["frame_range"][1] - selection["frame_range"][0] == 1000
        assert [
            plan["native_block_length"]
            for plan in result.metadata["sampling_plans"]
        ] == [40, 40]

    @pytest.mark.parametrize(
        "keyword,value,message",
        [
            ("folds", 1, "folds"),
            ("sample_size", 20.5, "sample_size"),
            ("n_permutations", True, "n_permutations"),
            ("time_stride_b", 0, "time_stride_b"),
            ("alpha", np.inf, "alpha"),
        ],
    )
    def test_invalid_inference_controls_are_refused(self, keyword, value, message):
        rng = np.random.default_rng(56)
        a, b = rng.normal(size=(40, 2)), rng.normal(size=(40, 2))
        with pytest.raises(ValueError, match=message):
            classifier_two_sample(a, b, **{keyword: value})


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


class TestMmdSamplingUnits:
    def test_a_permutation_never_splits_a_sampling_unit(self):
        units = [
            np.arange(0, 4),
            np.arange(4, 10),
            np.arange(10, 13),
            np.arange(13, 20),
        ]
        rng = np.random.default_rng(30)
        for _ in range(100):
            left, right = _mmd_unit_assignment(units, 2, rng)
            for unit in units:
                assert np.isin(unit, left).all() or np.isin(unit, right).all()

    def test_probability_mass_stays_on_its_observation(self):
        mass = np.array([0.8, 0.2, 0.5, 0.5])
        left = np.array([0, 2])
        right = np.array([1, 3])

        signed = _mmd_signed_weights(mass, left, right)

        np.testing.assert_allclose(
            signed,
            [0.8 / 1.3, -0.2 / 0.7, 0.5 / 1.3, -0.5 / 0.7],
        )
        assert signed.sum() == pytest.approx(0.0)

    def test_too_few_units_withholds_only_the_p_value(self):
        a = ou(120, 3, 10.0, np.random.default_rng(31))
        b = ou(120, 3, 10.0, np.random.default_rng(32))
        with pytest.warns(UserWarning, match="p-value.*withheld"):
            result = maximum_mean_discrepancy(
                a, b,
                correlation_time_frames_a=20.0,
                correlation_time_frames_b=20.0,
                n_permutations=20,
                sample_size=120,
                random_state=0,
            )

        assert result.metadata["permutation_units"] == [3, 3]
        assert result.metadata["distinct_labelings"] == 10
        assert result.statistic > 0.0
        assert result.p_value is None
        assert result.distinguishable is None
        assert result.p_value_withheld
        assert "MMD²" in result.summary()
        assert result.to_dict()["distinguishable"] is None

    def test_complete_replicas_are_the_units(self):
        def replicas(seed):
            rng = np.random.default_rng(seed)
            return np.vstack([ou(50, 4, 10.0, rng) for _ in range(8)])

        labels = np.repeat(np.arange(8), 50)
        result = maximum_mean_discrepancy(
            replicas(33), replicas(34),
            replica_labels_a=labels,
            replica_labels_b=labels,
            n_permutations=99,
            sample_size=500,
            random_state=0,
        )

        assert result.metadata["permutation_units"] == [8, 8]
        assert result.metadata["permutation_unit_sizes"] == [[50] * 8, [50] * 8]
        assert result.effective_samples == pytest.approx((8.0, 8.0))
        assert not result.p_value_withheld
        assert not result.distinguishable

    def test_default_subsample_is_contiguous_and_replans_the_tested_frames(self):
        a = ou(1200, 3, 10.0, np.random.default_rng(44))
        b = ou(1200, 3, 10.0, np.random.default_rng(45))
        result = maximum_mean_discrepancy(
            a,
            b,
            correlation_time_frames_a=20.0,
            correlation_time_frames_b=20.0,
            n_permutations=20,
            random_state=0,
        )

        assert result.n_samples == (1000, 1000)
        assert result.metadata["permutation_units"] == [25, 25]
        for selection in result.metadata["sample_selection"]:
            assert selection["original_frames"] == 1200
            assert selection["sampling_strategy"] == "contiguous window"
            assert selection["frame_range"][1] - selection["frame_range"][0] == 1000
        assert [
            plan["native_block_length"]
            for plan in result.metadata["sampling_plans"]
        ] == [40, 40]

    def test_complete_replicas_survive_computational_subsampling(self):
        labels = np.repeat(np.arange(8), 50)
        rng = np.random.default_rng(35)
        a, b = rng.normal(size=(400, 3)), rng.normal(size=(400, 3))
        result = maximum_mean_discrepancy(
            a, b,
            replica_labels_a=labels,
            replica_labels_b=labels,
            n_permutations=20,
            sample_size=220,
            random_state=0,
        )

        assert result.n_samples == (200, 200)
        assert result.metadata["permutation_units"] == [4, 4]
        assert result.metadata["permutation_unit_sizes"] == [[50] * 4, [50] * 4]
        assert not result.p_value_withheld

    def test_repeated_permutations_cannot_exceed_exact_unit_resolution(self):
        rng = np.random.default_rng(39)
        labels = np.repeat(np.arange(4), 20)
        a = rng.normal(size=(80, 3))
        b = rng.normal(8.0, 1.0, size=(80, 3))
        result = maximum_mean_discrepancy(
            a,
            b,
            replica_labels_a=labels,
            replica_labels_b=labels,
            n_permutations=200,
            sample_size=80,
            random_state=0,
        )

        assert result.metadata["distinct_labelings"] == 35
        assert result.metadata["p_value_resolution"] == pytest.approx(1 / 35)
        assert result.p_value >= 1 / 35

    def test_too_few_requested_permutations_withholds_inference(self):
        rng = np.random.default_rng(43)
        a, b = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
        with pytest.warns(UserWarning, match="resolution.*alpha"):
            result = maximum_mean_discrepancy(
                a,
                b,
                sampling_kind_a="iid",
                sampling_kind_b="iid",
                n_permutations=19,
                random_state=0,
            )

        assert result.metadata["p_value_resolution"] == pytest.approx(0.05)
        assert result.statistic > 0.0
        assert result.p_value is None

    def test_probability_concentrated_in_too_few_blocks_withholds_inference(self):
        rng = np.random.default_rng(38)
        a, b = rng.normal(size=(200, 3)), rng.normal(size=(200, 3))
        weights_b = np.concatenate(
            [np.full(40, 0.8 / 40), np.full(160, 0.2 / 160)]
        )
        with pytest.warns(UserWarning, match="effective.*p-value"):
            result = maximum_mean_discrepancy(
                a,
                b,
                weights_b=weights_b,
                correlation_time_frames_a=20.0,
                correlation_time_frames_b=20.0,
                n_permutations=20,
                sample_size=200,
                random_state=0,
            )

        assert result.metadata["permutation_units"] == [5, 5]
        assert result.metadata["frame_weight_effective_samples"][1] > 10
        assert result.effective_samples[1] < MINIMUM_MMD_UNITS
        assert result.statistic > 0.0
        assert result.p_value is None

    @pytest.mark.parametrize(
        "keyword,value,message",
        [
            ("sample_size", 20.5, "sample_size"),
            ("n_permutations", True, "n_permutations"),
            ("time_stride_a", 0, "time_stride_a"),
            ("alpha", np.nan, "alpha"),
        ],
    )
    def test_invalid_inference_controls_are_refused(self, keyword, value, message):
        rng = np.random.default_rng(36)
        a, b = rng.normal(size=(40, 2)), rng.normal(size=(40, 2))
        with pytest.raises(ValueError, match=message):
            maximum_mean_discrepancy(a, b, **{keyword: value})

    def test_nonfinite_coordinates_are_refused(self):
        rng = np.random.default_rng(37)
        a, b = rng.normal(size=(40, 2)), rng.normal(size=(40, 2))
        b[3, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            maximum_mean_discrepancy(a, b)


class TestMmdCorrelatedCalibration:
    """Predeclared calibration: null 0-3/20; power at least 8/10."""

    settings = {
        "n_permutations": 99,
        "sample_size": 500,
        "sampling_kind_a": "trajectory",
        "sampling_kind_b": "trajectory",
        "correlation_time_frames_a": 20.0,
        "correlation_time_frames_b": 20.0,
    }

    def test_block_null_replaces_the_reproduced_row_failure(self):
        blocked_calls = 0
        row_calls = 0
        for seed in range(20):
            a = ou(500, 4, 10.0, np.random.default_rng(100 + seed))
            b = ou(500, 4, 10.0, np.random.default_rng(200 + seed))
            blocked = maximum_mean_discrepancy(
                a, b, random_state=seed, **self.settings
            )
            blocked_calls += int(blocked.distinguishable)
            if seed < 8:
                row = maximum_mean_discrepancy(
                    a, b,
                    n_permutations=99,
                    sample_size=500,
                    sampling_kind_a="iid",
                    sampling_kind_b="iid",
                    random_state=seed,
                )
                row_calls += int(row.distinguishable)

        assert row_calls == 8, "row permutation reproduces the audited failure"
        assert blocked_calls <= 3, "outside the 95% binomial null band"

    def test_block_null_retains_power(self):
        calls = 0
        for seed in range(10):
            a = ou(500, 4, 10.0, np.random.default_rng(300 + seed))
            b = ou(500, 4, 10.0, np.random.default_rng(400 + seed), mean=0.5)
            result = maximum_mean_discrepancy(
                a, b, random_state=seed, **self.settings
            )
            calls += int(result.distinguishable)
        assert calls >= 8

    @pytest.mark.parametrize(
        "case",
        ["unequal lengths", "unequal weights", "trajectory versus iid", "circular"],
    )
    def test_other_sampling_designs_remain_calibrated(self, case):
        if case == "unequal lengths":
            a = ou(500, 4, 10.0, np.random.default_rng(1))
            b = ou(700, 4, 10.0, np.random.default_rng(2))
            kwargs = {
                "sample_size": 700,
                "correlation_time_frames_a": 20.0,
                "correlation_time_frames_b": 20.0,
            }
        elif case == "unequal weights":
            rng = np.random.default_rng(3)
            a = rng.normal(size=(320, 4))
            b = rng.normal(size=(470, 4))
            kwargs = {
                "weights_a": np.linspace(0.2, 2.0, 320),
                "weights_b": np.linspace(2.0, 0.2, 470) ** 2,
                "sample_size": 500,
                "sampling_kind_a": "iid",
                "sampling_kind_b": "iid",
            }
        elif case == "trajectory versus iid":
            rng = np.random.default_rng(4)
            a = ou(500, 4, 10.0, rng)
            b = rng.normal(size=(650, 4))
            kwargs = {
                "sample_size": 700,
                "sampling_kind_a": "trajectory",
                "sampling_kind_b": "iid",
                "correlation_time_frames_a": 20.0,
            }
        else:
            rng = np.random.default_rng(7)
            a = np.pi + 0.5 * ou(500, 4, 10.0, rng)
            b = np.pi + 0.5 * ou(500, 4, 10.0, rng)
            a = (a + np.pi) % (2.0 * np.pi) - np.pi
            b = (b + np.pi) % (2.0 * np.pi) - np.pi
            kwargs = {
                "circular": True,
                "sample_size": 500,
                "correlation_time_frames_a": 20.0,
                "correlation_time_frames_b": 20.0,
            }

        result = maximum_mean_discrepancy(
            a, b, n_permutations=99, random_state=0, **kwargs
        )

        assert result.p_value is not None
        assert result.p_value >= 0.05
        assert min(result.metadata["permutation_units"]) >= MINIMUM_MMD_UNITS


class TestClassifierCorrelatedCalibration:
    """Predeclared calibration: null 0-3/20; power at least 8/10."""

    settings = {
        "n_permutations": 99,
        "sample_size": 500,
        "sampling_kind_a": "trajectory",
        "sampling_kind_b": "trajectory",
        "correlation_time_frames_a": 20.0,
        "correlation_time_frames_b": 20.0,
    }

    @staticmethod
    def _legacy_row_result(a, b, random_state):
        """The replaced shuffled-row CV and independent-row normal null."""
        from scipy.stats import norm
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        rng = np.random.default_rng(random_state)
        seed = int(rng.integers(0, 2**31 - 1))
        features = np.vstack([a, b])
        labels = np.repeat([0, 1], [len(a), len(b)])
        predictions = np.zeros(labels.size)
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        for train, test in splitter.split(features, labels):
            forest = RandomForestClassifier(
                n_estimators=200,
                random_state=seed,
                n_jobs=1,
                min_samples_leaf=2,
            )
            forest.fit(features[train], labels[train])
            predictions[test] = forest.predict_proba(features[test])[:, 1]
        accuracy = np.mean((predictions >= 0.5) == labels)
        p_value = norm.sf(2.0 * np.sqrt(labels.size) * (accuracy - 0.5))
        return float(roc_auc_score(labels, predictions)), float(p_value)

    def test_grouped_folds_replace_the_reproduced_row_leakage(self):
        grouped_calls = 0
        grouped_auc = []
        legacy_calls = 0
        legacy_auc = []
        for seed in range(20):
            a = ou(500, 4, 10.0, np.random.default_rng(100 + seed))
            b = ou(500, 4, 10.0, np.random.default_rng(200 + seed))
            grouped = classifier_two_sample(
                a, b, random_state=seed, **self.settings
            )
            grouped_calls += int(grouped.distinguishable)
            grouped_auc.append(grouped.effect)
            if seed < 10:
                auc, p_value = self._legacy_row_result(a, b, seed)
                legacy_auc.append(auc)
                legacy_calls += int(p_value < 0.05)

        assert legacy_calls == 10, "shuffled rows reproduce the audited failure"
        assert min(legacy_auc) > 0.65
        assert grouped_calls <= 3, "outside the 95% binomial null band"
        assert 0.4 <= np.mean(grouped_auc) <= 0.6

    def test_grouped_classifier_retains_power(self):
        calls = 0
        for seed in range(10):
            a = ou(500, 4, 10.0, np.random.default_rng(300 + seed))
            b = ou(
                500,
                4,
                10.0,
                np.random.default_rng(400 + seed),
                mean=0.7,
            )
            result = classifier_two_sample(
                a, b, random_state=seed, **self.settings
            )
            calls += int(result.distinguishable)
        assert calls >= 8

    @pytest.mark.parametrize(
        "case",
        ["unequal lengths", "unequal weights", "trajectory versus iid", "circular"],
    )
    def test_other_sampling_designs_remain_calibrated(self, case):
        if case == "unequal lengths":
            a = ou(500, 4, 10.0, np.random.default_rng(1))
            b = ou(700, 4, 10.0, np.random.default_rng(2))
            kwargs = {
                "sample_size": 700,
                "correlation_time_frames_a": 20.0,
                "correlation_time_frames_b": 20.0,
            }
        elif case == "unequal weights":
            rng = np.random.default_rng(3)
            a = rng.normal(size=(320, 4))
            b = rng.normal(size=(470, 4))
            kwargs = {
                "weights_a": np.linspace(0.2, 2.0, 320),
                "weights_b": np.linspace(2.0, 0.2, 470) ** 2,
                "sample_size": 500,
                "sampling_kind_a": "iid",
                "sampling_kind_b": "iid",
            }
        elif case == "trajectory versus iid":
            rng = np.random.default_rng(4)
            a = ou(500, 4, 10.0, rng)
            b = rng.normal(size=(650, 4))
            kwargs = {
                "sample_size": 700,
                "sampling_kind_a": "trajectory",
                "sampling_kind_b": "iid",
                "correlation_time_frames_a": 20.0,
            }
        else:
            rng = np.random.default_rng(7)
            a = np.pi + 0.5 * ou(500, 4, 10.0, rng)
            b = np.pi + 0.5 * ou(500, 4, 10.0, rng)
            a = (a + np.pi) % (2.0 * np.pi) - np.pi
            b = (b + np.pi) % (2.0 * np.pi) - np.pi
            kwargs = {
                "circular": True,
                "sample_size": 500,
                "correlation_time_frames_a": 20.0,
                "correlation_time_frames_b": 20.0,
            }

        result = classifier_two_sample(
            a, b, n_permutations=99, random_state=0, **kwargs
        )

        assert result.p_value is not None
        assert result.p_value >= 0.05
        assert min(result.metadata["sampling_units"]) >= 4


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
