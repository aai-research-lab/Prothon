"""Tests for back-calculated observables and agreement with experiment.

Two things here are easy to get wrong in ways that look right: averaging a
sixth-power observable linearly, and comparing a chi-squared against 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_ingest import as_residues, build

from prothon.validate import (
    KARPLUS_VUISTER_BAX,
    average_observable,
    end_to_end,
    fret_efficiency,
    j_coupling_hn_ha,
    pairwise_distance,
    pre_distance,
    radius_of_gyration,
    score_observable,
)

SEQ = "ACDEFHIKLMNP"


@pytest.fixture(scope="module")
def traj():
    return build(as_residues(SEQ), n_frames=400, seed=1)


class TestAveraging:
    def test_linear_is_the_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(5.0, 1.0, 5000)
        assert average_observable(values) == pytest.approx(values.mean())

    def test_the_sixth_power_average_is_not_the_mean(self):
        """The case that matters, and the reason each observable declares how
        it averages.

        A distance distribution with a rare compact state: 90% at 5.0 nm and
        10% at 1.5 nm. A PRE reports the sixth-power average, which is 2.19 nm;
        the linear mean is 4.64 nm. Two and a half nanometres of error, in the
        direction of missing exactly the rare compact state that PRE exists to
        detect.
        """
        rng = np.random.default_rng(0)
        distances = np.where(rng.random(20000) < 0.10, 1.5, 5.0)
        linear = average_observable(distances, averaging="linear")
        sixth = average_observable(distances, averaging="r6")
        assert linear == pytest.approx(4.65, abs=0.05)
        assert sixth == pytest.approx(2.19, abs=0.05)
        assert linear - sixth > 2.0

    def test_the_two_agree_on_a_narrow_distribution(self):
        """Which is why a rigid test case cannot catch the mistake."""
        rng = np.random.default_rng(1)
        distances = rng.normal(4.0, 0.02, 5000)
        assert average_observable(distances, averaging="r6") == pytest.approx(
            average_observable(distances, averaging="linear"), abs=0.01
        )

    def test_weights_are_used(self):
        values = np.array([1.0, 9.0])
        assert average_observable(values, weights=[0.9, 0.1]) == pytest.approx(1.8)

    def test_a_negative_distance_is_refused(self):
        with pytest.raises(ValueError, match="positive distances"):
            average_observable(np.array([1.0, -1.0]), averaging="r6")

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="Unknown averaging"):
            average_observable(np.array([1.0]), averaging="geometric")


class TestObservables:
    def test_radius_of_gyration_is_positive_and_per_frame(self, traj):
        rg = radius_of_gyration(traj)
        assert rg.shape == (traj.n_frames,)
        assert (rg > 0).all()

    def test_a_compact_structure_has_a_smaller_radius(self):
        extended = build(as_residues(SEQ), n_frames=20, seed=2)
        compact = build(as_residues(SEQ), n_frames=20, seed=2, compact_from=2)
        assert radius_of_gyration(compact).mean() < radius_of_gyration(extended).mean()

    def test_end_to_end_is_shorter_than_the_contour_length(self, traj):
        distance = end_to_end(traj)
        assert distance.shape == (traj.n_frames,)
        assert (distance < 0.4 * len(SEQ)).all()

    def test_karplus_reproduces_the_textbook_values(self):
        """A helix gives about 4 Hz and a sheet about 9 Hz. If those move, the
        coefficients or the phase are wrong."""
        a, b, c = KARPLUS_VUISTER_BAX

        def karplus(phi_degrees):
            theta = np.radians(phi_degrees - 60.0)
            return a * np.cos(theta) ** 2 + b * np.cos(theta) + c

        assert karplus(-57) == pytest.approx(3.7, abs=0.3)    # alpha helix
        assert karplus(-139) == pytest.approx(9.1, abs=0.3)   # beta sheet

    def test_j_couplings_are_in_a_physical_range(self, traj):
        couplings, residues = j_coupling_hn_ha(traj)
        assert couplings.shape[0] == traj.n_frames
        assert couplings.shape[1] == residues.size
        # The Karplus curve is bounded by its coefficients.
        assert couplings.min() >= -0.5 and couplings.max() <= 12.0

    def test_fret_efficiency_is_bounded_and_falls_with_distance(self, traj):
        efficiency = fret_efficiency(traj, "name CA and resid 0",
                                     "name CA and resid 11", r0=4.0)
        assert ((efficiency >= 0) & (efficiency <= 1)).all()
        near = fret_efficiency(traj, "name CA and resid 0",
                               "name CA and resid 1", r0=4.0)
        assert near.mean() > efficiency.mean()

    def test_fret_at_the_forster_radius_is_one_half(self, traj):
        distances = pairwise_distance(traj, "name CA and resid 0",
                                      "name CA and resid 11")
        efficiency = fret_efficiency(traj, "name CA and resid 0",
                                     "name CA and resid 11",
                                     r0=float(distances.mean()))
        assert efficiency.mean() == pytest.approx(0.5, abs=0.1)

    def test_a_zero_forster_radius_is_refused(self, traj):
        with pytest.raises(ValueError, match="must be positive"):
            fret_efficiency(traj, 0, 5, r0=0.0)

    def test_an_empty_selection_is_refused(self, traj):
        with pytest.raises(ValueError, match="matched no atoms"):
            pairwise_distance(traj, "name ZZ", "name CA")

    def test_pre_uses_the_sixth_power(self, traj):
        both = (
            pre_distance(traj, "name CA and resid 0", "name CA and resid 11"),
            pairwise_distance(traj, "name CA and resid 0",
                              "name CA and resid 11").mean(),
        )
        assert both[0] <= both[1] + 1e-9      # r6 can never exceed the mean


class TestAgreement:
    @staticmethod
    def perfect(n, n_points=20, seed=0, spread=2.0):
        """An ensemble whose true average is the experimental value."""
        rng = np.random.default_rng(seed)
        experimental = rng.normal(8.0, 1.5, n_points)
        per_frame = experimental[None, :] + rng.normal(0, spread, (n, n_points))
        return per_frame, experimental, np.full(n_points, 0.5)

    def test_a_perfect_ensemble_does_not_score_one(self):
        """The reason a floor is reported at all.

        chi2_red = 1 is the usual target, and for an ensemble it is wrong in
        both directions: a perfect ensemble of twenty conformations scores
        about 0.8 and a perfect ensemble of five thousand scores about 0.0.
        Fitting either to 1.0 is fitting to noise.
        """
        small = score_observable(*self.perfect(20), random_state=0)
        large = score_observable(*self.perfect(5000), random_state=0)
        assert small.chi2_reduced > 5 * large.chi2_reduced
        assert large.chi2_reduced < 0.1

    def test_a_perfect_ensemble_is_within_its_floor(self):
        for n in (50, 500, 5000):
            result = score_observable(*self.perfect(n, seed=n), random_state=0)
            assert result.within_floor, f"n={n} scored outside its own floor"

    def test_a_biased_ensemble_is_not(self):
        per_frame, experimental, sigma = self.perfect(2000, seed=3)
        per_frame = per_frame + 2.0          # two sigma out, systematically
        result = score_observable(per_frame, experimental, sigma, random_state=0)
        assert not result.within_floor
        assert result.chi2_reduced > 10

    def test_the_floor_falls_as_the_ensemble_grows(self):
        floors = [
            score_observable(*self.perfect(n, seed=7), random_state=0).floor
            for n in (50, 500, 5000)
        ]
        assert floors[0] > floors[1] > floors[2]

    def test_floor_distribution_and_upper_tail_threshold_are_recorded(self):
        result = score_observable(
            *self.perfect(500), sampling_kind="iid", random_state=0
        )
        assert result.floor_distribution.shape == (20,)
        assert result.floor_threshold == pytest.approx(
            np.quantile(result.floor_distribution, 0.95)
        )
        assert result.floor_assessable

    def test_trajectory_blocks_replace_random_rows(self):
        rng = np.random.default_rng(17)
        tau = 20.0
        phi = np.exp(-1.0 / tau)
        per_frame = np.empty((1000, 4))
        per_frame[0] = rng.normal(size=4)
        noise = rng.normal(size=(1000, 4)) * np.sqrt(1.0 - phi**2)
        for frame in range(1, 1000):
            per_frame[frame] = phi * per_frame[frame - 1] + noise[frame]
        experimental = per_frame.mean(axis=0)
        uncertainty = np.ones(4)

        iid = score_observable(
            per_frame,
            experimental,
            uncertainty,
            sampling_kind="iid",
            random_state=0,
        )
        blocked = score_observable(
            per_frame,
            experimental,
            uncertainty,
            correlation_time_frames=20.0,
            random_state=0,
        )
        assert blocked.floor > 10.0 * iid.floor
        assert blocked.metadata["floor_strategy"] == "temporal blocks"

    def test_too_few_blocks_withhold_the_within_floor_verdict(self):
        per_frame, experimental, uncertainty = self.perfect(100)
        with pytest.warns(UserWarning, match="verdicts are withheld"):
            result = score_observable(
                per_frame,
                experimental,
                uncertainty,
                correlation_time_frames=20.0,
                random_state=0,
            )
        assert not result.floor_assessable
        assert result.within_floor is None
        assert "verdict withheld" in result.summary()

    def test_it_names_the_worst_points(self):
        per_frame, experimental, sigma = self.perfect(500, n_points=10, seed=4)
        per_frame[:, 3] += 5.0
        result = score_observable(
            per_frame, experimental, sigma, labels=np.arange(1, 11), random_state=0
        )
        assert result.worst[0][0] == 4        # one-based label of column 3
        assert "largest deviations" in result.summary()

    def test_mismatched_lengths_are_refused(self):
        per_frame, experimental, sigma = self.perfect(100, n_points=10)
        with pytest.raises(ValueError, match="do not describe the same"):
            score_observable(per_frame, experimental[:5], sigma[:5])

    def test_a_missing_uncertainty_is_refused(self):
        """A chi-squared without uncertainties is a sum of squares in
        arbitrary units, and the floor it is compared against means nothing."""
        per_frame, experimental, sigma = self.perfect(100)
        with pytest.raises(ValueError, match="must be positive"):
            score_observable(per_frame, experimental, np.zeros_like(sigma))

    def test_too_few_conformations_are_refused(self):
        per_frame, experimental, sigma = self.perfect(6)
        with pytest.raises(ValueError, match="independent conformations"):
            score_observable(per_frame, experimental, sigma)

    def test_sixth_power_scoring_is_available(self):
        rng = np.random.default_rng(5)
        distances = rng.uniform(2.0, 6.0, (2000, 5))
        experimental = average_observable(distances, averaging="r6")
        result = score_observable(
            distances, experimental, np.full(5, 0.1),
            averaging="r6", random_state=0,
        )
        assert result.chi2_reduced < 1e-6      # scored against its own average

    def test_results_serialise(self):
        import json

        result = score_observable(*self.perfect(200), random_state=0)
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["within_floor"] is True
        assert len(payload["residuals"]) == 20
