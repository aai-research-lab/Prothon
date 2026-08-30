"""The figures and the tables must not drift apart.

Two of the manuscript figures are measured in the figure script itself, by the
same calls the library makes, so they cannot disagree with the implementation.
The calibration panel cannot be: a thousand replicates at six correlation times
is an hour of compute, and a figure script that takes an hour stops being run.

Those numbers therefore live as constants in ``scripts/figures.py``, and this
is what stops them going stale. A stale table has already survived a squash
here once, invisible to the whole suite and to a clean documentation build. The
same number now appears in a figure, a documentation table and a manuscript
table, which is three places for it to rot in.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("matplotlib")

from figures import CALIBRATION  # noqa: E402


def _correlation_table() -> dict[int, tuple[int, float, float]]:
    """The τ table out of docs/calibration.md, keyed by correlation time.

    Parsed rather than duplicated: a copy here would be a fourth place for the
    number to be wrong.
    """
    text = (ROOT / "docs" / "calibration.md").read_text(encoding="utf-8")
    rows: dict[int, tuple[int, float, float]] = {}
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        tau, independent, frame, block = match.groups()
        rows[int(tau)] = (int(independent), float(frame), float(block))
    return rows


class TestTheFigureAgreesWithTheDocumentation:
    def test_the_table_was_found_at_all(self):
        """A parser that silently matches nothing would pass every test below."""
        rows = _correlation_table()
        assert set(rows) == set(CALIBRATION["tau"])

    @pytest.mark.parametrize("index", range(len(CALIBRATION["tau"])))
    def test_every_row_matches(self, index):
        rows = _correlation_table()
        tau = CALIBRATION["tau"][index]
        independent, frame, block = rows[tau]
        assert independent == CALIBRATION["independent"][index]
        assert frame == pytest.approx(CALIBRATION["frame_permutation"][index])
        assert block == pytest.approx(CALIBRATION["block_permutation"][index])


class TestTheFigureAgreesWithItself:
    def test_the_block_null_is_calibrated_across_the_whole_range(self):
        """The claim the figure is drawn to make, asserted on its own data."""
        block = CALIBRATION["block_permutation"]
        assert max(block) < CALIBRATION["nominal"]
        assert max(block) - min(block) < 1.0

    def test_the_frame_null_is_not(self):
        frame = CALIBRATION["frame_permutation"]
        assert frame[0] == pytest.approx(CALIBRATION["nominal"], abs=1.0)
        assert frame[-1] > 99.0

    def test_independent_conformations_is_the_exact_ar1_sample_size(self):
        """Not M/tau, whatever the manuscript caption said.

        The Ornstein-Uhlenbeck process is parameterised by its *relaxation*
        time, and the column is the exact effective sample size for the
        resulting AR(1) series, M(1-phi)/(1+phi) with phi = exp(-1/tau). That
        equals M divided by the *integrated* autocorrelation time, which for
        AR(1) is about twice the relaxation time -- so a caption reading M/tau
        overstates the column by a factor approaching two.

        The ubiquitin table's column is genuinely M/tau_hat, tau_hat being the
        integrated time the estimator returns. Two right quantities under one
        name, which is how the caption came to describe neither.
        """
        import math

        for tau, independent in zip(
            CALIBRATION["tau"], CALIBRATION["independent"]
        ):
            phi = math.exp(-1.0 / tau)
            exact = 2000 * (1 - phi) / (1 + phi)
            assert independent == pytest.approx(exact, abs=1.0)
