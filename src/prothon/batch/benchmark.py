"""Comparing many ensembles against one reference, and saying what that is worth.

A generative model is judged by how closely its conformations resemble a
reference — usually molecular dynamics, sometimes an experimentally derived
ensemble. Every paper doing this invents its own arrangement of metrics, and
the numbers are not comparable across papers.

This runs the same comparison for every model against the same reference and
reports the same things: how far apart they are, whether that difference is
larger than the sampling can resolve, and whether the model *misses* states or
*invents* them — two failures that a symmetric distance scores alike and that
call for opposite work.

**Sample size is part of the result, not a footnote.** Models emit what they
emit: two hundred and fifty conformations from one, tens of thousands from
another. Measured against a 20,000-frame reference, two ensembles drawn from
the same distribution give a noise floor of 0.064 at 5,000 conformations and
0.109 at 50 — and a model with a real half-sigma bias scores 0.216 at 5,000 and
0.129 at 50. **A table of raw dissimilarities ranks the thinly sampled model
first.** Every row here therefore carries its floor, and the table reports the
margin above it rather than the distance alone.

**Refusing is a result.** Where an ensemble is worth too few independent
conformations to support a comparison, the row says so instead of carrying a
number. A benchmark that fills every cell regardless of whether the sampling
supports one is how a field ends up comparing noise.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..compare.coverage import precision_recall
from ..compare.dissimilarity import dissimilarity
from ..ingest import Ensemble
from ..represent.order_parameters import resolve_order_parameter
from ..utils import get_logger

logger = get_logger("benchmark")

__all__ = ["BenchmarkResult", "BenchmarkRow", "benchmark"]


@dataclass
class BenchmarkRow:
    """One model measured against the reference, for one target and measure."""

    target: str
    model: str
    order_parameter: str
    n_reference: int = 0
    n_model: int = 0
    dissimilarity: float | None = None
    noise_floor: float | None = None
    noise_floor_threshold: float | None = None
    noise_floor_assessable: bool = True
    n_significant: int | None = None
    n_features: int | None = None
    precision: float | None = None
    recall: float | None = None
    floor_precision: float | None = None
    floor_recall: float | None = None
    missed: list[int] = field(default_factory=list)
    invented: list[int] = field(default_factory=list)
    refused: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def margin(self) -> float | None:
        """How far the difference clears the resolution limit.

        The number to rank on. A raw dissimilarity rewards a model for being
        thinly sampled, because a small sample has a high floor and a depressed
        distance; the margin does not.
        """
        if (
            self.dissimilarity is None
            or self.noise_floor is None
            or not self.noise_floor_assessable
        ):
            return None
        threshold = (
            self.noise_floor
            if self.noise_floor_threshold is None
            else self.noise_floor_threshold
        )
        return self.dissimilarity - threshold

    @property
    def resolved(self) -> bool | None:
        if not self.noise_floor_assessable:
            return None
        margin = self.margin
        return margin is not None and margin > 0

    @property
    def verdict(self) -> str:
        if self.refused:
            return "refused"
        if not self.noise_floor_assessable:
            return "too few independent units for a floor verdict"
        if not self.resolved:
            return "indistinguishable from the reference at this sampling"
        parts = []
        if self.missed:
            parts.append(f"misses states at {len(self.missed)} residues")
        if self.invented:
            parts.append(f"invents states at {len(self.invented)} residues")
        return "; ".join(parts) if parts else "differs, neither coverage nor fidelity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "model": self.model,
            "order_parameter": self.order_parameter,
            "n_reference": self.n_reference,
            "n_model": self.n_model,
            "dissimilarity": self.dissimilarity,
            "noise_floor": self.noise_floor,
            "noise_floor_threshold": self.noise_floor_threshold,
            "noise_floor_assessable": self.noise_floor_assessable,
            "margin": self.margin,
            "resolved": self.resolved,
            "n_significant": self.n_significant,
            "n_features": self.n_features,
            "precision": self.precision,
            "recall": self.recall,
            "floor_precision": self.floor_precision,
            "floor_recall": self.floor_recall,
            "missed": self.missed,
            "invented": self.invented,
            "refused": self.refused,
            "verdict": self.verdict,
            **self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Every model against every target."""

    rows: list[BenchmarkRow]
    reference_label: str = "reference"
    order_parameters: tuple[str, ...] = ()

    def for_model(self, model: str) -> list[BenchmarkRow]:
        return [r for r in self.rows if r.model == model]

    def table(self, order_parameter: str | None = None) -> str:
        """A markdown table, ordered by margin.

        Ranked on the margin above the floor rather than on the dissimilarity,
        because the dissimilarity rewards a model for sampling thinly.
        """
        rows = [
            r for r in self.rows
            if order_parameter is None or r.order_parameter == order_parameter
        ]
        if not rows:
            return "_no comparisons_"

        header = (
            "| target | model | n | d | floor | margin | precision | recall | verdict |"
        )
        lines = [header, "|---" * 9 + "|"]
        for row in sorted(
            rows,
            key=lambda r: (r.target, -(r.margin if r.margin is not None else -1)),
        ):
            if row.refused:
                lines.append(
                    f"| {row.target} | {row.model} | {row.n_model} | — | — | — | — | — "
                    f"| {row.refused} |"
                )
                continue
            threshold = (
                row.noise_floor
                if row.noise_floor_threshold is None
                else row.noise_floor_threshold
            )
            margin = "—" if row.margin is None else f"{row.margin:+.3f}"
            lines.append(
                f"| {row.target} | {row.model} | {row.n_model} | "
                f"{row.dissimilarity:.3f} | {threshold:.3f} | {margin} | "
                f"{row.precision:.3f} | {row.recall:.3f} | {row.verdict} |"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        refused = sum(1 for r in self.rows if r.refused)
        unresolved = sum(1 for r in self.rows if not r.refused and r.resolved is False)
        unassessed = sum(1 for r in self.rows if not r.refused and r.resolved is None)
        lines = [
            f"{len(self.rows)} comparisons against {self.reference_label}",
            f"  {refused} refused for want of sampling",
            f"  {unresolved} not resolvable above the noise floor",
            f"  {unassessed} without enough independent units for a floor verdict",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        from .. import __version__

        return {
            "prothon_version": __version__,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reference": self.reference_label,
            "order_parameters": list(self.order_parameters),
            "rows": [r.to_dict() for r in self.rows],
        }

    def write(self, directory: str | os.PathLike) -> Path:
        """Write the table and the full results, and return the directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "benchmark.json").write_text(
            json.dumps(self.to_dict(), indent=2, default=float), encoding="utf-8"
        )
        report = ["# Benchmark", "", self.summary(), ""]
        for name in self.order_parameters:
            report += [f"## {name.upper()}", "", self.table(name), ""]
        (path / "benchmark.md").write_text("\n".join(report), encoding="utf-8")
        logger.info("Wrote %s", path / "benchmark.md")
        return path


def _compare_one(
    reference: Ensemble,
    model: Ensemble,
    order_parameter: str,
    target: str,
    reference_reps,
    random_state,
    **kwargs,
) -> BenchmarkRow:
    """One model against the reference, catching a refusal as a result."""
    from ..study import Prothon

    spec = resolve_order_parameter(order_parameter)
    row = BenchmarkRow(
        target=target,
        model=model.label,
        order_parameter=spec.name,
        n_reference=reference.n_frames,
        n_model=model.n_frames,
    )

    try:
        study = Prothon.from_ensembles([reference, model], random_state=random_state)
        left, right, index = study._align_columns(
            reference_reps,
            study.compute_ensemble_representation(spec.name)[1],
            0, 1, spec.name,
        )
        low = float(min(left.min(), right.min()))
        high = float(max(left.max(), right.max()))

        result = dissimilarity(
            left, right, low, high,
            circular=spec.circular,
            random_state=random_state,
            order_parameter=spec.name,
            # Conformations from a generative model are independent draws, not
            # a trajectory, so there is no correlation to block against and
            # blocking would cost resolution for nothing.
            block_permutation=kwargs.pop("block_permutation", False),
            **kwargs,
        )
        coverage = precision_recall(
            left, right,
            circular=spec.circular,
            feature_index=index,
            random_state=random_state,
            order_parameter=spec.name,
        )
    except ValueError as error:
        row.refused = str(error).split(".")[0]
        logger.warning("%s / %s: %s", target, model.label, row.refused)
        return row

    row.dissimilarity = result.global_dissimilarity
    row.noise_floor = result.noise_floor
    row.noise_floor_threshold = result.noise_floor_threshold
    row.noise_floor_assessable = result.noise_floor_assessable
    row.n_significant = result.n_significant
    row.n_features = int(result.local_dissimilarity.size)
    row.precision = coverage.mean_precision
    row.recall = coverage.mean_recall
    row.floor_precision = coverage.mean_floor_precision
    row.floor_recall = coverage.mean_floor_recall
    row.missed = [int(i) for i in coverage.missed()]
    row.invented = [int(i) for i in coverage.invented()]
    row.metadata = {
        "p_values_withheld": bool(result.p_values_withheld),
        "effective_samples": list(result.effective_samples),
    }
    return row


def benchmark(
    reference: Ensemble,
    models: Sequence[Ensemble],
    order_parameters: str | Sequence[str] = "cbcn",
    target: str = "target",
    random_state: int | None = None,
    output_dir: str | os.PathLike | None = None,
    **kwargs,
) -> BenchmarkResult:
    """Compare several ensembles against one reference on equal terms.

    Parameters
    ----------
    reference
        The ensemble being matched. Not interchangeable with the models:
        precision and recall swap when it is.
    models
        The ensembles being assessed.
    order_parameters
        One or more local order parameters.
    target
        A name for this system, used in the table when several are run.
    output_dir
        Where to write ``benchmark.md`` and ``benchmark.json``.

    Returns
    -------
    BenchmarkResult
    """
    from ..study import Prothon
    from ..utils import split_list_arg

    names = split_list_arg(order_parameters)
    if not names:
        raise ValueError("No order parameters requested.")
    if not models:
        raise ValueError("No models to compare against the reference.")

    rows: list[BenchmarkRow] = []
    for name in names:
        spec = resolve_order_parameter(name)
        # The reference is measured once per measure and reused, rather than
        # recomputed for every model.
        reference_reps = Prothon.from_ensembles(
            [reference, models[0]], random_state=random_state
        ).compute_ensemble_representation(spec.name)[0]

        for model in models:
            logger.info("%s: %s vs %s [%s]", target, reference.label, model.label, name)
            rows.append(
                _compare_one(
                    reference, model, spec.name, target,
                    reference_reps, random_state, **dict(kwargs),
                )
            )

    result = BenchmarkResult(
        rows=rows, reference_label=reference.label,
        order_parameters=tuple(names)
    )
    if output_dir:
        result.write(output_dir)
    return result
