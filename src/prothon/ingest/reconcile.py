"""Making two ensembles comparable when they are not the same molecule.

A per-residue dissimilarity profile is only meaningful if column *k* of one
ensemble's representation describes the same position as column *k* of the
other's. Version 2.1 guaranteed that by refusing anything else: the ensembles
had to share a topology, and differing feature counts were an error.

That is the right default and the wrong limit. The questions worth asking are
mostly about ensembles that differ -- a mutant against its wild type, a
construct that resolves a loop against one that does not, a coarse-grained
model against an all-atom one, an ortholog against an ortholog. Methods built
on superposition cannot ask them at all, because there is no common coordinate
frame to superpose into. Local order parameters have no such difficulty: what
they need is a map between residues, and a sequence alignment is that map.

Two things this module insists on.

**The map is between residues, and columns are derived from it.** A column is
not always a residue. ``sasa`` has one column per residue, but ``cbcn`` has one
per C-beta -- so glycines are absent, and a mutation to or from glycine shifts
every column after it. ``caba`` and ``cata`` are windows of three and four
consecutive alpha carbons, so a column exists only where the whole window does.
Mapping columns directly would be right for one of them and quietly wrong for
the rest.

**A weak alignment is refused, not reported.** Below roughly 25% identity an
alignment of two protein sequences is not evidence that the positions
correspond. Comparing across such a map still produces a per-residue profile;
it is simply a profile of residues that are not the ones named, which is worse
than no answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..core.representation import resolve_order_parameter
from ..utils import get_logger
from .sequence import (
    MINIMUM_COVERAGE,
    TWILIGHT_IDENTITY,
    Alignment,
    align,
    chain_sequences,
)

if TYPE_CHECKING:  # pragma: no cover
    from .ensemble import Ensemble

logger = get_logger("ingest.reconcile")

__all__ = ["Correspondence", "Substitution", "feature_residues", "reconcile"]


@dataclass(frozen=True)
class Substitution:
    """One position where the two ensembles hold different residues."""

    residue_a: int
    residue_b: int
    letter_a: str
    letter_b: str

    def __str__(self) -> str:
        # One-based, because that is how a mutation is named in a paper.
        return f"{self.letter_a}{self.residue_a + 1}{self.letter_b}"


def feature_residues(topology, order_parameter: str) -> list[tuple[int, ...]]:
    """Residue indices behind each column of a representation.

    Returns one tuple per column: a single residue for the per-residue
    parameters, and the window of consecutive alpha carbons for the angular
    ones.

    This has to stay in step with the ``compute_*`` functions it describes, and
    a test asserts that the count it returns matches the width of the matrix
    they produce.
    """
    spec = resolve_order_parameter(order_parameter)
    top = getattr(topology, "topology", topology)

    if spec.name in ("cbcn", "cacn"):
        atom = "CB" if spec.name == "cbcn" else "CA"
        return [(top.atom(int(a)).residue.index,) for a in top.select(f"name {atom}")]

    if spec.name in ("caba", "cata"):
        residues = [top.atom(int(a)).residue.index for a in top.select("name CA")]
        width = 3 if spec.name == "caba" else 4
        return [tuple(residues[i : i + width]) for i in range(len(residues) - width + 1)]

    if spec.name == "sasa":
        return [(r.index,) for r in top.residues]

    raise ValueError(
        f"No feature-to-residue map defined for {spec.name!r}."
    )


@dataclass
class Correspondence:
    """Which residue of one ensemble is which residue of the other.

    Attributes
    ----------
    pairs
        ``(k, 2)`` array of corresponding residue indices.
    identity
        Fraction of corresponding positions holding the same residue.
    coverage
        Fraction of the shorter sequence that found a counterpart.
    substitutions
        Positions that correspond but differ. For a point mutant this is the
        mutation, named as a paper would name it.
    unmatched_a, unmatched_b
        Residues present in one ensemble with no counterpart in the other.
    """

    pairs: np.ndarray
    identity: float
    coverage: float
    substitutions: list[Substitution]
    unmatched_a: np.ndarray
    unmatched_b: np.ndarray
    labels: tuple[str, str] = ("a", "b")
    alignments: list[Alignment] = field(default_factory=list)

    @property
    def n_aligned(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def is_identical(self) -> bool:
        """Whether the two ensembles are the same molecule, residue for residue."""
        return (
            not self.substitutions
            and self.unmatched_a.size == 0
            and self.unmatched_b.size == 0
        )

    def residue_map(self) -> dict[int, int]:
        return {int(i): int(j) for i, j in self.pairs}

    def columns_for(
        self, order_parameter: str, topology_a, topology_b
    ) -> tuple[np.ndarray, np.ndarray]:
        """Column indices to take from each representation so they line up.

        A column survives only if every residue in its window has a counterpart
        *and* those counterparts form a window that exists on the other side.
        The second condition is what a deletion breaks: three consecutive alpha
        carbons in one ensemble may map to three residues in the other that are
        no longer consecutive, and the virtual bond angle between them is not
        the same quantity.
        """
        mapping = self.residue_map()
        features_a = feature_residues(topology_a, order_parameter)
        features_b = feature_residues(topology_b, order_parameter)
        lookup_b = {window: index for index, window in enumerate(features_b)}

        take_a: list[int] = []
        take_b: list[int] = []
        for index, window in enumerate(features_a):
            if not all(residue in mapping for residue in window):
                continue
            counterpart = tuple(mapping[residue] for residue in window)
            match = lookup_b.get(counterpart)
            if match is not None:
                take_a.append(index)
                take_b.append(match)

        logger.debug(
            "%s: %d of %d columns comparable",
            order_parameter, len(take_a), len(features_a),
        )
        return np.array(take_a, dtype=int), np.array(take_b, dtype=int)

    def summary(self) -> str:
        left, right = self.labels
        lines = [f"{left} vs {right}: {self.n_aligned} residues correspond, "
                 f"{self.identity:.1%} identity, {self.coverage:.1%} coverage"]
        if self.substitutions:
            named = ", ".join(str(s) for s in self.substitutions[:8])
            more = "" if len(self.substitutions) <= 8 else f" (+{len(self.substitutions) - 8} more)"
            lines.append(f"  substitutions: {named}{more}")
        if self.unmatched_a.size:
            lines.append(f"  only in {left}: {self.unmatched_a.size} residues")
        if self.unmatched_b.size:
            lines.append(f"  only in {right}: {self.unmatched_b.size} residues")
        if self.is_identical:
            lines.append("  the same molecule; no reconciliation was needed")
        return "\n".join(lines)


def reconcile(
    ensemble_a: Ensemble,
    ensemble_b: Ensemble,
    min_identity: float = TWILIGHT_IDENTITY,
    min_coverage: float = MINIMUM_COVERAGE,
    per_chain: bool = True,
) -> Correspondence:
    """Build the residue correspondence between two ensembles.

    Parameters
    ----------
    ensemble_a, ensemble_b
        The ensembles to reconcile.
    min_identity
        Refuse below this fraction of identical corresponding positions. The
        default is the twilight zone, where sequence similarity stops being
        evidence of positional equivalence.
    min_coverage
        Refuse unless this fraction of the shorter sequence is aligned.
        Identity on its own is not a sufficient guard: with free end gaps, two
        unrelated sequences align on a handful of positions at high identity.
        Lower it deliberately to compare a domain against the protein
        containing it.
    per_chain
        Align chain against chain in order. Concatenating the chains of a
        complex into one string invites the aligner to slide one chain against
        another, which is cheap in score and nonsense as a map.

    Raises
    ------
    ValueError
        If the chain counts differ under ``per_chain``, or the alignment comes
        back below ``min_identity``. Both cases are refusals rather than
        warnings: the alternative is a per-residue profile whose residues are
        not the ones it names.
    """
    top_a = ensemble_a.trajectory.topology
    top_b = ensemble_b.trajectory.topology
    labels = (ensemble_a.label, ensemble_b.label)

    if per_chain:
        chains_a = chain_sequences(top_a)
        chains_b = chain_sequences(top_b)
        if len(chains_a) != len(chains_b):
            raise ValueError(
                f"{labels[0]} has {len(chains_a)} protein chain(s) and {labels[1]} "
                f"has {len(chains_b)}. Prothon will not guess which chain "
                f"corresponds to which; select matching chains first, or pass "
                f"per_chain=False to align them as one sequence."
            )
        blocks = list(zip(chains_a, chains_b))
    else:
        blocks = [(chain_sequences(top_a) and _flatten(top_a), _flatten(top_b))]
        blocks = [(_flatten(top_a), _flatten(top_b))]

    pairs: list[tuple[int, int]] = []
    substitutions: list[Substitution] = []
    alignments: list[Alignment] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()

    for (seq_a, idx_a), (seq_b, idx_b) in blocks:
        alignment = align(seq_a, seq_b)
        alignments.append(alignment)
        for position_a, position_b in alignment.columns:
            residue_a = int(idx_a[position_a])
            residue_b = int(idx_b[position_b])
            pairs.append((residue_a, residue_b))
            matched_a.add(residue_a)
            matched_b.add(residue_b)
            if seq_a[position_a] != seq_b[position_b]:
                substitutions.append(
                    Substitution(residue_a, residue_b, seq_a[position_a], seq_b[position_b])
                )

    if not pairs:
        raise ValueError(
            f"No residues of {labels[0]} correspond to any of {labels[1]}. "
            f"These do not appear to be the same protein."
        )

    pair_array = np.array(sorted(pairs), dtype=int)
    identity = 1.0 - len(substitutions) / len(pairs)

    shorter = min(
        sum(len(seq) for seq, _ in (chains_a if per_chain else [blocks[0][0]])),
        sum(len(seq) for seq, _ in (chains_b if per_chain else [blocks[0][1]])),
    ) if per_chain else min(len(blocks[0][0][0]), len(blocks[0][1][0]))
    coverage = len(pairs) / shorter if shorter else 0.0

    if coverage < min_coverage:
        raise ValueError(
            f"Only {coverage:.1%} of the shorter sequence aligns between {labels[0]} "
            f"and {labels[1]} ({len(pairs)} of {shorter} residues), below the "
            f"{min_coverage:.0%} floor. An alignment covering this little describes a "
            f"fragment, not a molecule, and its identity says nothing: two unrelated "
            f"sequences align on a couple of positions at high identity. Pass "
            f"min_coverage= to compare a domain against the protein containing it."
        )

    if identity < min_identity:
        raise ValueError(
            f"{labels[0]} and {labels[1]} align at {identity:.1%} identity, below the "
            f"{min_identity:.0%} floor. Below roughly a quarter, a sequence alignment "
            f"is not evidence that the positions correspond, and a per-residue "
            f"comparison across this map would name residues it is not describing. "
            f"Pass min_identity= to override if the correspondence is known by "
            f"other means."
        )

    all_a = {r for seq, idx in chain_sequences(top_a) for r in map(int, idx)}
    all_b = {r for seq, idx in chain_sequences(top_b) for r in map(int, idx)}

    correspondence = Correspondence(
        pairs=pair_array,
        identity=float(identity),
        coverage=float(coverage),
        substitutions=substitutions,
        unmatched_a=np.array(sorted(all_a - matched_a), dtype=int),
        unmatched_b=np.array(sorted(all_b - matched_b), dtype=int),
        labels=labels,
        alignments=alignments,
    )
    logger.info("%s", correspondence.summary().replace("\n", "; "))
    return correspondence


def _flatten(topology) -> tuple[str, np.ndarray]:
    """Every protein residue of a topology as one sequence."""
    from .sequence import sequence_of

    return sequence_of(topology)
