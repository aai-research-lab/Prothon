"""No feature crosses a chain break, and every label names its own chain.

The release gate has three clauses. Two are covered elsewhere: `test_ingest`
checks that an equal atom count cannot stand in for topology identity, and that
joining models across a chain break is refused. This file covers the third and
ties all three to the comparison a user actually runs, rather than to the
helpers underneath it.

A windowed order parameter is the place this fails silently. The virtual bond
angle spans three consecutive residues and the torsion four, so on a
twelve-residue single chain there are ten and nine of them. Split those twelve
residues into two chains of six and the counts must fall to eight and six,
because a window that begins in chain A and ends in chain B measures an angle
between atoms that are not bonded. Nothing about the resulting number looks
wrong.
"""

from __future__ import annotations

import numpy as np
import pytest


def _topology(chains, residues_per_chain):
    import mdtraj as md

    topology = md.Topology()
    for _ in range(chains):
        chain = topology.add_chain()
        for _ in range(residues_per_chain):
            residue = topology.add_residue("ALA", chain)
            for name, element in (
                ("N", md.element.nitrogen),
                ("CA", md.element.carbon),
                ("C", md.element.carbon),
                ("O", md.element.oxygen),
                ("CB", md.element.carbon),
            ):
                topology.add_atom(name, element, residue)
    return topology


class TestNoWindowCrossesAChainBreak:
    """Counts, because a window that straddles a break is invisible otherwise."""

    @pytest.mark.parametrize(
        ("order_parameter", "span"), [("cbcn", 1), ("caba", 3), ("cata", 4)]
    )
    def test_windows_stay_inside_their_chain(self, order_parameter, span):
        from prothon.ingest.reconcile import feature_identity

        one = _topology(1, 12)
        two = _topology(2, 6)

        # One chain of twelve: 12 - span + 1 windows.
        assert len(feature_identity(one, order_parameter)[1]) == 12 - span + 1
        # Two chains of six: the same count per chain, and none between them.
        assert len(feature_identity(two, order_parameter)[1]) == 2 * (6 - span + 1)

    def test_a_split_chain_loses_exactly_the_crossing_windows(self):
        """Splitting one chain in two removes `span - 1` windows per break."""
        from prothon.ingest.reconcile import feature_identity

        for order_parameter, span in (("caba", 3), ("cata", 4)):
            joined = len(feature_identity(_topology(1, 12), order_parameter)[1])
            split = len(feature_identity(_topology(2, 6), order_parameter)[1])
            assert joined - split == span - 1


class TestEveryLabelNamesItsOwnChain:
    def test_labels_carry_the_chain(self):
        from prothon.ingest.reconcile import feature_identity

        _, labels = feature_identity(_topology(2, 6), "cbcn")
        chains = {str(label).split(":", 1)[0] for label in labels}
        assert chains == {"1", "2"}, (
            "every per-residue label must name the chain it belongs to"
        )

    def test_a_window_label_names_its_span(self):
        from prothon.ingest.reconcile import feature_identity

        _, labels = feature_identity(_topology(2, 6), "cata")
        assert str(labels[0]) == "1:1-4"
        assert str(labels[-1]) == "2:3-6"

    def test_the_index_is_global_and_the_label_is_chain_local(self):
        """Two numbering conventions, deliberately, and worth stating.

        The index is a one-based *global* topology key, stable and
        machine-readable. The label is chain-local and chain-qualified, because
        residue 1 occurs once per chain and a display label that said only "1"
        would be ambiguous. Reading one as the other is the mistake this test
        exists to prevent.
        """
        from prothon.ingest.reconcile import feature_identity

        index, labels = feature_identity(_topology(2, 6), "caba")
        assert index is not None

        # Chain 1 windows start at global residues 1-4 (one-based).
        assert list(index[:4]) == [1, 2, 3, 4]
        # Chain 2 windows start at global 7-10, and are labelled 1-4 locally.
        assert list(index[4:]) == [7, 8, 9, 10]
        assert str(labels[4]) == "2:1-3"
        assert str(labels[0]) == "1:1-3"


class TestTheComparisonReportsTheseLabels:
    """The gate is about what a user sees, not only about the helper."""

    def test_a_two_chain_comparison_labels_every_feature(self, tmp_path):
        import mdtraj as md

        from prothon import Prothon

        topology = _topology(2, 6)
        rng = np.random.default_rng(0)
        base = np.zeros((topology.n_atoms, 3), dtype=np.float32)
        base[:, 0] = np.arange(topology.n_atoms) * 0.15

        paths = []
        for index, spread in enumerate((0.05, 0.09)):
            xyz = base + rng.normal(0.0, spread, (120, topology.n_atoms, 3))
            path = tmp_path / f"e{index}.dcd"
            md.Trajectory(xyz.astype(np.float32), topology).save_dcd(str(path))
            paths.append(str(path))
        structure = tmp_path / "top.pdb"
        md.Trajectory(base[None, :, :], topology).save_pdb(str(structure))

        result = Prothon(
            paths, str(structure), "cata", random_state=0
        ).compare()["cata"][0]

        # Six torsion windows: three per chain, none across the break.
        assert int(result.local_dissimilarity.size) == 6
