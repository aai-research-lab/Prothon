"""Relative solvent accessibility, half-sphere exposure and the chi1 torsion.

Three order parameters that describe a residue's exposure and side-chain
state rather than its contacts: `rsa` normalises SASA by residue type, the
`hse*` family splits a residue's neighbourhood into the half its side chain
points into and the half it does not, and `chi1` is the first side-chain
torsion. Every geometry here is built by hand so that the expected answer is
known before the code runs.
"""

from __future__ import annotations

import logging

import mdtraj as md
import numpy as np
import pytest

from prothon.compare.dissimilarity import dissimilarity
from prothon.represent import order_parameters as op

NEW = ("rsa", "hseu", "hsed", "hsea", "chi1")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _add_residue(top, chain, name, atoms):
    residue = top.add_residue(name, chain, resSeq=top.n_residues + 1)
    elements = {"N": md.element.nitrogen, "O": md.element.oxygen, "S": md.element.sulfur}
    return [top.add_atom(a, elements.get(a[0], md.element.carbon), residue) for a in atoms]


def _ala_gly_xyz():
    """ALA, GLY and an unknown residue type, crudely placed, one frame."""
    top = md.Topology()
    chain = top.add_chain()
    _add_residue(top, chain, "ALA", ["N", "CA", "C", "O", "CB"])
    _add_residue(top, chain, "GLY", ["N", "CA", "C", "O"])
    _add_residue(top, chain, "XYZ", ["N", "CA", "C", "O"])
    xyz = np.array([[
        [0.00, 0.00, 0.00], [0.15, 0.00, 0.00], [0.30, 0.05, 0.00],
        [0.30, 0.20, 0.00], [0.15, -0.15, 0.05],
        [0.45, 0.00, 0.00], [0.60, 0.00, 0.00], [0.75, 0.05, 0.00], [0.75, 0.20, 0.00],
        [0.90, 0.00, 0.00], [1.05, 0.00, 0.00], [1.20, 0.05, 0.00], [1.20, 0.20, 0.00],
    ]], dtype=np.float32)
    return md.Trajectory(xyz, top)


def _helix(names, has_cb, n_frames=1):
    """A rough helical coil; side chains point radially outward."""
    top = md.Topology()
    chain = top.add_chain()
    xyz = []
    for i, (name, cb) in enumerate(zip(names, has_cb)):
        theta = np.radians(100.0 * i)
        ca = np.array([0.23 * np.cos(theta), 0.23 * np.sin(theta), 0.15 * i])
        atoms = ["N", "CA", "C", "O"] + (["CB"] if cb else [])
        _add_residue(top, chain, name, atoms)
        xyz += [ca + [-0.05, 0, -0.03], ca, ca + [0.05, 0, 0.03], ca + [0.05, 0.1, 0.03]]
        if cb:
            xyz.append(ca + np.array([np.cos(theta), np.sin(theta), 0.0]) * 0.15)
    xyz = np.repeat(np.array([xyz], dtype=np.float32), n_frames, axis=0)
    return md.Trajectory(xyz, top)


def _split_geometry():
    """REF at the origin with its C-beta along +x; one neighbour at +x, one at -x."""
    top = md.Topology()
    chain = top.add_chain()
    xyz = []
    for ca, cb in (
        ([0.0, 0.0, 0.0], [0.15, 0.0, 0.0]),
        ([0.30, 0.5, 0.5], [0.0, 0.15, 0.0]),
        ([-0.30, 0.5, 0.5], [0.0, 0.15, 0.0]),
    ):
        ca = np.array(ca)
        _add_residue(top, chain, "ALA", ["N", "CA", "C", "O", "CB"])
        xyz += [ca + [0, 0.1, 0], ca, ca + [0.1, 0.1, 0], ca + [0.1, 0.2, 0], ca + cb]
    return md.Trajectory(np.array([xyz], dtype=np.float32), top)


def _chi1_residue(chi_angles, name="SER", xg="OG"):
    """One residue per frame-row of `chi_angles`, with chi1 set exactly.

    N sits in the plane z=0, CA at the origin, CB on +z; Xg is placed at
    azimuth `chi` about the CA->CB axis, so N-CA-CB-Xg is exactly `chi`.
    """
    top = md.Topology()
    chain = top.add_chain()
    _add_residue(top, chain, name, ["N", "CA", "C", "O", "CB", xg])
    frames = []
    for chi in np.atleast_1d(chi_angles):
        cb = np.array([0.0, 0.0, 0.153])
        frames.append([
            [0.146, 0.0, -0.05],                      # N
            [0.0, 0.0, 0.0],                          # CA
            [-0.07, 0.13, -0.05],                     # C
            [-0.05, 0.25, -0.05],                     # O
            cb,                                       # CB
            cb + 0.143 * np.array([np.cos(chi), np.sin(chi), 0.5]),  # Xg
        ])
    return md.Trajectory(np.array(frames, dtype=np.float32), top)


def _independent_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def _rigidly_moved(traj, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    moved = traj.xyz.astype(np.float64) @ q.T + rng.normal(size=3)
    return md.Trajectory(moved.astype(np.float32), traj.topology)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def warnings_logged():
    handler = _Capture()
    logger = logging.getLogger("prothon")
    logger.addHandler(handler)
    yield handler.messages
    logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestRegistration:
    @pytest.mark.parametrize("name", NEW)
    def test_each_is_registered_and_computable(self, name):
        assert name in op.ORDER_PARAMETERS
        assert name in op._COMPUTE
        assert op.ORDER_PARAMETERS[name].per_residue

    def test_only_chi1_is_circular(self):
        # Declaring circular wrongly is silent: a linear kernel on chi1 would
        # split the trans rotamer at +/-pi into two false populations.
        assert op.ORDER_PARAMETERS["chi1"].circular
        for name in ("rsa", "hseu", "hsed", "hsea"):
            assert not op.ORDER_PARAMETERS[name].circular

    @pytest.mark.parametrize("name", NEW)
    def test_feature_map_matches_the_computed_columns(self, name):
        traj = _helix(["SER", "GLY", "LEU", "VAL", "ALA"], [True, False, True, True, True])
        if name == "chi1":
            traj = _chi1_residue([1.0])
        matrix = op.compute_representation(traj, name)
        assert matrix.shape[1] == len(op._feature_residue_windows(traj.topology, name))


# ---------------------------------------------------------------------------
# rsa
# ---------------------------------------------------------------------------
class TestRsa:
    def test_is_sasa_divided_by_the_reference_maximum(self):
        traj = _ala_gly_xyz()
        rsa = op.compute_rsa(traj)
        sasa = op.compute_sasa(traj)
        expected = sasa[0, 0] / (op.RSA_MAX_ASA["ALA"]["theoretical"] * 0.01)
        assert rsa.shape == (1, 3)
        assert np.isclose(rsa[0, 0], expected)

    def test_glycine_is_an_ordinary_column(self):
        assert np.isfinite(op.compute_rsa(_ala_gly_xyz())[0, 1])

    def test_unknown_residue_is_nan_and_says_so(self, warnings_logged):
        rsa = op.compute_rsa(_ala_gly_xyz())
        assert np.isnan(rsa[0, 2])
        assert any("XYZ" in message for message in warnings_logged)

    def test_alternative_scale_changes_the_value(self):
        traj = _ala_gly_xyz()
        theoretical = op.compute_rsa(traj)[0, 0]
        rose = op.compute_rsa(traj, scale="rose")[0, 0]
        assert np.isclose(rose / theoretical, 129.0 / 118.1)

    def test_unknown_scale_is_refused(self):
        with pytest.raises(ValueError, match="scale"):
            op.compute_rsa(_ala_gly_xyz(), scale="nonsense")

    def test_every_standard_residue_has_every_scale(self):
        assert len(op.RSA_MAX_ASA) == 20
        for entry in op.RSA_MAX_ASA.values():
            assert set(entry) == {"theoretical", "empirical", "miller", "rose"}


# ---------------------------------------------------------------------------
# hse
# ---------------------------------------------------------------------------
class TestHse:
    def test_neighbours_are_split_by_the_side_chain_direction(self):
        traj = _split_geometry()
        assert op.compute_hseu(traj)[0, 0] == 1
        assert op.compute_hsed(traj)[0, 0] == 1

    def test_reversing_the_side_chain_swaps_the_halves(self):
        traj = _split_geometry()
        atoms = list(traj.topology.atoms)
        cb = [a.index for a in atoms if a.name == "CB"][0]
        ca_plus_x = [a.index for a in atoms if a.name == "CA"][1]
        xyz = traj.xyz.copy()
        xyz[0, ca_plus_x] = [2.0, 0.5, 0.5]  # out of range: only the -x neighbour is left
        one_sided = md.Trajectory(xyz, traj.topology)
        assert op.compute_hseu(one_sided)[0, 0] == 0
        assert op.compute_hsed(one_sided)[0, 0] == 1
        xyz[0, cb] = [-0.15, 0.0, 0.0]  # now the side chain points at that neighbour
        reversed_ = md.Trajectory(xyz, traj.topology)
        assert op.compute_hseu(reversed_)[0, 0] == 1
        assert op.compute_hsed(reversed_)[0, 0] == 0

    def test_mid_chain_glycine_uses_the_pseudo_c_beta(self):
        traj = _helix(["ALA", "GLY", "LEU", "VAL", "GLY"], [True, False, True, True, False])
        assert np.isfinite(op.compute_hseu(traj)[0, 1])

    def test_terminal_glycine_has_no_direction(self):
        traj = _helix(["ALA", "GLY", "LEU", "VAL", "GLY"], [True, False, True, True, False])
        assert np.isnan(op.compute_hseu(traj)[0, 4])
        assert np.isnan(op.compute_hsed(traj)[0, 4])

    def test_anisotropy_is_bounded_and_matches_its_definition(self):
        traj = _helix(["ALA", "SER", "LEU", "VAL", "ALA", "LYS"], [True] * 6)
        u, d, a = (op.compute_hseu(traj), op.compute_hsed(traj), op.compute_hsea(traj))
        assert np.allclose(a, (u - d) / (u + d))
        assert np.all((a >= -1) & (a <= 1))

    def test_counts_never_exceed_the_other_residues(self):
        traj = _helix(["ALA", "SER", "LEU", "VAL", "ALA", "LYS"], [True] * 6)
        total = op.compute_hseu(traj) + op.compute_hsed(traj)
        assert np.all(total <= traj.n_residues - 1)

    @pytest.mark.parametrize("name", ["hseu", "hsed", "hsea"])
    def test_rigid_motion_changes_nothing(self, name):
        traj = _helix(["ALA", "GLY", "LEU", "VAL", "SER", "GLY"],
                      [True, False, True, True, True, False])
        before = op.compute_representation(traj, name)
        after = op.compute_representation(_rigidly_moved(traj), name)
        assert np.array_equal(np.isnan(before), np.isnan(after))
        assert np.allclose(before, after, equal_nan=True)

    def test_frame_chunking_does_not_change_the_answer(self, monkeypatch):
        traj = _helix(["ALA", "SER", "LEU", "VAL"], [True] * 4, n_frames=7)
        rng = np.random.default_rng(3)
        traj.xyz += rng.normal(0, 0.02, traj.xyz.shape).astype(np.float32)
        whole = op.compute_hseu(traj)
        monkeypatch.setattr(op, "_BLOCK_BYTES", 1)  # one frame per chunk
        assert np.array_equal(whole, op.compute_hseu(traj))


# ---------------------------------------------------------------------------
# chi1
# ---------------------------------------------------------------------------
class TestChi1:
    @pytest.mark.parametrize("degrees", [-179.0, -60.0, 0.0, 60.0, 179.0])
    def test_recovers_a_known_angle(self, degrees):
        chi = np.radians(degrees)
        got = op.compute_chi1(_chi1_residue([chi]))[0, 0]
        assert np.isclose(np.angle(np.exp(1j * (got - chi))), 0.0, atol=1e-4)

    def test_agrees_with_an_independent_dihedral_formula(self):
        traj = _chi1_residue([0.7, -2.1, 3.0])
        quads, _ = op._chi1_atom_quads(traj.topology)
        for frame in range(traj.n_frames):
            p = traj.xyz[frame].astype(np.float64)
            expected = _independent_dihedral(*(p[i] for i in quads[0]))
            assert np.isclose(op.compute_chi1(traj)[frame, 0], expected, atol=1e-5)

    @pytest.mark.parametrize("name, xg", [("ILE", "CG1"), ("THR", "OG1"), ("VAL", "CG1")])
    def test_branched_residues_use_the_priority_gamma_atom(self, name, xg):
        traj = _chi1_residue([1.0], name=name, xg=xg)
        assert np.isclose(op.compute_chi1(traj)[0, 0], 1.0, atol=1e-4)

    def test_glycine_and_alanine_have_no_column(self):
        top = md.Topology()
        chain = top.add_chain()
        _add_residue(top, chain, "GLY", ["N", "CA", "C", "O"])
        _add_residue(top, chain, "ALA", ["N", "CA", "C", "O", "CB"])
        assert op._feature_residue_windows(top, "chi1") == []

    def test_a_missing_gamma_atom_is_excluded_with_a_warning(self, warnings_logged):
        top = md.Topology()
        chain = top.add_chain()
        _add_residue(top, chain, "CYS", ["N", "CA", "C", "O", "CB"])
        quads, windows = op._chi1_atom_quads(top)
        assert windows == [] and quads.size == 0
        assert any("CYS" in message for message in warnings_logged)

    def test_no_chi1_anywhere_is_refused(self):
        with pytest.raises(ValueError, match="chi1"):
            op.compute_chi1(_ala_gly_xyz())

    def test_rigid_motion_changes_nothing(self):
        traj = _chi1_residue([0.3, 2.9, -2.9])
        before = op.compute_chi1(traj)
        after = op.compute_chi1(_rigidly_moved(traj))
        assert np.allclose(np.angle(np.exp(1j * (before - after))), 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Through the comparison
# ---------------------------------------------------------------------------
class TestThroughDissimilarity:
    """The order parameters are only useful if the comparison can see them."""

    def test_a_rotamer_switch_is_detected(self):
        rng = np.random.default_rng(0)
        g_minus = op.compute_chi1(_chi1_residue(rng.normal(np.radians(-60), 0.15, 300)))
        trans = op.compute_chi1(_chi1_residue(rng.normal(np.pi, 0.15, 300)))
        result = dissimilarity(g_minus, trans, -np.pi, np.pi, x_num=60, s_num=3,
                               circular=True, random_state=0)
        assert result.n_significant == 1

    def test_the_trans_rotamer_is_one_population_across_the_wraparound(self):
        # Two samples of the same trans rotamer straddle +/-pi. With the circular
        # declaration they are the same; this is the failure circular=True exists
        # to prevent.
        rng = np.random.default_rng(1)
        a = op.compute_chi1(_chi1_residue(rng.normal(np.pi, 0.15, 300)))
        b = op.compute_chi1(_chi1_residue(rng.normal(np.pi, 0.15, 300)))
        result = dissimilarity(a, b, -np.pi, np.pi, x_num=60, s_num=3,
                               circular=True, random_state=0)
        assert result.n_significant == 0
