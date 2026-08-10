"""
Representation module for the Prothon package.

This module provides functions for:
  • Loading individual trajectory files.
  • Computing ensemble representations based on a chosen local structural measure.
    Supported measures:
       - C-beta contact number (cbcn)
       - C-alpha contact number (cacn)
       - Virtual C-alpha–C-alpha bond angles (caba)
       - Virtual C-alpha–C-alpha torsion angles (cata)
       - Solvent accessible surface area (sasa)
"""

import numpy as np
from mdtraj import load, load_dcd, compute_angles, compute_dihedrals, compute_distances, shrake_rupley
from itertools import combinations

def load_ensemble(file: str, topology: str):
    """
    Load a single trajectory file using MDTraj.

    Parameters
    ----------
    file : str
        Filename of the trajectory.
    topology : str
        Topology file (PDB format).

    Returns
    -------
    traj : mdtraj.Trajectory
        The loaded trajectory.
    """
    if file.strip().lower().endswith('.dcd'):
        return load_dcd(file, topology)
    else:
        return load(file, topology)

def compute_cbcn(traj, verbose: bool=False):
    """
    Compute the C-beta contact number (cbcn) for a given trajectory.

    Uses a smooth cutoff defined by 1/(1 + exp(CONST*(distance - r0))).
    To avoid overflow, the argument to the exponential is clipped between -700 and 700.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Loaded trajectory.
    verbose : bool, optional

    Returns
    -------
    cbcn : np.ndarray, shape=(n_frames, n_residues)
        Matrix containing cbcn values.
    """
    CONST = 50  # 1/nm
    r0 = 1      # nm
    cb_indices = traj.topology.select("name == 'CB'")
    if verbose:
        print(f"[CBcN] Found {len(cb_indices)} CB atoms.")
    cb_pairs = np.array(
        [(i, j) for (i, j) in combinations(cb_indices, 2)
         if abs(traj.topology.atom(i).residue.index - traj.topology.atom(j).residue.index) > 2]
    )
    cbcn = []
    for idx in cb_indices:
        pairs = cb_pairs[[idx in pair for pair in cb_pairs]]
        dists = compute_distances(traj, pairs)
        arg = CONST * (dists - r0)
        arg = np.clip(arg, -700, 700)
        contacts = np.sum(1.0 / (1 + np.exp(arg)), axis=1)
        cbcn.append(contacts)
    return np.transpose(np.array(cbcn))

def compute_cacn(traj, verbose: bool=False):
    """
    Compute the C-alpha contact number (cacn) for a given trajectory.

    Parameters
    ----------
    traj : mdtraj.Trajectory
    verbose : bool, optional

    Returns
    -------
    cacn : np.ndarray, shape=(n_frames, n_residues)
    """
    CONST = 50
    r0 = 1
    ca_indices = traj.topology.select("name == 'CA'")
    cacn_pairs = np.array(
        [(i, j) for (i, j) in combinations(ca_indices, 2)
         if abs(traj.topology.atom(i).residue.index - traj.topology.atom(j).residue.index) > 2]
    )
    cacn = []
    for idx in ca_indices:
        pairs = cacn_pairs[[idx in pair for pair in cacn_pairs]]
        dists = compute_distances(traj, pairs)
        arg = CONST * (dists - r0)
        arg = np.clip(arg, -700, 700)
        contacts = np.sum(1.0 / (1 + np.exp(arg)), axis=1)
        cacn.append(contacts)
    return np.transpose(np.array(cacn))

def compute_caba(traj):
    """
    Compute the virtual C-alpha–C-alpha bond angles (caba) for a trajectory.

    Parameters
    ----------
    traj : mdtraj.Trajectory

    Returns
    -------
    caba : np.ndarray, shape=(n_frames, n_angles)
    """
    ca_indices = traj.topology.select("name == 'CA'")
    n_angles = len(ca_indices) - 2
    angle_indices = [(ca_indices[i], ca_indices[i+1], ca_indices[i+2]) for i in range(n_angles)]
    return compute_angles(traj, angle_indices)

def compute_cata(traj):
    """
    Compute the virtual C-alpha–C-alpha torsion angles (cata) for a trajectory.

    Parameters
    ----------
    traj : mdtraj.Trajectory

    Returns
    -------
    cata : np.ndarray, shape=(n_frames, n_dihedrals)
    """
    ca_indices = traj.topology.select("name == 'CA'")
    n_dihedrals = len(ca_indices) - 3
    dihedral_indices = [(ca_indices[i], ca_indices[i+1], ca_indices[i+2], ca_indices[i+3])
                        for i in range(n_dihedrals)]
    return compute_dihedrals(traj, dihedral_indices)

def compute_sasa(traj):
    """
    Compute the solvent accessible surface area (sasa) per residue for a trajectory.

    Parameters
    ----------
    traj : mdtraj.Trajectory

    Returns
    -------
    sasa : np.ndarray, shape=(n_frames, n_residues)
    """
    return shrake_rupley(traj, mode='residue')

def compute_ensemble_representation(traj_files, topology, measure, verbose: bool=False):
    """
    Compute the ensemble representation for each trajectory file provided.

    Parameters
    ----------
    traj_files : list of str
        Filenames of trajectory files.
    topology : str
        PDB file for the topology.
    measure : str
        Chosen measure: 'cbcn', 'cacn', 'caba', 'cata', or 'sasa'.
    verbose : bool, optional

    Returns
    -------
    reps : list of np.ndarray
        List of representation matrices.
    """
    measure = measure.lower()
    reps = []
    for file in traj_files:
        if verbose:
            print(f"[Representation] Loading {file} ...")
        traj = load_ensemble(file, topology)
        if measure == 'cbcn':
            rep = compute_cbcn(traj, verbose)
        elif measure == 'cacn':
            rep = compute_cacn(traj, verbose)
        elif measure == 'caba':
            rep = compute_caba(traj)
        elif measure == 'cata':
            rep = compute_cata(traj)
        elif measure == 'sasa':
            rep = compute_sasa(traj)
        else:
            raise ValueError(f"Unsupported measure: {measure}")
        reps.append(rep)
    return reps

