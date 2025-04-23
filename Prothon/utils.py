"""
Utility functions for the Prothon package.
"""

import mdtraj as md
from glob import glob
from typing import Union, List

def load_trajectories(traj_input: Union[str, List[str], tuple], topology: str) -> md.Trajectory:
    """
    Load and join multiple trajectory files using MDTraj.

    Parameters
    ----------
    traj_input : str, list, or tuple
        If a string, it can be a comma-separated list of files or a glob pattern.
    topology : str
        The topology file (PDB format) for the trajectory.

    Returns
    -------
    traj : mdtraj.Trajectory
        The combined trajectory.
    """
    if isinstance(traj_input, str):
        # If comma-separated list, split it; otherwise assume glob pattern or single file.
        if ',' in traj_input:
            files = [f.strip() for f in traj_input.split(',')]
        else:
            files = glob(traj_input)
            if not files:
                files = [traj_input]
    elif isinstance(traj_input, (list, tuple)):
        files = list(traj_input)
    else:
        raise ValueError("traj_input must be a string, list, or tuple of filenames.")

    traj_list = []
    for file in files:
        if file.lower().endswith('.dcd'):
            traj = md.load_dcd(file, top=topology)
        else:
            traj = md.load(file, top=topology)
        traj_list.append(traj)

    if len(traj_list) == 1:
        return traj_list[0]
    else:
        return md.join(traj_list)

