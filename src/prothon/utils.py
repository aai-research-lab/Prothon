"""Shared helpers: logging, and loading trajectories from mixed inputs.

Version 2.0 printed progress with bare ``print`` calls guarded by a ``verbose``
flag threaded through every function. That works until Prothon is imported by
something else, at which point the output cannot be silenced, redirected or
levelled. Logging is configured once, here, and the ``verbose`` arguments
survive only as a way to raise the level.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from glob import glob

import mdtraj as md

from .quiet import quiet_c_output

__all__ = ["configure_logging", "get_logger", "load_trajectories", "split_list_arg"]

_ROOT = "prothon"


def get_logger(name: str) -> logging.Logger:
    """Logger for one Prothon module, under the shared ``prothon`` root."""
    return logging.getLogger(f"{_ROOT}.{name}")


def configure_logging(verbose: bool = False) -> None:
    """Attach a handler to the Prothon logger, once.

    Adding a handler unconditionally is how a library ends up printing every
    message twice after a second import, so an existing handler is left alone.
    Nothing is attached to the root logger: a program embedding Prothon keeps
    control of its own logging.
    """
    logger = logging.getLogger(_ROOT)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[prothon] %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    # Compiled trajectory readers announce themselves on stdout from C. That
    # is noise in ordinary use and a diagnostic when something is wrong, so it
    # is suppressed unless the run is verbose.
    from .quiet import set_quiet

    set_quiet(not verbose)


def split_list_arg(value: str | Sequence[str] | None) -> list[str]:
    """Normalise a comma-separated string or a sequence into a list.

    The CLI hands over ``"cbcn,sasa"``; the Python API hands over
    ``["cbcn", "sasa"]``. Both should reach the same code.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def load_trajectories(
    traj_input: str | Iterable[str], topology: str
) -> md.Trajectory:
    """Load one or more trajectory files and join them into a single trajectory.

    Parameters
    ----------
    traj_input
        A filename, a comma-separated string of filenames, a glob pattern, or
        an iterable of filenames.
    topology
        Topology file shared by all of them.

    Notes
    -----
    This concatenates its inputs, which is the right thing for continuing
    replicates of one condition and the wrong thing for separate ensembles --
    joining those would average away the very difference being measured.
    :class:`~prothon.Prothon` therefore keeps ensembles apart and does not use
    this function; it is here for callers assembling one ensemble from several
    files.
    """
    if isinstance(traj_input, str):
        if "," in traj_input:
            files = split_list_arg(traj_input)
        else:
            files = sorted(glob(traj_input)) or [traj_input]
    else:
        files = [str(item) for item in traj_input]

    if not files:
        raise ValueError("No trajectory files given.")

    with quiet_c_output():
        loaded = [md.load(path, top=topology) for path in files]
    return loaded[0] if len(loaded) == 1 else md.join(loaded)
