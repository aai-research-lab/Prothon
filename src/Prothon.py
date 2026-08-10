"""Backward-compatibility shim for ``import Prothon``.

The distribution and import name are lowercase from 2.1 onward, matching Python
packaging convention. Code written against 2.0 says ``from Prothon import
Prothon``, so that keeps working and says once, on import, where the package
moved to.

**Why this is a module and not a package.** It was ``src/Prothon/__init__.py``,
which differs from ``src/prothon/__init__.py`` only in the case of one letter.
macOS and Windows use case-insensitive filesystems by default, so those are one
path, and checking out the tree writes both files on top of each other: the
directory arrives named ``Prothon``, every module lands inside it, and
``import prothon`` finds nothing. A single module named ``Prothon.py`` does not
collide with a directory named ``prothon``, because the names differ by more
than case.

There is a test that no two tracked paths differ only in case, so the next
thing to try this is caught before a release rather than on someone's laptop.
"""

from __future__ import annotations

import warnings

from prothon import *  # noqa: F401,F403
from prothon import Prothon, __version__  # noqa: F401
from prothon.utils import load_trajectories  # noqa: F401

warnings.warn(
    "Importing 'Prothon' is deprecated; the package is now imported as "
    "'prothon' (lowercase). 'from prothon import Prothon' gives the same "
    "class. The old name will be removed in 3.0.",
    DeprecationWarning,
    stacklevel=2,
)
