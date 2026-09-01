"""Compatibility shim for the 2.2 layout. Removed in 3.0.

`core` held eight modules whose only shared property was not being `ingest`:
featurisation, comparison, sampling statistics and plotting. A reader looking
for the function that compares two ensembles had to already know it lived in
`core.dissimilarity`. The modules now sit under names that say what they do.

    prothon.core.representation    ->  prothon.represent.order_parameters
    prothon.core.metrics           ->  prothon.compare.distance
    prothon.core.dissimilarity     ->  prothon.compare.dissimilarity
    prothon.core.ensemble_metrics  ->  prothon.compare.joint
    prothon.core.precision_recall  ->  prothon.compare.coverage
    prothon.core.correlation       ->  prothon.sampling.correlation
    prothon.core.plotting          ->  prothon.plot.figures
    prothon.core.prothon_core      ->  prothon.study

`from prothon import Prothon` is unchanged and always was the supported entry
point; only the module paths beneath it have moved.
"""

from __future__ import annotations

import importlib
import sys
import warnings

__all__ = ["MOVED"]

#: Old module name to new, for anything still importing the 2.2 paths.
MOVED = {
    "correlation": "prothon.sampling.correlation",
    "dissimilarity": "prothon.compare.dissimilarity",
    "ensemble_metrics": "prothon.compare.joint",
    "metrics": "prothon.compare.distance",
    "plotting": "prothon.plot.figures",
    "precision_recall": "prothon.compare.coverage",
    "prothon_core": "prothon.study",
    "representation": "prothon.represent.order_parameters",
}


def _install_aliases() -> None:
    """Register every old name in ``sys.modules``, so both import forms work.

    A module ``__getattr__`` is enough for ``from prothon.core import
    dissimilarity``, and is *not* enough for ``import
    prothon.core.dissimilarity``. The second goes through the import system's
    finder, which looks in ``sys.modules`` and on the path and never consults
    ``__getattr__``, so it raises ``ModuleNotFoundError`` for a name the first
    form resolves happily.

    The gap is easy to miss because testing both in one process hides it: the
    first import populates ``sys.modules`` and the second then finds it there.
    A conda-forge build caught this, running ``import
    prothon.core.dissimilarity`` in a fresh interpreter.

    Registering the aliases eagerly costs importing the eight modules, all of
    which ``prothon/__init__`` imports anyway. Nothing is saved by being lazy
    about a package that only exists for code written against 2.2.
    """
    for old_name, target in MOVED.items():
        module = importlib.import_module(target)
        sys.modules[f"{__name__}.{old_name}"] = module
        globals()[old_name] = module


warnings.warn(
    "prothon.core was split in 2.3 and is removed in 3.0. The modules moved "
    "to prothon.represent, prothon.compare, prothon.sampling, prothon.plot "
    "and prothon.study; see prothon.core.MOVED for the mapping. The supported "
    "entry point, `from prothon import Prothon`, is unchanged.",
    DeprecationWarning,
    stacklevel=2,
)
_install_aliases()
