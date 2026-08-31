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


def __getattr__(name: str):
    """Resolve `prothon.core.<old>` to its new home, once, with a warning."""
    target = MOVED.get(name)
    if target is None:
        raise AttributeError(f"module 'prothon.core' has no attribute {name!r}")
    warnings.warn(
        f"prothon.core.{name} moved to {target} in 2.3 and this alias is "
        f"removed in 3.0. The supported entry point, `from prothon import "
        f"Prothon`, is unchanged.",
        DeprecationWarning,
        stacklevel=2,
    )
    module = importlib.import_module(target)
    sys.modules[f"{__name__}.{name}"] = module
    return module
