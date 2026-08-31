"""Conformations to numbers.

Each conformation becomes a row of local order parameters -- one value per
residue -- so an ensemble of *M* conformations of an *N*-residue protein is an
*M* x *N* matrix. Everything downstream reads that matrix rather than
coordinates, which is why no superposition is ever needed.
"""

from .order_parameters import (
    ORDER_PARAMETERS,
    OrderParameter,
    compute_ensemble_representation,
    compute_representation,
    describe_order_parameter,
    resolve_order_parameter,
)

__all__ = [
    "ORDER_PARAMETERS",
    "OrderParameter",
    "compute_ensemble_representation",
    "compute_representation",
    "describe_order_parameter",
    "resolve_order_parameter",
]
