"""Studies over many ensembles at once.

  - :mod:`benchmark` -- several ensembles against one reference, on equal
    terms, with the sampling reported alongside every number
"""

from .benchmark import BenchmarkResult, BenchmarkRow, benchmark

__all__ = ["BenchmarkResult", "BenchmarkRow", "benchmark"]
