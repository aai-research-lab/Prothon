"""
Prothon package

A Python package for efficient representation and comparison of protein
ensembles using local order parameter probability distributions.

Based on:
    Adekunle Aina, Shawn C.C. Hsueh, and Steven S. Plotkin. 
    PROTHON: A Local Order Parameter-Based Method for Efficient Comparison of Protein Ensembles. 
    J. Chem. Inf. Model.
"""

from .core import Prothon
from .utils import load_trajectories

__version__ = "2.0.0"

