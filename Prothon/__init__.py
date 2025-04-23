"""
Prothon Package

A Python package for efficient comparison of protein ensembles using local order parameters.

Enhanced features include:
  • Automatic output saving into method-specific directories.
  • Support for multiple methods (e.g., "cbcn,sasa") in one run.
  • Saving matrix representation data and generating heatmap plots with whole-number ticks.
  • Automatic generation of both global and local dissimilarity plots (with black defaults for line plots and a colour-coded bar plot matching dimensionality reduction colours).
  • Dimensionality reduction (options: PCA, MDS, and t-SNE) on ensemble representation data. The 2D scatter plots display all ensembles in distinct colours.
  • An API to retrieve all generated data and replot figures with custom styling.
  
Based on:
    Adekunle Aina, Shawn C.C. Hsueh, and Steven S. Plotkin.
    PROTHON: A Local Order Parameter-Based Method for Efficient Comparison of Protein Ensembles.
    J. Chem. Inf. Model.
"""

from .core.prothon_core import Prothon
from .utils import load_trajectories

__version__ = "2.0.0"

