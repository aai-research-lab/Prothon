# Prothon

Prothon is a Python package for efficient comparison of protein conformational ensembles
using local structural properties. It represents each ensemble as a vector of probability 
density functions (pdfs) estimated from local order parameters (e.g. Cβ contact numbers, Cα 
contact numbers, bond angles, torsion angles, or SASA) and computes dissimilarity via the 
Jensen–Shannon distance.

## Installation

Clone the repository and install using pip:

```bash
git clone https://github.com/PlotkinLab/Prothon.git
cd Prothon
pip install .

