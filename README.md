# Prothon

**Prothon** is a Python package for the efficient comparison of protein ensembles using local order parameters.  
It is based on the work:  
_Adekunle Aina, Shawn C.C. Hsueh, and Steven S. Plotkin.  
PROTHON: A Local Order Parameter-Based Method for Efficient Comparison of Protein Ensembles.  
_J. Chem. Inf. Model._

## Features

- **Ensemble Representation:** Compute ensemble representations using local structural measures such as:
  - C-beta contact number (**cbcn**)
  - C-alpha contact number (**cacn**)
  - Virtual C-alpha–C-alpha bond angles (**caba**)
  - Virtual C-alpha–C-alpha torsion angles (**cata**)
  - Solvent Accessible Surface Area (**sasa**)
  
- **Dissimilarity Analysis:** Automatically computes both local (per-residue/feature) and global dissimilarity between ensembles using a Jensen–Shannon distance–based metric with statistical significance testing.

- **Output Generation:**  
  - Saves ensemble matrix representations as CSV files  
  - Generates heatmap plots of the ensemble matrices (with color bar ticks at regular intervals)  
  - Produces both bar and line plots for global dissimilarity (the bar plot uses a fixed color palette matching the dimensionality reduction plot, and the line plot defaults to black)  
  - Creates individual and combined local dissimilarity plots with x-axis tick marks spaced at regular intervals (e.g., 2, 4, 6; 5, 10, 15; or 10, 20, 30)

- **Dimensionality Reduction:**  
  - Offers dimensionality reduction (PCA, MDS, and t-SNE) on the ensemble representation data.  
  - Generates 2D scatter plots with each ensemble shown in a distinct color, using the fixed palette (first 12 colors: red, gold, darkgreen, blue, darkorchid, lightcoral, orange, lime, deepskyblue, magenta, navy, cyan; additional ensembles get random colors).  
  - The reduced data and plot are saved into the same measure output directory (e.g., `cbcn_output`).

- **API Access and Replotting:**  
  - Retrieve raw ensemble representations, dissimilarity results, and dimensionality-reduction data.  
  - Replot and customize all generated figures via provided API methods.

## Installation

Ensure that you have Python 3 installed. Install the package using pip from the package root (where setup.py is located):

```bash
pip install .
```

## Command-Line Usage

After installation, the prothon command becomes available in your shell. The following example demonstrates a typical run using the CLI:

```bash
prothon -traj "Q99.dcd,Q95.dcd,Q85.dcd,Q75.dcd" -top topology.pdb -m cbcn --dimred tsne 
```

### Explanation of CLI arguments:

-    -traj or --trajectories
    A comma-separated list (or glob pattern) of trajectory files.

-    -top or --topology
    The topology file in PDB format.

-    -m or --methods
    A comma-separated list of the representation measures to use (e.g., "cbcn,sasa"). In the above example, only cbcn is used.

-    -d or --dimred
    A comma-separated list of dimensionality reduction techniques to perform (e.g., "pca,mds,tsne"). In the example, only tsne is requested.

-    -o or --output
    The root output directory where all generated data and plots are saved. If not provided, the package creates measure-specific directories (e.g., cbcn_output).

-    -v or --verbose
    Increases the verbosity for detailed processing messages.

The CLI will save the matrix representations, dissimilarity plots (global and local, including a combined local plot), and (if requested) the dimensionality reduction plot in the output directory.

## API Usage

You can also use Prothon programmatically from within Python. Below is an example:

```python
from Prothon import Prothon
import matplotlib.pyplot as plt

# List of trajectory files and topology file path
traj_list = ["Q99.dcd", "Q95.dcd", "Q85.dcd", "Q75.dcd"]
topology = "topology.pdb"

# Create a Prothon instance with desired output directory and verbose output.
prothon = Prothon(traj_files=traj_list, topology=topology, output_dir="my_outputs", verbose=True)

# Compare ensembles using the cbcn measure
# Use dimensionality reduction with all three techniques by default:
results = prothon.compare_ensembles(methods="cbcn", ref=0, dimred="pca,mds,tsne")

# Retrieve and work with the computed data:
cbcn_data = prothon.get_representation_data("cbcn")
comparison_results = prothon.get_comparison_results("cbcn")
dimred_results = prothon.get_dimred_results("cbcn")

# To replot a global dissimilarity plot with custom styling:
fig_global = prothon.replot_global_dissimilarity("cbcn", plot_type="bar", color=None,
                                                 xlabel="Ensemble Index", ylabel="Global Dissimilarity",
                                                 title="Custom CBcN Global Dissimilarity")
plt.show()

# To replot local dissimilarity for ensemble 1 with customized tick intervals:
fig_local = prothon.replot_local_dissimilarity("cbcn", ensemble_index=1, color="black",
                                               xlabel="Residue Index", ylabel="Local Dissimilarity",
                                               title="Custom CBcN Local Dissimilarity for Ensemble 1")
plt.show()

# Dimensionality reduction scatter plot data can be accessed and re-plotted too.
# For example, using PCA:
dimred_pca = dimred_results.get("pca")
if dimred_pca:
    reduced_data = dimred_pca["reduced_data"]
    labels = dimred_pca["labels"]
    # Custom plotting code can be applied here if required.
```

## License

Prothon is released under the GNU GPL license.

## Citation
Adekunle Aina, Shawn C.C. Hsueh, and Steven S. Plotkin.  
PROTHON: A Local Order Parameter-Based Method for Efficient Comparison of Protein Ensembles.  
_J. Chem. Inf. Model._

