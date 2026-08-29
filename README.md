<div align="center">

# Prothon

**How different are two protein ensembles — and is the difference real?**

[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.3c00145-blue)](https://doi.org/10.1021/acs.jcim.3c00145)
[![PyPI](https://img.shields.io/pypi/v/prothon-ensembles?label=pypi)](https://pypi.org/project/prothon-ensembles/)
[![Python](https://img.shields.io/pypi/pyversions/prothon-ensembles)](https://pypi.org/project/prothon-ensembles/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Tests](https://github.com/aai-research-lab/Prothon/actions/workflows/tests.yml/badge.svg)](https://github.com/aai-research-lab/Prothon/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/aai-research-lab/Prothon/branch/main/graph/badge.svg)](https://codecov.io/gh/aai-research-lab/Prothon)
[![Docs](https://img.shields.io/readthedocs/prothon?label=docs)](https://prothon.readthedocs.io)

[**Documentation**](https://prothon.readthedocs.io) ·
[**Quick start**](https://prothon.readthedocs.io/en/latest/getting_started.html) ·
[**The statistics**](https://prothon.readthedocs.io/en/latest/statistics.html) ·
[**Calibration**](https://prothon.readthedocs.io/en/latest/calibration.html) ·
[**Cite**](#citation)

</div>

---

```bash
prothon -traj wild_type.dcd,mutant.dcd -top topology.pdb -p cbcn
```

```
CBCN (reference: ensemble 0)
  ensemble 1: d = 0.2841 (floor 0.0472) — 34/76 residues differ
```

Prothon represents each conformational ensemble as a vector of probability
distributions over **local order parameters** — contact numbers, virtual bond
and torsion angles, solvent accessibility — and measures the distance between
corresponding distributions.

Because the representation is local, no structural superposition is required.
Two consequences follow, and they are the reasons to use it.

**It is linear in ensemble size, not quadratic.** Methods that compare
ensembles through pairwise RMSD must superpose every structure against every
other. Prothon never superposes anything, so ensembles of tens of thousands of
conformations are ordinary rather than prohibitive.

**It can compare ensembles that are not the same molecule.** A superposition
needs a common coordinate frame, and a wild type and its mutant do not have
one. Local order parameters need only a map between residues, which a sequence
alignment provides.

## It reports what it cannot resolve

Two independent halves of a *single* ensemble have a non-zero distance between
them, because a finite sample never reproduces a continuous distribution
exactly. That self-distance is the resolution limit of the comparison. Prothon
measures it, prints it beside every result, and draws it on every figure. A
difference smaller than the floor is reported as unresolvable rather than as a
small difference.

Significance is decided against a permutation null — pool the conformations of
both ensembles, relabel them at random, measure — which is the exact
distribution of the statistic when the ensembles are the same and assumes
nothing about the shape of anything (Good 2005). Per-residue p-values are
corrected for multiplicity (Benjamini and Hochberg 1995), because a
300-residue protein tested at α = 0.05 yields fifteen false positives by
construction.

**What gets relabelled is a block, not a frame.** Consecutive conformations in
a trajectory are nearly the same conformation, so a null built by relabelling
individual frames is far too narrow. Prothon estimates the correlation time
from the data and relabels contiguous blocks of it. The difference is not
subtle — on data where both ensembles are drawn from the *same* distribution,
against a nominal 5%:

| correlation time τ (frames) | frame permutation | block permutation |
|---|---|---|
| 1 | 5.5% | 1.7% |
| 5 | 72.1% | 2.3% |
| 20 | 99.0% | 2.2% |
| 50 | 99.9% | 2.3% |

The block rate is flat across the range rather than degrading with τ, so a
result does not depend on the user knowing their correlation time. Where a
trajectory holds too few independent blocks to build a null from, Prothon
reports the floor and withholds the p-value rather than printing one it cannot
support.

The [calibration page](https://prothon.readthedocs.io/en/latest/calibration.html)
carries the full measurement, and the
[statistics page](https://prothon.readthedocs.io/en/latest/statistics.html)
sets out what is and is not corrected for.

## Install

```bash
pip install prothon-ensembles
prothon --info
```

The distribution is `prothon-ensembles` because PyPI's `prothon` was registered
in 2020 by an unrelated project. **The import name and the command are both
`prothon`.** A conda-forge package named `prothon` is in review.

## Use it

From the command line:

```bash
prothon -traj a.dcd,b.dcd,c.dcd -top top.pdb -p cbcn,cata -o results --seed 0
```

| flag | meaning |
|---|---|
| `-traj` | Trajectory files, one per ensemble, comma-separated. Never concatenated. |
| `-top` | Topology (PDB), shared by all of them. |
| `-m` | Order parameters: `cbcn`, `cacn`, `caba`, `cata`, `sasa`. |
| `--metric` | Distance: `jsd` (default), `wasserstein`, `ks`. |
| `--s-num` | Split-half repeats behind the noise floor. |
| `-r` | Reference ensemble index (default 0). |
| `-o` | Output root. Each measure writes `<measure>_output/`. |
| `-d` | Projections: `pca`, `mds`, `tsne`. Off by default. |
| `--seed` | Set it, and the run is reproducible. |

Or from Python:

```python
from prothon import Prothon

study = Prothon(["wild_type.dcd", "mutant.dcd"], "topology.pdb", random_state=0)
results = study.compare_ensembles(order_parameters="cbcn")

comparison = results["cbcn"][0]
comparison.global_dissimilarity     # 0.2841
comparison.noise_floor              # 0.0472 — the resolution limit
comparison.resolved                 # True: the difference clears the floor
comparison.significant              # bool array, one per residue
comparison.local_dissimilarity      # per residue, zero where not significant
comparison.raw_local_dissimilarity  # per residue, unmasked
comparison.correlation_time         # 18.4 frames, estimated from the data
comparison.p_values_withheld        # False: the sampling supported a test

print(study.summary())
```

Each measure writes a directory containing the representation matrices as CSV,
heatmaps, global and per-residue figures, and a `manifest.json` recording the
inputs, parameters, seed and version that produced them.

More, with real output, on the
[examples page](https://prothon.readthedocs.io/en/latest/examples.html).

## The order parameters

| name | quantity | units | circular |
|---|---|---|---|
| `cbcn` | C-beta contact number, smooth cutoff | contacts | |
| `cacn` | C-alpha contact number, smooth cutoff | contacts | |
| `caba` | Virtual Cα–Cα–Cα bond angle | rad | |
| `cata` | Virtual Cα torsion angle | rad | yes |
| `sasa` | Per-residue solvent accessible surface area | nm² | |

Torsions live on a circle, so they are estimated with a von Mises kernel on a
grid spanning a full turn. Each measure declares this and every estimator reads
it — a linear treatment of circular data is wrong by large factors and says
nothing about it.

## What else it does

- **Comparison across different molecules.** `prothon.ingest` builds a residue
  correspondence from an affine-gap alignment under BLOSUM62 (Needleman and
  Wunsch 1970; Gotoh 1982; Henikoff and Henikoff 1992), so a mutant, a
  truncated construct or a coarse-grained model can be compared against a
  reference. Columns are derived from the residue map rather than assumed — a
  mutation to glycine removes a C-beta and renumbers every `cbcn` column after
  it.
- **Ensembles from the Protein Ensemble Database**, by accession:
  `Ensemble.from_ped("PED00024")`. Comparing a model against an
  experimentally determined ensemble asks whether it reproduces what the
  measurements support, rather than whether it reproduces someone else's force
  field.
- **Weighted ensembles.** Conformer probabilities from a deposited ensemble, or
  frame weights from a reweighted simulation, reach the density estimate. The
  effective sample size (Kish 1965) sizes the noise floor: a thousand frames in
  which one conformer carries half the probability is worth four independent
  samples.
- **A choice of distance.** Jensen–Shannon (Lin 1991; Endres and Schindelin
  2003), Wasserstein-1, which reports in the feature's own units and needs no
  grid or bandwidth (Villani 2009), and the Kolmogorov–Smirnov statistic —
  Kuiper's (1960) on circular features, which is invariant to where the circle
  is cut where KS is not.
- **Whole-ensemble comparison.** Maximum mean discrepancy and a classifier
  two-sample test, which see differences in the relationship *between* residues
  that no per-residue metric can.
- **Coverage and fidelity.** Precision and recall, per residue (after Sajjadi
  et al. 2018 and Kynkäänniemi et al. 2019), distinguishing an ensemble that
  misses a state from one that invents one — two failures that any symmetric
  distance scores alike and that need opposite work.

## Benchmarking several ensembles

```bash
prothon compare -e bioemu/ alphaflow/ bbflow/ -r md.xtc -t target.pdb --report table
```

One table, one reference, the same treatment for each model — with each row
carrying the noise floor for *its own* sample size, because a smaller sample
has a higher floor and a depressed distance, and a table of raw distances
therefore ranks a thinly sampled model first. Rows report whether a model
*misses* states or *invents* them, per residue, and a model whose sampling
cannot support a comparison gets a row saying so rather than a number. See the
[benchmarking page](https://prothon.readthedocs.io/en/latest/benchmark.html).

- **Scoring against experiment.** Radius of gyration, end-to-end distance,
  PRE distances, FRET efficiencies and ³J couplings computed from the
  coordinates, and scored against measurements beside a floor — because a
  perfect ensemble of twenty conformations scores χ²_red = 0.77 and a perfect
  ensemble of five thousand scores 0.00, so fitting either to 1.0 is fitting
  to noise. See the
  [validation page](https://prothon.readthedocs.io/en/latest/validate.html).

## What it costs

Linear in the number of conformations and quadratic in chain length, both
measured rather than asserted. Fifty thousand conformations of a hundred
residues take about seven seconds to represent and peak at 1.25 GB; a
comparison is nearly independent of ensemble size, because ensembles are
subsampled before the permutation null. Full tables on the
[performance page](https://prothon.readthedocs.io/en/latest/performance.html).

## Citation

> Aina, A.; Hsueh, S. C. C.; Plotkin, S. S. PROTHON: A Local Order
> Parameter-Based Method for Efficient Comparison of Protein Ensembles.
> *J. Chem. Inf. Model.* **2023**, *63* (11), 3453–3461.
> DOI: [10.1021/acs.jcim.3c00145](https://doi.org/10.1021/acs.jcim.3c00145)

```bibtex
@article{aina2023prothon,
  author  = {Aina, Adekunle and Hsueh, Shawn C. C. and Plotkin, Steven S.},
  title   = {PROTHON: A Local Order Parameter-Based Method for Efficient
             Comparison of Protein Ensembles},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {63},
  number  = {11},
  pages   = {3453--3461},
  year    = {2023},
  doi     = {10.1021/acs.jcim.3c00145},
}
```

## References

Every method above is referenced in full on the
[references page](https://prothon.readthedocs.io/en/latest/references.html).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The version 1 code accompanying the
2023 paper is preserved unchanged under `legacy/`.

## License

MIT. See [LICENSE](LICENSE).

Earlier releases were distributed under GPL-3.0, and copies obtained under that
licence remain governed by it; relicensing is not retroactive and takes nothing
away from anyone holding one.

---

<div align="center">

Built in the [AAI Research Lab](https://aai-research-lab.github.io) at
California State University Dominguez Hills, on MDTraj, NumPy, SciPy,
scikit-learn and Matplotlib.

</div>
