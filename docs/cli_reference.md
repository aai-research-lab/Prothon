# Command line

```bash
prothon -traj a.dcd,b.dcd -top top.pdb -m cbcn
```

## Arguments

| flag | default | meaning |
|---|---|---|
| `-traj`, `--trajectories` | required | Trajectory files, one per ensemble, comma-separated. |
| `-top`, `--topology` | required | Topology file (PDB), shared by all of them. |
| `-m`, `--methods` | `cbcn` | Comma-separated measures. |
| `--metric` | `jsd` | Per-residue distance: `jsd`, `wasserstein`, `ks`. |
| `-r`, `--ref` | `0` | Reference ensemble index. |
| `-o`, `--output` | working directory | Output root; each measure writes `<measure>_output/`. |
| `-d`, `--dimred` | `none` | Projections: `pca`, `mds`, `tsne`. |
| `--x-num` | `100` | Grid points per estimated density. |
| `--s-num` | `5` | Split-half repeats behind the noise floor. |
| `--alpha` | `0.05` | False-discovery rate for the per-residue test. |
| `--seed` | none | Random seed. Set it for a reproducible run. |
| `--legacy-statistics` | off | Reproduce the historical statistics, for regenerating a published figure. |
| `--json` | off | Print full results as JSON instead of a summary. |
| `-v`, `--verbose` | off | Verbose logging. |
| `--info` | | Print measures, metrics and detected backends, then exit. |
| `--version` | | Print the version and exit. |

## Notes on the defaults

**Dimensionality reduction is off.** The projection is a visualisation rather
than part of the measurement, and MDS builds a dense frame-by-frame distance
matrix — asking for tens of gigabytes on a real trajectory. MDS is refused
above 5,000 frames with a message naming the memory it would have needed, and a
refusal does not discard the comparison that already succeeded.

**Each `-traj` file is one ensemble.** They are never concatenated.

**Blocks, not frames.** The null relabels contiguous blocks of conformations,
with the block length set from a correlation time estimated from the data. This
needs no flag and no knowledge of the system. It does need the frames to be in
the order they were generated — a trajectory written in a shuffled order has no
correlation time to find.

Where the sampling cannot support a p-value the summary says so, and the exit
code is still 0: an honest refusal is a successful run.

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 2 | the study was described wrongly — unknown measure, missing file, reference out of range. The message names what to change; there is no traceback, because a traceback would bury it. |

## Examples

Several measures and metrics, reproducibly:

```bash
prothon -traj wt.dcd,mut.dcd -top top.pdb \
        -m cbcn,cata,sasa --metric wasserstein \
        -o results --seed 0 --s-num 10
```

Machine-readable output:

```bash
prothon -traj a.dcd,b.dcd -top top.pdb --json > results.json
```

Regenerating a figure published under the historical statistics:

```bash
prothon -traj a.dcd,b.dcd -top top.pdb --legacy-statistics
```
