# Command line

```bash
prothon compare  --ensembles wt.xtc mutant.xtc --topology top.pdb
prothon compare  --ensembles bioemu/ alphaflow/ --reference md.xtc --report table
prothon validate --ensembles md.xtc -t top.pdb --experimental rg.txt
prothon info
```

Every flag is generated from a single schema, so the command line and the
Python API cannot drift apart: `--random-state` on the command line is
`random_state` in Python, because both read the same row. Long and short forms
work everywhere, and `--ensembles a,b` and `--ensembles a b` are the same
request.

## What `--ensembles` accepts

A source is whatever holds the conformations, and the kind is decided by
inspection rather than by a flag:

| source | example |
|---|---|
| a trajectory, with `--topology` | `md.xtc` |
| a directory of single-model PDBs | `bioemu_out/` |
| a glob | `'samples/*.pdb'` |
| a multi-model PDB | `nmr_entry.pdb` |
| a PED accession | `PED00024` |
| one ensemble within a PED entry | `PED00001:e002` |

They mix freely: `--ensembles md.xtc PED00024 bioemu_out/` is a valid
comparison of a simulation, a deposited ensemble and a generative model.

`--topology` takes one path shared by every source, or one per source in the
same order — which is what comparing different molecules needs, since a mutant
has its own:

```bash
prothon compare -e wt.xtc mut.xtc -t wt.pdb mut.pdb
```

**Each source is one ensemble.** They are never concatenated — joining two
conditions averages away the difference being measured.

## `prothon compare`

| flag | short | default | meaning |
|---|---|---|---|
| `--config` | `-c` | | A study in a YAML file. Flags override it. |
| `--ensembles` | `-e` | required | Sources to compare, unless `--config` names them. |
| `--topology` | `-t` | | One shared path, or one per ensemble in the same order. |
| `--reference` | `-r` | `0` | An index into `--ensembles`, or a source of its own. |
| `--order-parameters` | `-p` | `cbcn` | `cbcn`, `cacn`, `caba`, `cata`, `sasa`. |
| `--metric` | `-m` | `jsd` | `jsd`, `wasserstein`, `ks`. |
| `--random-state` | `-s` | | Seed. Set it and the run is reproducible. |
| `--n-permutations` | | `100` | Relabellings behind the null. |
| `--s-num` | | `5` | Split-half repeats behind the noise floor. |
| `--x-num` | | `100` | Grid points per density. |
| `--alpha` | | `0.05` | False-discovery rate. |
| `--no-block-permutation` | | | Treat frames as independent. |
| `--legacy-statistics` | | | Reproduce the historical statistics. |
| `--report` | | `summary` | `summary`, or `table` for the ranked view. |
| `--save-config` | | | Write the study this command describes to a file. |
| `--output-dir` | `-o` | | Where to write results. |
| `--dimred` | `-d` | `none` | `pca`, `mds`, `tsne`. |
| `--json` | | | Results as JSON. |
| `--verbose` | `-v` | | Verbose logging. |

A reference given as a source is prepended and becomes ensemble 0:

```bash
prothon compare -e mutant.xtc double.xtc -r wildtype.xtc -t top.pdb
```

### Benchmarking is a view, not a command

Several ensembles against one reference is `compare --report table`:

```bash
prothon compare -e bioemu/ alphaflow/ bbflow/ -r md.xtc -t target.pdb \
                -p cbcn -s 0 -o results --report table
```

Same comparison, same estimator, same floor — presented as a table ranked by
the margin above each ensemble's own noise floor rather than by raw distance,
with coverage and fidelity beside each row. There is no separate `benchmark`
command, because there is no separate calculation, and two commands for one
operation is how they come to disagree. See [benchmarking](benchmark.md).

### A study in a file

```bash
prothon compare --config study.yml
prothon compare -e wt.xtc mut.xtc -t top.pdb --save-config study.yml
```

Flags, a file and the Python API all build the same `Study` object and run
that, so none of them can offer a setting the others do not. A study expresses
what flags cannot — a topology, a label and a weight vector per ensemble — and
every key is checked against the schema, so a misspelled setting is refused
rather than ignored. See [the study](config.md).

## `prothon validate`

| flag | meaning |
|---|---|
| `--observable` | `rg`, `end_to_end` or `j_hn_ha`. |
| `--experimental` | Measured values: one column, or two of value and uncertainty. |
| `--uncertainty` | A file, or one number applied to every measurement. |

```bash
prothon validate -e md.xtc -t top.pdb --observable rg --experimental rg.txt
```

Uncertainties are required. A chi-squared without them is a sum of squares in
arbitrary units. See [validation](validate.md).

## `prothon info`

Order parameters, metrics, the sources `--ensembles` accepts, and the detected
backends.

## Exit codes

| code | meaning |
|---|---|
| 0 | success — including an honest refusal to report a p-value |
| 2 | the study was described wrongly. The message names what to change; there is no traceback, because a traceback would bury it |

## Superseded flags

Commands written against version 2 still run, and warn once:

```text
prothon -traj a.dcd,b.dcd -top top.pdb -m cbcn --seed 0
```

| old | now |
|---|---|
| `-traj` | `--ensembles` / `-e` |
| `-top` | `--topology` / `-t` |
| `-m` | `--order-parameters` / `-p` |
| `--seed` | `--random-state` / `-s` |
| `--info` | `prothon info` |

## Notes on the defaults

**Dimensionality reduction is off.** The projection is a visualisation rather
than part of the measurement, and MDS builds a dense frame-by-frame distance
matrix — tens of gigabytes on a real trajectory. It is refused above 5,000
frames with a message naming the memory it would have needed, and a refusal
does not discard the comparison that already succeeded.

**Blocking needs no flag.** The null relabels contiguous blocks of
conformations, with the block length set from a correlation time estimated from
the data. It does need the frames to be in the order they were generated.
`--no-block-permutation` turns it off for genuinely independent ensembles.
