# The study as a file

A command line is a good way to ask a question once and a poor way to record
one. The flags that produced a figure live in a shell history nobody reads, on
one machine, and the study cannot be re-run by somebody who has the data but
not the terminal session.

```bash
prothon compare --config study.yml
```

```yaml
description: wild type against the F5G mutant, three replicates each

ensembles:
  - source: wt.xtc
    topology: wt.pdb
    label: wild type
  - source: mut.xtc
    topology: mut.pdb
    label: F5G

reference: wild type

compare:
  order_parameters: [cbcn, cata]
  metric: jsd
  random_state: 0
  n_permutations: 200
  s_num: 10

output_dir: results
```

The file is the study: something to commit beside the manuscript, diff when it
changes, and hand to a collaborator.

## What a file expresses that a flag cannot

**A topology per ensemble.** `--topology` is one path for every source, which
is right when comparing conditions of one system and wrong for everything else.
A mutant has its own topology; so does an ortholog.

**A label per ensemble.** Figures and tables read better with "wild type" than
with `sim_run3_final.xtc`, and a label survives the file being moved.

**Weights per ensemble.** A reweighted simulation carries per-frame weights in
a separate file, and there is no sensible flag for that.

```yaml
ensembles:
  - source: md.xtc
    topology: system.pdb
    label: unbiased
  - source: metad.xtc
    topology: system.pdb
    label: reweighted
    weights: weights.txt      # one per frame
    stride: 10
  - source: PED00024
    label: deposited          # carries its own topology
```

## Every key is checked

A configuration that silently ignores what it does not recognise is a file that
lies. A misspelled `random_state` would leave the study unseeded and say
nothing, and the run would look fine.

```yaml
compare:
  random_seed: 0
```

```
prothon: study.yml: unknown setting 'random_seed' under 'compare'. A setting
that is silently ignored is worse than one that is refused: a misspelled
'random_state' would leave the study unseeded and say nothing. Did you mean
random_state?
```

The same applies to a top-level key, to a key inside an ensemble, and to a
reference naming something that is not there.

## The keys

At the top level:

| key | meaning |
|---|---|
| `ensembles` | A list, one entry per ensemble. Required, at least two. |
| `reference` | A label, an index, or a source. Defaults to the first. |
| `compare` | Any setting `prothon compare` accepts, by its long name. |
| `output_dir` | Where results are written. |
| `description` | Free text, recorded in the manifest. |

Within one ensemble:

| key | meaning |
|---|---|
| `source` | Required. A trajectory, directory, glob, multi-model PDB, or PED accession. |
| `topology` | For sources that need one. |
| `label` | Used in figures, tables and messages. |
| `weights` | A file of per-frame weights. |
| `stride` | Take every *n*-th frame. |

Under `compare`, every long flag name works: `order_parameters`, `metric`,
`random_state`, `n_permutations`, `s_num`, `x_num`, `alpha`, `report`,
`dimred`, `no_block_permutation`, `legacy_statistics`. The names are the same
because both come from one schema.

An entry may also be a bare source, when nothing else needs saying:

```yaml
ensembles:
  - md.xtc
  - PED00024
  - bioemu_out/
```

## Flags override the file

So a study can be re-run with one thing changed, without editing it:

```bash
prothon compare --config study.yml --random-state 7
prothon compare --config study.yml --report table -o rerun/
```

Anything given explicitly on the command line wins; anything left at its
default comes from the file.

## The manifest records the study

Every run writes the study that produced it into `manifest.json`, so a result
found later carries the question it answered rather than only the answer:

```json
"study": {
  "path": "/work/study.yml",
  "description": "wild type against the F5G mutant",
  "ensembles": [{"source": "wt.xtc", "label": "wild type", ...}],
  "settings": {"order_parameters": "cbcn", "random_state": 0}
}
```

## From Python

```python
from prothon import Prothon
from prothon.config import load_study, resolve_ensembles

study = load_study("study.yml")
ensembles = resolve_ensembles(study)

Prothon(
    ensembles=ensembles,
    random_state=study.settings.get("random_state"),
    output_dir=study.output_dir,
    study=study,
).compare_ensembles(**study.settings)
```
