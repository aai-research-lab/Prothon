# The study

A command line is a good way to ask a question once and a poor way to record
one. The flags that produced a figure live in a shell history nobody reads, on
one machine, and the study cannot be re-run by somebody who has the data but
not the terminal session.

**So a study is the thing, and each interface is a way of writing one down.**
The command line parses flags into a study; a file is read into one; Python
constructs one directly. All of them then run the same object — which is what
keeps them from drifting, because a setting reachable from one interface and
not another is a bug that cannot happen when there is only one place for
settings to live.

| how you ask | what happens |
|---|---|
| `prothon compare -e a b -t top.pdb` | flags → a study → run |
| `prothon compare --config s.yml` | a file → a study → run |
| `prothon compare --config s.yml -s 7` | a file, with the flag applied → run |
| `Study.from_file("s.yml").run()` | the same object |

```bash
prothon compare --config study.yml
```

```yaml
description: wild type against the F5G mutant, three replicates each

ensembles:
  - ensemble: wt.xtc
    topology: wt.pdb
    label: wild type
  - ensemble: mut.xtc
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

## Writing one down

It runs the other way too. A command line typed once becomes a study that can
be committed:

```bash
prothon compare -e wt.xtc mut.xtc -t top.pdb -p cbcn -s 0 --save-config study.yml
```

```yaml
# Written by Prothon. Run with: prothon compare --config study.yml
ensembles:
- ensemble: wt.xtc
  topology: top.pdb
- ensemble: mut.xtc
  topology: top.pdb
compare:
  order_parameters: cbcn
  random_state: 0
```

Only what was actually given is written. A flag left at its default is a flag
nobody chose, and recording it would produce a file full of settings that look
deliberate and are not.

The file does not record where it came from, either: a rewritten study that
carried its own `config:` path would point at a different file.

## What a study expresses that a flag cannot

**A topology per ensemble.** `--topology` is one path for every source, which
is right when comparing conditions of one system and wrong for everything else.
A mutant has its own topology; so does an ortholog.

**A label per ensemble.** Figures and tables read better with "wild type" than
with `sim_run3_final.xtc`, and a label survives the file being moved.

**Weights per ensemble.** A reweighted simulation carries per-frame weights in
a separate file, and there is no sensible flag for that.

```yaml
ensembles:
  - ensemble: md.xtc
    topology: system.pdb
    label: unbiased
  - ensemble: metad.xtc
    topology: system.pdb
    label: reweighted
    weights: weights.txt      # one per frame
    stride: 10
  - ensemble: PED00024
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
| `ensemble` | Required. A trajectory, directory, glob, multi-model PDB, or PED accession. |
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
{
  "study": {
    "path": "/work/study.yml",
    "description": "wild type against the F5G mutant",
    "ensembles": [
      {"ensemble": "wt.xtc", "topology": "wt.pdb", "label": "wild type"}
    ],
    "compare": {"order_parameters": "cbcn", "random_state": 0}
  }
}
```

## From Python

The same object, constructed directly or read from a file:

```python
from prothon.config import Study

study = Study.from_file("study.yml")
comparison = study.run()
print(comparison.summary())
```

```python
study = Study(
    ensembles=[
        {"ensemble": "wt.xtc", "topology": "wt.pdb", "label": "wild type"},
        {"ensemble": "mut.xtc", "topology": "mut.pdb", "label": "F5G"},
    ],
    reference="wild type",
    settings={"order_parameters": "cbcn", "random_state": 0},
)
study.run()
study.save("study.yml")       # and write it down
```

`study.resolve()` returns the loaded ensembles without running anything, for
when you want to do something else with them.

## The older key name

`source:` was the first name for `ensemble:`. Files written against it keep
working.
