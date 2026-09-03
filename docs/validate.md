# Scoring against experiment

Comparing two ensembles says how they differ. It cannot say which is right.
Answering that means predicting what an experiment would have measured and
checking against what it did — the difference between a difference detector and
an arbiter.

```python
from prothon.validate import radius_of_gyration, score_observable

rg = radius_of_gyration(ensemble.trajectory)          # per frame, nm
result = score_observable(
    rg[:, None], experimental=[2.71], uncertainty=[0.08], observable="Rg",
)
print(result.summary())
```

```
Rg: chi2_red = 0.31 (floor mean 0.44, q95 0.73) — agrees to within its own sampling
```

`radius_of_gyration` is protein-only by default, so explicit solvent, ions, a
ligand, or a membrane cannot silently dominate a SAXS-like Rg. Pass an MDTraj
selection when the experimental object is deliberately a larger complex:

```python
complex_rg = radius_of_gyration(traj, selection="not water")
```

`end_to_end` uses the terminal alpha carbons of one protein chain. A multichain
topology is refused unless `chain=` names a chain (by index or PDB ID), or
`selection=` explicitly defines the intended endpoints. In a `Prothon` study,
selecting the chain at ingestion with `chains=` or `--chains` applies the same
choice consistently to comparison and validation.

## A perfect ensemble does not score one

A reduced chi-squared of 1 is the usual target. For an ensemble it is wrong in
both directions, because the prediction is itself an estimate made from a
finite sample. Measured on a synthetic ensemble whose true average *is* the
experimental value, so the only error is sampling:

| conformations | χ²_red of a perfect ensemble |
|---|---|
| 20 | 0.77 |
| 50 | 0.33 |
| 200 | 0.08 |
| 1000 | 0.02 |
| 5000 | 0.00 |

A perfect ensemble of twenty conformations scores 0.77 and a perfect ensemble
of five thousand scores 0.00. Fitting either to 1.0 is fitting to noise: the
first is being asked to reproduce experimental scatter it cannot know about,
and the second to reproduce scatter it has already averaged away.

So every score is reported beside a **floor**. Independent generated structures
split random frames; trajectories split complete temporal blocks; supplied
replica labels split complete independent replicas. One half's prediction is
scored against the other's. The mean describes what sampling contributes and
the 95th percentile decides `within_floor`. When fewer than eight independent
units are available, `within_floor` is `None` rather than an invented verdict.

`score_observable` defaults to `sampling_kind="trajectory"` because a numeric
matrix does not carry its provenance. Pass `sampling_kind="iid"` for genuinely
independent generated conformations, or `replica_labels=` when the rows contain
several independent runs.

## Averaging is not always over the value

A PRE reports a distance, but the relaxation goes as the inverse sixth power,
so the ensemble average is $\langle r^{-6}\rangle^{-1/6}$ and not $\langle
r\rangle$. On a distribution with a rare compact state — 90% at 5.0 nm, 10% at
1.5 nm — the linear mean is **4.64 nm** and the sixth-power average is **2.19
nm**. Two and a half nanometres, in the direction of missing exactly the rare
compact state that PRE exists to detect.

```python
pre_distance(traj, "name CA and resid 42", "name H and resid 76")   # r^-6
```

FRET is averaged the same way — over the efficiency, which is what the photons
report, rather than over the distance. Take the mean of what `fret_efficiency`
returns, not the efficiency of the mean distance.

Note that a rigid test case cannot catch this mistake: on a narrow distribution
the two averages agree to within a hundredth of a nanometre.

## What is computed

Each of these is a closed-form function of the coordinates, so the value is a
consequence of the ensemble rather than of somebody's regression.

| observable | units | averaging |
|---|---|---|
| `radius_of_gyration` | nm | linear |
| `end_to_end` | nm | linear |
| `pairwise_distance` | nm | linear |
| `pre_distance` | nm | **r⁻⁶** |
| `fret_efficiency` | — | linear, over the efficiency |
| `j_coupling_hn_ha` | Hz | linear |

The Karplus relation for ³J(HN,HA) uses the Vuister and Bax (1993)
coefficients, giving about 4 Hz in a helix and 9 Hz in a sheet.
Because a backbone phi angle is undefined for the first residue, the first
coupling usually belongs to residue 2. Both `Prothon.validate("j_hn_ha", ...)`
and `prothon validate --observable j_hn_ha` preserve those residue identities:
`AgreementResult.feature_index` uses stable one-based positions and `labels`
adds chain identity for multichain inputs.

## What is not computed, and why

| observable | use instead |
|---|---|
| chemical shifts | SPARTA+, SHIFTX2, UCBShift |
| SAXS profile | CRYSOL, FoXS, Pepsi-SAXS |
| RDCs | PALES, or a singular-value alignment tensor fit |

Each of these needs something that is not in the coordinates: an empirical
predictor trained on a database, an explicit solvent layer, or an alignment
tensor fitted to the very data being compared against. Reimplementing any of
them badly would be worse than not having them.

`score_observable` takes predicted values from anywhere, so chemical shifts
from SPARTA+ or a SAXS profile from CRYSOL are scored here on the same footing,
with the same floor:

```python
shifts = np.loadtxt("sparta_predictions.txt")     # (n_frames, n_residues)
score_observable(shifts, measured, uncertainty, observable="CA shift")
```

## Uncertainties are required

`score_observable` refuses without them. A chi-squared computed without
experimental uncertainties is a sum of squares in arbitrary units, and the
floor it would be compared against means nothing.

## What it refuses

- An ensemble worth fewer than ten independent conformations. An ensemble
  average from that describes those conformations rather than a distribution.
- Zero or negative uncertainties.
- A prediction whose shape does not match the measurements.
