# How long is long enough? Q99, `cbcn`

Contiguous prefixes of the Q99 trajectory, 70 features. The floor is the smallest dissimilarity that much sampling can resolve; precision and recall are the prefix measured against the whole trajectory.

| conformations | floor | τ | slope | independent | precision | recall |
|---|---|---|---|---|---|---|
| 250 | 0.1051 | 5+ | — | 46 | 0.953 | 0.921 |
| 500 | 0.0811 | 17+ | 0.97 | 30 | 0.957 | 0.935 |
| 1000 | 0.0628 | 21+ | 0.85 | 47 | 0.955 | 0.943 |
| 2000 | 0.0499 | 19+ | 0.57 | 108 | 0.958 | 0.944 |
| 5000 | 0.0359 | 45+ | 0.48 | 110 | 0.954 | 0.954 |

`slope` is the slope of log τ against log n across four nested prefixes. Zero means the answer does not depend on how much data it was given, which is what a settled estimate looks like; one means the estimate is reporting the trajectory length rather than the correlation. `—` means the prefix was too short to fit three sub-prefixes, so no trend could be fitted and nothing is claimed. A `τ` marked `+` is a **lower bound**, and the `independent` column beside it is correspondingly an upper bound. 5 of 5 lengths are flagged.

## Where the crossing is

The Q99 against Q95 dissimilarity is **0.0618**.

It first exceeds the floor at **2000 conformations**, where the floor is 0.0499. At the length below that the floor is larger than the dissimilarity, so the two ensembles are not distinguishable at all and the value returned would be sampling rather than structure.

## Missed states, not invented ones

A prefix has visited a subset of what the whole trajectory visited, so it should be missing states and inventing none. Recall should therefore rise with length while precision stays near its floor. That is a prediction with a correct answer, which makes this a test of the implementation rather than an illustration of it.

- Recall rises: 0.921 at 250 conformations to 0.954 at 5000.
- Precision moves less than recall: 0.953 to 0.954.
