# Calibrated thresholds

Measured on null pairs at 2000 replicates per correlation time. Both ensembles come from one distribution, so every call is wrong by construction and the correct rate is exactly 5%.

| τ | threshold for 5% | rate at the nominal threshold | rate on fresh nulls | testable |
|---|---|---|---|---|
| 1 | 0.0250 | 10.3% | 4.3% | 100% |
| 5 | 0.0150 | 13.3% | 3.8% | 100% |
| 10 | 0.0200 | 13.1% | 5.6% | 100% |
| 25 | 0.0200 | 12.9% | 5.1% | 100% |
| 50 | 0.0150 | 15.9% | 5.3% | 82% |

**The third column is the promise and the fourth is whether it is kept.** The threshold is measured on one set of nulls and applied to a second, disjoint set. If the fourth column reads near 5% the correction holds on data it has never seen, which is the only evidence worth having. If it reads near the second column instead, the correction has not worked and nothing here should be shipped.

Reproduce with `python scripts/calibrate_threshold.py`.
