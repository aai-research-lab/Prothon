# What a linear treatment of a circular feature costs

**The cost is not the same for every metric, and it is not where the
documentation used to say it was.** Wasserstein fails badly, by up to 85x.
Kolmogorov-Smirnov fails in a different way, reporting the coordinate
convention rather than the data. Jensen-Shannon, the default, barely moves in
this construction -- at most 1.1x -- because two tight populations either side
of the wrap have almost no overlap under *either* treatment, so the distance is
near its maximum before the treatment is chosen.

That last result is worth stating plainly, because the obvious argument for a
circular density estimator -- that a population straddling the wrap arrives as
two separated modes -- is sound and yet does not produce a large error here.
Both populations split the same way, and the overlap survives.

Two tight von Mises populations a fixed angular distance apart, 2000 draws each, 100 replicates. The same pair is placed once away from the wrap at ±π and once straddling it. Only the position changes; the separation does not.

## Wasserstein-1

| separation | κ | spread (rad) | away | across, circular | across, linear | ratio |
|---|---|---|---|---|---|---|
| 0.05 | 10 | 0.316 | 0.0512 | 0.0504 | 0.3417 | 6.8× |
| 0.05 | 30 | 0.183 | 0.0511 | 0.0497 | 0.6347 | 12.8× |
| 0.05 | 100 | 0.100 | 0.0504 | 0.0504 | 1.2014 | 23.8× |
| 0.05 | 400 | 0.050 | 0.0500 | 0.0500 | 2.3618 | 47.3× |
| 0.05 | 1600 | 0.025 | 0.0500 | 0.0501 | 4.2467 | 84.8× |
| 0.10 | 10 | 0.316 | 0.1009 | 0.1001 | 0.6769 | 6.8× |
| 0.10 | 30 | 0.183 | 0.1010 | 0.0996 | 1.2491 | 12.5× |
| 0.10 | 100 | 0.100 | 0.1004 | 0.1004 | 2.3131 | 23.0× |
| 0.10 | 400 | 0.050 | 0.1000 | 0.1000 | 4.1854 | 41.9× |
| 0.10 | 1600 | 0.025 | 0.1000 | 0.1001 | 5.8975 | 58.9× |
| 0.20 | 10 | 0.316 | 0.2008 | 0.1999 | 1.3335 | 6.7× |
| 0.20 | 30 | 0.183 | 0.2010 | 0.1995 | 2.4022 | 12.0× |
| 0.20 | 100 | 0.100 | 0.2004 | 0.2004 | 4.0916 | 20.4× |
| 0.20 | 400 | 0.050 | 0.2000 | 0.2000 | 5.7988 | 29.0× |
| 0.20 | 1600 | 0.025 | 0.2000 | 0.2001 | 6.0827 | 30.4× |
| 0.40 | 10 | 0.316 | 0.4007 | 0.3999 | 2.5239 | 6.3× |
| 0.40 | 30 | 0.183 | 0.4010 | 0.3995 | 4.1534 | 10.4× |
| 0.40 | 100 | 0.100 | 0.4004 | 0.4004 | 5.5912 | 14.0× |
| 0.40 | 400 | 0.050 | 0.4000 | 0.4000 | 5.8828 | 14.7× |
| 0.40 | 1600 | 0.025 | 0.4000 | 0.4001 | 5.8831 | 14.7× |

Largest overestimate across the wrap: **84.8×**, at a separation of 0.05 rad and κ = 1600. The `away` column is the circular distance away from the wrap, and the linear treatment reproduces it to four decimals there — which is why the failure survives inspection. It depends on where the population sits, and on whether the population is narrow enough to fall on one side of the wrap rather than straddling it.

## Jensen–Shannon

| separation | κ | spread (rad) | away | across, circular | across, linear | ratio |
|---|---|---|---|---|---|---|
| 0.05 | 10 | 0.316 | 0.0758 | 0.0742 | 0.0593 | 0.8× |
| 0.05 | 30 | 0.183 | 0.1200 | 0.1140 | 0.0996 | 0.9× |
| 0.05 | 100 | 0.100 | 0.2079 | 0.1970 | 0.1776 | 0.9× |
| 0.05 | 400 | 0.050 | 0.3886 | 0.3526 | 0.3381 | 1.0× |
| 0.05 | 1600 | 0.025 | 0.6329 | 0.5554 | 0.6160 | 1.1× |
| 0.10 | 10 | 0.316 | 0.1336 | 0.1306 | 0.1180 | 0.9× |
| 0.10 | 30 | 0.183 | 0.2262 | 0.2168 | 0.1974 | 0.9× |
| 0.10 | 100 | 0.100 | 0.3935 | 0.3739 | 0.3456 | 0.9× |
| 0.10 | 400 | 0.050 | 0.6798 | 0.6333 | 0.6209 | 1.0× |
| 0.10 | 1600 | 0.025 | 0.9217 | 0.8794 | 0.9268 | 1.1× |
| 0.20 | 10 | 0.316 | 0.2535 | 0.2481 | 0.2336 | 0.9× |
| 0.20 | 30 | 0.183 | 0.4254 | 0.4113 | 0.3845 | 0.9× |
| 0.20 | 100 | 0.100 | 0.6853 | 0.6608 | 0.6338 | 1.0× |
| 0.20 | 400 | 0.050 | 0.9470 | 0.9299 | 0.9329 | 1.0× |
| 0.20 | 1600 | 0.025 | 0.9996 | 0.9990 | 0.9999 | 1.0× |
| 0.40 | 10 | 0.316 | 0.4732 | 0.4649 | 0.4489 | 1.0× |
| 0.40 | 30 | 0.183 | 0.7258 | 0.7110 | 0.6923 | 1.0× |
| 0.40 | 100 | 0.100 | 0.9496 | 0.9411 | 0.9387 | 1.0× |
| 0.40 | 400 | 0.050 | 0.9999 | 0.9999 | 0.9999 | 1.0× |
| 0.40 | 1600 | 0.025 | 1.0000 | 1.0000 | 1.0000 | 1.0× |

Largest overestimate across the wrap: **1.1×**, at a separation of 0.05 rad and κ = 1600. The `away` column is the circular distance away from the wrap, and the linear treatment reproduces it to four decimals there — which is why the failure survives inspection. It depends on where the population sits, and on whether the population is narrow enough to fall on one side of the wrap rather than straddling it.

## Kolmogorov–Smirnov, and where the origin sits

The same two populations, rotated together through twelve positions. Rotating both is a change of coordinate convention and nothing else, so a statistic that moves under it is reporting the convention.

- Linear KS: the statistic moves by **0.0051** between the best and worst origin.
- Kuiper's, the circular branch: **0.0000**.

Kuiper's statistic is invariant to the choice of origin by construction, which is why it replaces KS for circular features rather than the circular distance being bolted onto KS.
