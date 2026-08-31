"""What a comparison is worth, given how the data was sampled.

This is the part with no counterpart in comparable tools, and the reason for
most of the numbers in the documentation. A distance between two ensembles
carries a contribution from the finiteness of each sample; a significance test
over trajectory frames is invalid unless the frames are exchangeable; and a
correlation time estimated from a short trajectory is a lower bound that looks
like a value.
"""

from .correlation import (
    MINIMUM_BLOCKS,
    CorrelationEstimate,
    block_labels,
    correlation_time,
    correlation_time_estimate,
    effective_frames,
    plan_blocks,
)

__all__ = [
    "MINIMUM_BLOCKS",
    "CorrelationEstimate",
    "block_labels",
    "correlation_time",
    "correlation_time_estimate",
    "effective_frames",
    "plan_blocks",
]
