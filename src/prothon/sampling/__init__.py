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
    CorrelationProfile,
    block_labels,
    correlation_profile,
    correlation_time,
    correlation_time_estimate,
    effective_frames,
    plan_blocks,
)
from .floor import (
    FLOOR_QUANTILE,
    MINIMUM_FLOOR_REPEATS,
    MINIMUM_FLOOR_UNITS,
    FloorPlan,
    floor_unit_count,
    plan_floor,
    split_half_floor,
)
from .null import permutation_null, studentised_p_values
from .statistics import (
    benjamini_hochberg,
    effective_sample_size,
    random_sample,
)

__all__ = [
    "MINIMUM_BLOCKS",
    "FLOOR_QUANTILE",
    "MINIMUM_FLOOR_REPEATS",
    "MINIMUM_FLOOR_UNITS",
    "FloorPlan",
    "benjamini_hochberg",
    "effective_sample_size",
    "floor_unit_count",
    "permutation_null",
    "random_sample",
    "split_half_floor",
    "studentised_p_values",
    "CorrelationEstimate",
    "CorrelationProfile",
    "block_labels",
    "correlation_profile",
    "correlation_time",
    "correlation_time_estimate",
    "effective_frames",
    "plan_blocks",
    "plan_floor",
]
