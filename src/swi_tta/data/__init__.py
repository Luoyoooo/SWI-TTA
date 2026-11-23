"""Data loading, spatial/period splitting, and input-aligned period resampling."""

from __future__ import annotations

from swi_tta.data.alignment import (
    compute_period_residual_curve,
    get_valid_station_mask,
    prepare_aligned_input,
)
from swi_tta.data.loader import (
    DEEP_BASE_MODELS,
    SHALLOW_STRUCTURE_TYPES,
    SWIDatasetBatch,
    load_csrm,
    load_deep_subset,
    load_longbeach,
    load_shallow_subset,
)
from swi_tta.data.splits import (
    apply_period_holdout_mask,
    create_period_holdout_split,
    create_spatial_block_split,
    load_split_file,
    save_split_file,
)

__all__ = [
    "DEEP_BASE_MODELS",
    "SHALLOW_STRUCTURE_TYPES",
    "SWIDatasetBatch",
    "apply_period_holdout_mask",
    "compute_period_residual_curve",
    "create_period_holdout_split",
    "create_spatial_block_split",
    "get_valid_station_mask",
    "load_csrm",
    "load_deep_subset",
    "load_longbeach",
    "load_shallow_subset",
    "load_split_file",
    "prepare_aligned_input",
    "save_split_file",
]
