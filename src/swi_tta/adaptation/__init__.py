"""Test-time adaptation methods, station-independent output refinement, and early-stopping controls."""

from __future__ import annotations

from swi_tta.adaptation.baselines import (
    run_norm_stats_tta,
    run_output_space_refinement,
    run_zeroshot_inference,
)
from swi_tta.adaptation.gated_cycle import (
    TAU_QUANTILE_GRID,
    build_valid_channel_mask,
    compute_cycle_loss_from_outputs,
    compute_gate_weights,
    compute_r_depth,
    compute_r_depth_per_station,
    compute_source_tau_quantile,
    run_gated_cycle_tta,
)
from swi_tta.adaptation.safe_tta import (
    DEFAULT_VELOCITY_BOUNDS,
    compute_resolvable_depth_from_kept,
    create_stop_period_split,
    run_fixed_step_early_stopping,
    run_holdout_early_stopping,
    run_ideal_early_stopping,
    run_output_refinement_from_profile,
    run_safe_tta,
)

__all__ = [
    "DEFAULT_VELOCITY_BOUNDS",
    "TAU_QUANTILE_GRID",
    "build_valid_channel_mask",
    "compute_cycle_loss_from_outputs",
    "compute_gate_weights",
    "compute_r_depth",
    "compute_r_depth_per_station",
    "compute_resolvable_depth_from_kept",
    "compute_source_tau_quantile",
    "create_stop_period_split",
    "run_fixed_step_early_stopping",
    "run_gated_cycle_tta",
    "run_holdout_early_stopping",
    "run_ideal_early_stopping",
    "run_norm_stats_tta",
    "run_output_refinement_from_profile",
    "run_output_space_refinement",
    "run_safe_tta",
    "run_zeroshot_inference",
]
