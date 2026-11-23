"""Unified evaluation metrics and field-array dispersion evaluation."""

from __future__ import annotations

from swi_tta.evaluation.field_eval import evaluate_field_predictions, forward_field_batch_mp
from swi_tta.evaluation.metrics import (
    compute_per_sample_rmse_res,
    compute_resolvable_depth,
    disba_misfit,
    disba_misfit_ho,
    disba_misfit_per_sample,
    ref_corr,
    summarize_ref_corr,
    surr_gap,
    t_infer,
    vs_rel_err,
    vs_rel_err_res,
    vs_rmse,
    vs_rmse_res,
)

__all__ = [
    "compute_per_sample_rmse_res",
    "compute_resolvable_depth",
    "disba_misfit",
    "disba_misfit_ho",
    "disba_misfit_per_sample",
    "evaluate_field_predictions",
    "forward_field_batch_mp",
    "ref_corr",
    "summarize_ref_corr",
    "surr_gap",
    "t_infer",
    "vs_rel_err",
    "vs_rel_err_res",
    "vs_rmse",
    "vs_rmse_res",
]
