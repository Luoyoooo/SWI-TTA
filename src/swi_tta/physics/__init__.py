"""Exact surface-wave dispersion physics referee based on ``disba``."""

from __future__ import annotations

from swi_tta.physics.disba_referee import (
    DEFAULT_DEEP_DC_KM_S,
    DEFAULT_SHALLOW_DC_KM_S,
    MAX_DISBA_WORKERS,
    VS_SAFE_FLOOR_KM_S,
    batch_forward_dispersion,
    compute_masked_misfits_diag,
    depth_to_thickness,
    forward_dispersion,
    forward_dispersion_diag,
    forward_field_batch_diag,
    resolve_regime_dc,
    summarize_forward_diagnostics,
    vs_to_vp_rho,
)

__all__ = [
    "DEFAULT_DEEP_DC_KM_S",
    "DEFAULT_SHALLOW_DC_KM_S",
    "MAX_DISBA_WORKERS",
    "VS_SAFE_FLOOR_KM_S",
    "batch_forward_dispersion",
    "compute_masked_misfits_diag",
    "depth_to_thickness",
    "forward_dispersion",
    "forward_dispersion_diag",
    "forward_field_batch_diag",
    "resolve_regime_dc",
    "summarize_forward_diagnostics",
    "vs_to_vp_rho",
]
