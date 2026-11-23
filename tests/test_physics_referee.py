"""Unit tests for the exact disba physics referee and LVZ root-skipping diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from swi_tta.evaluation.metrics import disba_misfit_ho
from swi_tta.physics.disba_referee import (
    DEFAULT_DEEP_DC_KM_S,
    DEFAULT_SHALLOW_DC_KM_S,
    depth_to_thickness,
    forward_dispersion,
    forward_dispersion_diag,
    resolve_regime_dc,
    vs_to_vp_rho,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_vs_to_vp_rho_brocher_and_mantle_switch() -> None:
    depth = np.array([0.0, 50.0, 120.0, 121.0, 200.0], dtype=np.float64)
    vs = np.array([0.8, 3.5, 4.4, 4.5, 4.6], dtype=np.float64)
    vp, rho = vs_to_vp_rho(vs, depth=depth)

    assert np.all(np.isfinite(vp)) and np.all(np.isfinite(rho))
    assert np.all(vp > np.sqrt(2.0) * vs)
    assert np.all(rho >= 1.0)
    # Mantle switch strictly applies above 120 km depth
    assert np.isclose(vp[3], 1.79 * vs[3], atol=1e-12)
    assert np.isclose(vp[4], 1.79 * vs[4], atol=1e-12)
    # Crustal Brocher Eq. 9 differs from linear 1.79*Vs
    assert not np.isclose(vp[1], 1.79 * vs[1], atol=1e-2)


def test_depth_to_thickness_positive_and_exact() -> None:
    depth = np.linspace(0.0, 0.69, 70)
    thick = depth_to_thickness(depth)
    assert thick.shape == depth.shape
    assert np.all(thick >= 1e-4)
    assert np.allclose(thick[:-1], np.diff(depth), atol=1e-12)
    assert np.isclose(thick[-1], thick[-2], atol=1e-12)


def test_resolve_regime_dc_defaults_and_overrides() -> None:
    # Zero silent fallback: omitting all dc specifications must raise ValueError
    with pytest.raises(ValueError):
        resolve_regime_dc("shallow")
    with pytest.raises(ValueError):
        resolve_regime_dc("deep")

    dc_cfg = {"shallow_converged": DEFAULT_SHALLOW_DC_KM_S, "deep_converged": DEFAULT_DEEP_DC_KM_S}
    assert resolve_regime_dc("shallow", dc_cfg=dc_cfg) == pytest.approx(DEFAULT_SHALLOW_DC_KM_S)
    assert resolve_regime_dc("deep", dc_cfg=dc_cfg) == pytest.approx(DEFAULT_DEEP_DC_KM_S)
    assert resolve_regime_dc("shallow", dc=1e-4) == pytest.approx(1e-4)
    assert resolve_regime_dc("shallow", dc_shallow=5e-5) == pytest.approx(5e-5)
    assert resolve_regime_dc("deep", dc_deep=1e-3) == pytest.approx(1e-3)


def test_forward_dispersion_and_diag_are_bit_exact() -> None:
    depth = np.linspace(0.0, 0.69, 70)
    vs = 0.55 + 1.4 * depth
    periods = np.linspace(0.30, 1.80, 15)

    ph_base, gp_base = forward_dispersion(vs, depth, periods, dc=DEFAULT_SHALLOW_DC_KM_S, compute_group=True)
    diag = forward_dispersion_diag(vs, depth, periods, dc=DEFAULT_SHALLOW_DC_KM_S, compute_group=True)

    assert float(np.max(np.abs(ph_base - diag["phase_vel"]))) == 0.0
    assert float(np.max(np.abs(gp_base - diag["group_vel"]))) == 0.0
    assert diag["is_complete"] is True
    assert diag["is_empty"] is False
    assert diag["actual_dc"] == pytest.approx(DEFAULT_SHALLOW_DC_KM_S)


def test_live_shallow_lvz_root_skipping_reproduction() -> None:
    rs_path = REPO_ROOT / "results_summary" / "root_skipping_diagnostics.json"
    rs = json.loads(rs_path.read_text(encoding="utf-8"))
    depth_km = np.asarray(rs["typical_lvz_depth_km"], dtype=np.float64)
    ex = rs["typical_lvz_examples"][0]

    vs_profile = np.asarray(ex["profiles"]["vs_refined"], dtype=np.float64)
    periods_s = np.asarray(ex["periods_s"], dtype=np.float64)
    phase_obs = np.asarray(ex["phase_obs_km_s"], dtype=np.float64)
    ho_mask = np.asarray(ex["is_heldout"], dtype=np.float64)
    obs_mask = np.ones_like(ho_mask)

    res_coarse = forward_dispersion_diag(vs_profile, depth_km, periods_s, compute_group=False, dc=DEFAULT_DEEP_DC_KM_S)
    res_conv = forward_dispersion_diag(vs_profile, depth_km, periods_s, compute_group=False, dc=DEFAULT_SHALLOW_DC_KM_S)

    ho_coarse = disba_misfit_ho(res_coarse["phase_vel"], phase_obs, obs_mask, ho_mask)
    ho_conv = disba_misfit_ho(res_conv["phase_vel"], phase_obs, obs_mask, ho_mask)

    assert ho_coarse > 0.20, f"Expected coarse dc root-skipping (>20%), got {ho_coarse:.4f}"
    assert ho_conv < 0.05, f"Expected converged dc recovery (<5%), got {ho_conv:.4f}"
