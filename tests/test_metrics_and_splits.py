"""Unit tests for evaluation metrics and deterministic spatial/period split invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from swi_tta.data.splits import (
    apply_period_holdout_mask,
    create_period_holdout_split,
    create_spatial_block_split,
)
from swi_tta.evaluation.metrics import (
    compute_resolvable_depth,
    disba_misfit,
    disba_misfit_ho,
    vs_rel_err_res,
    vs_rmse,
    vs_rmse_res,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_vs_rmse_res_ignores_depths_below_z_res() -> None:
    depth = np.linspace(0.0, 0.70, 71)
    vs_true = np.stack([0.6 + 1.0 * depth, 0.8 + 0.9 * depth], axis=0)
    vs_pred = vs_true.copy()
    z_res = np.array([0.30, 0.50], dtype=np.float64)

    # Perturb strictly below each station's resolvable depth
    vs_pred[0, depth > 0.30] += 3.0
    vs_pred[1, depth > 0.50] += 3.0

    full = vs_rmse(vs_pred, vs_true)
    res = vs_rmse_res(vs_pred, vs_true, depth, z_res)
    rel = vs_rel_err_res(vs_pred, vs_true, depth, z_res)

    assert full["overall"] > 1.0
    assert res["overall"] == pytest.approx(0.0, abs=1e-12)
    assert rel["median"] == pytest.approx(0.0, abs=1e-12)


def test_compute_resolvable_depth_formula() -> None:
    depth = np.linspace(0.0, 1.0, 101)
    periods = np.array([[0.3, 0.6, 1.2, 1.5]], dtype=np.float64)
    phase = np.array([[0.5, 0.7, 0.9, 1.0]], dtype=np.float64)
    mask = np.array([[1, 1, 1, 0]], dtype=np.uint8)  # 1.5s is masked out -> T_max=1.2s, c=0.9
    zr = compute_resolvable_depth(periods, phase, mask, depth)
    assert zr[0] == pytest.approx(0.9 * 1.2 / 3.0, abs=1e-12)


def test_disba_misfit_and_holdout_isolation() -> None:
    d_obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    d_pred = np.array([1.1, 2.0, 3.0, 4.0], dtype=np.float64)
    obs_m = np.array([1, 1, 1, 1], dtype=np.float64)
    ho_m = np.array([0, 1, 1, 0], dtype=np.float64)

    # Index 0 has 10% error (kept), indices 1-2 have 0% error (held-out)
    assert disba_misfit(d_pred, d_obs, mask=obs_m) == pytest.approx(0.05, abs=1e-12)
    assert disba_misfit_ho(d_pred, d_obs, obs_m, ho_m) == pytest.approx(0.0, abs=1e-12)


def test_spatial_block_and_period_holdout_splits_are_disjoint_and_deterministic() -> None:
    rng = np.random.default_rng(123)
    coords = rng.uniform(0.0, 10.0, size=(300, 2))
    sp1 = create_spatial_block_split(coords, ratio_a=0.30, grid_nx=5, grid_ny=5, seed=7)
    sp2 = create_spatial_block_split(coords, ratio_a=0.30, grid_nx=5, grid_ny=5, seed=7)
    assert sp1 == sp2
    assert set(sp1["indices_a"]).isdisjoint(set(sp1["indices_b"]))
    assert set(sp1["blocks_a"]).isdisjoint(set(sp1["blocks_b"]))
    assert len(sp1["indices_a"]) + len(sp1["indices_b"]) == 300

    obs_mask = np.ones((12, 20, 2), dtype=np.uint8)
    ho1 = create_period_holdout_split(obs_mask, holdout_ratio=0.20, seed=11, keep_endpoints=True)
    ho2 = create_period_holdout_split(obs_mask, holdout_ratio=0.20, seed=11, keep_endpoints=True)
    assert ho1 == ho2
    kept_m, ho_m = apply_period_holdout_mask(obs_mask, ho1["heldout_indices"])
    assert int(np.sum((kept_m > 0) & (ho_m > 0))) == 0
    assert np.array_equal(kept_m + ho_m, obs_mask)
    # Endpoints preserved
    for idxs in ho1["heldout_indices"]:
        assert 0 not in idxs and 19 not in idxs


def test_committed_split_manifests_integrity() -> None:
    splits_dir = REPO_ROOT / "data" / "splits"
    lb = json.loads((splits_dir / "longbeach_blocks.json").read_text(encoding="utf-8"))
    cs = json.loads((splits_dir / "csrm_blocks.json").read_text(encoding="utf-8"))
    ho = json.loads((splits_dir / "heldout_periods.json").read_text(encoding="utf-8"))

    assert lb["num_total"] == 5297
    assert set(lb["indices_a"]).isdisjoint(set(lb["indices_b"]))
    assert cs["num_total"] == 12705
    assert set(cs["indices_a"]).isdisjoint(set(cs["indices_b"]))
    assert "longbeach" in ho and "csrm" in ho
    assert len(ho["longbeach"]["heldout_indices"]) == 5297
    assert len(ho["csrm"]["heldout_indices"]) == 12705
