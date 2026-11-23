#!/usr/bin/env python3
"""Pure-CPU smoke verification script (target wall time <= 30 s).

Verifies:
1. Package installation and module imports (``swi_tta``).
2. Integrity of ``results_summary/manifest_sha256.json`` and calibrated noise-floor table.
3. Exact ``disba`` forward referee physics and Brocher/mantle empirical relations.
4. Live CPU demonstration of the shallow low-velocity-zone (LVZ) ``disba``
   root-skipping pitfall: default ``dc = 0.005 km/s`` vs converged ``dc = 2.5e-5 km/s``.
5. Core evaluation metrics and zero-leakage input alignment invariants.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import swi_tta
from swi_tta.data.alignment import prepare_aligned_input
from swi_tta.data.splits import apply_period_holdout_mask, create_period_holdout_split, create_spatial_block_split
from swi_tta.evaluation.metrics import disba_misfit_ho, vs_rmse_res
from swi_tta.physics.disba_referee import (
    DEFAULT_DEEP_DC_KM_S,
    DEFAULT_SHALLOW_DC_KM_S,
    forward_dispersion_diag,
    vs_to_vp_rho,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_summary_manifest() -> None:
    manifest_path = REPO_ROOT / "results_summary" / "manifest_sha256.json"
    assert manifest_path.exists(), f"Missing summary manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for fname, meta in manifest["files"].items():
        fp = REPO_ROOT / "results_summary" / fname
        assert fp.exists(), f"Missing summary file: {fp}"
        digest = _sha256(fp)
        assert digest == meta["sha256"], f"SHA-256 mismatch for {fname}"

    cal = json.loads((REPO_ROOT / "results_summary" / "calibration_and_domains.json").read_text(encoding="utf-8"))
    dl = cal["calibrated_noise_floors"]["deep_lithospheric"]
    assert abs(dl["sigma_hat_d"] - 0.02416589723283842) < 1e-12
    assert abs(dl["ratio_sigma_over_gt_eval"] - 1.8184119278999296) < 1e-10
    print(
        f"[1/4] Summary manifest verified ({len(manifest['files'])} files, "
        f"{manifest['total_size_mb']:.2f} MB); Deep Lithospheric sigma_hat_D = {dl['sigma_hat_d']:.5f} ({dl['ratio_sigma_over_gt_eval']:.2f}x GT)."
    )


def verify_live_lvz_root_skipping() -> None:
    rs = json.loads((REPO_ROOT / "results_summary" / "root_skipping_diagnostics.json").read_text(encoding="utf-8"))
    depth_km = np.asarray(rs["typical_lvz_depth_km"], dtype=np.float64)
    ex = rs["typical_lvz_examples"][0]
    vs_profile = np.asarray(ex["profiles"]["vs_refined"], dtype=np.float64)
    periods_s = np.asarray(ex["periods_s"], dtype=np.float64)
    phase_obs = np.asarray(ex["phase_obs_km_s"], dtype=np.float64)
    is_ho = np.asarray(ex["is_heldout"], dtype=bool)

    vp, rho = vs_to_vp_rho(vs_profile, depth_km)
    assert np.all(np.isfinite(vp)) and np.all(np.isfinite(rho)) and np.all(vp > vs_profile)

    # Compare coarse default dc=0.005 vs converged shallow dc=2.5e-5 LIVE on CPU
    res_coarse = forward_dispersion_diag(
        vs_profile, depth_km, periods_s, mode=0, compute_group=False, dc=DEFAULT_DEEP_DC_KM_S
    )
    res_conv = forward_dispersion_diag(
        vs_profile, depth_km, periods_s, mode=0, compute_group=False, dc=DEFAULT_SHALLOW_DC_KM_S
    )

    obs_mask = np.ones_like(phase_obs, dtype=np.float64)
    ho_mask = is_ho.astype(np.float64)

    ho_coarse = disba_misfit_ho(res_coarse["phase_vel"], phase_obs, obs_mask, ho_mask)
    ho_conv = disba_misfit_ho(res_conv["phase_vel"], phase_obs, obs_mask, ho_mask)

    assert ho_coarse > 0.20, f"Expected coarse dc root-skipping (>20% HO misfit), got {ho_coarse:.4f}"
    assert ho_conv < 0.05, f"Expected converged dc recovery (<5% HO misfit), got {ho_conv:.4f}"
    print(
        f"[2/4] Live shallow LVZ root-skipping demo (sample #{ex['sample_idx']}, {ex['structure_type']}): "
        f"default dc={DEFAULT_DEEP_DC_KM_S:g} -> HO misfit = {ho_coarse*100:.2f}% (Mode-1 jump) vs "
        f"converged dc={DEFAULT_SHALLOW_DC_KM_S:g} -> HO misfit = {ho_conv*100:.2f}% (Mode-0 recovered)."
    )


def verify_metrics_and_splits() -> None:
    depth = np.linspace(0.0, 0.70, 70)
    vs_true = 0.5 + 1.2 * depth
    vs_pred = vs_true.copy()
    vs_pred[depth > 0.40] += 5.0  # Perturb solely below z_res=0.40 km
    rmse_res = vs_rmse_res(vs_pred[None, :], vs_true[None, :], depth, np.array([0.40]))["overall"]
    assert abs(rmse_res) < 1e-12, f"Expected 0.0 RMSE within z_res, got {rmse_res}"

    rng = np.random.default_rng(42)
    coords = rng.uniform(0.0, 1.0, size=(200, 2))
    sp = create_spatial_block_split(coords, ratio_a=0.30, grid_nx=4, grid_ny=4, seed=42)
    assert len(set(sp["indices_a"]).intersection(sp["indices_b"])) == 0
    assert len(sp["indices_a"]) + len(sp["indices_b"]) == 200

    obs_mask = np.ones((5, 25, 2), dtype=np.uint8)
    ho_sp = create_period_holdout_split(obs_mask, holdout_ratio=0.20, seed=42)
    kept_m, ho_m = apply_period_holdout_mask(obs_mask, ho_sp["heldout_indices"])
    assert int(np.sum((kept_m > 0) & (ho_m > 0))) == 0
    assert int(np.sum(kept_m + ho_m)) == int(np.sum(obs_mask))
    print("[3/4] Metrics (Resolvable-Depth Vs RMSE) and deterministic spatial/period split invariants verified.")


def verify_zero_leakage_alignment() -> None:
    grid_t = np.linspace(0.25, 2.0, 30, dtype=np.float32)
    obs_t = np.linspace(0.30, 1.80, 20, dtype=np.float32)
    ph = (0.4 + 0.5 * obs_t)[None, :]
    gp = np.zeros_like(ph)
    obs_mask = np.ones((1, 20, 2), dtype=np.float32)
    ho_mask = np.zeros((1, 20, 2), dtype=np.float32)
    ho_mask[:, [2, 7, 12, 17], :] = 1.0  # Held-out periods masked out

    ph_corrupt = ph.copy()
    ph_corrupt[:, [2, 7, 12, 17]] = 999.0  # Corrupt held-out values aggressively

    x_clean, _ = prepare_aligned_input(obs_t[None, :], ph, gp, obs_mask, ho_mask, grid_t, remove_group=True)
    x_corrupt, _ = prepare_aligned_input(obs_t[None, :], ph_corrupt, gp, obs_mask, ho_mask, grid_t, remove_group=True)
    max_diff = float(np.max(np.abs(x_clean - x_corrupt)))
    assert max_diff == 0.0, f"Held-out period leakage detected: max_diff={max_diff}"
    print("[4/4] Zero-leakage input alignment verified (corrupted held-out periods max_diff = 0.0).")


def main() -> int:
    t0 = time.perf_counter()
    print(f"swi_tta v{swi_tta.__version__} — Pure-CPU Installation & Physics Verification")
    verify_summary_manifest()
    verify_live_lvz_root_skipping()
    verify_metrics_and_splits()
    verify_zero_leakage_alignment()
    elapsed = time.perf_counter() - t0
    print(f"ALL CHECKS PASSED in {elapsed:.2f} s (pure CPU).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
