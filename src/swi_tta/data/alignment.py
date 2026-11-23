"""Input-aligned period resampling and valid-station filtering utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def get_valid_station_mask(obs_mask: np.ndarray, min_valid_periods: int = 10) -> np.ndarray:
    """Return boolean mask of stations with at least ``min_valid_periods`` active periods."""
    per_period_valid = (obs_mask[:, :, 0] > 0.5) | (obs_mask[:, :, 1] > 0.5)
    counts = np.sum(per_period_valid, axis=1)
    return counts >= int(min_valid_periods)


def _extract_channel_kept_mask(
    obs_mask: np.ndarray,
    heldout_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    om = np.asarray(obs_mask) > 0.5
    if heldout_mask is None:
        return om[:, :, 0], om[:, :, 1]
    hm = np.asarray(heldout_mask) > 0.5
    if hm.ndim == 2:
        return om[:, :, 0] & (~hm), om[:, :, 1] & (~hm)
    if hm.ndim == 3:
        return om[:, :, 0] & (~hm[:, :, 0]), om[:, :, 1] & (~hm[:, :, 1])
    raise ValueError(f"Unsupported heldout_mask rank: {hm.ndim}")


def prepare_aligned_input(
    period: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    heldout_mask: Optional[np.ndarray],
    train_grid: np.ndarray,
    remove_group: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate kept observed dispersion onto the canonical training period grid.

    Strictly excludes held-out periods prior to interpolation and performs no
    extrapolation outside each station's observed period bounds (masking out-of-band
    entries with ``-1.0``).

    Returns
    -------
    x : np.ndarray
        Aligned input tensor of shape ``(N, 3, G)`` (float32).
    pad_mask : np.ndarray
        Boolean padding mask of shape ``(N, G)`` where ``True`` marks absent tokens.
    """
    per = np.asarray(period, dtype=np.float64)
    ph = np.asarray(phase_vel, dtype=np.float64)
    gp = np.asarray(group_vel, dtype=np.float64)
    kp0, kp1 = _extract_channel_kept_mask(obs_mask, heldout_mask)

    g = np.asarray(train_grid, dtype=np.float64)
    n = per.shape[0]
    m = len(g)

    per_out = np.broadcast_to(g[None, :], (n, m)).copy()
    ph_out = np.full((n, m), -1.0, dtype=np.float32)
    gp_out = np.full((n, m), -1.0, dtype=np.float32)

    for i in range(n):
        m0 = kp0[i] & (per[i] > 0)
        if np.any(m0):
            t_cand = per[i, m0]
            v_cand = ph[i, m0]
            fin = np.isfinite(t_cand) & np.isfinite(v_cand) & (v_cand > 0)
            if np.sum(fin) >= 2:
                tk, vk = t_cand[fin], v_cand[fin]
                ord0 = np.argsort(tk)
                tk, vk = tk[ord0], vk[ord0]
                in_rng = (g >= tk[0] - 1e-6) & (g <= tk[-1] + 1e-6)
                if np.any(in_rng):
                    ph_out[i, in_rng] = np.interp(g[in_rng], tk, vk).astype(np.float32)

        if not remove_group:
            m1 = kp1[i] & (per[i] > 0)
            if np.any(m1):
                t_cand = per[i, m1]
                v_cand = gp[i, m1]
                fin = np.isfinite(t_cand) & np.isfinite(v_cand) & (v_cand > 0)
                if np.sum(fin) >= 2:
                    tk, vk = t_cand[fin], v_cand[fin]
                    ord1 = np.argsort(tk)
                    tk, vk = tk[ord1], vk[ord1]
                    in_rng = (g >= tk[0] - 1e-6) & (g <= tk[-1] + 1e-6)
                    if np.any(in_rng):
                        gp_out[i, in_rng] = np.interp(g[in_rng], tk, vk).astype(np.float32)

    x = np.stack([per_out.astype(np.float32), ph_out, gp_out], axis=1)
    x[x <= 0] = -1.0
    pad_mask = (x[:, 1, :] == -1.0) & (x[:, 2, :] == -1.0)
    return x, pad_mask


def compute_period_residual_curve(
    periods: np.ndarray,
    ph_pred: np.ndarray,
    ph_obs: np.ndarray,
    eval_mask_phase: np.ndarray,
    n_bins: int = 28,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize relative phase-velocity residual ``(v_pred - v_obs) / v_obs`` across period bins."""
    valid = (eval_mask_phase > 0.5) & (periods > 0) & (ph_obs > 0) & np.isfinite(ph_pred) & (ph_pred > 0)
    t_v = periods[valid].astype(np.float64)
    r_v = ((ph_pred[valid] - ph_obs[valid]) / ph_obs[valid]).astype(np.float64)

    if len(t_v) == 0:
        return {"bins": [], "overall_mean": float("nan"), "overall_std": float("nan")}

    lo = float(np.min(t_v)) if t_min is None else float(t_min)
    hi = float(np.max(t_v)) if t_max is None else float(t_max)
    edges = np.linspace(lo, hi, int(n_bins) + 1)

    bins: List[Dict[str, Any]] = []
    bin_abs_means: List[float] = []
    bin_stds: List[float] = []
    for b in range(int(n_bins)):
        sel = (t_v >= edges[b]) & (t_v < edges[b + 1]) if b < int(n_bins) - 1 else (t_v >= edges[b]) & (t_v <= edges[b + 1])
        cnt = int(np.sum(sel))
        if cnt == 0:
            continue
        rv = r_v[sel]
        bm = float(np.mean(rv))
        bs = float(np.std(rv))
        bin_abs_means.append(abs(bm))
        bin_stds.append(bs)
        bins.append(
            {
                "t_lo": float(edges[b]),
                "t_hi": float(edges[b + 1]),
                "t_center": float(0.5 * (edges[b] + edges[b + 1])),
                "count": cnt,
                "mean_rel": bm,
                "median_rel": float(np.median(rv)),
                "std_rel": bs,
                "q25_rel": float(np.percentile(rv, 25)),
                "q75_rel": float(np.percentile(rv, 75)),
            }
        )

    mean_abs_bias = float(np.mean(bin_abs_means)) if bin_abs_means else float("nan")
    mean_scatter_std = float(np.mean(bin_stds)) if bin_stds else float("nan")
    return {
        "n_points": int(len(t_v)),
        "t_range": [lo, hi],
        "overall_mean_rel": float(np.mean(r_v)),
        "overall_median_rel": float(np.median(r_v)),
        "overall_std_rel": float(np.std(r_v)),
        "mean_abs_bin_bias": mean_abs_bias,
        "mean_bin_scatter_std": mean_scatter_std,
        "bias_to_scatter_ratio": float(mean_abs_bias / max(mean_scatter_std, 1e-12)),
        "bins": bins,
    }
