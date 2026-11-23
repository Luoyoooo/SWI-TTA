"""Unified evaluation metrics for surface-wave dispersion inversion diagnostics.

Metrics include:
- ``vs_rmse``: Full-depth shear-wave velocity RMSE (km/s), optionally stratified by geology.
- ``compute_resolvable_depth`` & ``vs_rmse_res``: Resolvable-depth RMSE within
  ``[0, z_res]``, where ``z_res = min(c(T_max) * T_max / 3, z_model_max)``.
- ``vs_rel_err`` & ``vs_rel_err_res``: Median, p95, and p99 relative velocity errors.
- ``disba_misfit`` & ``disba_misfit_ho``: Relative RMS dispersion misfit on observed
  and strictly held-out period points.
- ``surr_gap``: Physical-vs-surrogate residual discrepancy.
- ``ref_corr`` & ``summarize_ref_corr``: Depth-slice Pearson correlation against
  published reference velocity models.
- ``t_infer``: Amortized per-station inference time (ms).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from scipy.stats import pearsonr

_DEPTH_KEY_RE = re.compile(r"^depth_([0-9]+(?:\.[0-9]+)?)km$")


def vs_rmse(
    vs_pred: np.ndarray,
    vs_true: np.ndarray,
    structure_types: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compute full-depth shear-wave velocity RMSE (km/s)."""
    pred = np.asarray(vs_pred, dtype=np.float64)
    true = np.asarray(vs_true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"vs_pred shape {pred.shape} does not match vs_true shape {true.shape}")

    if pred.ndim == 1:
        pred = pred[None, :]
        true = true[None, :]

    sample_rmse = np.sqrt(np.mean((pred - true) ** 2, axis=-1))
    result: Dict[str, float] = {
        "overall": float(np.mean(sample_rmse)),
        "overall_global": float(np.sqrt(np.mean((pred - true) ** 2))),
    }

    if structure_types is not None:
        st_arr = np.asarray(structure_types)
        if len(st_arr) != pred.shape[0]:
            raise ValueError("structure_types length must match sample count N")
        for st in sorted(set(st_arr.tolist())):
            idx = st_arr == st
            result[str(st)] = float(np.mean(sample_rmse[idx]))

    return result


def compute_resolvable_depth(
    periods: np.ndarray,
    phase_vel: np.ndarray,
    phase_mask: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    """Compute per-station resolvable depth ``z_res = min(c(T_max) * T_max / 3, z_max)``."""
    per = np.asarray(periods, dtype=np.float64)
    ph = np.asarray(phase_vel, dtype=np.float64)
    pm = np.asarray(phase_mask) > 0.5
    d_arr = np.asarray(depths, dtype=np.float64)
    z_max = float(d_arr[-1])
    z_min_fallback = float(d_arr[0])

    n = per.shape[0]
    z_res = np.full(n, z_max, dtype=np.float64)
    for i in range(n):
        valid = pm[i] & (per[i] > 0.0) & np.isfinite(per[i]) & (ph[i] > 0.0) & np.isfinite(ph[i])
        if np.any(valid):
            t_v = per[i, valid]
            c_v = ph[i, valid]
            k_max = int(np.argmax(t_v))
            zr = float(c_v[k_max] * t_v[k_max] / 3.0)
            z_res[i] = max(z_min_fallback, min(zr, z_max))
        else:
            z_res[i] = z_min_fallback
    return z_res


def compute_per_sample_rmse_res(
    vs_pred: np.ndarray,
    vs_true: np.ndarray,
    depths: np.ndarray,
    z_res: np.ndarray,
) -> np.ndarray:
    """Compute per-sample resolvable-depth Vs RMSE array of shape ``(N,)``."""
    pred = np.asarray(vs_pred, dtype=np.float64)
    true = np.asarray(vs_true, dtype=np.float64)
    d_arr = np.asarray(depths, dtype=np.float64)
    zr = np.asarray(z_res, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError(f"vs_pred {pred.shape} and vs_true {true.shape} must both be 2D (N, D)")
    if len(d_arr) != pred.shape[1]:
        raise ValueError(f"depths length {len(d_arr)} does not match D={pred.shape[1]}")
    if len(zr) != pred.shape[0]:
        raise ValueError(f"z_res length {len(zr)} does not match N={pred.shape[0]}")

    n = pred.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lmask = d_arr <= (zr[i] + 1e-6)
        if not np.any(lmask):
            lmask[0] = True
        diff = pred[i, lmask] - true[i, lmask]
        out[i] = float(np.sqrt(np.mean(diff ** 2)))
    return out


def vs_rmse_res(
    vs_pred: np.ndarray,
    vs_true: np.ndarray,
    depths: np.ndarray,
    z_res: np.ndarray,
    structure_types: Optional[Sequence[str]] = None,
    id_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute shear-wave velocity RMSE restricted to each sample's resolvable depth ``[0, z_res]``."""
    pred = np.asarray(vs_pred, dtype=np.float64)
    true = np.asarray(vs_true, dtype=np.float64)
    d_arr = np.asarray(depths, dtype=np.float64)
    zr = np.asarray(z_res, dtype=np.float64)

    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError(f"vs_pred {pred.shape} and vs_true {true.shape} must both be (N, D)")
    if len(d_arr) != pred.shape[1]:
        raise ValueError(f"depths length {len(d_arr)} does not match profile layers {pred.shape[1]}")
    if len(zr) != pred.shape[0]:
        raise ValueError(f"z_res length {len(zr)} does not match sample count {pred.shape[0]}")

    n = pred.shape[0]
    sample_rmse = np.empty(n, dtype=np.float64)
    sq_total = 0.0
    cnt_total = 0

    for i in range(n):
        lmask = d_arr <= (zr[i] + 1e-6)
        if not np.any(lmask):
            lmask[0] = True
        diff = pred[i, lmask] - true[i, lmask]
        sq = diff ** 2
        sample_rmse[i] = float(np.sqrt(np.mean(sq)))
        sq_total += float(np.sum(sq))
        cnt_total += int(diff.size)

    result: Dict[str, Any] = {
        "overall": float(np.mean(sample_rmse)),
        "overall_global": float(np.sqrt(sq_total / max(cnt_total, 1))),
        "sample_rmse_median": float(np.median(sample_rmse)),
        "mean_z_res_km": float(np.mean(zr)),
        "median_z_res_km": float(np.median(zr)),
    }

    if structure_types is not None:
        st_arr = np.asarray(structure_types)
        if len(st_arr) != n:
            raise ValueError("structure_types length must match sample count N")
        by_type: Dict[str, float] = {}
        for st in sorted(set(st_arr.tolist())):
            idx = st_arr == st
            v = float(np.mean(sample_rmse[idx]))
            result[str(st)] = v
            by_type[str(st)] = v
        result["by_type"] = by_type

        if id_types is not None:
            id_set = set(str(x) for x in id_types)
            is_id = np.isin(st_arr, list(id_set))
            if np.any(is_id):
                result["id_overall"] = float(np.mean(sample_rmse[is_id]))
            if np.any(~is_id):
                result["ood_overall"] = float(np.mean(sample_rmse[~is_id]))

    return result


def vs_rel_err(
    vs_pred: np.ndarray,
    vs_true: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute median, p95, and p99 relative velocity error across full depth."""
    pred = np.asarray(vs_pred, dtype=np.float64)
    true = np.asarray(vs_true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"vs_pred shape {pred.shape} does not match vs_true shape {true.shape}")

    rel = np.abs(pred - true) / np.maximum(np.abs(true), eps)
    sample_rel = np.mean(rel, axis=-1) if rel.ndim > 1 else rel

    return {
        "median": float(np.median(sample_rel)),
        "p95": float(np.percentile(sample_rel, 95)),
        "p99": float(np.percentile(sample_rel, 99)),
        "pointwise_median": float(np.median(rel)),
        "pointwise_p95": float(np.percentile(rel, 95)),
        "pointwise_p99": float(np.percentile(rel, 99)),
    }


def vs_rel_err_res(
    vs_pred: np.ndarray,
    vs_true: np.ndarray,
    depths: np.ndarray,
    z_res: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute median, p95, and p99 relative velocity error within ``[0, z_res]``."""
    pred = np.asarray(vs_pred, dtype=np.float64)
    true = np.asarray(vs_true, dtype=np.float64)
    d_arr = np.asarray(depths, dtype=np.float64)
    zr = np.asarray(z_res, dtype=np.float64)

    sample_rels: List[float] = []
    pw_rels: List[np.ndarray] = []
    for i in range(pred.shape[0]):
        lmask = d_arr <= (zr[i] + 1e-6)
        if not np.any(lmask):
            lmask[0] = True
        rel_i = np.abs(pred[i, lmask] - true[i, lmask]) / np.maximum(np.abs(true[i, lmask]), eps)
        sample_rels.append(float(np.mean(rel_i)))
        pw_rels.append(rel_i)

    sr = np.asarray(sample_rels, dtype=np.float64)
    pw = np.concatenate(pw_rels)
    return {
        "median": float(np.median(sr)),
        "p95": float(np.percentile(sr, 95)),
        "p99": float(np.percentile(sr, 99)),
        "pointwise_median": float(np.median(pw)),
        "pointwise_p95": float(np.percentile(pw, 95)),
        "pointwise_p99": float(np.percentile(pw, 99)),
    }


def disba_misfit(
    d_pred: np.ndarray,
    d_obs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> float:
    """Compute relative RMS dispersion misfit ``sqrt(mean(((d_pred - d_obs) / d_obs)^2))``."""
    pred = np.asarray(d_pred, dtype=np.float64)
    obs = np.asarray(d_obs, dtype=np.float64)
    if pred.shape != obs.shape:
        raise ValueError(f"d_pred shape {pred.shape} does not match d_obs shape {obs.shape}")

    if mask is None:
        valid = obs > eps
    else:
        m = np.asarray(mask)
        if m.shape != obs.shape:
            raise ValueError(f"mask shape {m.shape} does not match d_obs shape {obs.shape}")
        valid = (m > 0.5) & (np.abs(obs) > eps) & np.isfinite(pred) & np.isfinite(obs)

    if not np.any(valid):
        return float("nan")

    rel_sq = ((pred[valid] - obs[valid]) / obs[valid]) ** 2
    return float(np.sqrt(np.mean(rel_sq)))


def disba_misfit_per_sample(
    d_pred: np.ndarray,
    d_obs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute per-sample relative RMS dispersion misfit with shape ``(N,)``."""
    pred = np.asarray(d_pred, dtype=np.float64)
    obs = np.asarray(d_obs, dtype=np.float64)
    if pred.ndim == 1:
        return np.array([disba_misfit(pred, obs, mask, eps=eps)], dtype=np.float64)

    n = pred.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        m_i = mask[i] if mask is not None else None
        out[i] = disba_misfit(pred[i], obs[i], m_i, eps=eps)
    return out


def disba_misfit_ho(
    d_pred: np.ndarray,
    d_obs: np.ndarray,
    obs_mask: np.ndarray,
    heldout_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute relative RMS dispersion misfit exclusively on held-out period points."""
    om = np.asarray(obs_mask) > 0.5
    hm = np.asarray(heldout_mask) > 0.5
    combined_mask = (om & hm).astype(np.float64)
    return disba_misfit(d_pred=d_pred, d_obs=d_obs, mask=combined_mask, eps=eps)


def surr_gap(
    d_disba: np.ndarray,
    d_surr: np.ndarray,
    d_obs: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute discrepancy between exact ``disba`` misfit and surrogate misfit."""
    m_disba = disba_misfit(d_disba, d_obs, mask=mask)
    m_surr = disba_misfit(d_surr, d_obs, mask=mask)
    return {
        "disba_misfit": m_disba,
        "surr_misfit": m_surr,
        "surr_gap": float(m_disba - m_surr),
        "abs_gap": float(abs(m_disba - m_surr)),
    }


def ref_corr(
    vs_pred: np.ndarray,
    vs_ref: np.ndarray,
    depths: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute depth-slice Pearson correlations against a published reference model."""
    pred = np.asarray(vs_pred, dtype=np.float64)
    ref = np.asarray(vs_ref, dtype=np.float64)
    if pred.shape != ref.shape or pred.ndim != 2:
        raise ValueError(f"vs_pred {pred.shape} and vs_ref {ref.shape} must both be (N, D)")

    corrs = []
    slice_corrs: Dict[str, float] = {}
    for d_idx in range(pred.shape[1]):
        p_col = pred[:, d_idx]
        r_col = ref[:, d_idx]
        if np.std(p_col) < 1e-12 or np.std(r_col) < 1e-12:
            c = 0.0
        else:
            c_val, _ = pearsonr(p_col, r_col)
            c = float(c_val)
        corrs.append(c)
        key = f"depth_{depths[d_idx]:.3f}km" if depths is not None else f"layer_{d_idx}"
        slice_corrs[key] = c

    slice_corrs["mean_pearson_r"] = float(np.mean(corrs))
    slice_corrs["median_pearson_r"] = float(np.median(corrs))
    return slice_corrs


def summarize_ref_corr(
    rc_dict: Dict[str, Any],
    z_min: float,
    z_max: float,
) -> Dict[str, Any]:
    """Aggregate depth-slice Pearson correlations over the resolvable depth interval."""
    res_vals: List[float] = []
    res_slices: Dict[str, float] = {}
    for k, v in rc_dict.items():
        m = _DEPTH_KEY_RE.match(k)
        if m is None:
            continue
        z = float(m.group(1))
        if (z_min - 1e-6) <= z <= (z_max + 1e-6) and np.isfinite(v):
            res_vals.append(float(v))
            res_slices[k] = float(v)
    return {
        "z_range_km": [float(z_min), float(z_max)],
        "num_resolvable_slices": len(res_vals),
        "resolvable_mean_pearson_r": float(np.mean(res_vals)) if res_vals else float("nan"),
        "resolvable_median_pearson_r": float(np.median(res_vals)) if res_vals else float("nan"),
        "full_mean_pearson_r": float(rc_dict.get("mean_pearson_r", float("nan"))),
        "full_median_pearson_r": float(rc_dict.get("median_pearson_r", float("nan"))),
        "resolvable_slices": res_slices,
    }


def t_infer(total_elapsed_sec: float, num_stations: int) -> float:
    """Compute per-station inference time in milliseconds."""
    if num_stations <= 0:
        raise ValueError("num_stations must be positive")
    return float(total_elapsed_sec * 1000.0 / num_stations)
