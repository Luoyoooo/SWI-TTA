"""Parallel field-array dispersion evaluation with explicit ``disba`` step size."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from swi_tta.evaluation.metrics import (
    disba_misfit,
    disba_misfit_ho,
    disba_misfit_per_sample,
    ref_corr,
)
from swi_tta.physics.disba_referee import MAX_DISBA_WORKERS, forward_dispersion


def _mp_worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"


def _forward_one_station(args: Tuple[Any, ...]) -> Tuple[int, np.ndarray, np.ndarray]:
    if len(args) == 7:
        idx, vs_1d, depth_1d, periods_1d, mask_2d, compute_group, dc = args
    else:
        idx, vs_1d, depth_1d, periods_1d, mask_2d, compute_group = args
        dc = 0.005
    p_len = len(periods_1d)
    ph_out = np.zeros(p_len, dtype=np.float64)
    gp_out = np.zeros(p_len, dtype=np.float64)

    valid_p = (periods_1d > 0) & ((mask_2d[:, 0] > 0.5) | (mask_2d[:, 1] > 0.5))
    if (not np.any(valid_p)) or (np.max(vs_1d) <= 0.05) or np.any(~np.isfinite(vs_1d)):
        return idx, ph_out, gp_out

    vs_safe = np.maximum(np.asarray(vs_1d, dtype=np.float64), 0.05)
    t_sub = np.asarray(periods_1d[valid_p], dtype=np.float64)
    ph_sub, gp_sub = forward_dispersion(
        vs=vs_safe,
        depth=depth_1d,
        periods=t_sub,
        compute_group=compute_group,
        dc=float(dc),
    )
    ph_out[valid_p] = ph_sub
    gp_out[valid_p] = gp_sub
    return idx, ph_out, gp_out


def forward_field_batch_mp(
    vs_batch: np.ndarray,
    depth_1d: np.ndarray,
    periods: np.ndarray,
    obs_mask: np.ndarray,
    compute_group: bool = True,
    max_workers: int = 12,
    dc: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run parallel ``disba`` forward modeling on active observed periods only."""
    n, p_len = periods.shape
    ph_all = np.zeros((n, p_len), dtype=np.float64)
    gp_all = np.zeros((n, p_len), dtype=np.float64)

    dc_val = float(dc)
    eff_workers = min(int(max_workers), MAX_DISBA_WORKERS)
    tasks = [
        (i, vs_batch[i], depth_1d, periods[i], obs_mask[i], compute_group, dc_val)
        for i in range(n)
    ]
    if eff_workers <= 1 or n <= 16:
        _mp_worker_init()
        for t in tasks:
            i, ph_i, gp_i = _forward_one_station(t)
            ph_all[i] = ph_i
            gp_all[i] = gp_i
    else:
        with ProcessPoolExecutor(max_workers=eff_workers, initializer=_mp_worker_init) as ex:
            for i, ph_i, gp_i in ex.map(_forward_one_station, tasks, chunksize=64):
                ph_all[i] = ph_i
                gp_all[i] = gp_i
    return ph_all, gp_all


def evaluate_field_predictions(
    vs_pred: np.ndarray,
    pred_depths: np.ndarray,
    periods: np.ndarray,
    phase_obs: np.ndarray,
    group_obs: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    ho_mask: np.ndarray,
    ref_vs: Optional[np.ndarray] = None,
    ref_depths: Optional[np.ndarray] = None,
    indices_tuning: Optional[Sequence[int]] = None,
    indices_report: Optional[Sequence[int]] = None,
    station_subset: Optional[Sequence[int]] = None,
    compute_group: bool = True,
    max_workers: int = 12,
    dc: float = 0.005,
) -> Dict[str, Any]:
    """Evaluate predicted field velocity profiles with exact ``disba`` forward misfits."""
    if station_subset is not None:
        sub = np.asarray(station_subset, dtype=np.int64)
        vs_eval = vs_pred if vs_pred.shape[0] == len(sub) else vs_pred[sub]
        per_e = periods[sub]
        ph_e = phase_obs[sub]
        gp_e = group_obs[sub]
        om_e = obs_mask[sub]
        km_e = kept_mask[sub]
        hm_e = ho_mask[sub]
        ref_e = ref_vs[sub] if ref_vs is not None else None
    else:
        vs_eval = vs_pred
        per_e = periods
        ph_e = phase_obs
        gp_e = group_obs
        om_e = obs_mask
        km_e = kept_mask
        hm_e = ho_mask
        ref_e = ref_vs

    ph_pred, gp_pred = forward_field_batch_mp(
        vs_batch=vs_eval,
        depth_1d=pred_depths,
        periods=per_e,
        obs_mask=om_e,
        compute_group=compute_group,
        max_workers=max_workers,
        dc=dc,
    )

    d_pred = np.stack([ph_pred, gp_pred], axis=-1)
    d_obs = np.stack([ph_e, gp_e], axis=-1)

    def _stats_on_slice(sl: np.ndarray) -> Dict[str, Any]:
        dp, do = d_pred[sl], d_obs[sl]
        km_s, hm_s, om_s = km_e[sl], hm_e[sl], om_e[sl]

        ps_kept = disba_misfit_per_sample(dp, do, km_s)
        ps_ho = disba_misfit_per_sample(dp, do, hm_s)
        valid_kept = ps_kept[np.isfinite(ps_kept)]
        valid_ho = ps_ho[np.isfinite(ps_ho)]
        res: Dict[str, Any] = {
            "num_stations": int(len(sl)),
            "num_valid_kept_stations": int(len(valid_kept)),
            "num_valid_ho_stations": int(len(valid_ho)),
            "disba_misfit": float(disba_misfit(dp, do, km_s)),
            "disba_misfit_ho": float(disba_misfit_ho(dp, do, om_s, hm_s)),
            "disba_misfit_all": float(disba_misfit(dp, do, om_s)),
            "station_misfit_mean": float(np.mean(valid_kept)) if len(valid_kept) else float("nan"),
            "station_misfit_std": float(np.std(valid_kept)) if len(valid_kept) else float("nan"),
            "station_misfit_median": float(np.median(valid_kept)) if len(valid_kept) else float("nan"),
            "station_misfit_ho_mean": float(np.mean(valid_ho)) if len(valid_ho) else float("nan"),
            "station_misfit_ho_std": float(np.std(valid_ho)) if len(valid_ho) else float("nan"),
            "station_misfit_ho_median": float(np.median(valid_ho)) if len(valid_ho) else float("nan"),
        }

        if ref_e is not None and ref_depths is not None:
            d_max_overlap = min(float(pred_depths[-1]), float(ref_depths[-1]))
            keep_ref_mask = ref_depths <= d_max_overlap + 1e-6
            overlap_depths = ref_depths[keep_ref_mask]
            vs_s = vs_eval[sl]
            sta_valid = np.any(om_s > 0.5, axis=(1, 2)) & (np.max(vs_s, axis=1) > 0.05)
            if np.any(sta_valid):
                vs_interp = np.empty((int(np.sum(sta_valid)), len(overlap_depths)), dtype=np.float64)
                for ii, row in enumerate(vs_s[sta_valid]):
                    vs_interp[ii] = np.interp(overlap_depths, pred_depths, row)
                ref_sub = ref_e[sl][sta_valid][:, keep_ref_mask]
                res["ref_corr"] = ref_corr(vs_interp, ref_sub, depths=overlap_depths)
        return res

    out: Dict[str, Any] = {}
    all_idx = np.arange(len(vs_eval), dtype=np.int64)
    out["all"] = _stats_on_slice(all_idx)
    if station_subset is None:
        if indices_tuning is not None:
            out["tuning"] = _stats_on_slice(np.asarray(indices_tuning, dtype=np.int64))
        if indices_report is not None:
            out["report"] = _stats_on_slice(np.asarray(indices_report, dtype=np.int64))
    return out
