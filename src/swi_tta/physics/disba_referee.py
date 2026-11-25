"""Exact ``disba`` surface-wave forward referee and root-search diagnostics.

Implements the physical forward operator used across all evaluation metrics and
stopping rules:
- Converts 1D shear-wave velocity ``vs`` (km/s) to compressional velocity ``vp``
  (km/s) and density ``rho`` (g/cm^3) using the Brocher (2005) empirical
  polynomials (with constant mantle Vp/Vs = 1.79 below 120 km depth, matching
  OpenSWI).
- Evaluates Rayleigh/Love phase and group dispersion via ``disba``.
- Supports explicit per-regime root-search step sizes (``dc``):
  ``2.5e-5`` km/s for shallow profiles (preventing fundamental-mode root
  skipping in profiles containing deep low-velocity zones) and ``5e-3`` km/s
  for deep lithospheric profiles.
- Provides a diagnostic wrapper that is bit-exact with the base wrapper while
  recording actual ``dc``, retry flags, returned period counts, and completeness.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
from disba import GroupDispersion, PhaseDispersion

def _get_disba_misfit_fns():
    from swi_tta.evaluation.metrics import disba_misfit, disba_misfit_per_sample
    return disba_misfit, disba_misfit_per_sample

DEFAULT_SHALLOW_DC_KM_S: float = 0.000025
DEFAULT_DEEP_DC_KM_S: float = 0.005
MAX_DISBA_WORKERS: int = 20
VS_SAFE_FLOOR_KM_S: float = 0.05


def _mp_worker_init() -> None:
    """Restrict BLAS/Numba threads inside worker processes to avoid oversubscription."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"


def vs_to_vp_rho(
    vs: np.ndarray,
    depth: Optional[np.ndarray] = None,
    mantle_depth_km: float = 120.0,
    mantle_vp_vs_ratio: float = 1.79,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute compressional velocity ``vp`` (km/s) and density ``rho`` (g/cm^3) from ``vs``.

    Follows Brocher (2005) Eq. 9 (Vp from Vs) for crustal/uppermost-mantle depths
    (``depth <= mantle_depth_km``), linear mantle ratio ``vp = 1.79 * vs`` below
    ``mantle_depth_km``, and Brocher (2005) Eq. 1 (Nafe-Drake polynomial) for density.

    Parameters
    ----------
    vs : np.ndarray
        Shear-wave velocity array in km/s with shape ``(..., L)``.
    depth : Optional[np.ndarray]
        Depth coordinates in km with shape ``(L,)`` or broadcastable to ``vs``.
    mantle_depth_km : float, default=120.0
        Depth threshold (km) where Brocher Eq. 9 switches to constant mantle Vp/Vs.
    mantle_vp_vs_ratio : float, default=1.79
        Mantle Vp/Vs ratio applied for ``depth > mantle_depth_km``.

    Returns
    -------
    vp, rho : Tuple[np.ndarray, np.ndarray]
        Compressional velocity (km/s) and bulk density (g/cm^3), same shape as ``vs``.
    """
    vs_arr = np.asarray(vs, dtype=np.float64)

    vp_brocher = (
        0.9409
        + 2.0947 * vs_arr
        - 0.8206 * (vs_arr ** 2)
        + 0.2683 * (vs_arr ** 3)
        - 0.0251 * (vs_arr ** 4)
    )

    if depth is not None:
        d_arr = np.asarray(depth, dtype=np.float64)
        vp_mantle = mantle_vp_vs_ratio * vs_arr
        vp = np.where(d_arr > mantle_depth_km, vp_mantle, vp_brocher)
    else:
        vp = vp_brocher

    vp = np.maximum(vp, np.sqrt(2.0) * vs_arr + 0.05)

    rho = (
        1.6612 * vp
        - 0.4721 * (vp ** 2)
        + 0.0671 * (vp ** 3)
        - 0.0043 * (vp ** 4)
        + 0.000106 * (vp ** 5)
    )
    rho = np.maximum(rho, 1.0)
    return vp, rho


def depth_to_thickness(depth: np.ndarray) -> np.ndarray:
    """Convert a 1D depth grid (km) to positive layer thicknesses (km) for ``disba``."""
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 1 or d.size < 2:
        raise ValueError(f"depth must be a 1D array with length >= 2; got shape {d.shape}")
    thick = np.empty_like(d)
    thick[:-1] = np.diff(d)
    thick[-1] = thick[-2]
    return np.maximum(thick, 1e-4)


def resolve_regime_dc(
    domain: str,
    *,
    dc: Optional[float] = None,
    dc_shallow: Optional[float] = None,
    dc_deep: Optional[float] = None,
    dc_cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """Resolve the converged ``disba`` root-search step ``dc`` for a depth regime.

    Parameters
    ----------
    domain : str
        Regime identifier, either ``"shallow"`` or ``"deep"``.
    dc : Optional[float]
        Explicit scalar fallback step size in km/s.
    dc_shallow : Optional[float]
        Explicit shallow step size override in km/s.
    dc_deep : Optional[float]
        Explicit deep step size override in km/s.
    dc_cfg : Optional[Dict[str, Any]]
        Configuration dictionary containing ``disba_dc_shallow`` / ``disba_dc_deep``
        (or ``shallow_converged`` / ``deep_converged``).
    """
    if domain not in ("shallow", "deep"):
        raise ValueError(f"domain must be 'shallow' or 'deep'; got {domain!r}")
    if domain == "shallow" and dc_shallow is not None:
        return float(dc_shallow)
    if domain == "deep" and dc_deep is not None:
        return float(dc_deep)
    if dc_cfg is not None:
        candidates = (
            ("disba_dc_shallow", "dc_shallow", "shallow_converged")
            if domain == "shallow"
            else ("disba_dc_deep", "dc_deep", "deep_converged")
        )
        for key in candidates:
            if key in dc_cfg and dc_cfg[key] is not None:
                return float(dc_cfg[key])
    if dc is not None:
        return float(dc)
    raise ValueError(
        f"No disba dc provided for {domain!r} regime: pass dc_{domain}, dc_cfg, or explicit dc."
    )


def forward_dispersion(
    vs: np.ndarray,
    depth: np.ndarray,
    periods: np.ndarray,
    vp: Optional[np.ndarray] = None,
    rho: Optional[np.ndarray] = None,
    mode: int = 0,
    wave: str = "rayleigh",
    dc: float = 0.005,
    compute_group: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Rayleigh or Love phase and group velocity curves for one 1D profile.

    Parameters
    ----------
    vs : np.ndarray
        1D shear-wave velocity profile (km/s), shape ``(L,)``.
    depth : np.ndarray
        1D depth grid (km), shape ``(L,)``.
    periods : np.ndarray
        Target periods (s), shape ``(P,)``. Entries ``<= 0`` are filled with ``0.0``.
    vp, rho : Optional[np.ndarray]
        Optional Vp (km/s) and density (g/cm^3); derived via :func:`vs_to_vp_rho` if omitted.
    mode : int, default=0
        Surface-wave mode index (``0`` = fundamental mode).
    wave : str, default="rayleigh"
        Wave type (``"rayleigh"`` or ``"love"``).
    dc : float, default=0.005
        Phase-velocity root-search step size (km/s). Use ``2.5e-5`` for shallow LVZ safety.
    compute_group : bool, default=True
        Whether to evaluate group velocity in addition to phase velocity.

    Returns
    -------
    phase_vel, group_vel : Tuple[np.ndarray, np.ndarray]
        Forward dispersion curves (km/s), each of shape ``(P,)``.
    """
    vs_1d = np.asarray(vs, dtype=np.float64)
    depth_1d = np.asarray(depth, dtype=np.float64)
    t_arr = np.asarray(periods, dtype=np.float64)

    if vp is None or rho is None:
        vp_calc, rho_calc = vs_to_vp_rho(vs_1d, depth_1d)
        vp_1d = vp_calc if vp is None else np.asarray(vp, dtype=np.float64)
        rho_1d = rho_calc if rho is None else np.asarray(rho, dtype=np.float64)
    else:
        vp_1d = np.asarray(vp, dtype=np.float64)
        rho_1d = np.asarray(rho, dtype=np.float64)

    thick_1d = depth_to_thickness(depth_1d)
    phase_out = np.zeros_like(t_arr, dtype=np.float64)
    group_out = np.zeros_like(t_arr, dtype=np.float64)

    valid_mask = (t_arr > 0.0) & np.isfinite(t_arr)
    if not np.any(valid_mask):
        return phase_out, group_out

    t_valid = t_arr[valid_mask]
    t_unique = np.unique(t_valid)

    for cur_dc in (dc, dc * 0.2):
        try:
            pd = PhaseDispersion(thick_1d, vp_1d, vs_1d, rho_1d, dc=cur_dc)
            c_res = pd(t_unique, mode=mode, wave=wave)
            if c_res.period.size > 0:
                phase_out[valid_mask] = np.interp(t_valid, c_res.period, c_res.velocity)
                break
        except Exception:
            continue

    if compute_group:
        for cur_dc in (dc, dc * 0.2):
            try:
                gd = GroupDispersion(thick_1d, vp_1d, vs_1d, rho_1d, dc=cur_dc)
                u_res = gd(t_unique, mode=mode, wave=wave)
                if u_res.period.size > 0:
                    group_out[valid_mask] = np.interp(t_valid, u_res.period, u_res.velocity)
                    break
            except Exception:
                continue

    return phase_out, group_out


def _worker_forward(args: Tuple[Any, ...]) -> Tuple[np.ndarray, np.ndarray]:
    vs_i, depth_i, t_i, mode, wave, dc, compute_group = args
    return forward_dispersion(
        vs=vs_i,
        depth=depth_i,
        periods=t_i,
        vp=None,
        rho=None,
        mode=mode,
        wave=wave,
        dc=dc,
        compute_group=compute_group,
    )


def batch_forward_dispersion(
    vs_batch: np.ndarray,
    depth_batch: np.ndarray,
    periods_batch: np.ndarray,
    mode: int = 0,
    wave: str = "rayleigh",
    dc: float = 0.005,
    compute_group: bool = True,
    max_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate :func:`forward_dispersion` over a batch of profiles in parallel."""
    vs_b = np.asarray(vs_batch, dtype=np.float64)
    n = vs_b.shape[0]

    d_b = np.asarray(depth_batch, dtype=np.float64)
    if d_b.ndim == 1:
        d_b = np.broadcast_to(d_b[None, :], vs_b.shape)

    t_b = np.asarray(periods_batch, dtype=np.float64)
    if t_b.ndim == 1:
        t_b = np.broadcast_to(t_b[None, :], (n, t_b.shape[0]))

    tasks = [(vs_b[i], d_b[i], t_b[i], mode, wave, dc, compute_group) for i in range(n)]

    eff_workers = min(int(max_workers), MAX_DISBA_WORKERS)
    if eff_workers <= 1 or n <= 4:
        results = [_worker_forward(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=eff_workers, initializer=_mp_worker_init) as ex:
            results = list(ex.map(_worker_forward, tasks, chunksize=max(1, n // (eff_workers * 4))))

    phase_batch = np.stack([r[0] for r in results], axis=0)
    group_batch = np.stack([r[1] for r in results], axis=0)
    return phase_batch, group_batch


def _count_matched_unique(
    req_unique: np.ndarray,
    ret_periods: np.ndarray,
    ret_velocities: np.ndarray,
) -> int:
    """Count how many requested unique periods were validly returned by ``disba``."""
    n_req = int(req_unique.size)
    if n_req == 0 or ret_periods.size == 0:
        return 0
    rp = np.asarray(ret_periods, dtype=np.float64).reshape(-1)
    rv = np.asarray(ret_velocities, dtype=np.float64).reshape(-1)
    good = np.isfinite(rp) & np.isfinite(rv) & (rv > 0.0)
    if not np.any(good):
        return 0
    rp = rp[good]
    rp_s = rp[np.argsort(rp)]

    idx = np.searchsorted(rp_s, req_unique)
    idx_hi = np.clip(idx, 0, rp_s.size - 1)
    idx_lo = np.clip(idx - 1, 0, rp_s.size - 1)
    min_d = np.minimum(np.abs(rp_s[idx_hi] - req_unique), np.abs(rp_s[idx_lo] - req_unique))
    tol = np.maximum(1e-6 * np.abs(req_unique), 1e-9)
    return int(np.sum(min_d <= tol))


def forward_dispersion_diag(
    vs: np.ndarray,
    depth: np.ndarray,
    periods: np.ndarray,
    vp: Optional[np.ndarray] = None,
    rho: Optional[np.ndarray] = None,
    mode: int = 0,
    wave: str = "rayleigh",
    dc: float = 0.005,
    compute_group: bool = True,
    phase_dispersion_cls: Any = PhaseDispersion,
    group_dispersion_cls: Any = GroupDispersion,
) -> Dict[str, Any]:
    """Diagnostic single-profile forward solver bit-exact with :func:`forward_dispersion`."""
    t0 = time.perf_counter()
    vs_1d = np.asarray(vs, dtype=np.float64)
    depth_1d = np.asarray(depth, dtype=np.float64)
    t_arr = np.asarray(periods, dtype=np.float64)

    if vp is None or rho is None:
        vp_calc, rho_calc = vs_to_vp_rho(vs_1d, depth_1d)
        vp_1d = vp_calc if vp is None else np.asarray(vp, dtype=np.float64)
        rho_1d = rho_calc if rho is None else np.asarray(rho, dtype=np.float64)
    else:
        vp_1d = np.asarray(vp, dtype=np.float64)
        rho_1d = np.asarray(rho, dtype=np.float64)

    thick_1d = depth_to_thickness(depth_1d)
    phase_out = np.zeros_like(t_arr, dtype=np.float64)
    group_out = np.zeros_like(t_arr, dtype=np.float64)

    valid_mask = (t_arr > 0.0) & np.isfinite(t_arr)
    req_dc = float(dc)
    if not np.any(valid_mask):
        return {
            "phase_vel": phase_out,
            "group_vel": group_out,
            "requested_dc": req_dc,
            "phase_actual_dc": req_dc,
            "group_actual_dc": req_dc if compute_group else float("nan"),
            "actual_dc": req_dc,
            "phase_retry_triggered": False,
            "group_retry_triggered": False,
            "retry_triggered": False,
            "n_requested_unique": 0,
            "n_requested_points": 0,
            "phase_n_returned_unique": 0,
            "group_n_returned_unique": 0,
            "phase_is_complete": False,
            "group_is_complete": False if compute_group else True,
            "is_complete": False,
            "phase_is_empty": True,
            "group_is_empty": True if compute_group else False,
            "is_empty": True,
            "phase_ret_periods": np.empty(0, dtype=np.float64),
            "phase_ret_velocities": np.empty(0, dtype=np.float64),
            "group_ret_periods": np.empty(0, dtype=np.float64),
            "group_ret_velocities": np.empty(0, dtype=np.float64),
            "elapsed_sec": float(time.perf_counter() - t0),
        }

    t_valid = t_arr[valid_mask]
    t_unique = np.unique(t_valid)
    n_req_u = int(t_unique.size)

    ph_actual_dc = float(req_dc * 0.2)
    ph_retry = False
    ph_ret_p = np.empty(0, dtype=np.float64)
    ph_ret_v = np.empty(0, dtype=np.float64)
    for attempt_idx, cur_dc in enumerate((req_dc, req_dc * 0.2)):
        if attempt_idx > 0:
            ph_retry = True
        try:
            pd = phase_dispersion_cls(thick_1d, vp_1d, vs_1d, rho_1d, dc=cur_dc)
            c_res = pd(t_unique, mode=mode, wave=wave)
            if c_res.period.size > 0:
                phase_out[valid_mask] = np.interp(t_valid, c_res.period, c_res.velocity)
                ph_actual_dc = float(getattr(pd, "dc", cur_dc))
                ph_ret_p = np.asarray(c_res.period, dtype=np.float64).reshape(-1)
                ph_ret_v = np.asarray(c_res.velocity, dtype=np.float64).reshape(-1)
                break
        except Exception:
            continue

    ph_n_ret_u = _count_matched_unique(t_unique, ph_ret_p, ph_ret_v)
    ph_is_empty = bool(ph_ret_p.size == 0 or ph_n_ret_u == 0)
    ph_is_complete = bool((not ph_is_empty) and (ph_n_ret_u == n_req_u))

    gp_actual_dc = float("nan")
    gp_retry = False
    gp_ret_p = np.empty(0, dtype=np.float64)
    gp_ret_v = np.empty(0, dtype=np.float64)
    gp_n_ret_u = 0
    gp_is_empty = False
    gp_is_complete = True

    if compute_group:
        gp_actual_dc = float(req_dc * 0.2)
        for attempt_idx, cur_dc in enumerate((req_dc, req_dc * 0.2)):
            if attempt_idx > 0:
                gp_retry = True
            try:
                gd = group_dispersion_cls(thick_1d, vp_1d, vs_1d, rho_1d, dc=cur_dc)
                u_res = gd(t_unique, mode=mode, wave=wave)
                if u_res.period.size > 0:
                    group_out[valid_mask] = np.interp(t_valid, u_res.period, u_res.velocity)
                    gp_actual_dc = float(getattr(gd, "dc", cur_dc))
                    gp_ret_p = np.asarray(u_res.period, dtype=np.float64).reshape(-1)
                    gp_ret_v = np.asarray(u_res.velocity, dtype=np.float64).reshape(-1)
                    break
            except Exception:
                continue
        gp_n_ret_u = _count_matched_unique(t_unique, gp_ret_p, gp_ret_v)
        gp_is_empty = bool(gp_ret_p.size == 0 or gp_n_ret_u == 0)
        gp_is_complete = bool((not gp_is_empty) and (gp_n_ret_u == n_req_u))

    elapsed = float(time.perf_counter() - t0)
    actual_dc = min(ph_actual_dc, gp_actual_dc) if compute_group else ph_actual_dc
    return {
        "phase_vel": phase_out,
        "group_vel": group_out,
        "requested_dc": req_dc,
        "phase_actual_dc": float(ph_actual_dc),
        "group_actual_dc": float(gp_actual_dc),
        "actual_dc": float(actual_dc),
        "phase_retry_triggered": bool(ph_retry),
        "group_retry_triggered": bool(gp_retry),
        "retry_triggered": bool(ph_retry or (compute_group and gp_retry)),
        "n_requested_unique": n_req_u,
        "n_requested_points": int(t_valid.size),
        "phase_n_returned_unique": int(ph_n_ret_u),
        "group_n_returned_unique": int(gp_n_ret_u),
        "phase_is_complete": ph_is_complete,
        "group_is_complete": gp_is_complete,
        "is_complete": bool(ph_is_complete and gp_is_complete),
        "phase_is_empty": ph_is_empty,
        "group_is_empty": gp_is_empty,
        "is_empty": bool(ph_is_empty or (compute_group and gp_is_empty)),
        "phase_ret_periods": ph_ret_p,
        "phase_ret_velocities": ph_ret_v,
        "group_ret_periods": gp_ret_p,
        "group_ret_velocities": gp_ret_v,
        "elapsed_sec": elapsed,
    }


def _forward_one_station_diag(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    idx, vs_1d, depth_1d, periods_1d, mask_2d, compute_group, dc, ph_cls, gp_cls = args
    p_len = len(periods_1d)
    ph_out = np.zeros(p_len, dtype=np.float64)
    gp_out = np.zeros(p_len, dtype=np.float64)
    dc_f = float(dc)

    m_ph = (periods_1d > 0) & (mask_2d[:, 0] > 0.5)
    m_gp = (periods_1d > 0) & (mask_2d[:, 1] > 0.5) if compute_group else np.zeros(p_len, dtype=bool)
    valid_p = (periods_1d > 0) & ((mask_2d[:, 0] > 0.5) | (mask_2d[:, 1] > 0.5))
    t_ph_u = np.unique(np.asarray(periods_1d[m_ph], dtype=np.float64)) if np.any(m_ph) else np.empty(0, dtype=np.float64)
    t_gp_u = np.unique(np.asarray(periods_1d[m_gp], dtype=np.float64)) if np.any(m_gp) else np.empty(0, dtype=np.float64)
    n_ph_req_ch = int(t_ph_u.size)
    n_gp_req_ch = int(t_gp_u.size)

    if (not np.any(valid_p)) or (np.max(vs_1d) <= VS_SAFE_FLOOR_KM_S) or np.any(~np.isfinite(vs_1d)):
        n_req_u = int(np.unique(np.asarray(periods_1d[valid_p], dtype=np.float64)).size) if np.any(valid_p) else 0
        return (
            idx, ph_out, gp_out,
            dc_f, dc_f if compute_group else float("nan"), dc_f,
            False, False, False,
            n_req_u, 0, 0,
            n_ph_req_ch, 0, n_gp_req_ch, 0,
            False, False, True, 0.0,
        )

    vs_safe = np.maximum(np.asarray(vs_1d, dtype=np.float64), VS_SAFE_FLOOR_KM_S)
    t_sub = np.asarray(periods_1d[valid_p], dtype=np.float64)
    dres = forward_dispersion_diag(
        vs=vs_safe,
        depth=depth_1d,
        periods=t_sub,
        compute_group=compute_group,
        dc=dc_f,
        phase_dispersion_cls=ph_cls,
        group_dispersion_cls=gp_cls,
    )
    ph_out[valid_p] = dres["phase_vel"]
    gp_out[valid_p] = dres["group_vel"]

    n_ph_ret_ch = _count_matched_unique(t_ph_u, dres["phase_ret_periods"], dres["phase_ret_velocities"])
    n_gp_ret_ch = (
        _count_matched_unique(t_gp_u, dres["group_ret_periods"], dres["group_ret_velocities"])
        if compute_group
        else 0
    )
    has_any_ch = (n_ph_req_ch + n_gp_req_ch) > 0
    is_complete_obs = bool(
        has_any_ch
        and (n_ph_ret_ch == n_ph_req_ch)
        and ((not compute_group) or (n_gp_ret_ch == n_gp_req_ch))
    )
    return (
        idx, ph_out, gp_out,
        dres["phase_actual_dc"], dres["group_actual_dc"], dres["actual_dc"],
        dres["phase_retry_triggered"], dres["group_retry_triggered"], dres["retry_triggered"],
        dres["n_requested_unique"], dres["phase_n_returned_unique"], dres["group_n_returned_unique"],
        n_ph_req_ch, n_ph_ret_ch, n_gp_req_ch, n_gp_ret_ch,
        dres["is_complete"], is_complete_obs, dres["is_empty"], dres["elapsed_sec"],
    )


def forward_field_batch_diag(
    vs_batch: np.ndarray,
    depth_1d: np.ndarray,
    periods: np.ndarray,
    obs_mask: np.ndarray,
    *,
    compute_group: bool = True,
    dc: float = 0.005,
    max_workers: int = 12,
    executor: Optional[ProcessPoolExecutor] = None,
    phase_dispersion_cls: Any = PhaseDispersion,
    group_dispersion_cls: Any = GroupDispersion,
) -> Dict[str, Any]:
    """Parallel diagnostic batch forward modeling returning dispersion curves and audit arrays."""
    n, p_len = periods.shape
    ph_all = np.zeros((n, p_len), dtype=np.float64)
    gp_all = np.zeros((n, p_len), dtype=np.float64)
    phase_actual_dc = np.full(n, float(dc), dtype=np.float64)
    group_actual_dc = np.full(n, float(dc) if compute_group else float("nan"), dtype=np.float64)
    actual_dc = np.full(n, float(dc), dtype=np.float64)
    phase_retry = np.zeros(n, dtype=bool)
    group_retry = np.zeros(n, dtype=bool)
    retry = np.zeros(n, dtype=bool)
    n_req_u = np.zeros(n, dtype=np.int32)
    ph_n_ret_u = np.zeros(n, dtype=np.int32)
    gp_n_ret_u = np.zeros(n, dtype=np.int32)
    ph_n_req_ch = np.zeros(n, dtype=np.int32)
    ph_n_ret_ch = np.zeros(n, dtype=np.int32)
    gp_n_req_ch = np.zeros(n, dtype=np.int32)
    gp_n_ret_ch = np.zeros(n, dtype=np.int32)
    is_complete = np.zeros(n, dtype=bool)
    is_complete_obs = np.zeros(n, dtype=bool)
    is_empty = np.zeros(n, dtype=bool)
    elapsed_sec = np.zeros(n, dtype=np.float64)

    dc_val = float(dc)
    eff_workers = min(int(max_workers), MAX_DISBA_WORKERS)
    custom_cls = (phase_dispersion_cls is not PhaseDispersion) or (group_dispersion_cls is not GroupDispersion)

    tasks = [
        (i, vs_batch[i], depth_1d, periods[i], obs_mask[i], bool(compute_group), dc_val, phase_dispersion_cls, group_dispersion_cls)
        for i in range(n)
    ]

    def _consume(row: Tuple[Any, ...]) -> None:
        (
            i, ph_i, gp_i,
            p_dc, g_dc, a_dc,
            p_rt, g_rt, a_rt,
            n_rq, p_nr, g_nr,
            p_rq_c, p_nr_c, g_rq_c, g_nr_c,
            comp, comp_obs, emp, el,
        ) = row
        ph_all[i] = ph_i
        gp_all[i] = gp_i
        phase_actual_dc[i] = p_dc
        group_actual_dc[i] = g_dc
        actual_dc[i] = a_dc
        phase_retry[i] = p_rt
        group_retry[i] = g_rt
        retry[i] = a_rt
        n_req_u[i] = n_rq
        ph_n_ret_u[i] = p_nr
        gp_n_ret_u[i] = g_nr
        ph_n_req_ch[i] = p_rq_c
        ph_n_ret_ch[i] = p_nr_c
        gp_n_req_ch[i] = g_rq_c
        gp_n_ret_ch[i] = g_nr_c
        is_complete[i] = comp
        is_complete_obs[i] = comp_obs
        is_empty[i] = emp
        elapsed_sec[i] = el

    if custom_cls or eff_workers <= 1 or n <= 16:
        _mp_worker_init()
        for t in tasks:
            _consume(_forward_one_station_diag(t))
    elif executor is not None:
        chunk = max(1, min(64, n // max(eff_workers * 4, 1)))
        for row in executor.map(_forward_one_station_diag, tasks, chunksize=chunk):
            _consume(row)
    else:
        chunk = max(1, min(64, n // max(eff_workers * 4, 1)))
        with ProcessPoolExecutor(max_workers=eff_workers, initializer=_mp_worker_init) as ex:
            for row in ex.map(_forward_one_station_diag, tasks, chunksize=chunk):
                _consume(row)

    return {
        "phase_vel": ph_all,
        "group_vel": gp_all,
        "requested_dc": dc_val,
        "phase_actual_dc": phase_actual_dc,
        "group_actual_dc": group_actual_dc,
        "actual_dc": actual_dc,
        "phase_retry_triggered": phase_retry,
        "group_retry_triggered": group_retry,
        "retry_triggered": retry,
        "n_requested_unique": n_req_u,
        "phase_n_returned_unique": ph_n_ret_u,
        "group_n_returned_unique": gp_n_ret_u,
        "phase_n_req_ch": ph_n_req_ch,
        "phase_n_ret_ch": ph_n_ret_ch,
        "group_n_req_ch": gp_n_req_ch,
        "group_n_ret_ch": gp_n_ret_ch,
        "is_complete": is_complete,
        "is_complete_obs": is_complete_obs,
        "is_empty": is_empty,
        "elapsed_sec": elapsed_sec,
    }


def summarize_forward_diagnostics(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize per-sample forward diagnostics into a JSON-serializable dictionary."""
    n = int(len(diag["actual_dc"]))
    is_comp = np.asarray(diag["is_complete"], dtype=bool)
    is_comp_obs = np.asarray(diag["is_complete_obs"], dtype=bool)
    is_emp = np.asarray(diag["is_empty"], dtype=bool)
    rt = np.asarray(diag["retry_triggered"], dtype=bool)
    adc = np.asarray(diag["actual_dc"], dtype=np.float64)
    uniq_dc, cnt_dc = np.unique(adc, return_counts=True)
    return {
        "n_samples": n,
        "requested_dc": float(diag["requested_dc"]),
        "actual_dc_histogram": {f"{float(u):g}": int(c) for u, c in zip(uniq_dc, cnt_dc)},
        "n_retry_triggered": int(np.sum(rt)),
        "retry_rate": float(np.mean(rt)) if n > 0 else 0.0,
        "n_incomplete": int(np.sum(~is_comp)),
        "incomplete_rate": float(np.mean(~is_comp)) if n > 0 else 0.0,
        "n_incomplete_obs": int(np.sum(~is_comp_obs)),
        "incomplete_obs_rate": float(np.mean(~is_comp_obs)) if n > 0 else 0.0,
        "n_empty": int(np.sum(is_emp)),
        "empty_rate": float(np.mean(is_emp)) if n > 0 else 0.0,
    }


def compute_masked_misfits_diag(
    vs_pred: np.ndarray,
    depths: np.ndarray,
    periods: np.ndarray,
    phase_obs: np.ndarray,
    group_obs: np.ndarray,
    obs_mask: np.ndarray,
    *,
    compute_group: bool,
    dc: float = 0.005,
    workers: int = 12,
    extra_masks: Optional[Dict[str, np.ndarray]] = None,
    executor: Optional[ProcessPoolExecutor] = None,
    phase_dispersion_cls: Any = PhaseDispersion,
    group_dispersion_cls: Any = GroupDispersion,
) -> Dict[str, Any]:
    """Run one diagnostic forward pass and compute pooled and per-sample masked misfits."""
    diag = forward_field_batch_diag(
        vs_batch=vs_pred,
        depth_1d=depths,
        periods=periods,
        obs_mask=obs_mask,
        compute_group=compute_group,
        dc=float(dc),
        max_workers=int(workers),
        executor=executor,
        phase_dispersion_cls=phase_dispersion_cls,
        group_dispersion_cls=group_dispersion_cls,
    )
    ph_f = diag["phase_vel"]
    gp_f = diag["group_vel"]
    if compute_group:
        d_fwd = np.stack([ph_f, gp_f], axis=-1)
        d_obs = np.stack([phase_obs, group_obs], axis=-1)
    else:
        d_fwd = np.stack([ph_f, np.zeros_like(ph_f)], axis=-1)
        d_obs = np.stack([phase_obs, np.zeros_like(phase_obs)], axis=-1)

    def _mask2(m: np.ndarray) -> np.ndarray:
        if compute_group:
            return m > 0.5
        return np.stack([m[:, :, 0] > 0.5, np.zeros_like(m[:, :, 0], dtype=bool)], axis=-1)

    disba_misfit, disba_misfit_per_sample = _get_disba_misfit_fns()
    m_obs2 = _mask2(obs_mask)
    out: Dict[str, Any] = {
        "obs_pooled": float(disba_misfit(d_fwd, d_obs, mask=m_obs2)),
        "obs_ps": disba_misfit_per_sample(d_fwd, d_obs, mask=m_obs2),
        "diag": diag,
        "diag_summary": summarize_forward_diagnostics(diag),
    }
    if extra_masks:
        for tag, m_sub in extra_masks.items():
            m2 = _mask2(m_sub)
            out[f"{tag}_pooled"] = float(disba_misfit(d_fwd, d_obs, mask=m2))
            out[f"{tag}_ps"] = disba_misfit_per_sample(d_fwd, d_obs, mask=m2)
    return out
