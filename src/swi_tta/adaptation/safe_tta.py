"""Pre-Registered Safe-TTA, Holdout ES, Fixed-Step ES, and Ideal ES controllers.

All four safety mechanisms of Pre-Registered Safe-TTA are independently toggleable:
1. Discrepancy-principle stopping (exact ``disba`` kept-period misfit <= alpha * sigma_hat_D).
2. Resolvable-depth freezing (zero gradient and zero update for layers deeper than z_res).
3. Trust-region projection onto ``[v_init * (1 - Delta), v_init * (1 + Delta)] ∩ [v_lo, v_hi]``.
4. Exact ``disba`` final verification with automatic rollback to the initial profile on
   misfit increase, empty return, incomplete return, or non-finite residual.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from disba import GroupDispersion, PhaseDispersion

from swi_tta.adaptation.gated_cycle import (
    build_valid_channel_mask,
    compute_r_depth_per_station,
)
from swi_tta.evaluation.metrics import (
    compute_per_sample_rmse_res,
    compute_resolvable_depth,
    disba_misfit_per_sample,
)
from swi_tta.models.surrogate import SurrogateEnsemble
from swi_tta.physics.disba_referee import (
    MAX_DISBA_WORKERS,
    _mp_worker_init,
    forward_field_batch_diag,
    resolve_regime_dc,
)

DEFAULT_VELOCITY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "shallow": (0.08, 4.45),
    "deep": (0.15, 4.98),
}

DEFAULT_ROUTING_BOUNDS: Dict[str, Tuple[float, Optional[float]]] = {
    "shallow": (2.0, None),
    "deep": (1.25, 3.0),
}

_UNSET = object()


def compute_max_frozen_delta(
    vs_out: np.ndarray,
    vs_init: np.ndarray,
    depths: np.ndarray,
    z_res_kept: np.ndarray,
) -> np.ndarray:
    """Compute per-station maximum absolute change strictly deeper than ``z_res_kept``."""
    v_out = np.asarray(vs_out, dtype=np.float64)
    v_in = np.asarray(vs_init, dtype=np.float64)
    d_arr = np.asarray(depths, dtype=np.float64).reshape(-1)
    zr = np.asarray(z_res_kept, dtype=np.float64).reshape(-1)
    n_sta = v_out.shape[0]
    frozen_mask = d_arr[None, :] > (zr[:, None] + 1e-9)
    abs_diff = np.abs(v_out - v_in)
    out = np.zeros(n_sta, dtype=np.float64)
    for i in range(n_sta):
        if np.any(frozen_mask[i]):
            out[i] = float(np.max(abs_diff[i, frozen_mask[i]]))
    return out


def _normalize_kept_mask_2ch(
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    *,
    remove_group: bool,
) -> np.ndarray:
    om = np.asarray(obs_mask) > 0.5
    km = np.asarray(kept_mask)
    if km.ndim == 2:
        km_2ch = np.stack([km > 0.5, km > 0.5], axis=-1)
    elif km.ndim == 3:
        km_2ch = km > 0.5
    else:
        raise ValueError(f"kept_mask must have rank 2 or 3; got {km.ndim}")
    out = om & km_2ch
    if remove_group:
        out[:, :, 1] = False
    return out


def _scrub_to_kept_only(
    periods: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    *,
    remove_group: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Structurally erase all held-out period values before any downstream computation."""
    km_2ch = _normalize_kept_mask_2ch(obs_mask, kept_mask, remove_group=remove_group)
    any_kept = km_2ch[:, :, 0] | km_2ch[:, :, 1]
    per_clean = np.where(any_kept, np.asarray(periods, dtype=np.float32), -1.0).astype(np.float32)
    ph_clean = np.where(km_2ch[:, :, 0], np.asarray(phase_vel, dtype=np.float32), -1.0).astype(np.float32)
    gp_clean = np.where(km_2ch[:, :, 1], np.asarray(group_vel, dtype=np.float32), -1.0).astype(np.float32)
    return per_clean, ph_clean, gp_clean, km_2ch


def compute_resolvable_depth_from_kept(
    periods: np.ndarray,
    phase_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    """Compute per-station resolvable depth ``z_res`` strictly from kept phase-velocity points."""
    km_2ch = _normalize_kept_mask_2ch(obs_mask, kept_mask, remove_group=True)
    pm_kept = km_2ch[:, :, 0]
    per_kept = np.where(pm_kept, np.asarray(periods, dtype=np.float64), -1.0)
    ph_kept = np.where(pm_kept, np.asarray(phase_vel, dtype=np.float64), -1.0)
    return compute_resolvable_depth(per_kept, ph_kept, pm_kept, depths)


def create_stop_period_split(
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    *,
    compute_group: bool,
    stop_ratio: float = 0.20,
    seed: int = 20261121,
    keep_endpoints: bool = True,
    min_valid_points: int = 5,
    station_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically split each station's kept periods into optimization and internal stop sets."""
    km_2ch = _normalize_kept_mask_2ch(obs_mask, kept_mask, remove_group=(not compute_group))
    n_sta, _, _ = km_2ch.shape
    train_mask = km_2ch.copy()
    stop_mask = np.zeros_like(km_2ch, dtype=bool)

    channels = (0, 1) if compute_group else (0,)
    base_seed = int(seed)
    sta_ids = (
        np.arange(n_sta, dtype=np.int64)
        if station_indices is None
        else np.asarray(station_indices, dtype=np.int64).reshape(-1)
    )
    if sta_ids.size != n_sta:
        raise ValueError(f"station_indices length {sta_ids.size} != n_sta {n_sta}")

    for i in range(n_sta):
        sid_i = int(sta_ids[i])
        for ch in channels:
            valid_idx = np.where(km_2ch[i, :, ch])[0]
            nv = int(valid_idx.size)
            if nv < int(min_valid_points):
                continue
            candidates = valid_idx[1:-1] if (keep_endpoints and nv > 2) else valid_idx
            n_stop = min(int(round(float(stop_ratio) * nv)), int(candidates.size))
            if n_stop <= 0:
                continue
            sta_seed = (base_seed + sid_i * 1000003 + ch * 97) & 0x7FFFFFFF
            rng = np.random.RandomState(sta_seed)
            picked = rng.choice(candidates, size=n_stop, replace=False)
            stop_mask[i, picked, ch] = True
            train_mask[i, picked, ch] = False

    return train_mask, stop_mask


def _eval_disba_subset_diag(
    vs_batch: np.ndarray,
    depths: np.ndarray,
    per_clean: np.ndarray,
    ph_clean: np.ndarray,
    gp_clean: np.ndarray,
    eval_mask_2ch: np.ndarray,
    *,
    compute_group: bool,
    dc: float,
    workers: int,
    executor: Optional[ProcessPoolExecutor] = None,
    phase_dispersion_cls: Any = PhaseDispersion,
    group_dispersion_cls: Any = GroupDispersion,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    diag = forward_field_batch_diag(
        vs_batch=vs_batch,
        depth_1d=depths,
        periods=per_clean.astype(np.float64),
        obs_mask=eval_mask_2ch.astype(np.uint8),
        compute_group=compute_group,
        dc=float(dc),
        max_workers=min(int(workers), MAX_DISBA_WORKERS),
        executor=executor,
        phase_dispersion_cls=phase_dispersion_cls,
        group_dispersion_cls=group_dispersion_cls,
    )
    ph_f = diag["phase_vel"]
    gp_f = diag["group_vel"]
    if compute_group:
        d_fwd = np.stack([ph_f, gp_f], axis=-1)
        d_obs = np.stack([ph_clean.astype(np.float64), gp_clean.astype(np.float64)], axis=-1)
        m2 = eval_mask_2ch
    else:
        d_fwd = np.stack([ph_f, np.zeros_like(ph_f)], axis=-1)
        d_obs = np.stack([ph_clean.astype(np.float64), np.zeros_like(ph_clean, dtype=np.float64)], axis=-1)
        m2 = np.stack([eval_mask_2ch[:, :, 0], np.zeros_like(eval_mask_2ch[:, :, 0], dtype=bool)], axis=-1)

    misfit_ps = disba_misfit_per_sample(d_fwd, d_obs, mask=m2)
    elapsed = float(time.perf_counter() - t0)
    return {
        "misfit_ps": np.asarray(misfit_ps, dtype=np.float64),
        "is_complete": np.asarray(diag["is_complete"], dtype=bool),
        "is_complete_obs": np.asarray(diag["is_complete_obs"], dtype=bool),
        "is_empty": np.asarray(diag["is_empty"], dtype=bool),
        "actual_dc": np.asarray(diag["actual_dc"], dtype=np.float64),
        "retry_triggered": np.asarray(diag["retry_triggered"], dtype=bool),
        "elapsed_sec": elapsed,
        "n_calls": int(len(vs_batch)),
    }


def _set_deterministic_cuda(device: str) -> Optional[Tuple[bool, bool, bool, bool]]:
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        return None
    prev = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    return prev


def _restore_deterministic_cuda(prev_flags: Optional[Tuple[bool, bool, bool, bool]]) -> None:
    if prev_flags is not None:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        ) = prev_flags


def run_safe_tta(
    vs_init_aligned: np.ndarray,
    ens: SurrogateEnsemble,
    depths: np.ndarray,
    periods: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    domain: str,
    *,
    alpha: float,
    sigma_hat_d: float,
    delta_trust: float,
    dc: Optional[float] = None,
    dc_shallow: Optional[float] = None,
    dc_deep: Optional[float] = None,
    dc_cfg: Optional[Dict[str, Any]] = None,
    lr: float = 0.02,
    num_steps: int = 50,
    check_interval: int = 5,
    lambda_prior: float = 0.05,
    v_bounds: Optional[Tuple[float, float]] = None,
    batch_size: int = 512,
    seed: int = 0,
    device: str = "cpu",
    remove_group: Optional[bool] = None,
    compute_group: Optional[bool] = None,
    disba_workers: int = 4,
    enable_discrepancy_stop: bool = True,
    enable_zres_freeze: bool = True,
    enable_trust_region: bool = True,
    enable_exact_verification: bool = True,
    record_all_step_profiles: bool = False,
    phase_dispersion_cls: Any = PhaseDispersion,
    group_dispersion_cls: Any = GroupDispersion,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run Pre-Registered Safe-TTA starting from precomputed input-aligned profiles."""
    t_total0 = time.perf_counter()
    torch.manual_seed(int(seed))

    if domain not in ("shallow", "deep"):
        raise ValueError(f"domain must be 'shallow' or 'deep'; got {domain!r}")
    rm_grp = (domain == "shallow") if remove_group is None else bool(remove_group)
    comp_grp = (domain == "deep") if compute_group is None else bool(compute_group)
    v_lo, v_hi = DEFAULT_VELOCITY_BOUNDS[domain] if v_bounds is None else (float(v_bounds[0]), float(v_bounds[1]))

    vs_init = np.asarray(vs_init_aligned, dtype=np.float32)
    if vs_init.ndim != 2:
        raise ValueError(f"vs_init_aligned must be 2D (N, D); got {vs_init.shape}")
    n_sta, n_layers = vs_init.shape
    d_arr = np.asarray(depths, dtype=np.float64).reshape(-1)
    if d_arr.size != n_layers:
        raise ValueError(f"depths length {d_arr.size} != profile layers {n_layers}")

    per_clean, ph_clean, gp_clean, km_2ch = _scrub_to_kept_only(
        periods, phase_vel, group_vel, obs_mask, kept_mask, remove_group=rm_grp
    )
    z_res_kept = compute_resolvable_depth_from_kept(per_clean, ph_clean, km_2ch, km_2ch, d_arr)
    editable_mask_np = (d_arr[None, :] <= (z_res_kept[:, None] + 1e-9)) if enable_zres_freeze else np.ones((n_sta, n_layers), dtype=bool)
    any_kept_station = np.any(km_2ch, axis=(1, 2))

    ens.eval().to(device)
    for p in ens.parameters():
        p.requires_grad_(False)
    g_min = float(ens.members[0].grid_dense[0].item())
    per_q = np.where(per_clean > 0.0, per_clean, g_min).astype(np.float32)
    d_obs_2ch = np.stack([ph_clean, gp_clean], axis=1).astype(np.float32)

    stop_thresh = float(alpha) * float(sigma_hat_d)
    dc_val = resolve_regime_dc(domain, dc=dc, dc_shallow=dc_shallow, dc_deep=dc_deep, dc_cfg=dc_cfg)
    eff_workers = min(int(disba_workers), MAX_DISBA_WORKERS)
    custom_cls = (phase_dispersion_cls is not PhaseDispersion) or (group_dispersion_cls is not GroupDispersion)

    check_steps = [st for st in range(0, int(num_steps) + 1, int(check_interval))]
    if int(num_steps) not in check_steps:
        check_steps.append(int(num_steps))

    stop_step = np.full(n_sta, int(num_steps), dtype=np.int32)
    stopped_by_disc = np.zeros(n_sta, dtype=bool)
    init_incomplete_skipped = np.zeros(n_sta, dtype=bool)
    reverted = np.zeros(n_sta, dtype=bool)
    revert_reason: List[str] = ["none"] * n_sta
    incomplete_check_counts = np.zeros(n_sta, dtype=np.int32)
    m_check_history: Dict[int, np.ndarray] = {st: np.full(n_sta, np.nan, dtype=np.float64) for st in check_steps}
    step_profiles_all: Optional[Dict[int, np.ndarray]] = (
        {st: np.empty_like(vs_init) for st in check_steps} if record_all_step_profiles else None
    )
    if step_profiles_all is not None:
        step_profiles_all[0][:] = vs_init

    vs_cand = vs_init.copy()
    m_final_cand = np.full(n_sta, np.nan, dtype=np.float64)
    complete_final_cand = np.zeros(n_sta, dtype=bool)
    empty_final_cand = np.zeros(n_sta, dtype=bool)
    has_final_eval = np.zeros(n_sta, dtype=bool)

    t_gpu_refine_sec = 0.0
    t_disba_check_sec = 0.0
    n_disba_calls = 0

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    prev_flags = _set_deterministic_cuda(device)

    executor: Optional[ProcessPoolExecutor] = None
    if (not custom_cls) and eff_workers > 1 and n_sta > 16:
        executor = ProcessPoolExecutor(max_workers=eff_workers, initializer=_mp_worker_init)

    try:
        ev0 = _eval_disba_subset_diag(
            vs_init,
            d_arr,
            per_clean,
            ph_clean,
            gp_clean,
            km_2ch,
            compute_group=comp_grp,
            dc=dc_val,
            workers=eff_workers,
            executor=executor,
            phase_dispersion_cls=phase_dispersion_cls,
            group_dispersion_cls=group_dispersion_cls,
        )
        t_disba_check_sec += ev0["elapsed_sec"]
        n_disba_calls += ev0["n_calls"]

        m_init = ev0["misfit_ps"]
        init_comp = ev0["is_complete_obs"] & np.isfinite(m_init)
        m_check_history[0][:] = m_init

        init_incomplete_skipped = (~init_comp) | (~any_kept_station)
        incomplete_check_counts[~init_comp] += 1

        active_global = ~init_incomplete_skipped
        stop_step[init_incomplete_skipped] = 0
        for i in np.where(init_incomplete_skipped)[0].tolist():
            revert_reason[i] = "initial_incomplete_skipped"
            m_final_cand[i] = m_init[i]
            complete_final_cand[i] = bool(init_comp[i])
            empty_final_cand[i] = bool(ev0["is_empty"][i])
            has_final_eval[i] = True

        if enable_discrepancy_stop:
            stop0_mask = active_global & (m_init <= stop_thresh)
            if np.any(stop0_mask):
                active_global[stop0_mask] = False
                stopped_by_disc[stop0_mask] = True
                stop_step[stop0_mask] = 0
                m_final_cand[stop0_mask] = m_init[stop0_mask]
                complete_final_cand[stop0_mask] = True
                empty_final_cand[stop0_mask] = False
                has_final_eval[stop0_mask] = True

        bs = max(1, int(batch_size))
        for s in range(0, n_sta, bs):
            e = min(s + bs, n_sta)
            active_b = active_global[s:e].copy()
            if not np.any(active_b):
                if step_profiles_all is not None:
                    for st in check_steps:
                        if st > 0:
                            step_profiles_all[st][s:e] = vs_init[s:e]
                continue

            t_g0 = time.perf_counter()
            vs0_t = torch.from_numpy(vs_init[s:e]).to(device)
            delta = torch.zeros_like(vs0_t, requires_grad=True)
            frozen_delta = torch.zeros_like(vs0_t)
            pq_t = torch.from_numpy(per_q[s:e]).to(device)
            per_t = torch.from_numpy(per_clean[s:e]).to(device)
            dobs_t = torch.from_numpy(d_obs_2ch[s:e]).to(device)
            km_t = torch.from_numpy(km_2ch[s:e]).to(device)
            edit_t = torch.from_numpy(editable_mask_np[s:e]).to(device)

            if enable_trust_region:
                tr_lo_raw = torch.clamp(vs0_t * (1.0 - float(delta_trust)), min=v_lo, max=v_hi)
                tr_hi_raw = torch.clamp(vs0_t * (1.0 + float(delta_trust)), min=v_lo, max=v_hi)
                tr_lo = torch.minimum(tr_lo_raw, tr_hi_raw)
                tr_hi = torch.maximum(tr_lo_raw, tr_hi_raw)
            else:
                tr_lo = torch.full_like(vs0_t, v_lo)
                tr_hi = torch.full_like(vs0_t, v_hi)

            vm = build_valid_channel_mask(per_t, dobs_t, km_t)
            d_safe = torch.where(vm, dobs_t, torch.ones_like(dobs_t)).clamp(min=0.05)
            vm_f = vm.to(dtype=torch.float32)
            n_val_i = vm_f.sum(dim=(1, 2))
            has_val_i = (n_val_i > 0.0).to(dtype=torch.float32)
            n_val_denom = n_val_i.clamp(min=1.0)

            opt = torch.optim.Adam([delta], lr=float(lr))
            if use_cuda:
                torch.cuda.synchronize(device)
            t_gpu_refine_sec += float(time.perf_counter() - t_g0)

            for st_idx in range(1, int(num_steps) + 1):
                if not np.any(active_b):
                    if step_profiles_all is not None and st_idx in step_profiles_all:
                        step_profiles_all[st_idx][s:e] = vs_cand[s:e]
                    continue

                t_step0 = time.perf_counter()
                active_t = torch.from_numpy(active_b).to(device)
                active_f = active_t.to(dtype=torch.float32)

                opt.zero_grad(set_to_none=True)
                vs_raw = vs0_t + delta
                vs_proj = torch.clamp(vs_raw, min=tr_lo, max=tr_hi) if enable_trust_region else torch.clamp(vs_raw, min=v_lo, max=v_hi)
                vs_cur = torch.where(edit_t, vs_proj, vs0_t) if enable_zres_freeze else vs_proj

                mn, _, _ = ens.predict_with_uncertainty(vs_cur, pq_t)
                pred_m = torch.where(vm, mn, d_safe)
                rel_sq = ((pred_m - d_safe) / d_safe) ** 2
                loss_c_i = (vm_f * rel_sq).sum(dim=(1, 2)) / n_val_denom * has_val_i
                prior_i = float(lambda_prior) * compute_r_depth_per_station(vs_cur, domain) if lambda_prior > 0 else 0.0
                loss = ((loss_c_i + prior_i) * active_f).sum()
                loss.backward()

                if delta.grad is not None:
                    if enable_zres_freeze:
                        delta.grad.masked_fill_(~edit_t, 0.0)
                    delta.grad.masked_fill_(~active_t.unsqueeze(1), 0.0)

                opt.step()

                with torch.no_grad():
                    if enable_trust_region:
                        vs_post = torch.clamp(vs0_t + delta.data, min=tr_lo, max=tr_hi)
                        delta.data.copy_(vs_post - vs0_t)
                    if enable_zres_freeze:
                        delta.data.masked_fill_(~edit_t, 0.0)
                    if torch.any(~active_t):
                        delta.data[~active_t] = frozen_delta[~active_t]

                    vs_after = (
                        torch.clamp(vs0_t + delta.data, min=tr_lo, max=tr_hi)
                        if enable_trust_region
                        else torch.clamp(vs0_t + delta.data, min=v_lo, max=v_hi)
                    )
                    if enable_zres_freeze:
                        vs_after = torch.where(edit_t, vs_after, vs0_t)
                    vs_after_np = vs_after.cpu().numpy()

                if use_cuda:
                    torch.cuda.synchronize(device)
                t_gpu_refine_sec += float(time.perf_counter() - t_step0)

                vs_cand[s:e][active_b] = vs_after_np[active_b]
                if step_profiles_all is not None and st_idx in step_profiles_all:
                    step_profiles_all[st_idx][s:e] = vs_cand[s:e]

                if enable_discrepancy_stop and (st_idx % int(check_interval) == 0 or st_idx == int(num_steps)):
                    act_loc = np.where(active_b)[0]
                    if act_loc.size > 0:
                        act_glob = s + act_loc
                        ev_st = _eval_disba_subset_diag(
                            vs_after_np[act_loc],
                            d_arr,
                            per_clean[act_glob],
                            ph_clean[act_glob],
                            gp_clean[act_glob],
                            km_2ch[act_glob],
                            compute_group=comp_grp,
                            dc=dc_val,
                            workers=eff_workers,
                            executor=executor,
                            phase_dispersion_cls=phase_dispersion_cls,
                            group_dispersion_cls=group_dispersion_cls,
                        )
                        t_disba_check_sec += ev_st["elapsed_sec"]
                        n_disba_calls += ev_st["n_calls"]

                        m_st = ev_st["misfit_ps"]
                        comp_st = ev_st["is_complete_obs"]
                        emp_st = ev_st["is_empty"]
                        m_check_history[st_idx][act_glob] = m_st

                        for k_i, (li, gi) in enumerate(zip(act_loc.tolist(), act_glob.tolist())):
                            is_c = bool(comp_st[k_i])
                            m_val = float(m_st[k_i])
                            if not is_c:
                                incomplete_check_counts[gi] += 1
                                if st_idx == int(num_steps):
                                    m_final_cand[gi] = m_val
                                    complete_final_cand[gi] = False
                                    empty_final_cand[gi] = bool(emp_st[k_i])
                                    has_final_eval[gi] = True
                            else:
                                if np.isfinite(m_val) and (m_val <= stop_thresh):
                                    active_b[li] = False
                                    stopped_by_disc[gi] = True
                                    stop_step[gi] = int(st_idx)
                                    vs_cand[gi] = vs_after_np[li]
                                    frozen_delta[li] = delta.data[li].clone()
                                    m_final_cand[gi] = m_val
                                    complete_final_cand[gi] = True
                                    empty_final_cand[gi] = False
                                    has_final_eval[gi] = True
                                elif st_idx == int(num_steps):
                                    m_final_cand[gi] = m_val
                                    complete_final_cand[gi] = True
                                    empty_final_cand[gi] = False
                                    has_final_eval[gi] = True

        need_final = (~has_final_eval) & (~init_incomplete_skipped)
        if np.any(need_final) and enable_exact_verification:
            nf_idx = np.where(need_final)[0]
            ev_fin = _eval_disba_subset_diag(
                vs_cand[nf_idx],
                d_arr,
                per_clean[nf_idx],
                ph_clean[nf_idx],
                gp_clean[nf_idx],
                km_2ch[nf_idx],
                compute_group=comp_grp,
                dc=dc_val,
                workers=eff_workers,
                executor=executor,
                phase_dispersion_cls=phase_dispersion_cls,
                group_dispersion_cls=group_dispersion_cls,
            )
            t_disba_check_sec += ev_fin["elapsed_sec"]
            n_disba_calls += ev_fin["n_calls"]
            m_final_cand[nf_idx] = ev_fin["misfit_ps"]
            complete_final_cand[nf_idx] = ev_fin["is_complete_obs"]
            empty_final_cand[nf_idx] = ev_fin["is_empty"]
            has_final_eval[nf_idx] = True
            incomplete_check_counts[nf_idx[~ev_fin["is_complete_obs"]]] += 1
            if int(num_steps) in m_check_history:
                m_check_history[int(num_steps)][nf_idx] = ev_fin["misfit_ps"]

        vs_out = vs_cand.copy()
        if enable_exact_verification:
            for i in range(n_sta):
                if init_incomplete_skipped[i]:
                    vs_out[i] = vs_init[i]
                    continue
                if stopped_by_disc[i] and stop_step[i] == 0:
                    vs_out[i] = vs_init[i]
                    continue
                if empty_final_cand[i]:
                    vs_out[i] = vs_init[i]
                    reverted[i] = True
                    revert_reason[i] = "disba_empty_return"
                elif not complete_final_cand[i]:
                    vs_out[i] = vs_init[i]
                    reverted[i] = True
                    revert_reason[i] = "disba_incomplete_return"
                elif not np.isfinite(m_final_cand[i]):
                    vs_out[i] = vs_init[i]
                    reverted[i] = True
                    revert_reason[i] = "nonfinite_misfit"
                elif m_final_cand[i] > m_init[i]:
                    vs_out[i] = vs_init[i]
                    reverted[i] = True
                    revert_reason[i] = "misfit_increased"

        if use_cuda:
            torch.cuda.synchronize(device)
        peak_b = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        _restore_deterministic_cuda(prev_flags)

    t_total_sec = float(time.perf_counter() - t_total0)
    reason_counts: Dict[str, int] = {}
    for r in revert_reason:
        reason_counts[r] = reason_counts.get(r, 0) + 1

    info: Dict[str, Any] = {
        "method": "pre_registered_safe_tta",
        "domain": domain,
        "dc": dc_val,
        "alpha": float(alpha),
        "sigma_hat_d": float(sigma_hat_d),
        "stop_threshold": float(stop_thresh),
        "delta_trust": float(delta_trust),
        "lr": float(lr),
        "num_steps": int(num_steps),
        "check_interval": int(check_interval),
        "lambda_prior": float(lambda_prior),
        "v_bounds": [float(v_lo), float(v_hi)],
        "switches": {
            "enable_discrepancy_stop": bool(enable_discrepancy_stop),
            "enable_zres_freeze": bool(enable_zres_freeze),
            "enable_trust_region": bool(enable_trust_region),
            "enable_exact_verification": bool(enable_exact_verification),
        },
        "n_stations": int(n_sta),
        "n_initial_incomplete_skipped": int(np.sum(init_incomplete_skipped)),
        "n_stopped_by_discrepancy": int(np.sum(stopped_by_disc)),
        "stop_rate": float(np.mean(stopped_by_disc)) if n_sta else 0.0,
        "n_stopped_at_step0": int(np.sum(stopped_by_disc & (stop_step == 0))),
        "step0_stop_rate": float(np.mean(stopped_by_disc & (stop_step == 0))) if n_sta else 0.0,
        "n_reverted": int(np.sum(reverted)),
        "revert_rate": float(np.mean(reverted)) if n_sta else 0.0,
        "revert_reason_counts": reason_counts,
        "n_disba_calls": int(n_disba_calls),
        "t_gpu_refine_sec": float(t_gpu_refine_sec),
        "t_disba_check_sec": float(t_disba_check_sec),
        "t_total_sec": t_total_sec,
        "peak_vram_gb": float(peak_b / (1024 ** 3)),
        "stop_step": stop_step,
        "stopped_by_discrepancy": stopped_by_disc,
        "initial_incomplete_skipped": init_incomplete_skipped,
        "reverted": reverted,
        "revert_reason": revert_reason,
        "incomplete_check_counts": incomplete_check_counts,
        "z_res_kept": z_res_kept,
        "m_init": m_init,
        "m_final_cand": m_final_cand,
        "m_check_history": m_check_history,
        **({"step_profiles": step_profiles_all} if step_profiles_all is not None else {}),
    }
    return vs_out, info


def run_output_refinement_from_profile(
    vs_init_aligned: np.ndarray,
    ens: SurrogateEnsemble,
    periods: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    domain: str,
    *,
    depths: Optional[np.ndarray] = None,
    z_res_kept: Optional[np.ndarray] = None,
    enable_zres_freeze: bool = False,
    lr: float = 0.02,
    num_steps: int = 50,
    lambda_prior: float = 0.05,
    v_bounds: Optional[Tuple[float, float]] = None,
    batch_size: int = 512,
    snapshot_steps: Optional[Sequence[int]] = None,
    seed: int = 0,
    device: str = "cpu",
    remove_group: Optional[bool] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Station-independent output refinement from precomputed initial profiles.

    When ``enable_zres_freeze=True``, gradients and updates are strictly zeroed for
    all layers deeper than ``z_res_kept`` so frozen layers remain bit-identical to
    ``vs_init_aligned``.
    """
    torch.manual_seed(int(seed))
    if domain not in ("shallow", "deep"):
        raise ValueError(f"domain must be 'shallow' or 'deep'; got {domain!r}")
    rm_grp = (domain == "shallow") if remove_group is None else bool(remove_group)
    v_lo, v_hi = DEFAULT_VELOCITY_BOUNDS[domain] if v_bounds is None else (float(v_bounds[0]), float(v_bounds[1]))

    vs_init = np.asarray(vs_init_aligned, dtype=np.float32)
    n_sta, n_layers = vs_init.shape

    per_clean, ph_clean, gp_clean, km_2ch = _scrub_to_kept_only(
        periods, phase_vel, group_vel, obs_mask, kept_mask, remove_group=rm_grp
    )
    all_masked_np = ~np.any(km_2ch, axis=(1, 2))

    editable_mask_np: Optional[np.ndarray] = None
    z_res_used: Optional[np.ndarray] = None
    d_arr: Optional[np.ndarray] = None
    if enable_zres_freeze:
        if depths is None:
            raise ValueError("depths must be provided when enable_zres_freeze=True")
        d_arr = np.asarray(depths, dtype=np.float64).reshape(-1)
        if d_arr.size != n_layers:
            raise ValueError(f"depths length {d_arr.size} != profile layers {n_layers}")
        z_res_used = (
            compute_resolvable_depth_from_kept(per_clean, ph_clean, km_2ch, km_2ch, d_arr)
            if z_res_kept is None
            else np.asarray(z_res_kept, dtype=np.float64).reshape(-1)
        )
        if z_res_used.size != n_sta:
            raise ValueError(f"z_res_kept length {z_res_used.size} != n_sta {n_sta}")
        editable_mask_np = d_arr[None, :] <= (z_res_used[:, None] + 1e-9)

    ens.eval().to(device)
    for p in ens.parameters():
        p.requires_grad_(False)

    g_min = float(ens.members[0].grid_dense[0].item())
    per_q = np.where(per_clean > 0.0, per_clean, g_min).astype(np.float32)
    d_obs_2ch = np.stack([ph_clean, gp_clean], axis=1).astype(np.float32)

    snap_set = sorted({int(st) for st in snapshot_steps}) if snapshot_steps is not None else []
    snapshots: Dict[int, np.ndarray] = {st: np.empty_like(vs_init) for st in snap_set}
    vs_refined = np.empty_like(vs_init)

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    prev_flags = _set_deterministic_cuda(device)

    try:
        t0 = time.perf_counter()
        bs = max(1, int(batch_size))
        for s in range(0, n_sta, bs):
            e = min(s + bs, n_sta)
            vs0_t = torch.from_numpy(vs_init[s:e]).to(device)
            delta = torch.zeros_like(vs0_t, requires_grad=True)
            pq_t = torch.from_numpy(per_q[s:e]).to(device)
            per_t = torch.from_numpy(per_clean[s:e]).to(device)
            dobs_t = torch.from_numpy(d_obs_2ch[s:e]).to(device)
            km_t = torch.from_numpy(km_2ch[s:e]).to(device)
            all_m = torch.from_numpy(all_masked_np[s:e]).to(device)
            edit_t = (
                torch.from_numpy(editable_mask_np[s:e]).to(device)
                if editable_mask_np is not None
                else None
            )

            if 0 in snapshots:
                with torch.no_grad():
                    if edit_t is not None:
                        v0_sn = vs0_t.clone()
                    else:
                        v0_sn = torch.clamp(vs0_t, min=v_lo, max=v_hi)
                        if torch.any(all_m):
                            v0_sn[all_m] = 0.0
                    snapshots[0][s:e] = v0_sn.cpu().numpy()

            vm = build_valid_channel_mask(per_t, dobs_t, km_t)
            d_safe = torch.where(vm, dobs_t, torch.ones_like(dobs_t)).clamp(min=0.05)
            vm_f = vm.to(dtype=torch.float32)
            n_val_i = vm_f.sum(dim=(1, 2))
            has_val_i = (n_val_i > 0.0).to(dtype=torch.float32)
            n_val_denom = n_val_i.clamp(min=1.0)

            opt = torch.optim.Adam([delta], lr=float(lr))
            for st_idx in range(1, int(num_steps) + 1):
                opt.zero_grad(set_to_none=True)
                vs_proj = torch.clamp(vs0_t + delta, min=v_lo, max=v_hi)
                vs_cur = torch.where(edit_t, vs_proj, vs0_t) if edit_t is not None else vs_proj
                mn, _, _ = ens.predict_with_uncertainty(vs_cur, pq_t)
                pred_m = torch.where(vm, mn, d_safe)
                rel_sq = ((pred_m - d_safe) / d_safe) ** 2
                loss_c_i = (vm_f * rel_sq).sum(dim=(1, 2)) / n_val_denom * has_val_i
                prior_i = float(lambda_prior) * compute_r_depth_per_station(vs_cur, domain) if lambda_prior > 0 else 0.0
                loss = ((loss_c_i + prior_i) * has_val_i).sum()
                loss.backward()
                if delta.grad is not None and edit_t is not None:
                    delta.grad.masked_fill_(~edit_t, 0.0)
                opt.step()
                if edit_t is not None:
                    with torch.no_grad():
                        delta.data.masked_fill_(~edit_t, 0.0)

                if st_idx in snapshots:
                    with torch.no_grad():
                        vs_sn = torch.clamp(vs0_t + delta, min=v_lo, max=v_hi)
                        if edit_t is not None:
                            vs_sn = torch.where(edit_t, vs_sn, vs0_t)
                            if torch.any(all_m):
                                vs_sn[all_m] = vs0_t[all_m]
                        elif torch.any(all_m):
                            vs_sn[all_m] = 0.0
                        snapshots[st_idx][s:e] = vs_sn.cpu().numpy()

            with torch.no_grad():
                vs_fin = torch.clamp(vs0_t + delta, min=v_lo, max=v_hi)
                if edit_t is not None:
                    vs_fin = torch.where(edit_t, vs_fin, vs0_t)
                    if torch.any(all_m):
                        vs_fin[all_m] = vs0_t[all_m]
                elif torch.any(all_m):
                    vs_fin[all_m] = 0.0
                vs_refined[s:e] = vs_fin.cpu().numpy()

        if use_cuda:
            torch.cuda.synchronize(device)
        t_ref = float(time.perf_counter() - t0)
        peak_b = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0
    finally:
        _restore_deterministic_cuda(prev_flags)

    out_info: Dict[str, Any] = {
        "method": "output_space_refinement",
        "domain": domain,
        "num_steps": int(num_steps),
        "lr": float(lr),
        "lambda_prior": float(lambda_prior),
        "enable_zres_freeze": bool(enable_zres_freeze),
        "t_gpu_refine_sec": t_ref,
        "peak_vram_gb": float(peak_b / (1024 ** 3)),
        **({"step_snapshots": snapshots} if snapshot_steps is not None else {}),
    }
    if enable_zres_freeze and d_arr is not None and z_res_used is not None:
        out_info["z_res_kept"] = z_res_used
        out_info["max_frozen_delta"] = compute_max_frozen_delta(vs_refined, vs_init, d_arr, z_res_used)
    return vs_refined, out_info


def run_holdout_early_stopping(
    vs_init_aligned: np.ndarray,
    ens: SurrogateEnsemble,
    depths: np.ndarray,
    periods: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    domain: str,
    *,
    dc: Optional[float] = None,
    dc_shallow: Optional[float] = None,
    dc_deep: Optional[float] = None,
    dc_cfg: Optional[Dict[str, Any]] = None,
    lr: float = 0.02,
    num_steps: int = 50,
    check_interval: int = 5,
    lambda_prior: float = 0.05,
    v_bounds: Optional[Tuple[float, float]] = None,
    batch_size: int = 512,
    stop_split_ratio: float = 0.20,
    stop_split_seed: int = 20261121,
    stop_split_keep_endpoints: bool = True,
    station_indices: Optional[Sequence[int]] = None,
    enable_zres_freeze: bool = False,
    freeze_depth_factor: float = 1.0,
    enable_exact_verification: bool = False,
    m_init_kept: Optional[np.ndarray] = None,
    init_complete: Optional[np.ndarray] = None,
    seed: int = 0,
    device: str = "cpu",
    remove_group: Optional[bool] = None,
    compute_group: Optional[bool] = None,
    disba_workers: int = 4,
    executor: Optional[ProcessPoolExecutor] = None,
    phase_dispersion_cls: Any = PhaseDispersion,
    group_dispersion_cls: Any = GroupDispersion,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-station internal Holdout ES using 20% of kept periods evaluated via ``disba``.

    Optional safety guards (enabled by default in ``run_operational_protocol``):
    - ``enable_zres_freeze``: freezes layers deeper than ``z_res_kept`` (zero update).
    - ``enable_exact_verification``: evaluates the selected snapshot on all kept periods
      with exact converged ``disba`` and reverts to ``vs_init_aligned`` if empty,
      incomplete, non-finite, or failing to lower full kept-period misfit.
    """
    t_total0 = time.perf_counter()
    if domain not in ("shallow", "deep"):
        raise ValueError(f"domain must be 'shallow' or 'deep'; got {domain!r}")
    rm_grp = (domain == "shallow") if remove_group is None else bool(remove_group)
    comp_grp = (domain == "deep") if compute_group is None else bool(compute_group)
    d_arr = np.asarray(depths, dtype=np.float64).reshape(-1)
    vs_init = np.asarray(vs_init_aligned, dtype=np.float32)
    n_sta = int(len(vs_init))

    per_clean, ph_clean, gp_clean, km_2ch = _scrub_to_kept_only(
        periods, phase_vel, group_vel, obs_mask, kept_mask, remove_group=rm_grp
    )
    any_kept_station = np.any(km_2ch, axis=(1, 2))
    z_res_kept = compute_resolvable_depth_from_kept(per_clean, ph_clean, km_2ch, km_2ch, d_arr)

    train_kept_mask, stop_mask = create_stop_period_split(
        km_2ch,
        km_2ch,
        compute_group=comp_grp,
        stop_ratio=float(stop_split_ratio),
        seed=int(stop_split_seed),
        keep_endpoints=bool(stop_split_keep_endpoints),
        station_indices=station_indices,
    )

    snapshot_steps = list(range(0, int(num_steps) + 1, int(check_interval)))
    if int(num_steps) not in snapshot_steps:
        snapshot_steps.append(int(num_steps))

    _, ref_info = run_output_refinement_from_profile(
        vs_init_aligned=vs_init,
        ens=ens,
        periods=per_clean,
        phase_vel=ph_clean,
        group_vel=gp_clean,
        obs_mask=km_2ch,
        kept_mask=train_kept_mask,
        domain=domain,
        depths=d_arr,
        z_res_kept=z_res_kept * float(freeze_depth_factor),
        enable_zres_freeze=bool(enable_zres_freeze),
        lr=lr,
        num_steps=num_steps,
        lambda_prior=lambda_prior,
        v_bounds=v_bounds,
        batch_size=batch_size,
        snapshot_steps=snapshot_steps,
        seed=seed,
        device=device,
        remove_group=rm_grp,
    )
    snaps = ref_info["step_snapshots"]

    dc_val = resolve_regime_dc(domain, dc=dc, dc_shallow=dc_shallow, dc_deep=dc_deep, dc_cfg=dc_cfg)
    eff_workers = min(int(disba_workers), MAX_DISBA_WORKERS)
    custom_cls = (phase_dispersion_cls is not PhaseDispersion) or (group_dispersion_cls is not GroupDispersion)

    has_stop_pts = np.any(stop_mask, axis=(1, 2))
    stop_per = np.where(stop_mask[:, :, 0] | stop_mask[:, :, 1], per_clean, -1.0).astype(np.float32)
    stop_ph = np.where(stop_mask[:, :, 0], ph_clean, -1.0).astype(np.float32)
    stop_gp = np.where(stop_mask[:, :, 1], gp_clean, -1.0).astype(np.float32)

    stop_misfit_history: Dict[int, np.ndarray] = {}
    t_disba_sec = 0.0
    n_disba_calls = 0

    own_executor = executor is None and (not custom_cls) and eff_workers > 1 and n_sta > 16
    exec_pool: Optional[ProcessPoolExecutor] = (
        ProcessPoolExecutor(max_workers=eff_workers, initializer=_mp_worker_init)
        if own_executor
        else executor
    )

    reverted = np.zeros(n_sta, dtype=bool)
    revert_reason: List[str] = ["none"] * n_sta
    m_init_arr = np.full(n_sta, np.nan, dtype=np.float64)
    init_comp_arr = np.ones(n_sta, dtype=bool)
    m_final_kept = np.full(n_sta, np.nan, dtype=np.float64)

    try:
        for st in snapshot_steps:
            ev_st = _eval_disba_subset_diag(
                snaps[st],
                d_arr,
                stop_per,
                stop_ph,
                stop_gp,
                stop_mask,
                compute_group=comp_grp,
                dc=dc_val,
                workers=eff_workers,
                executor=exec_pool,
                phase_dispersion_cls=phase_dispersion_cls,
                group_dispersion_cls=group_dispersion_cls,
            )
            t_disba_sec += ev_st["elapsed_sec"]
            n_disba_calls += ev_st["n_calls"]
            mf = ev_st["misfit_ps"].copy()
            mf[~ev_st["is_complete_obs"]] = np.inf
            mf[~np.isfinite(mf)] = np.inf
            stop_misfit_history[st] = mf

        best_step = np.full(n_sta, int(snapshot_steps[0]), dtype=np.int32)
        best_val = np.full(n_sta, np.inf, dtype=np.float64)
        vs_out = vs_init.copy()

        for st in snapshot_steps:
            mf_st = stop_misfit_history[st]
            better = has_stop_pts & (mf_st < best_val)
            if np.any(better):
                best_val[better] = mf_st[better]
                best_step[better] = int(st)
                if int(st) == 0:
                    vs_out[better] = vs_init[better]
                else:
                    vs_out[better] = snaps[st][better]

        vs_out[~has_stop_pts] = vs_init[~has_stop_pts]
        best_step[~has_stop_pts] = 0

        if enable_exact_verification:
            if m_init_kept is not None and init_complete is not None:
                m_init_arr = np.asarray(m_init_kept, dtype=np.float64).copy()
                init_comp_arr = np.asarray(init_complete, dtype=bool).copy()
            else:
                ev0 = _eval_disba_subset_diag(
                    vs_init,
                    d_arr,
                    per_clean,
                    ph_clean,
                    gp_clean,
                    km_2ch,
                    compute_group=comp_grp,
                    dc=dc_val,
                    workers=eff_workers,
                    executor=exec_pool,
                    phase_dispersion_cls=phase_dispersion_cls,
                    group_dispersion_cls=group_dispersion_cls,
                )
                t_disba_sec += ev0["elapsed_sec"]
                n_disba_calls += ev0["n_calls"]
                m_init_arr = ev0["misfit_ps"]
                init_comp_arr = ev0["is_complete_obs"] & np.isfinite(m_init_arr)

            m_final_kept[:] = m_init_arr
            init_bad = (~init_comp_arr) | (~any_kept_station)
            for i in np.where(init_bad)[0].tolist():
                vs_out[i] = vs_init[i]
                best_step[i] = 0
                revert_reason[i] = "initial_incomplete_skipped"

            nz_mask = (~init_bad) & (best_step > 0)
            nz_idx = np.where(nz_mask)[0]
            if nz_idx.size > 0:
                ev_fin = _eval_disba_subset_diag(
                    vs_out[nz_idx],
                    d_arr,
                    per_clean[nz_idx],
                    ph_clean[nz_idx],
                    gp_clean[nz_idx],
                    km_2ch[nz_idx],
                    compute_group=comp_grp,
                    dc=dc_val,
                    workers=eff_workers,
                    executor=exec_pool,
                    phase_dispersion_cls=phase_dispersion_cls,
                    group_dispersion_cls=group_dispersion_cls,
                )
                t_disba_sec += ev_fin["elapsed_sec"]
                n_disba_calls += ev_fin["n_calls"]
                m_cand = ev_fin["misfit_ps"]
                c_obs = ev_fin["is_complete_obs"]
                is_emp = ev_fin["is_empty"]
                for k_i, gi in enumerate(nz_idx.tolist()):
                    if bool(is_emp[k_i]):
                        vs_out[gi] = vs_init[gi]
                        reverted[gi] = True
                        revert_reason[gi] = "disba_empty_return"
                        m_final_kept[gi] = m_init_arr[gi]
                    elif not bool(c_obs[k_i]):
                        vs_out[gi] = vs_init[gi]
                        reverted[gi] = True
                        revert_reason[gi] = "disba_incomplete_return"
                        m_final_kept[gi] = m_init_arr[gi]
                    elif not np.isfinite(m_cand[k_i]):
                        vs_out[gi] = vs_init[gi]
                        reverted[gi] = True
                        revert_reason[gi] = "nonfinite_misfit"
                        m_final_kept[gi] = m_init_arr[gi]
                    elif float(m_cand[k_i]) >= float(m_init_arr[gi]):
                        vs_out[gi] = vs_init[gi]
                        reverted[gi] = True
                        revert_reason[gi] = "misfit_not_improved"
                        m_final_kept[gi] = m_init_arr[gi]
                    else:
                        m_final_kept[gi] = float(m_cand[k_i])
    finally:
        if own_executor and exec_pool is not None:
            exec_pool.shutdown(wait=True)

    max_frozen_delta = compute_max_frozen_delta(vs_out, vs_init, d_arr, z_res_kept)
    reason_counts: Dict[str, int] = {}
    for r in revert_reason:
        reason_counts[r] = reason_counts.get(r, 0) + 1
    t_total_sec = float(time.perf_counter() - t_total0)

    return vs_out, {
        "method": "holdout_es",
        "domain": domain,
        "dc": dc_val,
        "enable_zres_freeze": bool(enable_zres_freeze),
        "enable_exact_verification": bool(enable_exact_verification),
        "stop_split_ratio": float(stop_split_ratio),
        "stop_split_seed": int(stop_split_seed),
        "snapshot_steps": snapshot_steps,
        "selected_step": best_step,
        "best_stop_misfit": best_val,
        "stop_misfit_history": stop_misfit_history,
        "train_kept_mask": train_kept_mask,
        "stop_mask": stop_mask,
        "z_res_kept": z_res_kept,
        "max_frozen_delta": max_frozen_delta,
        "m_init_kept": m_init_arr,
        "m_final_kept": m_final_kept,
        "reverted": reverted,
        "n_reverted": int(np.sum(reverted)),
        "revert_rate": float(np.mean(reverted)) if n_sta else 0.0,
        "revert_reason": revert_reason,
        "revert_reason_counts": reason_counts,
        "n_stations": n_sta,
        "n_disba_calls": int(n_disba_calls),
        "t_gpu_refine_sec": float(ref_info["t_gpu_refine_sec"]),
        "t_disba_check_sec": float(t_disba_sec),
        "t_total_sec": t_total_sec,
        "peak_vram_gb": float(ref_info["peak_vram_gb"]),
    }


def run_operational_protocol(
    vs_init_aligned: np.ndarray,
    ens: SurrogateEnsemble,
    depths: np.ndarray,
    periods: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    domain: str,
    *,
    sigma_hat_d: float,
    r0_low: Optional[float] = None,
    r0_high: Any = _UNSET,
    enable_zres_freeze: bool = False,
    freeze_depth_factor: float = 1.0,
    enable_holdout_es: bool = True,
    enable_exact_verification: bool = True,
    dc: Optional[float] = None,
    dc_shallow: Optional[float] = None,
    dc_deep: Optional[float] = None,
    dc_cfg: Optional[Dict[str, Any]] = None,
    lr: float = 0.02,
    num_steps: int = 50,
    check_interval: int = 5,
    lambda_prior: float = 0.05,
    v_bounds: Optional[Tuple[float, float]] = None,
    batch_size: int = 512,
    stop_split_ratio: float = 0.20,
    stop_split_seed: int = 20261121,
    stop_split_keep_endpoints: bool = True,
    seed: int = 0,
    device: str = "cpu",
    remove_group: Optional[bool] = None,
    compute_group: Optional[bool] = None,
    disba_workers: int = 4,
    phase_dispersion_cls: Any = PhaseDispersion,
    group_dispersion_cls: Any = GroupDispersion,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute the full four-step operational TTA protocol from aligned zero-shot profiles.

    Steps executed per station:
    1. Strictly scrub held-out evaluation periods (zero leakage) and resolve converged ``dc``.
    2. Evaluate initial exact kept-period misfit ``m_0`` and compute ``r_0 = m_0 / sigma_hat_d``.
    3. Route stations using regime-aware validation-locked bounds (default: shallow ``r_0 >= 2.0``,
       deep ``1.25 <= r_0 <= 3.0``; near-floor and deep ``r_0 > 3.0`` retain ``vs_init_aligned``).
    4. On routed stations only, run output-space refinement (without hard ``z > z_res``
       freezing by default, because Rayleigh sensitivity extends beyond ``lambda_max / 3``;
       optional freezing at ``freeze_depth_factor * z_res_kept`` is available via
       ``enable_zres_freeze=True``), select the iteration via internal 80/20 Holdout ES
       (when ``enable_holdout_es=True``), and verify final full kept-period misfit with
       converged ``disba``, reverting to ``vs_init_aligned`` on empty/incomplete/non-finite
       returns or non-improving misfit.
    """
    t_total0 = time.perf_counter()
    if domain not in ("shallow", "deep"):
        raise ValueError(f"domain must be 'shallow' or 'deep'; got {domain!r}")
    if float(sigma_hat_d) <= 0.0 or not np.isfinite(float(sigma_hat_d)):
        raise ValueError(f"sigma_hat_d must be positive finite; got {sigma_hat_d!r}")

    def_lo, def_hi = DEFAULT_ROUTING_BOUNDS[domain]
    lo_bound = float(def_lo if r0_low is None else r0_low)
    hi_bound: Optional[float] = def_hi if r0_high is _UNSET else (None if r0_high is None else float(r0_high))

    rm_grp = (domain == "shallow") if remove_group is None else bool(remove_group)
    comp_grp = (domain == "deep") if compute_group is None else bool(compute_group)
    d_arr = np.asarray(depths, dtype=np.float64).reshape(-1)
    vs_init = np.asarray(vs_init_aligned, dtype=np.float32)
    if vs_init.ndim != 2:
        raise ValueError(f"vs_init_aligned must be 2D (N, D); got {vs_init.shape}")
    n_sta, n_layers = vs_init.shape
    if d_arr.size != n_layers:
        raise ValueError(f"depths length {d_arr.size} != profile layers {n_layers}")

    per_clean, ph_clean, gp_clean, km_2ch = _scrub_to_kept_only(
        periods, phase_vel, group_vel, obs_mask, kept_mask, remove_group=rm_grp
    )
    any_kept_station = np.any(km_2ch, axis=(1, 2))
    z_res_kept = compute_resolvable_depth_from_kept(per_clean, ph_clean, km_2ch, km_2ch, d_arr)
    z_freeze_eff = z_res_kept * float(freeze_depth_factor)

    dc_val = resolve_regime_dc(domain, dc=dc, dc_shallow=dc_shallow, dc_deep=dc_deep, dc_cfg=dc_cfg)
    eff_workers = min(int(disba_workers), MAX_DISBA_WORKERS)
    custom_cls = (phase_dispersion_cls is not PhaseDispersion) or (group_dispersion_cls is not GroupDispersion)

    snapshot_steps = list(range(0, int(num_steps) + 1, int(check_interval)))
    if int(num_steps) not in snapshot_steps:
        snapshot_steps.append(int(num_steps))

    executor: Optional[ProcessPoolExecutor] = None
    if (not custom_cls) and eff_workers > 1 and n_sta > 16:
        executor = ProcessPoolExecutor(max_workers=eff_workers, initializer=_mp_worker_init)

    t_gpu_sec = 0.0
    t_disba_sec = 0.0
    n_disba_calls = 0
    peak_vram_gb = 0.0

    vs_out = vs_init.copy()
    selected_step = np.zeros(n_sta, dtype=np.int32)
    reverted = np.zeros(n_sta, dtype=bool)
    revert_reason: List[str] = ["none"] * n_sta
    route_action: List[str] = ["retain_zs"] * n_sta

    try:
        ev0 = _eval_disba_subset_diag(
            vs_init,
            d_arr,
            per_clean,
            ph_clean,
            gp_clean,
            km_2ch,
            compute_group=comp_grp,
            dc=dc_val,
            workers=eff_workers,
            executor=executor,
            phase_dispersion_cls=phase_dispersion_cls,
            group_dispersion_cls=group_dispersion_cls,
        )
        t_disba_sec += ev0["elapsed_sec"]
        n_disba_calls += ev0["n_calls"]

        m_init_kept = ev0["misfit_ps"]
        init_comp = ev0["is_complete_obs"] & np.isfinite(m_init_kept)
        r0 = m_init_kept / float(sigma_hat_d)
        m_final_kept = m_init_kept.copy()

        init_bad = (~init_comp) | (~any_kept_station)
        routed_mask = np.zeros(n_sta, dtype=bool)
        for i in range(n_sta):
            if init_bad[i]:
                route_action[i] = "init_incomplete_skipped"
                revert_reason[i] = "initial_incomplete_skipped"
            elif r0[i] < lo_bound:
                route_action[i] = "retain_zs"
            elif hi_bound is not None and r0[i] > hi_bound:
                route_action[i] = "flag_outlier"
            else:
                route_action[i] = "refine_hes" if enable_holdout_es else "refine_50step"
                routed_mask[i] = True

        routed_idx = np.where(routed_mask)[0]
        if routed_idx.size > 0:
            if enable_holdout_es:
                vs_r, r_info = run_holdout_early_stopping(
                    vs_init_aligned=vs_init[routed_idx],
                    ens=ens,
                    depths=d_arr,
                    periods=per_clean[routed_idx],
                    phase_vel=ph_clean[routed_idx],
                    group_vel=gp_clean[routed_idx],
                    obs_mask=km_2ch[routed_idx],
                    kept_mask=km_2ch[routed_idx],
                    domain=domain,
                    dc=dc_val,
                    lr=lr,
                    num_steps=num_steps,
                    check_interval=check_interval,
                    lambda_prior=lambda_prior,
                    v_bounds=v_bounds,
                    batch_size=batch_size,
                    stop_split_ratio=stop_split_ratio,
                    stop_split_seed=stop_split_seed,
                    stop_split_keep_endpoints=stop_split_keep_endpoints,
                    station_indices=routed_idx,
                    enable_zres_freeze=bool(enable_zres_freeze),
                    freeze_depth_factor=float(freeze_depth_factor),
                    enable_exact_verification=bool(enable_exact_verification),
                    m_init_kept=m_init_kept[routed_idx],
                    init_complete=init_comp[routed_idx],
                    seed=seed,
                    device=device,
                    remove_group=rm_grp,
                    compute_group=comp_grp,
                    disba_workers=eff_workers,
                    executor=executor,
                    phase_dispersion_cls=phase_dispersion_cls,
                    group_dispersion_cls=group_dispersion_cls,
                )
                vs_out[routed_idx] = vs_r
                selected_step[routed_idx] = r_info["selected_step"]
                reverted[routed_idx] = r_info["reverted"]
                m_final_kept[routed_idx] = r_info["m_final_kept"]
                for k_i, gi in enumerate(routed_idx.tolist()):
                    revert_reason[gi] = str(r_info["revert_reason"][k_i])
                t_gpu_sec += float(r_info["t_gpu_refine_sec"])
                t_disba_sec += float(r_info["t_disba_check_sec"])
                n_disba_calls += int(r_info["n_disba_calls"])
                peak_vram_gb = max(peak_vram_gb, float(r_info["peak_vram_gb"]))
            else:
                vs_r, r_info = run_output_refinement_from_profile(
                    vs_init_aligned=vs_init[routed_idx],
                    ens=ens,
                    periods=per_clean[routed_idx],
                    phase_vel=ph_clean[routed_idx],
                    group_vel=gp_clean[routed_idx],
                    obs_mask=km_2ch[routed_idx],
                    kept_mask=km_2ch[routed_idx],
                    domain=domain,
                    depths=d_arr,
                    z_res_kept=z_freeze_eff[routed_idx],
                    enable_zres_freeze=bool(enable_zres_freeze),
                    lr=lr,
                    num_steps=num_steps,
                    lambda_prior=lambda_prior,
                    v_bounds=v_bounds,
                    batch_size=batch_size,
                    snapshot_steps=None,
                    seed=seed,
                    device=device,
                    remove_group=rm_grp,
                )
                t_gpu_sec += float(r_info["t_gpu_refine_sec"])
                peak_vram_gb = max(peak_vram_gb, float(r_info["peak_vram_gb"]))
                vs_out[routed_idx] = vs_r
                selected_step[routed_idx] = int(num_steps)
                if enable_exact_verification:
                    ev_fin = _eval_disba_subset_diag(
                        vs_r,
                        d_arr,
                        per_clean[routed_idx],
                        ph_clean[routed_idx],
                        gp_clean[routed_idx],
                        km_2ch[routed_idx],
                        compute_group=comp_grp,
                        dc=dc_val,
                        workers=eff_workers,
                        executor=executor,
                        phase_dispersion_cls=phase_dispersion_cls,
                        group_dispersion_cls=group_dispersion_cls,
                    )
                    t_disba_sec += ev_fin["elapsed_sec"]
                    n_disba_calls += ev_fin["n_calls"]
                    m_cand = ev_fin["misfit_ps"]
                    c_obs = ev_fin["is_complete_obs"]
                    is_emp = ev_fin["is_empty"]
                    for k_i, gi in enumerate(routed_idx.tolist()):
                        if bool(is_emp[k_i]):
                            vs_out[gi] = vs_init[gi]
                            reverted[gi] = True
                            revert_reason[gi] = "disba_empty_return"
                            m_final_kept[gi] = m_init_kept[gi]
                        elif not bool(c_obs[k_i]):
                            vs_out[gi] = vs_init[gi]
                            reverted[gi] = True
                            revert_reason[gi] = "disba_incomplete_return"
                            m_final_kept[gi] = m_init_kept[gi]
                        elif not np.isfinite(m_cand[k_i]):
                            vs_out[gi] = vs_init[gi]
                            reverted[gi] = True
                            revert_reason[gi] = "nonfinite_misfit"
                            m_final_kept[gi] = m_init_kept[gi]
                        elif float(m_cand[k_i]) >= float(m_init_kept[gi]):
                            vs_out[gi] = vs_init[gi]
                            reverted[gi] = True
                            revert_reason[gi] = "misfit_not_improved"
                            m_final_kept[gi] = m_init_kept[gi]
                        else:
                            m_final_kept[gi] = float(m_cand[k_i])
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    max_frozen_delta = compute_max_frozen_delta(vs_out, vs_init, d_arr, z_res_kept)
    route_counts: Dict[str, int] = {}
    for ra in route_action:
        route_counts[ra] = route_counts.get(ra, 0) + 1
    reason_counts: Dict[str, int] = {}
    for rr in revert_reason:
        reason_counts[rr] = reason_counts.get(rr, 0) + 1

    n_routed = int(np.sum(routed_mask))
    unrouted_hes_calls = int(n_sta * len(snapshot_steps))
    call_savings_pct = (
        float((unrouted_hes_calls - n_disba_calls) / unrouted_hes_calls * 100.0)
        if unrouted_hes_calls > 0
        else 0.0
    )
    t_total_sec = float(time.perf_counter() - t_total0)

    info: Dict[str, Any] = {
        "method": "operational_four_step_protocol",
        "domain": domain,
        "dc": dc_val,
        "sigma_hat_d": float(sigma_hat_d),
        "r0_low": lo_bound,
        "r0_high": hi_bound,
        "switches": {
            "enable_zres_freeze": bool(enable_zres_freeze),
            "freeze_depth_factor": float(freeze_depth_factor),
            "enable_holdout_es": bool(enable_holdout_es),
            "enable_exact_verification": bool(enable_exact_verification),
        },
        "lr": float(lr),
        "num_steps": int(num_steps),
        "check_interval": int(check_interval),
        "lambda_prior": float(lambda_prior),
        "n_stations": n_sta,
        "n_routed": n_routed,
        "routed_fraction": float(n_routed / n_sta) if n_sta else 0.0,
        "route_counts": route_counts,
        "n_reverted": int(np.sum(reverted)),
        "revert_rate_all": float(np.mean(reverted)) if n_sta else 0.0,
        "revert_rate_routed": float(np.sum(reverted) / n_routed) if n_routed else 0.0,
        "revert_reason_counts": reason_counts,
        "max_frozen_delta_overall": float(np.max(max_frozen_delta)) if n_sta else 0.0,
        "n_disba_calls": int(n_disba_calls),
        "unrouted_holdout_es_disba_calls": unrouted_hes_calls,
        "disba_call_savings_vs_unrouted_hes_pct": call_savings_pct,
        "t_gpu_refine_sec": t_gpu_sec,
        "t_disba_check_sec": t_disba_sec,
        "t_total_sec": t_total_sec,
        "peak_vram_gb": peak_vram_gb,
        "r0": r0,
        "m_init_kept": m_init_kept,
        "m_final_kept": m_final_kept,
        "route_action": np.asarray(route_action, dtype="<U32"),
        "selected_step": selected_step,
        "z_res_kept": z_res_kept,
        "max_frozen_delta": max_frozen_delta,
        "reverted": reverted,
        "revert_reason": np.asarray(revert_reason, dtype="<U32"),
    }
    return vs_out, info


def run_fixed_step_early_stopping(
    *,
    k_steps: int,
    step_snapshots: Optional[Dict[int, np.ndarray]] = None,
    vs_init_aligned: Optional[np.ndarray] = None,
    ens: Optional[SurrogateEnsemble] = None,
    periods: Optional[np.ndarray] = None,
    phase_vel: Optional[np.ndarray] = None,
    group_vel: Optional[np.ndarray] = None,
    obs_mask: Optional[np.ndarray] = None,
    kept_mask: Optional[np.ndarray] = None,
    domain: str = "shallow",
    lr: float = 0.02,
    lambda_prior: float = 0.05,
    v_bounds: Optional[Tuple[float, float]] = None,
    batch_size: int = 512,
    seed: int = 0,
    device: str = "cpu",
    remove_group: Optional[bool] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fixed-step early-stopping control (from snapshots or standalone K-step run)."""
    k = int(k_steps)
    if step_snapshots is not None:
        if k not in step_snapshots:
            raise KeyError(f"step_snapshots missing step K={k}; available={sorted(step_snapshots.keys())}")
        return np.asarray(step_snapshots[k], dtype=np.float32).copy(), {
            "method": "fixed_step_es",
            "k_steps": k,
            "source": "snapshot",
        }

    if any(x is None for x in (vs_init_aligned, ens, periods, phase_vel, group_vel, obs_mask, kept_mask)):
        raise ValueError("Full inputs are required when step_snapshots is not supplied.")

    vs_k, info = run_output_refinement_from_profile(
        vs_init_aligned=vs_init_aligned,
        ens=ens,
        periods=periods,
        phase_vel=phase_vel,
        group_vel=group_vel,
        obs_mask=obs_mask,
        kept_mask=kept_mask,
        domain=domain,
        lr=lr,
        num_steps=k,
        lambda_prior=lambda_prior,
        v_bounds=v_bounds,
        batch_size=batch_size,
        snapshot_steps=None,
        seed=seed,
        device=device,
        remove_group=remove_group,
    )
    info["method"] = "fixed_step_es"
    info["k_steps"] = k
    info["source"] = "standalone"
    return vs_k, info


def run_ideal_early_stopping(
    step_snapshots: Dict[int, np.ndarray],
    vs_true: np.ndarray,
    depths: np.ndarray,
    z_res: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Oracle Ideal ES upper bound selecting per-sample minimum resolvable-depth Vs RMSE."""
    steps = sorted(int(s) for s in step_snapshots.keys())
    if not steps:
        raise ValueError("step_snapshots cannot be empty")

    n_sta = int(len(vs_true))
    step_rmse: Dict[int, np.ndarray] = {
        st: np.asarray(compute_per_sample_rmse_res(step_snapshots[st], vs_true, depths, z_res), dtype=np.float64)
        for st in steps
    }

    best_step = np.full(n_sta, steps[0], dtype=np.int32)
    best_rmse = step_rmse[steps[0]].copy()
    vs_oracle = np.asarray(step_snapshots[steps[0]], dtype=np.float32).copy()

    for st in steps[1:]:
        r_st = step_rmse[st]
        better = np.isfinite(r_st) & (r_st < best_rmse)
        if np.any(better):
            best_rmse[better] = r_st[better]
            best_step[better] = int(st)
            vs_oracle[better] = np.asarray(step_snapshots[st], dtype=np.float32)[better]

    return vs_oracle, {
        "method": "ideal_es",
        "steps": steps,
        "selected_step": best_step,
        "oracle_rmse_res_per_sample": best_rmse,
        "oracle_rmse_res_mean": float(np.mean(best_rmse)),
        "step_rmse_res": step_rmse,
    }
