"""Zero-shot baselines, normalization-stats TTA, and station-independent output refinement."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from swi_tta.adaptation.gated_cycle import (
    build_valid_channel_mask,
    compute_gate_weights,
    compute_r_depth_per_station,
)
from swi_tta.data.alignment import prepare_aligned_input
from swi_tta.models.adapters import AdaptedOpenSWITransformer, FeatureStatsCalibrator
from swi_tta.models.backbones import predict_batches, prepare_openswi_input
from swi_tta.models.surrogate import SurrogateEnsemble


def _prepare_kept_inputs(
    periods: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    input_mode: str = "raw",
    train_grid: Optional[np.ndarray] = None,
    remove_group: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    km_arr = np.asarray(kept_mask)
    km_2ch = np.stack([km_arr > 0.5, km_arr > 0.5], axis=-1) if km_arr.ndim == 2 else (km_arr > 0.5)
    om_2ch = (np.asarray(obs_mask) > 0.5) & km_2ch
    if remove_group:
        om_2ch[:, :, 1] = False

    per_clean = np.where(km_2ch[:, :, 0] | km_2ch[:, :, 1], periods, -1.0).astype(np.float32)
    ph_clean = np.where(om_2ch[:, :, 0], phase_vel, -1.0).astype(np.float32)
    gp_clean = np.where(om_2ch[:, :, 1], group_vel, -1.0).astype(np.float32)

    if input_mode == "aligned":
        if train_grid is None:
            raise ValueError("input_mode='aligned' requires train_grid")
        x_in, pad_in = prepare_aligned_input(
            period=per_clean,
            phase_vel=ph_clean,
            group_vel=gp_clean,
            obs_mask=om_2ch.astype(np.uint8),
            heldout_mask=None,
            train_grid=train_grid,
            remove_group=remove_group,
        )
    elif input_mode == "raw":
        x_in, pad_in = prepare_openswi_input(
            period=per_clean,
            phase_vel=ph_clean,
            group_vel=gp_clean,
            obs_mask=om_2ch.astype(np.uint8),
            heldout_mask=None,
            remove_group=remove_group,
        )
    else:
        raise ValueError(f"Unknown input_mode: {input_mode!r}")

    return x_in, pad_in, per_clean, ph_clean, gp_clean, om_2ch


def run_zeroshot_inference(
    backbone_model: nn.Module,
    periods: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    slice_dim: int,
    *,
    input_mode: str = "aligned",
    train_grid: Optional[np.ndarray] = None,
    batch_size: int = 512,
    device: str = "cpu",
    remove_group: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run Zero-Shot (Raw) or Input-Aligned Zero-Shot inference using kept periods only."""
    x_in, pad_in, _, _, _, _ = _prepare_kept_inputs(
        periods,
        phase_vel,
        group_vel,
        obs_mask,
        kept_mask,
        input_mode=input_mode,
        train_grid=train_grid,
        remove_group=remove_group,
    )
    t0 = time.perf_counter()
    vs_full, _, peak_b = predict_batches(backbone_model, x_in, pad_in, device=device, batch_size=batch_size)
    t_tot = time.perf_counter() - t0
    vs_out = vs_full[:, :slice_dim]
    return vs_out, {
        "method": "zero_shot_aligned" if input_mode == "aligned" else "zero_shot_raw",
        "input_mode": input_mode,
        "t_adapt_sec": 0.0,
        "t_infer_pure_sec": float(t_tot),
        "t_total_per_station_ms": float(t_tot * 1000.0 / max(len(x_in), 1)),
        "peak_vram_gb": float(peak_b / (1024 ** 3)),
    }


def run_norm_stats_tta(
    backbone_model: nn.Module,
    source_val_disp: np.ndarray,
    periods: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    slice_dim: int,
    *,
    blend: float = 0.65,
    num_passes: int = 3,
    input_mode: str = "raw",
    train_grid: Optional[np.ndarray] = None,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
    remove_group: bool = False,
    max_src_samples: int = 2048,
) -> Tuple[AdaptedOpenSWITransformer, np.ndarray, Dict[str, Any]]:
    """Gradient-free feature-normalization statistics alignment (BN-Only TTA baseline)."""
    torch.manual_seed(int(seed))
    rng = np.random.RandomState(int(seed))
    adapted = AdaptedOpenSWITransformer(copy.deepcopy(backbone_model)).to(device)
    adapted.configure_trainable_scope("frozen")

    n_src = min(int(max_src_samples), len(source_val_disp))
    d_sub = source_val_disp[:n_src]
    per_s = d_sub[:, 0, :].astype(np.float32)
    ph_s = d_sub[:, 1, :].astype(np.float32)
    gp_s = d_sub[:, 2, :].astype(np.float32)
    om_s = np.stack([(per_s > 0) & (ph_s > 0), (per_s > 0) & (gp_s > 0)], axis=-1)
    x_src, pad_src = prepare_openswi_input(per_s, ph_s, gp_s, obs_mask=om_s, remove_group=remove_group)

    feats_src = []
    adapted.eval()
    with torch.inference_mode():
        for s in range(0, n_src, batch_size):
            e = min(s + batch_size, n_src)
            feats_src.append(
                adapted.encode_pooled_feature(
                    torch.from_numpy(x_src[s:e]).to(device),
                    torch.from_numpy(pad_src[s:e]).to(device),
                )
            )
    f_all_src = torch.cat(feats_src, dim=0)
    mu_src = f_all_src.mean(dim=0)
    std_src = f_all_src.std(dim=0, unbiased=False).clamp(min=1e-5)

    x_in, pad_in, _, _, _, _ = _prepare_kept_inputs(
        periods, phase_vel, group_vel, obs_mask, kept_mask, input_mode=input_mode, train_grid=train_grid, remove_group=remove_group
    )
    x_t = torch.from_numpy(x_in).to(device)
    pad_t = torch.from_numpy(pad_in).to(device)

    calib = FeatureStatsCalibrator(mu_src, std_src, blend=blend).to(device)
    adapted.stats_calibrator = calib
    n_sta = len(x_in)

    t0 = time.perf_counter()
    calib.track_stats = True
    with torch.inference_mode():
        for _ in range(max(1, int(num_passes))):
            perm = rng.permutation(n_sta)
            for s in range(0, n_sta, batch_size):
                idx = torch.from_numpy(perm[s : min(s + batch_size, n_sta)]).to(device)
                _ = adapted.encode_pooled_feature(x_t[idx], pad_t[idx])
    calib.track_stats = False
    t_adapt = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = []
    with torch.inference_mode():
        for s in range(0, n_sta, max(batch_size, 512)):
            e = min(s + max(batch_size, 512), n_sta)
            out = adapted(x_t[s:e], pad_t[s:e])[:, :slice_dim]
            all_m = pad_t[s:e].all(dim=1)
            if torch.any(all_m):
                out[all_m] = 0.0
            preds.append(out.cpu().numpy())
    t_inf = time.perf_counter() - t1

    return adapted, np.concatenate(preds, axis=0), {
        "method": "bn_only_tta",
        "blend": float(blend),
        "num_passes": int(num_passes),
        "trainable_params": 0,
        "t_adapt_sec": float(t_adapt),
        "t_infer_pure_sec": float(t_inf),
        "t_total_per_station_ms": float((t_adapt + t_inf) * 1000.0 / max(n_sta, 1)),
    }


def run_output_space_refinement(
    backbone_model: nn.Module,
    ens: SurrogateEnsemble,
    periods: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: np.ndarray,
    kept_mask: np.ndarray,
    domain: str,
    slice_dim: int,
    *,
    input_mode: str = "aligned",
    train_grid: Optional[np.ndarray] = None,
    num_steps: int = 50,
    lr: float = 0.02,
    lambda_prior: float = 0.05,
    gate_mode: str = "none",
    tau: float = 0.02,
    init_model: Optional[nn.Module] = None,
    batch_size: int = 512,
    init_batch_size: Optional[int] = 512,
    grad_clip_per_station: Optional[float] = None,
    snapshot_steps: Optional[Sequence[int]] = None,
    seed: int = 0,
    device: str = "cpu",
    remove_group: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Station-independent surrogate-gradient output-space refinement (batch-invariant)."""
    torch.manual_seed(int(seed))
    x_in, pad_in, per_clean, ph_clean, gp_clean, om_2ch = _prepare_kept_inputs(
        periods,
        phase_vel,
        group_vel,
        obs_mask,
        kept_mask,
        input_mode=input_mode,
        train_grid=train_grid,
        remove_group=remove_group,
    )

    init_src = init_model if init_model is not None else backbone_model
    init_src.eval().to(device)
    ens.eval().to(device)
    for p in init_src.parameters():
        p.requires_grad_(False)
    for p in ens.parameters():
        p.requires_grad_(False)

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    prev_flags = None
    if use_cuda:
        prev_flags = (
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

    try:
        t0 = time.perf_counter()
        eff_init_bs = min(int(batch_size), int(init_batch_size)) if init_batch_size is not None else int(batch_size)
        vs_init, _, _ = predict_batches(init_src, x_in, pad_in, device=device, batch_size=max(1, eff_init_bs))
        vs_init = vs_init[:, :slice_dim]
        t_init = time.perf_counter() - t0

        g_min = float(ens.members[0].grid_dense[0].item())
        per_q = np.where(per_clean > 0.0, per_clean, g_min).astype(np.float32)
        d_obs_2ch = np.stack([ph_clean, gp_clean], axis=1).astype(np.float32)

        v_lo, v_hi = (0.08, 4.45) if domain == "shallow" else (0.15, 4.98)
        n_sta = len(x_in)
        vs_refined = np.empty_like(vs_init)
        snap_set = sorted({int(st) for st in snapshot_steps}) if snapshot_steps is not None else []
        snapshots = {st: np.empty_like(vs_init) for st in snap_set}

        t_ref0 = time.perf_counter()
        for s in range(0, n_sta, batch_size):
            e = min(s + batch_size, n_sta)
            vs0_t = torch.from_numpy(vs_init[s:e]).to(device)
            delta = torch.zeros_like(vs0_t, requires_grad=True)
            pq_t = torch.from_numpy(per_q[s:e]).to(device)
            per_t = torch.from_numpy(per_clean[s:e]).to(device)
            dobs_t = torch.from_numpy(d_obs_2ch[s:e]).to(device)
            km_t = torch.from_numpy(om_2ch[s:e]).to(device)
            all_m = torch.from_numpy(pad_in[s:e]).to(device).all(dim=1)

            if 0 in snapshots:
                with torch.no_grad():
                    v0_snap = torch.clamp(vs0_t, min=v_lo, max=v_hi)
                    if torch.any(all_m):
                        v0_snap[all_m] = 0.0
                    snapshots[0][s:e] = v0_snap.cpu().numpy()

            vm = build_valid_channel_mask(per_t, dobs_t, km_t)
            d_safe = torch.where(vm, dobs_t, torch.ones_like(dobs_t)).clamp(min=0.05)
            vm_f = vm.to(dtype=torch.float32)
            n_val_i = vm_f.sum(dim=(1, 2))
            has_val_i = (n_val_i > 0.0).to(dtype=torch.float32)
            n_val_denom = n_val_i.clamp(min=1.0)

            opt = torch.optim.Adam([delta], lr=float(lr))
            for st_idx in range(1, int(num_steps) + 1):
                opt.zero_grad(set_to_none=True)
                vs_cur = torch.clamp(vs0_t + delta, min=v_lo, max=v_hi)
                mn, _, sr = ens.predict_with_uncertainty(vs_cur, pq_t)
                pred_m = torch.where(vm, mn, d_safe)
                rel_sq = ((pred_m - d_safe) / d_safe) ** 2
                loss_raw_i = (vm_f * rel_sq).sum(dim=(1, 2)) / n_val_denom * has_val_i
                if gate_mode != "none":
                    w = compute_gate_weights(sr, vm, tau=tau, gate_mode=gate_mode, gate_granularity="point")
                    w_sum_i = w.sum(dim=(1, 2))
                    loss_gate_i = (w * rel_sq).sum(dim=(1, 2)) / w_sum_i.clamp(min=1e-6)
                    loss_c_i = torch.where(w_sum_i > 1e-6, loss_gate_i, loss_raw_i)
                else:
                    loss_c_i = loss_raw_i
                prior_i = float(lambda_prior) * compute_r_depth_per_station(vs_cur, domain) if lambda_prior > 0 else 0.0
                loss = (loss_c_i + prior_i).sum()
                loss.backward()
                if grad_clip_per_station is not None and float(grad_clip_per_station) > 0.0 and delta.grad is not None:
                    row_norm = delta.grad.norm(dim=1, keepdim=True).clamp(min=1e-12)
                    scale = (float(grad_clip_per_station) / row_norm).clamp(max=1.0)
                    delta.grad.mul_(scale)
                opt.step()

                if st_idx in snapshots:
                    with torch.no_grad():
                        vs_sn = torch.clamp(vs0_t + delta, min=v_lo, max=v_hi)
                        if torch.any(all_m):
                            vs_sn[all_m] = 0.0
                        snapshots[st_idx][s:e] = vs_sn.cpu().numpy()

            with torch.no_grad():
                vs_fin = torch.clamp(vs0_t + delta, min=v_lo, max=v_hi)
                if torch.any(all_m):
                    vs_fin[all_m] = 0.0
                vs_refined[s:e] = vs_fin.cpu().numpy()

        if use_cuda:
            torch.cuda.synchronize(device)
    finally:
        if prev_flags is not None:
            (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
            ) = prev_flags

    t_ref = time.perf_counter() - t_ref0
    peak_b = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0

    return vs_refined, {
        "method": "output_space_refinement",
        "gate_mode": gate_mode,
        "tau": float(tau) if gate_mode != "none" else None,
        "input_mode": input_mode,
        "num_steps": int(num_steps),
        "t_init_sec": float(t_init),
        "t_refine_sec": float(t_ref),
        "t_total_per_station_ms": float((t_init + t_ref) * 1000.0 / max(n_sta, 1)),
        "peak_vram_gb": float(peak_b / (1024 ** 3)),
        **({"step_snapshots": snapshots} if snapshot_steps is not None else {}),
    }
