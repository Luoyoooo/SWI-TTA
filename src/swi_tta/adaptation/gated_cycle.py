"""Uncertainty-gated physics cycle-consistent test-time adaptation (Low-DOF Adapter)."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from swi_tta.data.alignment import prepare_aligned_input
from swi_tta.models.adapters import AdaptedOpenSWITransformer
from swi_tta.models.backbones import prepare_openswi_input
from swi_tta.models.surrogate import SurrogateEnsemble

TAU_QUANTILE_GRID: Tuple[float, ...] = (0.5, 0.7, 0.9, 0.95, 0.99)


def compute_r_depth_per_station(
    vs_pred: torch.Tensor,
    domain: str,
    smooth_weight: float = 1.0,
    range_weight: float = 1.0,
) -> torch.Tensor:
    """Compute per-station depth-smoothness and physical-bounds regularization ``(B,)``."""
    d2 = vs_pred[:, 2:] - 2.0 * vs_pred[:, 1:-1] + vs_pred[:, :-2]
    smooth_term = d2.pow(2).mean(dim=1)
    if domain == "shallow":
        v_min, v_max = 0.10, 4.40
    elif domain == "deep":
        v_min, v_max = 0.20, 4.95
    else:
        raise ValueError(f"Unknown domain: {domain!r}")
    range_term = (torch.relu(v_min - vs_pred).pow(2) + torch.relu(vs_pred - v_max).pow(2)).mean(dim=1)
    return float(smooth_weight) * smooth_term + float(range_weight) * range_term


def compute_r_depth(
    vs_pred: torch.Tensor,
    domain: str,
    smooth_weight: float = 1.0,
    range_weight: float = 1.0,
) -> torch.Tensor:
    """Batch mean of :func:`compute_r_depth_per_station`."""
    return compute_r_depth_per_station(vs_pred, domain, smooth_weight=smooth_weight, range_weight=range_weight).mean()


def build_valid_channel_mask(
    periods: torch.Tensor,
    d_obs_2ch: torch.Tensor,
    kept_mask_2ch: torch.Tensor,
) -> torch.Tensor:
    """Construct boolean valid-point mask ``(B, 2, P)`` excluding missing and held-out entries."""
    p_pos = (periods > 0.0).unsqueeze(1)
    km_b = kept_mask_2ch.permute(0, 2, 1) > 0.5
    v_pos = d_obs_2ch > 0.0
    return p_pos & km_b & v_pos


def compute_gate_weights(
    sigma_rel: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: float,
    gate_mode: str = "soft",
    gate_granularity: str = "point",
) -> torch.Tensor:
    """Compute stop-gradient uncertainty gate weights from relative surrogate uncertainty."""
    with torch.no_grad():
        sig = sigma_rel.detach()
        vm = valid_mask
        vm_f = vm.to(dtype=sig.dtype)

        if gate_granularity == "station":
            cnt = vm_f.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
            sig_eff = ((sig * vm_f).sum(dim=(1, 2), keepdim=True) / cnt).expand_as(sig)
        elif gate_granularity == "point":
            sig_eff = sig
        else:
            raise ValueError(f"Unknown gate_granularity: {gate_granularity!r}")

        tau_safe = max(float(tau), 1e-6)
        if gate_mode == "hard":
            w = ((sig_eff < tau_safe) & vm).to(dtype=sig.dtype)
        elif gate_mode == "soft":
            w = torch.exp(-(sig_eff ** 2) / (tau_safe ** 2)) * vm_f
        elif gate_mode in ("none", "penalty"):
            w = vm_f
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode!r}")
        return w.detach()


def compute_cycle_loss_from_outputs(
    vs_pred: torch.Tensor,
    surr_mean: torch.Tensor,
    sigma_rel: torch.Tensor,
    periods: torch.Tensor,
    d_obs_2ch: torch.Tensor,
    kept_mask_2ch: torch.Tensor,
    domain: str,
    tau: float,
    gate_mode: str = "soft",
    gate_granularity: str = "point",
    lambda_prior: float = 0.05,
    beta_penalty: float = 0.0,
    min_valid_weight_ratio: float = 0.10,
    const_gate_weights: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """Compute uncertainty-gated physics cycle loss with low-weight step skipping."""
    vm = build_valid_channel_mask(periods, d_obs_2ch, kept_mask_2ch)
    n_valid = int(vm.sum().item())
    if n_valid == 0:
        return None, {"skipped": True, "reason": "no_valid_points", "sum_w": 0.0, "n_valid": 0}

    if const_gate_weights is not None:
        w = (const_gate_weights * vm.to(dtype=const_gate_weights.dtype)).detach()
    else:
        w = compute_gate_weights(
            sigma_rel=sigma_rel,
            valid_mask=vm,
            tau=tau,
            gate_mode=gate_mode,
            gate_granularity=gate_granularity,
        )

    sum_w = float(w.sum().item())
    ratio = sum_w / max(float(n_valid), 1.0)
    if ratio < float(min_valid_weight_ratio):
        return None, {
            "skipped": True,
            "reason": "low_valid_weight_ratio",
            "sum_w": sum_w,
            "n_valid": n_valid,
            "weight_ratio": ratio,
        }

    d_safe = torch.where(vm, d_obs_2ch, torch.ones_like(d_obs_2ch)).clamp(min=0.05)
    pred_masked = torch.where(vm, surr_mean, d_safe)
    rel_sq = ((pred_masked - d_safe) / d_safe) ** 2

    sum_w_t = (w * vm.to(dtype=w.dtype)).sum().clamp(min=1e-6)
    loss_cyc = (w * rel_sq).sum() / sum_w_t
    loss = loss_cyc

    loss_prior_val = 0.0
    if float(lambda_prior) > 0.0:
        r_d = compute_r_depth(vs_pred, domain=domain)
        loss = loss + float(lambda_prior) * r_d
        loss_prior_val = float(r_d.detach().item())

    loss_pen_val = 0.0
    if gate_mode == "penalty" and float(beta_penalty) > 0.0:
        sig_masked = torch.where(vm, sigma_rel, torch.zeros_like(sigma_rel))
        pen = (sig_masked.pow(2) * vm.to(dtype=sigma_rel.dtype)).sum() / max(float(n_valid), 1.0)
        loss = loss + float(beta_penalty) * pen
        loss_pen_val = float(pen.detach().item())

    return loss, {
        "skipped": False,
        "sum_w": sum_w,
        "n_valid": n_valid,
        "weight_ratio": ratio,
        "loss_cyc": float(loss_cyc.detach().item()),
        "loss_prior": loss_prior_val,
        "loss_penalty": loss_pen_val,
        "loss_total": float(loss.detach().item()),
    }


def compute_source_tau_quantile(
    model: AdaptedOpenSWITransformer,
    ens: SurrogateEnsemble,
    val_disp: np.ndarray,
    slice_dim: int,
    quantile: float = 0.90,
    period_range: Optional[Tuple[float, float]] = None,
    max_samples: int = 2048,
    device: str = "cpu",
    batch_size: int = 512,
    remove_group: bool = False,
    input_mode: str = "raw",
    train_grid: Optional[np.ndarray] = None,
) -> float:
    """Calibrate dimensionless gate threshold ``tau`` from source-validation surrogate uncertainty."""
    n_use = min(int(max_samples), len(val_disp))
    d_sub = val_disp[:n_use]
    per = d_sub[:, 0, :].astype(np.float32)
    ph = d_sub[:, 1, :].astype(np.float32)
    gp = d_sub[:, 2, :].astype(np.float32)

    om = np.stack([(per > 0) & (ph > 0), (per > 0) & (gp > 0)], axis=-1)
    if input_mode == "aligned":
        if train_grid is None:
            raise ValueError("input_mode='aligned' requires train_grid")
        x_in, pad_in = prepare_aligned_input(
            per, ph, gp, obs_mask=om, heldout_mask=None, train_grid=train_grid, remove_group=remove_group
        )
    elif input_mode == "raw":
        x_in, pad_in = prepare_openswi_input(per, ph, gp, obs_mask=om, remove_group=remove_group)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode!r}")

    g_min = float(ens.members[0].grid_dense[0].item())
    per_q = np.where(per > 0, per, g_min).astype(np.float32)
    t_lo = float(period_range[0]) if period_range is not None else -float("inf")
    t_hi = float(period_range[1]) if period_range is not None else float("inf")

    model.eval().to(device)
    ens.eval().to(device)
    sig_vals = []
    with torch.inference_mode():
        for s in range(0, n_use, batch_size):
            e = min(s + batch_size, n_use)
            xb = torch.from_numpy(x_in[s:e]).to(device)
            mb = torch.from_numpy(pad_in[s:e]).to(device)
            pq = torch.from_numpy(per_q[s:e]).to(device)
            vs_b = model(xb, mb)[:, :slice_dim]
            _, _, sr = ens.predict_with_uncertainty(vs_b, pq)
            in_band = (per[s:e] >= t_lo) & (per[s:e] <= t_hi)
            vm0 = (per[s:e] > 0) & (ph[s:e] > 0) & in_band
            if np.any(vm0):
                sig_vals.append(sr[:, 0, :].cpu().numpy()[vm0])
            if not remove_group:
                vm1 = (per[s:e] > 0) & (gp[s:e] > 0) & in_band
                if np.any(vm1):
                    sig_vals.append(sr[:, 1, :].cpu().numpy()[vm1])

    all_sig = np.concatenate(sig_vals) if sig_vals else np.array([0.01], dtype=np.float32)
    return float(np.quantile(all_sig, float(quantile)))


def _infer_adapted_tensors(
    adapted: nn.Module,
    x_t: torch.Tensor,
    pad_t: torch.Tensor,
    slice_dim: int,
    infer_batch_size: int = 512,
) -> np.ndarray:
    adapted.eval()
    n_sta = len(x_t)
    vs_preds = []
    with torch.inference_mode():
        for s in range(0, n_sta, int(infer_batch_size)):
            e = min(s + int(infer_batch_size), n_sta)
            out = adapted(x_t[s:e], pad_t[s:e])[:, :slice_dim]
            all_m = pad_t[s:e].all(dim=1)
            if torch.any(all_m):
                out[all_m] = 0.0
            vs_preds.append(out.cpu().numpy())
    return np.concatenate(vs_preds, axis=0)


def run_gated_cycle_tta(
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
    scope: str = "norm_adapter",
    gate_mode: str = "soft",
    gate_granularity: str = "point",
    tau: float = 0.02,
    lambda_prior: float = 0.05,
    beta_penalty: float = 0.0,
    min_valid_weight_ratio: float = 0.10,
    input_mode: str = "aligned",
    train_grid: Optional[np.ndarray] = None,
    num_steps: int = 50,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
    remove_group: bool = False,
    reset_backbone_copy: bool = True,
    snapshot_steps: Optional[Sequence[int]] = None,
    infer_batch_size: int = 512,
) -> Tuple[AdaptedOpenSWITransformer, np.ndarray, Dict[str, Any]]:
    """Run unlabeled fixed-step low-DOF adapter TTA using strictly kept observations."""
    torch.manual_seed(int(seed))
    rng = np.random.RandomState(int(seed))

    bb = copy.deepcopy(backbone_model) if reset_backbone_copy else backbone_model
    adapted = AdaptedOpenSWITransformer(bb).to(device)
    scope_info = adapted.configure_trainable_scope(scope)

    ens.eval().to(device)
    for p in ens.parameters():
        p.requires_grad_(False)

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

    g_min = float(ens.members[0].grid_dense[0].item())
    per_query_np = np.where(per_clean > 0.0, per_clean, g_min).astype(np.float32)
    d_obs_2ch_np = np.stack([ph_clean, gp_clean], axis=1).astype(np.float32)

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    x_t = torch.from_numpy(x_in).to(device)
    pad_t = torch.from_numpy(pad_in).to(device)
    per_t = torch.from_numpy(per_clean).to(device)
    pq_t = torch.from_numpy(per_query_np).to(device)
    dobs_t = torch.from_numpy(d_obs_2ch_np).to(device)
    km_t = torch.from_numpy(om_2ch).to(device)

    trainable_params = [p for p in adapted.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))

    n_sta = len(x_in)
    bs_eff = min(int(batch_size), n_sta)
    history: List[Dict[str, Any]] = []
    skipped_steps: List[Dict[str, Any]] = []
    snap_set = sorted({int(st) for st in snapshot_steps}) if snapshot_steps is not None else []
    snapshots: Dict[int, np.ndarray] = {}
    if 0 in snap_set:
        snapshots[0] = _infer_adapted_tensors(adapted, x_t, pad_t, slice_dim, max(bs_eff, int(infer_batch_size)))

    t_adapt0 = time.perf_counter()
    for step in range(1, int(num_steps) + 1):
        adapted.train()
        idx_np = np.arange(n_sta, dtype=np.int64) if n_sta <= bs_eff else rng.choice(n_sta, size=bs_eff, replace=False)
        idx_t = torch.from_numpy(idx_np).to(device)

        optimizer.zero_grad(set_to_none=True)
        vs_b = adapted(x_t[idx_t], pad_t[idx_t])[:, :slice_dim]
        mn, _, sr = ens.predict_with_uncertainty(vs_b, pq_t[idx_t])

        loss, s_info = compute_cycle_loss_from_outputs(
            vs_pred=vs_b,
            surr_mean=mn,
            sigma_rel=sr,
            periods=per_t[idx_t],
            d_obs_2ch=dobs_t[idx_t],
            kept_mask_2ch=km_t[idx_t],
            domain=domain,
            tau=tau,
            gate_mode=gate_mode,
            gate_granularity=gate_granularity,
            lambda_prior=lambda_prior,
            beta_penalty=beta_penalty,
            min_valid_weight_ratio=min_valid_weight_ratio,
        )
        s_info["step"] = int(step)
        if loss is None:
            skipped_steps.append(s_info)
        else:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip))
            optimizer.step()
        if step in snap_set:
            snapshots[step] = _infer_adapted_tensors(adapted, x_t, pad_t, slice_dim, max(bs_eff, int(infer_batch_size)))
        history.append(s_info)

    if use_cuda:
        torch.cuda.synchronize(device)
    t_adapt = time.perf_counter() - t_adapt0

    t_inf0 = time.perf_counter()
    vs_out = _infer_adapted_tensors(adapted, x_t, pad_t, slice_dim, max(bs_eff, int(infer_batch_size)))
    if use_cuda:
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter() - t_inf0
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0

    info = {
        "method": "low_dof_adapter",
        "domain": domain,
        "scope_info": scope_info,
        "gate_mode": gate_mode,
        "gate_granularity": gate_granularity,
        "input_mode": input_mode,
        "tau": float(tau),
        "lambda_prior": float(lambda_prior),
        "num_steps": int(num_steps),
        "num_executed_steps": int(num_steps - len(skipped_steps)),
        "num_skipped_steps": int(len(skipped_steps)),
        "skipped_step_ratio": float(len(skipped_steps) / max(int(num_steps), 1)),
        "history": history,
        "t_adapt_sec": float(t_adapt),
        "t_infer_pure_sec": float(t_inf),
        "t_total_per_station_ms": float((t_adapt + t_inf) * 1000.0 / max(n_sta, 1)),
        "peak_vram_gb": float(peak_bytes / (1024 ** 3)),
        **({"step_snapshots": snapshots} if snapshot_steps is not None else {}),
    }
    return adapted, vs_out, info
