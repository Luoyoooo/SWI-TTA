"""Unit tests for adapter parameter budget, stop-gradient gating, and Safe-TTA invariants."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest
import torch
import torch.nn as nn

from swi_tta.adaptation.baselines import run_output_space_refinement
from swi_tta.adaptation.gated_cycle import compute_gate_weights
from swi_tta.adaptation.safe_tta import run_safe_tta
from swi_tta.models.adapters import AdaptedOpenSWITransformer
from swi_tta.models.backbones import OpenSWITransformer
from swi_tta.models.surrogate import SurrogateEnsemble


class _ToySurrogateMember(nn.Module):
    """Lightweight differentiable surrogate member for fast CPU unit tests."""

    def __init__(self, vs_dim: int, grid_dense: np.ndarray, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("grid_dense", torch.from_numpy(grid_dense.astype(np.float32)))
        self.proj = nn.Linear(vs_dim, 2)
        with torch.no_grad():
            self.proj.weight.fill_(scale / max(vs_dim, 1))
            self.proj.bias.fill_(0.25)

    def forward(self, vs: torch.Tensor, query_periods: torch.Tensor) -> torch.Tensor:
        base = self.proj(vs).unsqueeze(-1)  # (B, 2, 1)
        t_term = 0.08 * query_periods.unsqueeze(1)
        return torch.clamp(base + t_term, min=0.1)


def _make_toy_ensemble(vs_dim: int = 20) -> SurrogateEnsemble:
    grid = np.linspace(0.25, 2.0, 24, dtype=np.float32)
    members = [
        _ToySurrogateMember(vs_dim, grid, scale=0.85),
        _ToySurrogateMember(vs_dim, grid, scale=0.95),
        _ToySurrogateMember(vs_dim, grid, scale=1.05),
    ]
    return SurrogateEnsemble(members)


class _ToyBackbone(nn.Module):
    def __init__(self, out_dim: int = 20) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        base = torch.linspace(0.5, 1.4, self.out_dim, device=x.device).unsqueeze(0).expand(b, -1)
        mean_v = x[:, 1, :].clamp(min=0.0).mean(dim=1, keepdim=True)
        return base + 0.05 * mean_v


def test_adapter_parameter_budget_and_step0_identity() -> None:
    torch.manual_seed(0)
    backbone = OpenSWITransformer(
        model_dim=256,
        num_heads=8,
        num_layers=6,
        output_dim=70,
        scale_factor=4.5,
    ).eval()
    adapted = AdaptedOpenSWITransformer(backbone).eval()
    stats = adapted.configure_trainable_scope("norm_adapter")
    assert stats["trainable_pct_of_backbone"] < 5.0
    assert stats["adapter_pct_of_backbone"] < 5.0

    trainable = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    total_backbone = sum(p.numel() for p in backbone.parameters())
    assert trainable / total_backbone < 0.05

    x = torch.rand(4, 3, 25)
    pad = torch.zeros(4, 25, dtype=torch.bool)
    with torch.no_grad():
        y_base = backbone(x, pad)
        y_adp = adapted(x, pad)
    assert float(torch.max(torch.abs(y_base - y_adp)).item()) == 0.0


def test_compute_gate_weights_is_strictly_stop_gradient() -> None:
    sigma_rel = torch.tensor([[[0.01, 0.03, 0.05], [0.02, 0.04, 0.01]]], requires_grad=True)
    valid_mask = torch.ones_like(sigma_rel, dtype=torch.bool)
    w = compute_gate_weights(sigma_rel, valid_mask, tau=0.02, gate_mode="soft")
    assert w.requires_grad is False
    assert torch.all((w >= 0.0) & (w <= 1.0))


def test_output_space_refinement_is_batch_size_invariant() -> None:
    vs_dim = 20
    backbone = _ToyBackbone(out_dim=vs_dim)
    ens = _make_toy_ensemble(vs_dim=vs_dim)

    rng = np.random.default_rng(7)
    n_sta, n_per = 6, 12
    periods = np.broadcast_to(np.linspace(0.3, 1.5, n_per, dtype=np.float32), (n_sta, n_per)).copy()
    phase = (0.7 + 0.3 * periods + rng.normal(0.0, 0.02, size=(n_sta, n_per))).astype(np.float32)
    group = np.zeros_like(phase)
    obs_mask = np.ones((n_sta, n_per, 2), dtype=np.uint8)
    obs_mask[:, :, 1] = 0
    kept_mask = obs_mask.copy()
    grid = np.linspace(0.3, 1.5, n_per, dtype=np.float32)

    out_bs6, _ = run_output_space_refinement(
        backbone, ens, periods, phase, group, obs_mask, kept_mask,
        domain="shallow", slice_dim=vs_dim, input_mode="aligned", train_grid=grid,
        num_steps=10, batch_size=6, device="cpu", remove_group=True,
    )
    out_bs2, _ = run_output_space_refinement(
        backbone, ens, periods, phase, group, obs_mask, kept_mask,
        domain="shallow", slice_dim=vs_dim, input_mode="aligned", train_grid=grid,
        num_steps=10, batch_size=2, device="cpu", remove_group=True,
    )
    assert float(np.max(np.abs(out_bs6 - out_bs2))) < 1e-6


def test_safe_tta_invariants_step0_zres_trust_revert_and_dc_spy() -> None:
    vs_dim = 20
    depths = np.linspace(0.0, 0.95, vs_dim, dtype=np.float64)
    ens = _make_toy_ensemble(vs_dim=vs_dim)
    n_sta, n_per = 3, 10
    periods = np.broadcast_to(np.linspace(0.3, 1.2, n_per, dtype=np.float32), (n_sta, n_per)).copy()
    phase = (0.8 + 0.2 * periods).astype(np.float32)  # z_res = 1.04 * 1.2 / 3 = 0.416 km
    group = np.zeros_like(phase)
    obs_mask = np.ones((n_sta, n_per, 2), dtype=np.uint8)
    obs_mask[:, :, 1] = 0
    kept_mask = obs_mask.copy()
    vs_init = np.broadcast_to(np.linspace(0.6, 1.4, vs_dim, dtype=np.float32), (n_sta, vs_dim)).copy()

    recorded_dcs: List[float] = []

    class _SpyPhaseDispersion:
        def __init__(self, thick: Any, vp: Any, vs: Any, rho: Any, dc: float = 0.005) -> None:
            self.dc = float(dc)
            recorded_dcs.append(float(dc))
            self._vs = np.asarray(vs, dtype=np.float64)

        def __call__(self, t_arr: np.ndarray, mode: int = 0, wave: str = "rayleigh") -> Any:
            # Returns velocity close to 0.8 + 0.2*T scaled slightly by mean Vs
            scale = float(np.mean(self._vs)) / 1.0
            return SimpleNamespace(period=np.asarray(t_arr, dtype=np.float64), velocity=(0.8 + 0.2 * np.asarray(t_arr)) * scale)

    # 1. Step-0 discrepancy stop -> exact identity (0.0 diff) & explicit dc forwarded
    vs_step0, info_step0 = run_safe_tta(
        vs_init, ens, depths, periods, phase, group, obs_mask, kept_mask, domain="shallow",
        alpha=100.0, sigma_hat_d=1.0, delta_trust=0.2, dc=2.5e-5, num_steps=10, check_interval=5,
        disba_workers=1, phase_dispersion_cls=_SpyPhaseDispersion,
    )
    assert float(np.max(np.abs(vs_step0 - vs_init))) == 0.0
    assert info_step0["step0_stop_rate"] == pytest.approx(1.0)
    assert len(recorded_dcs) > 0 and all(abs(d - 2.5e-5) < 1e-12 for d in recorded_dcs)

    # 2. z > z_res freeze and trust-region clamping (disable exact verification so candidate is kept)
    vs_ref, info_ref = run_safe_tta(
        vs_init, ens, depths, periods, phase * 1.4, group, obs_mask, kept_mask, domain="shallow",
        alpha=0.0001, sigma_hat_d=0.001, delta_trust=0.05, dc=2.5e-5, num_steps=15, check_interval=5,
        disba_workers=1, enable_exact_verification=False, phase_dispersion_cls=_SpyPhaseDispersion,
    )
    z_res_kept = info_ref["z_res_kept"]
    for i in range(n_sta):
        deep_layers = depths > (z_res_kept[i] + 1e-9)
        assert np.any(deep_layers)
        assert float(np.max(np.abs(vs_ref[i, deep_layers] - vs_init[i, deep_layers]))) == 0.0
        # Relative trust region: |delta| <= delta_trust * |vs_init|
        rel_step = np.abs(vs_ref[i] - vs_init[i]) / np.maximum(np.abs(vs_init[i]), 1e-8)
        assert float(np.max(rel_step)) <= 0.05 + 1e-6

    # 3. Exact verification revert when final forward returns incomplete periods
    call_counter = {"n": 0}

    class _IncompleteFinalPhase:
        def __init__(self, thick: Any, vp: Any, vs: Any, rho: Any, dc: float = 0.005) -> None:
            self.dc = float(dc)

        def __call__(self, t_arr: np.ndarray, mode: int = 0, wave: str = "rayleigh") -> Any:
            call_counter["n"] += 1
            t = np.asarray(t_arr, dtype=np.float64)
            # First n_sta calls (step 0) succeed cleanly with high misfit; subsequent calls return incomplete periods
            if call_counter["n"] <= n_sta:
                return SimpleNamespace(period=t, velocity=0.5 + 0.1 * t)
            return SimpleNamespace(period=t[:2], velocity=0.8 + 0.2 * t[:2])

    vs_rev, info_rev = run_safe_tta(
        vs_init, ens, depths, periods, phase, group, obs_mask, kept_mask, domain="shallow",
        alpha=0.001, sigma_hat_d=0.001, delta_trust=0.2, dc=2.5e-5, num_steps=5, check_interval=5,
        disba_workers=1, enable_exact_verification=True, phase_dispersion_cls=_IncompleteFinalPhase,
    )
    assert info_rev["revert_rate"] == pytest.approx(1.0)
    assert float(np.max(np.abs(vs_rev - vs_init))) == 0.0
