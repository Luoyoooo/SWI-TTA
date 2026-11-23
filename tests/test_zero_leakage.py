"""Zero-leakage tests verifying held-out periods cannot influence alignment or adaptation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from swi_tta.adaptation.baselines import run_output_space_refinement, run_zeroshot_inference
from swi_tta.adaptation.safe_tta import create_stop_period_split, run_holdout_early_stopping, run_safe_tta
from swi_tta.data.alignment import prepare_aligned_input
from swi_tta.models.surrogate import SurrogateEnsemble


class _ToySurrogateMember(nn.Module):
    def __init__(self, vs_dim: int, grid_dense: np.ndarray, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("grid_dense", torch.from_numpy(grid_dense.astype(np.float32)))
        self.proj = nn.Linear(vs_dim, 2)
        with torch.no_grad():
            self.proj.weight.fill_(scale / max(vs_dim, 1))
            self.proj.bias.fill_(0.25)

    def forward(self, vs: torch.Tensor, query_periods: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.proj(vs).unsqueeze(-1) + 0.08 * query_periods.unsqueeze(1), min=0.1)


class _ToyBackbone(nn.Module):
    def __init__(self, out_dim: int = 20) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        base = torch.linspace(0.5, 1.4, self.out_dim, device=x.device).unsqueeze(0).expand(b, -1)
        mean_v = x[:, 1, :].clamp(min=0.0).mean(dim=1, keepdim=True)
        return base + 0.05 * mean_v


class _FastPhaseStub:
    def __init__(self, thick: Any, vp: Any, vs: Any, rho: Any, dc: float = 0.005) -> None:
        self.dc = float(dc)
        self._scale = float(np.mean(vs))

    def __call__(self, t_arr: np.ndarray, mode: int = 0, wave: str = "rayleigh") -> Any:
        t = np.asarray(t_arr, dtype=np.float64)
        return SimpleNamespace(period=t, velocity=(0.7 + 0.25 * t) * self._scale)


def test_corrupting_heldout_periods_leaves_all_methods_bit_exact() -> None:
    vs_dim = 20
    n_sta, n_per = 4, 15
    depths = np.linspace(0.0, 0.95, vs_dim, dtype=np.float64)
    grid = np.linspace(0.25, 2.0, 20, dtype=np.float32)

    periods = np.broadcast_to(np.linspace(0.30, 1.80, n_per, dtype=np.float32), (n_sta, n_per)).copy()
    phase_clean = (0.75 + 0.25 * periods).astype(np.float32)
    group_clean = np.zeros_like(phase_clean)

    obs_mask = np.ones((n_sta, n_per, 2), dtype=np.uint8)
    obs_mask[:, :, 1] = 0
    ho_indices = [3, 7, 11]
    ho_mask = np.zeros_like(obs_mask)
    ho_mask[:, ho_indices, :] = 1
    kept_mask = (obs_mask & (1 - ho_mask)).astype(np.uint8)

    # Corrupt held-out period entries with extreme outliers and NaNs
    phase_corrupt = phase_clean.copy()
    phase_corrupt[:, 3] = 9999.0
    phase_corrupt[:, 7] = -500.0
    phase_corrupt[:, 11] = np.nan

    # 1. Input alignment
    x1, p1 = prepare_aligned_input(periods, phase_clean, group_clean, obs_mask, ho_mask, grid, remove_group=True)
    x2, p2 = prepare_aligned_input(periods, phase_corrupt, group_clean, obs_mask, ho_mask, grid, remove_group=True)
    assert float(np.max(np.abs(x1 - x2))) == 0.0
    assert np.array_equal(p1, p2)

    backbone = _ToyBackbone(out_dim=vs_dim)
    ens = SurrogateEnsemble([_ToySurrogateMember(vs_dim, grid, s) for s in (0.9, 1.0, 1.1)])

    # 2. Input-Aligned Zero-Shot
    zs1, _ = run_zeroshot_inference(
        backbone, periods, phase_clean, group_clean, obs_mask, kept_mask,
        slice_dim=vs_dim, input_mode="aligned", train_grid=grid, remove_group=True,
    )
    zs2, _ = run_zeroshot_inference(
        backbone, periods, phase_corrupt, group_clean, obs_mask, kept_mask,
        slice_dim=vs_dim, input_mode="aligned", train_grid=grid, remove_group=True,
    )
    assert float(np.max(np.abs(zs1 - zs2))) == 0.0

    # 3. Output-Space Refinement
    ref1, _ = run_output_space_refinement(
        backbone, ens, periods, phase_clean, group_clean, obs_mask, kept_mask,
        domain="shallow", slice_dim=vs_dim, input_mode="aligned", train_grid=grid,
        num_steps=10, remove_group=True,
    )
    ref2, _ = run_output_space_refinement(
        backbone, ens, periods, phase_corrupt, group_clean, obs_mask, kept_mask,
        domain="shallow", slice_dim=vs_dim, input_mode="aligned", train_grid=grid,
        num_steps=10, remove_group=True,
    )
    assert float(np.max(np.abs(ref1 - ref2))) == 0.0

    # 4. Pre-Registered Safe-TTA
    s1, _ = run_safe_tta(
        zs1, ens, depths, periods, phase_clean, group_clean, obs_mask, kept_mask,
        domain="shallow", alpha=1.25, sigma_hat_d=0.01, delta_trust=0.2, dc=2.5e-5,
        num_steps=10, check_interval=5, disba_workers=1, phase_dispersion_cls=_FastPhaseStub,
    )
    s2, _ = run_safe_tta(
        zs2, ens, depths, periods, phase_corrupt, group_clean, obs_mask, kept_mask,
        domain="shallow", alpha=1.25, sigma_hat_d=0.01, delta_trust=0.2, dc=2.5e-5,
        num_steps=10, check_interval=5, disba_workers=1, phase_dispersion_cls=_FastPhaseStub,
    )
    assert float(np.max(np.abs(s1 - s2))) == 0.0

    # 5. Holdout ES internal split strictly disjoint from evaluation held-out periods
    train_k, stop_k = create_stop_period_split(obs_mask, kept_mask, compute_group=False, stop_ratio=0.20)
    assert not np.any(stop_k & (ho_mask > 0))
    assert not np.any(train_k & stop_k)
    assert np.array_equal(train_k | stop_k, kept_mask > 0)

    h1, _ = run_holdout_early_stopping(
        zs1, ens, depths, periods, phase_clean, group_clean, obs_mask, kept_mask,
        domain="shallow", dc=2.5e-5, num_steps=10, check_interval=5, disba_workers=1,
        phase_dispersion_cls=_FastPhaseStub,
    )
    h2, _ = run_holdout_early_stopping(
        zs2, ens, depths, periods, phase_corrupt, group_clean, obs_mask, kept_mask,
        domain="shallow", dc=2.5e-5, num_steps=10, check_interval=5, disba_workers=1,
        phase_dispersion_cls=_FastPhaseStub,
    )
    assert float(np.max(np.abs(h1 - h2))) == 0.0
