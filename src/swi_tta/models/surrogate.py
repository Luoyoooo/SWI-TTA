"""Heterogeneous M=4 differentiable forward dispersion surrogate ensemble."""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_geom_grid(t_min: float, t_max: float, n: int) -> np.ndarray:
    grid = np.geomspace(t_min, t_max, n).astype(np.float32)
    grid[0] = np.float32(t_min)
    grid[-1] = np.float32(t_max)
    return grid


DEFAULT_SHALLOW_GRID: np.ndarray = _default_geom_grid(0.20, 10.00, 96)
DEFAULT_DEEP_GRID: np.ndarray = _default_geom_grid(1.00, 100.00, 96)

SURROGATE_ARCH_NAMES: Tuple[str, ...] = (
    "dilated_resnet",
    "se_pyramid",
    "unet1d",
    "depth_transformer",
)


def default_dense_grid(domain: str) -> np.ndarray:
    """Return the canonical 96-point logarithmic query period grid for a regime."""
    if domain == "shallow":
        return DEFAULT_SHALLOW_GRID.copy()
    if domain == "deep":
        return DEFAULT_DEEP_GRID.copy()
    raise ValueError(f"Unknown domain: {domain!r}")


def brocher_physics_features(vs: torch.Tensor, depth_1d: torch.Tensor) -> torch.Tensor:
    """Lift 1D Vs profile ``(B, D)`` into 7 differentiable physical input channels ``(B, 7, D)``."""
    vs_c = torch.clamp(vs, min=0.05, max=6.5)
    vp_b = 0.9409 + 2.0947 * vs_c - 0.8206 * (vs_c ** 2) + 0.2683 * (vs_c ** 3) - 0.0251 * (vs_c ** 4)
    deep_m = (depth_1d > 120.0).unsqueeze(0)
    vp = torch.where(deep_m, 1.79 * vs_c, vp_b)
    vp_c = torch.clamp(vp, min=0.2, max=12.0)
    rho = (
        1.6612 * vp_c
        - 0.4721 * (vp_c ** 2)
        + 0.0671 * (vp_c ** 3)
        - 0.0043 * (vp_c ** 4)
        + 0.000106 * (vp_c ** 5)
    )
    dvs = torch.zeros_like(vs)
    dvs[:, 1:] = vs[:, 1:] - vs[:, :-1]
    dvs2 = torch.zeros_like(vs)
    dvs2[:, 2:] = vs[:, 2:] - 2.0 * vs[:, 1:-1] + vs[:, :-2]
    z_max = max(float(depth_1d[-1]), 1.0)
    z_norm = (depth_1d / z_max).unsqueeze(0).expand_as(vs)
    z_log = (torch.log1p(depth_1d) / float(np.log1p(z_max))).unsqueeze(0).expand_as(vs)
    return torch.stack([vs, vp, rho, dvs, dvs2, z_norm, z_log], dim=1)


def catmull_rom_interp(grid_t: torch.Tensor, y_grid: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
    """Differentiable C1 Catmull-Rom cubic spline interpolation onto arbitrary query periods."""
    g_len = grid_t.shape[0]
    tq = torch.clamp(t_query, min=float(grid_t[0]), max=float(grid_t[-1]))
    idx = torch.searchsorted(grid_t, tq).clamp(1, g_len - 1)
    t0 = grid_t[idx - 1]
    t1 = grid_t[idx]
    u = (tq - t0) / (t1 - t0).clamp(min=1e-8)

    im1 = (idx - 2).clamp(0, g_len - 1)
    i0 = idx - 1
    i1 = idx
    ip2 = (idx + 1).clamp(0, g_len - 1)

    outs = []
    for ch in (0, 1):
        yc = y_grid[:, ch, :]
        p_m1 = torch.gather(yc, 1, im1)
        p_0 = torch.gather(yc, 1, i0)
        p_1 = torch.gather(yc, 1, i1)
        p_2 = torch.gather(yc, 1, ip2)
        u2 = u * u
        val = 0.5 * (
            (2.0 * p_0)
            + (-p_m1 + p_1) * u
            + (2.0 * p_m1 - 5.0 * p_0 + 4.0 * p_1 - p_2) * u2
            + (-p_m1 + 3.0 * p_0 - 3.0 * p_1 + p_2) * (u2 * u)
        )
        outs.append(val)
    return torch.stack(outs, dim=1)


class _ResBlock1D(nn.Module):
    def __init__(self, c: int, k: int = 5, d: int = 1) -> None:
        super().__init__()
        pad = (k - 1) * d // 2
        self.net = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv1d(c, c, k, padding=pad, dilation=d),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv1d(c, c, k, padding=pad, dilation=d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _MultiScaleSEBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        c3 = c // 3
        c_last = c - 2 * c3
        self.norm = nn.GroupNorm(8, c)
        self.b1 = nn.Conv1d(c, c3, 3, padding=1)
        self.b2 = nn.Conv1d(c, c3, 7, padding=3)
        self.b3 = nn.Conv1d(c, c_last, 15, padding=7)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c, c // 4),
            nn.GELU(),
            nn.Linear(c // 4, c),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv1d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm(x))
        m = F.gelu(torch.cat([self.b1(h), self.b2(h), self.b3(h)], dim=1))
        w = self.se(m).unsqueeze(-1)
        return x + self.proj(m * w)


class _PeriodGridRefiner(nn.Module):
    def __init__(self, in_dim: int, grid_len: int) -> None:
        super().__init__()
        self.grid_len = int(grid_len)
        self.proj = nn.Linear(in_dim, 16 * self.grid_len)
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 2, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.conv(self.proj(z).view(-1, 16, self.grid_len))


def _make_dense_head(in_c: int, feat_len: int, grid_len: int, hidden: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.GroupNorm(8, in_c),
        nn.GELU(),
        nn.Conv1d(in_c, 32, kernel_size=1),
        nn.GELU(),
        nn.Flatten(),
        nn.Linear(feat_len * 32, hidden),
        nn.GELU(),
        _PeriodGridRefiner(hidden, grid_len),
    )


class _BaseGridSurrogate(nn.Module):
    def __init__(self, depth_1d: np.ndarray, grid_dense: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("depth_1d", torch.tensor(depth_1d, dtype=torch.float32))
        self.register_buffer("grid_dense", torch.tensor(grid_dense, dtype=torch.float32))
        self.grid_len = int(len(grid_dense))
        self.depth_len = int(len(depth_1d))

    def _encode_raw_grid(self, vs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_grid(self, vs: torch.Tensor) -> torch.Tensor:
        raw = self._encode_raw_grid(vs)
        return F.softplus(raw.view(-1, 2, self.grid_len)) + 0.02

    def forward(self, vs: torch.Tensor, query_periods: torch.Tensor) -> torch.Tensor:
        return catmull_rom_interp(self.grid_dense, self.forward_grid(vs), query_periods)


class SurrogateDilatedResNet1D(_BaseGridSurrogate):
    """Multi-scale dilated 1D ResNet forward surrogate."""

    def __init__(self, depth_1d: np.ndarray, grid_dense: np.ndarray, width: int = 160) -> None:
        super().__init__(depth_1d, grid_dense)
        feat_len = self.depth_len if self.depth_len <= 80 else 75
        self.pool = nn.Identity() if self.depth_len <= 80 else nn.AdaptiveAvgPool1d(75)
        self.stem = nn.Conv1d(7, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            _ResBlock1D(width, 3, 1),
            _ResBlock1D(width, 5, 1),
            _ResBlock1D(width, 5, 2),
            _ResBlock1D(width, 5, 4),
            _ResBlock1D(width, 5, 8),
            _ResBlock1D(width, 3, 1),
        )
        self.head = _make_dense_head(width, feat_len, self.grid_len)

    def _encode_raw_grid(self, vs: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.blocks(self.stem(brocher_physics_features(vs, self.depth_1d))))
        return self.head(h)


class SurrogateSEPyramid1D(_BaseGridSurrogate):
    """Multi-kernel pyramid with squeeze-and-excitation attention forward surrogate."""

    def __init__(self, depth_1d: np.ndarray, grid_dense: np.ndarray, width: int = 168) -> None:
        super().__init__(depth_1d, grid_dense)
        feat_len = self.depth_len if self.depth_len <= 80 else 75
        self.pool = nn.Identity() if self.depth_len <= 80 else nn.AdaptiveAvgPool1d(75)
        self.stem = nn.Conv1d(7, width, kernel_size=5, padding=2)
        self.blocks = nn.Sequential(*[_MultiScaleSEBlock(width) for _ in range(5)])
        self.head = _make_dense_head(width, feat_len, self.grid_len)

    def _encode_raw_grid(self, vs: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.blocks(self.stem(brocher_physics_features(vs, self.depth_1d))))
        return self.head(h)


class SurrogateUNet1D(_BaseGridSurrogate):
    """1D encoder-decoder UNet forward surrogate."""

    def __init__(self, depth_1d: np.ndarray, grid_dense: np.ndarray, width: int = 128) -> None:
        super().__init__(depth_1d, grid_dense)
        feat_len = self.depth_len if self.depth_len <= 80 else 75
        self.pool = nn.Identity() if self.depth_len <= 80 else nn.AdaptiveAvgPool1d(75)
        self.enc1 = nn.Sequential(nn.Conv1d(7, width, 5, padding=2), _ResBlock1D(width, 5, 1))
        self.down1 = nn.Conv1d(width, width, 4, stride=2, padding=1)
        self.enc2 = _ResBlock1D(width, 5, 2)
        self.down2 = nn.Conv1d(width, width * 2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(_ResBlock1D(width * 2, 3, 1), _ResBlock1D(width * 2, 3, 2))
        self.up2 = nn.Conv1d(width * 2, width, 1)
        self.dec2 = _ResBlock1D(width, 5, 1)
        self.dec1 = _ResBlock1D(width, 5, 1)
        self.head = _make_dense_head(width, feat_len, self.grid_len)

    def _encode_raw_grid(self, vs: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(brocher_physics_features(vs, self.depth_1d))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        u2 = self.dec2(F.interpolate(self.up2(m), size=e2.shape[-1], mode="linear", align_corners=False) + e2)
        u1 = self.dec1(F.interpolate(u2, size=e1.shape[-1], mode="linear", align_corners=False) + e1)
        return self.head(self.pool(u1))


class SurrogateDepthTransformer(_BaseGridSurrogate):
    """Convolutional stem plus multi-head depth-axis Transformer forward surrogate."""

    def __init__(self, depth_1d: np.ndarray, grid_dense: np.ndarray, d_model: int = 144) -> None:
        super().__init__(depth_1d, grid_dense)
        seq_len = self.depth_len if self.depth_len <= 80 else 75
        self.pool_tr = nn.Identity() if self.depth_len <= 80 else nn.AdaptiveAvgPool1d(75)
        self.stem = nn.Sequential(nn.Conv1d(7, d_model, 5, padding=2), _ResBlock1D(d_model, 5, 1))
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=6,
            dim_feedforward=d_model * 3,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(seq_len * 32, 512),
            nn.GELU(),
            _PeriodGridRefiner(512, self.grid_len),
        )

    def _encode_raw_grid(self, vs: torch.Tensor) -> torch.Tensor:
        h = self.pool_tr(self.stem(brocher_physics_features(vs, self.depth_1d))).permute(0, 2, 1) + self.pos
        h = self.tr(h)
        return self.head(h)


def build_surrogate(arch: str, depth_1d: np.ndarray, grid_dense: np.ndarray) -> nn.Module:
    """Factory for individual forward surrogate architectures."""
    if arch == "dilated_resnet":
        return SurrogateDilatedResNet1D(depth_1d, grid_dense)
    if arch == "se_pyramid":
        return SurrogateSEPyramid1D(depth_1d, grid_dense)
    if arch == "unet1d":
        return SurrogateUNet1D(depth_1d, grid_dense)
    if arch == "depth_transformer":
        return SurrogateDepthTransformer(depth_1d, grid_dense)
    raise ValueError(f"Unknown surrogate architecture: {arch!r}")


class SurrogateEnsemble(nn.Module):
    """M-member differentiable forward surrogate ensemble returning mean and relative uncertainty."""

    def __init__(self, members: Sequence[nn.Module], eps_rel: float = 0.05) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.eps_rel = float(eps_rel)

    def forward_all(self, vs: torch.Tensor, query_periods: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(vs, query_periods) for m in self.members], dim=0)

    def predict_with_uncertainty(
        self,
        vs: torch.Tensor,
        query_periods: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_preds = self.forward_all(vs, query_periods)
        mean = torch.mean(all_preds, dim=0)
        var_abs = torch.var(all_preds, dim=0, unbiased=False)
        sigma_rel = torch.sqrt(var_abs + 1e-12) / torch.clamp(mean, min=self.eps_rel)
        return mean, var_abs, sigma_rel

    def predict_batches_np(
        self,
        vs_np: np.ndarray,
        periods_np: np.ndarray,
        device: str = "cpu",
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.eval()
        self.to(device)
        n = len(vs_np)
        means, vars_abs, sigs_rel = [], [], []
        g_min = float(self.members[0].grid_dense[0].item())
        per_safe = np.where(periods_np > 0, periods_np, g_min).astype(np.float32)

        with torch.inference_mode():
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                vb = torch.from_numpy(vs_np[s:e].astype(np.float32)).to(device)
                pb = torch.from_numpy(per_safe[s:e]).to(device)
                mn, va, sr = self.predict_with_uncertainty(vb, pb)
                means.append(mn.cpu().numpy())
                vars_abs.append(va.cpu().numpy())
                sigs_rel.append(sr.cpu().numpy())

        return (
            np.concatenate(means, axis=0),
            np.concatenate(vars_abs, axis=0),
            np.concatenate(sigs_rel, axis=0),
        )


def save_surrogate_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    arch: str,
    domain: str,
    depth_1d: np.ndarray,
    grid_dense: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": arch,
            "domain": domain,
            "depth_1d": np.asarray(depth_1d, dtype=np.float32),
            "grid_dense": np.asarray(grid_dense, dtype=np.float32),
            "meta": meta,
        },
        ckpt_path,
    )


def load_surrogate_member(ckpt_path: str, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_surrogate(payload["arch"], payload["depth_1d"], payload["grid_dense"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def load_surrogate_ensemble(ckpt_paths: Sequence[str], device: str = "cpu") -> SurrogateEnsemble:
    members = [load_surrogate_member(p, device=device)[0] for p in ckpt_paths]
    ens = SurrogateEnsemble(members)
    ens.to(device)
    ens.eval()
    return ens
