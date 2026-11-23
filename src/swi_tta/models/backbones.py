"""Surface-wave dispersion inversion backbones (OpenSWI-Transformer & DispFormer)."""

from __future__ import annotations

from collections import OrderedDict
import time
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv1d(nn.Conv1d):
    """Mask-normalized 1D convolution operating exclusively on valid period tokens."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["bias"] = False
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_masked = x * mask
        conv_out = super().forward(x_masked)
        with torch.no_grad():
            ones_kernel = torch.ones((1, 1, self.kernel_size[0]), device=x.device, dtype=x.dtype)
            valid_count = F.conv1d(
                mask.to(dtype=x.dtype),
                ones_kernel,
                bias=None,
                stride=self.stride[0],
                padding=self.padding[0],
                dilation=self.dilation[0],
            ).clamp(min=1e-6)
        return conv_out / valid_count


class LocalFeatureExtraction(nn.Module):
    """Three-layer masked local convolutional stem for phase or group dispersion curves."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.conv1 = MaskedConv1d(model_dim, model_dim, kernel_size=7, padding=3)
        self.conv2 = MaskedConv1d(model_dim, model_dim, kernel_size=5, padding=2)
        self.conv3 = MaskedConv1d(model_dim, model_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x, mask))
        x = self.relu(self.conv2(x, mask))
        x = self.relu(self.conv3(x, mask))
        return x


class OpenSWITransformer(nn.Module):
    """OpenSWI local-global Transformer inversion backbone (state-dict compatible)."""

    def __init__(
        self,
        model_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 3,
        output_dim: int = 70,
        scale_factor: float = 4.5,
    ) -> None:
        super().__init__()
        self.period_embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=model_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.phase_velocity_encoding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=model_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.group_velocity_encoding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=model_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.local_feature_extraction_phaseVelocity = LocalFeatureExtraction(model_dim=model_dim)
        self.local_feature_extraction_groupVelocity = LocalFeatureExtraction(model_dim=model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.fc_fuse = nn.Sequential(
            nn.Linear(2 * model_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Sigmoid(),
        )
        self.scale_factor = float(scale_factor)

    def forward(self, input_data: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        period_data = input_data[:, 0, :]
        phase_velocity = input_data[:, 1, :]
        group_velocity = input_data[:, 2, :]

        phase_solid_mask = (phase_velocity > 0).unsqueeze(1)
        group_solid_mask = (group_velocity > 0).unsqueeze(1)

        period_emb = self.period_embedding(period_data.unsqueeze(1))
        phase_emb = self.phase_velocity_encoding(phase_velocity.unsqueeze(1))
        group_emb = self.group_velocity_encoding(group_velocity.unsqueeze(1))

        phase_emb = self.local_feature_extraction_phaseVelocity(phase_emb, phase_solid_mask)
        group_emb = self.local_feature_extraction_groupVelocity(group_emb, group_solid_mask)

        combined = period_emb + phase_emb + group_emb
        fused = combined.permute(0, 2, 1)

        trans_out = self.transformer_encoder(fused, src_key_padding_mask=mask)
        trans_c = trans_out.permute(0, 2, 1)
        out_avg = self.global_pooling(trans_c)
        out_max = self.max_pooling(trans_c)
        feat = torch.cat([out_avg, out_max], dim=1).squeeze(-1)

        return self.fc_fuse(feat) * self.scale_factor


class PeriodPositionalEncoding(nn.Module):
    """Log-period sinusoidal positional encoding for DispFormer."""

    def __init__(self, model_dim: int, min_period: float = 0.005, max_period: float = 200.0) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.min_period = float(min_period)
        self.max_period = float(max_period)

    def forward(self, periods: torch.Tensor) -> torch.Tensor:
        device = periods.device
        p = periods.float()
        invalid_mask = (p <= 0) | torch.isnan(p)
        valid_p = p.clone()
        valid_p[invalid_mask] = self.min_period

        log_min = float(np.log(self.min_period))
        log_max = float(np.log(self.max_period))
        p_norm = (torch.log(valid_p) - log_min) / (log_max - log_min)
        position = p_norm.unsqueeze(-1)

        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, dtype=torch.float32, device=device)
            * -(float(np.log(10000.0)) / self.model_dim)
        )
        pos_enc = torch.zeros((p.shape[0], p.shape[1], self.model_dim), dtype=torch.float32, device=device)
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)
        pos_enc[invalid_mask] = 0.0
        return pos_enc


class DispFormerBackbone(nn.Module):
    """DispFormer Transformer inversion backbone (state-dict compatible)."""

    def __init__(
        self,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        output_dim: int = 400,
        scale_factor: float = 6.5,
    ) -> None:
        super().__init__()
        self.phase_encoding = nn.Linear(1, model_dim)
        self.group_encoding = nn.Linear(1, model_dim)
        self.period_position_encoding = PeriodPositionalEncoding(model_dim=model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.conv_fusion = nn.Conv1d(model_dim * 3, model_dim, kernel_size=3, padding=1)
        self.fc_fuse = nn.Linear(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.scale_factor = float(scale_factor)

    def forward(self, input_data: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        period_data = input_data[:, 0, :]
        phase_velocity = input_data[:, 1, :]
        group_velocity = input_data[:, 2, :]

        phase_emb = self.phase_encoding(phase_velocity.unsqueeze(-1))
        group_emb = self.group_encoding(group_velocity.unsqueeze(-1))
        period_emb = self.period_position_encoding(period_data)

        if mask is not None:
            phase_mask = phase_velocity <= 0
            group_mask = group_velocity <= 0
            phase_emb = phase_emb.masked_fill(phase_mask.unsqueeze(-1), 0.0)
            group_emb = group_emb.masked_fill(group_mask.unsqueeze(-1), 0.0)
            period_emb = period_emb.masked_fill(mask.unsqueeze(-1), 0.0)

        combined = torch.cat([period_emb, phase_emb, group_emb], dim=2).permute(0, 2, 1)
        fused = self.conv_fusion(combined)
        trans_out = self.transformer_encoder(fused.permute(2, 0, 1), src_key_padding_mask=mask)
        trans_out = trans_out.permute(1, 0, 2)
        time_fused = self.fc_fuse(trans_out)
        pooled = torch.mean(time_fused, dim=1)
        return self.fc_out(pooled) * self.scale_factor


def map_dispformer_official_400_to_301(vs_400: np.ndarray) -> np.ndarray:
    """Map pretrained DispFormer 400-layer output (0.5 km step) to 301-layer grid (1.0 km step).

    Averages adjacent 0.5 km layer pairs over 0-199 km and extends the bottom
    half-space velocity across 200-300 km.
    """
    arr = np.asarray(vs_400, dtype=np.float32)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 400:
        raise ValueError(f"Expected shape (N, 400) or (400,); got {arr.shape}")
    out = np.empty((arr.shape[0], 301), dtype=np.float32)
    out[:, :200] = 0.5 * (arr[:, 0:400:2] + arr[:, 1:400:2])
    out[:, 200:] = arr[:, -1:]
    return out[0] if squeeze else out


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    """Remove leading ``module.`` prefix from DataParallel state dictionaries."""
    cleaned: OrderedDict = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


def load_openswi_checkpoint(
    ckpt_path: str,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    output_dim: int,
    scale_factor: float,
    device: str = "cpu",
) -> OpenSWITransformer:
    """Instantiate and load an :class:`OpenSWITransformer` checkpoint."""
    model = OpenSWITransformer(
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_dim,
        scale_factor=scale_factor,
    )
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(strip_module_prefix(sd))
    model.to(device)
    model.eval()
    return model


def load_dispformer_checkpoint(
    ckpt_path: str,
    model_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 3,
    output_dim: int = 400,
    scale_factor: float = 6.5,
    device: str = "cpu",
) -> DispFormerBackbone:
    """Instantiate and load a :class:`DispFormerBackbone` checkpoint."""
    model = DispFormerBackbone(
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_dim,
        scale_factor=scale_factor,
    )
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(strip_module_prefix(sd))
    model.to(device)
    model.eval()
    return model


def prepare_openswi_input(
    period: np.ndarray,
    phase_vel: np.ndarray,
    group_vel: np.ndarray,
    obs_mask: Optional[np.ndarray] = None,
    heldout_mask: Optional[np.ndarray] = None,
    remove_group: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build raw ``(N, 3, P)`` dispersion tensor and boolean padding mask ``(N, P)``."""
    per = np.asarray(period, dtype=np.float32)
    ph = np.asarray(phase_vel, dtype=np.float32)
    gp = np.asarray(group_vel, dtype=np.float32)
    x = np.stack([per, ph, gp], axis=1).copy()

    if obs_mask is not None:
        om = np.asarray(obs_mask) > 0.5
        x[:, 1, :][~om[:, :, 0]] = -1.0
        x[:, 2, :][~om[:, :, 1]] = -1.0

    if heldout_mask is not None:
        hm = np.asarray(heldout_mask) > 0.5
        if hm.ndim == 3:
            x[:, 1, :][hm[:, :, 0]] = -1.0
            x[:, 2, :][hm[:, :, 1]] = -1.0
        else:
            x[:, 1, :][hm] = -1.0
            x[:, 2, :][hm] = -1.0

    if remove_group:
        x[:, 2, :] = -1.0

    x[x <= 0] = -1.0
    pad_mask = (x[:, 1, :] == -1.0) & (x[:, 2, :] == -1.0)
    return x, pad_mask


def predict_batches(
    model: nn.Module,
    input_data: np.ndarray,
    pad_mask: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> Tuple[np.ndarray, float, int]:
    """Run deterministic mini-batch inference and return ``(vs_pred, elapsed_sec, peak_bytes)``."""
    model.eval()
    model.to(device)
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    n = input_data.shape[0]
    outputs = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            xb = torch.from_numpy(input_data[s:e]).to(device, non_blocking=False)
            mb = torch.from_numpy(pad_mask[s:e]).to(device, non_blocking=False)
            out = model(xb, mb)
            all_masked = mb.all(dim=1)
            if torch.any(all_masked):
                out[all_masked] = 0.0
            outputs.append(out.detach().cpu().numpy())
    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0
    return np.concatenate(outputs, axis=0), float(elapsed), peak_bytes
