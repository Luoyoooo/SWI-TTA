"""Low-DOF residual adapters and feature-statistics calibrator for OpenSWI-Transformer."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from swi_tta.models.backbones import OpenSWITransformer


class _ResidualMLPAdapter(nn.Module):
    """Zero-initialized bottleneck residual adapter (exact identity at step 0)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, dim)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class _LogitResidualAdapter(nn.Module):
    """Zero-initialized pre-sigmoid logit residual adapter."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.net[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class FeatureStatsCalibrator(nn.Module):
    """Gradient-free running feature-statistics alignment for normalization-only TTA."""

    def __init__(
        self,
        mu_src: torch.Tensor,
        std_src: torch.Tensor,
        blend: float = 0.5,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.register_buffer("mu_src", mu_src.detach().clone())
        self.register_buffer("std_src", std_src.detach().clone().clamp(min=eps))
        self.register_buffer("mu_tgt", mu_src.detach().clone())
        self.register_buffer("std_tgt", std_src.detach().clone().clamp(min=eps))
        self.blend = float(blend)
        self.eps = float(eps)
        self.track_stats = False
        self.ema_momentum = 0.1

    def update_from_batch(self, feat: torch.Tensor) -> None:
        if feat.shape[0] < 2:
            return
        with torch.no_grad():
            b_mu = feat.mean(dim=0)
            b_std = feat.std(dim=0, unbiased=False).clamp(min=self.eps)
            m = self.ema_momentum
            self.mu_tgt.mul_(1.0 - m).add_(b_mu, alpha=m)
            self.std_tgt.mul_(1.0 - m).add_(b_std, alpha=m)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.track_stats:
            self.update_from_batch(feat)
        if self.blend <= 0.0:
            return feat
        aligned = ((feat - self.mu_tgt) / self.std_tgt.clamp(min=self.eps)) * self.std_src + self.mu_src
        return (1.0 - self.blend) * feat + self.blend * aligned


class AdaptedOpenSWITransformer(nn.Module):
    """Wrapper attaching <5%-parameter residual adapters to :class:`OpenSWITransformer`."""

    def __init__(
        self,
        backbone: OpenSWITransformer,
        r_in: int = 16,
        r_tr: int = 16,
        r_feat: int = 24,
        r_out: int = 24,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        first_conv = backbone.period_embedding[0]
        d_model = int(first_conv.out_channels)
        n_layers = len(backbone.transformer_encoder.layers)
        out_dim = int(backbone.fc_fuse[4].out_features)
        self.d_model = d_model
        self.n_layers = n_layers
        self.output_dim = out_dim

        if d_model >= 128:
            r_in = max(r_in, 24)
            r_tr = max(r_tr, 24)
            r_feat = max(r_feat, 32)
            r_out = max(r_out, 32)

        self.adapters = nn.ModuleDict(
            {
                "in_adapter": _ResidualMLPAdapter(d_model, r_in),
                "tr_adapters": nn.ModuleList([_ResidualMLPAdapter(d_model, r_tr) for _ in range(n_layers)]),
                "feat_adapter": _ResidualMLPAdapter(2 * d_model, r_feat),
                "logit_adapter": _LogitResidualAdapter(2 * d_model, r_out, out_dim),
            }
        )
        self.use_adapters: bool = True
        self.stats_calibrator: Optional[FeatureStatsCalibrator] = None

    @property
    def scale_factor(self) -> float:
        return self.backbone.scale_factor

    def encode_pooled_feature(self, input_data: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bb = self.backbone
        period_data = input_data[:, 0, :]
        phase_velocity = input_data[:, 1, :]
        group_velocity = input_data[:, 2, :]

        phase_solid_mask = (phase_velocity > 0).unsqueeze(1)
        group_solid_mask = (group_velocity > 0).unsqueeze(1)

        period_emb = bb.period_embedding(period_data.unsqueeze(1))
        phase_emb = bb.phase_velocity_encoding(phase_velocity.unsqueeze(1))
        group_emb = bb.group_velocity_encoding(group_velocity.unsqueeze(1))

        phase_emb = bb.local_feature_extraction_phaseVelocity(phase_emb, phase_solid_mask)
        group_emb = bb.local_feature_extraction_groupVelocity(group_emb, group_solid_mask)

        combined = period_emb + phase_emb + group_emb
        fused = combined.permute(0, 2, 1)

        valid_tok = (~mask).unsqueeze(-1).to(fused.dtype) if mask is not None else None
        if self.use_adapters:
            in_delta = self.adapters["in_adapter"](fused)
            if valid_tok is not None:
                in_delta = in_delta * valid_tok
            fused = fused + in_delta

        h = fused
        for idx, layer in enumerate(bb.transformer_encoder.layers):
            h = layer(h, src_key_padding_mask=mask)
            if self.use_adapters:
                tr_delta = self.adapters["tr_adapters"][idx](h)
                if valid_tok is not None:
                    tr_delta = tr_delta * valid_tok
                h = h + tr_delta

        if bb.transformer_encoder.norm is not None:
            h = bb.transformer_encoder.norm(h)

        trans_c = h.permute(0, 2, 1)
        out_avg = bb.global_pooling(trans_c)
        out_max = bb.max_pooling(trans_c)
        feat = torch.cat([out_avg, out_max], dim=1).squeeze(-1)

        if self.stats_calibrator is not None:
            feat = self.stats_calibrator(feat)

        if self.use_adapters:
            feat = feat + self.adapters["feat_adapter"](feat)
        return feat

    def forward(self, input_data: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.encode_pooled_feature(input_data, mask=mask)
        bb = self.backbone
        z = bb.fc_fuse[0](feat)
        z = bb.fc_fuse[1](z)
        z = bb.fc_fuse[2](z)
        z = bb.fc_fuse[3](z)
        logits = bb.fc_fuse[4](z)
        if self.use_adapters:
            logits = logits + self.adapters["logit_adapter"](feat)
        return bb.fc_fuse[5](logits) * bb.scale_factor

    def configure_trainable_scope(self, scope: str) -> Dict[str, int | float | str]:
        """Configure trainable parameter subset and verify adapter size < 5% of backbone."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.adapters.parameters():
            p.requires_grad_(False)

        if scope == "norm_adapter":
            self.use_adapters = True
            for p in self.adapters.parameters():
                p.requires_grad_(True)
            for mod in self.backbone.modules():
                if isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                    for p in mod.parameters():
                        p.requires_grad_(True)
        elif scope == "norm_only":
            self.use_adapters = False
            for mod in self.backbone.modules():
                if isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                    for p in mod.parameters():
                        p.requires_grad_(True)
        elif scope == "adapter_only":
            self.use_adapters = True
            for p in self.adapters.parameters():
                p.requires_grad_(True)
        elif scope == "full_params":
            self.use_adapters = False
            for p in self.backbone.parameters():
                p.requires_grad_(True)
        elif scope == "frozen":
            self.use_adapters = False
        else:
            raise ValueError(f"Unknown trainable scope: {scope!r}")

        bb_total = int(sum(p.numel() for p in self.backbone.parameters()))
        ad_total = int(sum(p.numel() for p in self.adapters.parameters()))
        norm_total = int(
            sum(
                p.numel()
                for mod in self.backbone.modules()
                if isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm))
                for p in mod.parameters()
            )
        )
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))

        if ad_total >= 0.05 * bb_total:
            raise AssertionError(f"Adapter parameter count {ad_total} exceeds 5% of backbone ({bb_total})")
        if scope == "norm_adapter" and trainable >= 0.05 * bb_total:
            raise AssertionError(f"Trainable parameter count {trainable} exceeds 5% of backbone ({bb_total})")

        return {
            "scope": scope,
            "backbone_params": bb_total,
            "adapter_params": ad_total,
            "norm_params": norm_total,
            "trainable_params": trainable,
            "adapter_pct_of_backbone": float(100.0 * ad_total / bb_total),
            "trainable_pct_of_backbone": float(100.0 * trainable / bb_total),
        }


def snapshot_frozen_parameters(modules: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
    """Capture an exact CPU copy of all ``requires_grad=False`` parameters."""
    snap: Dict[str, torch.Tensor] = {}
    for m_name, mod in modules.items():
        for p_name, p in mod.named_parameters():
            if not p.requires_grad:
                snap[f"{m_name}.{p_name}"] = p.detach().cpu().clone()
    return snap


def verify_frozen_parameters_unchanged(
    modules: Dict[str, nn.Module],
    snap: Dict[str, torch.Tensor],
) -> Tuple[bool, List[str]]:
    """Verify bit-exact preservation of frozen parameters after adaptation."""
    changed: List[str] = []
    for m_name, mod in modules.items():
        for p_name, p in mod.named_parameters():
            key = f"{m_name}.{p_name}"
            if key in snap and not torch.equal(p.detach().cpu(), snap[key]):
                changed.append(key)
    return len(changed) == 0, changed
