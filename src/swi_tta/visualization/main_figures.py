"""Main publication figures (Figures 1-9) and Graphical Abstract for C&G submission.

All figures are generated deterministically on CPU strictly from the compact
summary files in ``results_summary/``, with zero hardcoded local paths and zero
internal task/method identifiers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from swi_tta.visualization.style import (
    DOUBLE_COL_IN,
    DOUBLE_COL_MM,
    METHOD_STYLE,
    PALETTE,
    add_panel_label,
    apply_cg_style,
    mm_to_in,
    save_figure,
)


def _load_json(summary_dir: str, filename: str) -> Dict[str, Any]:
    with open(os.path.join(summary_dir, filename), "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_bin_labels(labels: Sequence[str]) -> List[str]:
    """Replace ASCII 'inf' with Unicode infinity symbol for publication axes."""
    return [str(lbl).replace("inf", "∞") for lbl in labels]


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_color: str = "#111111",
    fontsize_title: float = 7.4,
    fontsize_body: float = 6.5,
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=fontsize_title,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    ax.text(
        x + w / 2.0,
        y + h * 0.28,
        body,
        ha="center",
        va="center",
        fontsize=fontsize_body,
        color="#262626",
        linespacing=1.22,
        zorder=3,
    )


# ---------------------------------------------------------------------------
# Figure 1: Diagnostic Evaluation Framework & Four-Step Safe-TTA Protocol
# (Restrained geoscience editorial vector layout in true 1:1 millimetre space)
# ---------------------------------------------------------------------------

_F1_INK = "#111827"
_F1_TEXT = "#334155"
_F1_MUTED = "#64748B"
_F1_LINE = "#94A3B8"
_F1_BORDER = "#CBD5E1"
_F1_BAND = "#FAFAFB"
_F1_HDR = "#F1F5F9"
_F1_BADGE = "#334155"

_F1_NAVY = "#1F5A8A"
_F1_NAVY_BG = "#F2F6FA"
_F1_RUST = "#B8502A"
_F1_RUST_BG = "#FAF2EE"
_F1_SAGE = "#2E7D6F"
_F1_SAGE_BG = "#F1F7F5"


def _fmt_sci_tex(val: float) -> str:
    """Format a positive float as LaTeX mantissa x 10^{exp} for Figure 1 labels."""
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / (10.0 ** exp)
    return rf"{mant:.1f}\times 10^{{{exp}}}"


def _f1_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str = "#FFFFFF",
    ec: str = _F1_LINE,
    lw: float = 0.75,
    r: float = 1.8,
    z: int = 2,
) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=z,
        )
    )


def _f1_header_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    num: str,
    title: str,
    *,
    ec: str = _F1_LINE,
    hdr_fc: str = _F1_HDR,
    badge_fc: str = _F1_BADGE,
    lw: float = 0.78,
    hdr_h: float = 8.2,
    badge_r: float = 2.20,
    title_fs: float = 6.9,
    r: float = 1.8,
) -> Tuple[float, float, float, float]:
    """Draw a restrained flat card with uniform slate header and 1:1 true circular badge."""
    _f1_box(ax, x, y, w, h, fc="#FFFFFF", ec=ec, lw=lw, r=r, z=2)

    hy = y + h - hdr_h
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, hy),
            w,
            hdr_h,
            boxstyle=f"round,pad=0.0,rounding_size={r}",
            linewidth=0.0,
            facecolor=hdr_fc,
            zorder=3,
        )
    )
    ax.add_patch(patches.Rectangle((x + 0.15, hy), w - 0.30, hdr_h * 0.45, linewidth=0.0, facecolor=hdr_fc, zorder=3))
    ax.plot([x + 0.2, x + w - 0.2], [hy, hy], color=_F1_BORDER, lw=0.65, zorder=4)

    cx = x + 4.1
    cy = hy + hdr_h * 0.50
    ax.add_patch(patches.Circle((cx, cy), radius=badge_r, facecolor=badge_fc, edgecolor="none", zorder=5))
    ax.text(cx, cy - 0.08, num, ha="center", va="center", fontsize=6.4, fontweight="bold", color="white", zorder=6)

    ax.text(
        cx + badge_r + 1.5,
        cy - 0.04,
        title,
        ha="left",
        va="center",
        fontsize=title_fs,
        fontweight="bold",
        color=_F1_INK,
        zorder=6,
    )
    return (x + 2.0, y + 1.6, w - 4.0, h - hdr_h - 2.6)


def _f1_straight_arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = "#475569",
    lw: float = 0.95,
    head_len: float = 0.32,
    head_w: float = 0.17,
    z: int = 5,
) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=f"-|>,head_length={head_len},head_width={head_w}",
            color=color,
            lw=lw,
            shrinkA=0.0,
            shrinkB=0.0,
        ),
        zorder=z,
    )


def make_fig01_workflow_and_protocol(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 1: restrained vector diagnostic pipeline and 4-step operational protocol."""
    apply_cg_style()
    rs = _load_json(summary_dir, "root_skipping_diagnostics.json")
    dc_sh = float(rs["step_sizes_km_s"]["shallow_converged"])
    dc_dp = float(rs["step_sizes_km_s"]["deep_converged"])

    W = DOUBLE_COL_MM
    HA = 56.0
    HB = 74.0
    GAP_MM = 5.0
    TOTAL_H = HA + HB + GAP_MM

    fig = plt.figure(figsize=(mm_to_in(W), mm_to_in(TOTAL_H)))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[HA, HB],
        hspace=GAP_MM / ((HA + HB) / 2.0),
        left=0.004,
        right=0.996,
        top=0.994,
        bottom=0.006,
    )

    # Panel (a): Zero-Leakage Diagnostic Evaluation Pipeline
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0.0, W)
    ax_a.set_ylim(0.0, HA)
    ax_a.set_aspect("equal")
    ax_a.axis("off")

    _f1_box(ax_a, 0.8, 0.8, W - 1.6, HA - 1.6, fc=_F1_BAND, ec=_F1_BORDER, lw=0.70, r=2.4, z=0)
    ax_a.text(
        4.0,
        HA - 3.6,
        "(a) Zero-Leakage Diagnostic Evaluation Pipeline",
        fontsize=8.5,
        fontweight="bold",
        color=_F1_INK,
        va="top",
        ha="left",
        zorder=5,
    )

    cw = 29.0
    gap = 3.25
    x0 = 4.0
    cy0, ch = 3.8, 43.2

    # Card 1: Dispersion Input
    bx, by, bw, bh = _f1_header_card(ax_a, x0, cy0, cw, ch, "1", "Dispersion Input", title_fs=6.8)
    gx, gy, gw, gh = bx + 0.6, by + 9.2, bw - 1.2, bh - 10.2
    ax_a.plot([gx, gx, gx + gw], [gy + gh, gy, gy], color=_F1_MUTED, lw=0.70, zorder=4)
    tx = np.linspace(0.08, 0.92, 9)
    ty = 0.22 + 0.66 * (1.0 - np.exp(-2.6 * tx))
    kept_idx = [0, 1, 3, 4, 6, 7, 8]
    ho_idx = [2, 5]
    ax_a.plot(gx + tx * gw, gy + ty * gh, "-", color="#94A3B8", lw=0.80, zorder=4)
    ax_a.scatter(gx + tx[kept_idx] * gw, gy + ty[kept_idx] * gh, s=11, color=_F1_INK, zorder=5)
    ax_a.scatter(
        gx + tx[ho_idx] * gw,
        gy + ty[ho_idx] * gh,
        s=15,
        marker="D",
        facecolors="white",
        edgecolors=_F1_RUST,
        linewidths=0.95,
        zorder=6,
    )
    ax_a.text(gx + gw * 0.96, gy + gh * 0.06, "Kept / HO", ha="right", va="bottom", fontsize=6.0, color=_F1_MUTED, zorder=6)
    ax_a.text(x0 + cw / 2, by + 3.6, "Seal 20% interior\nbefore alignment", ha="center", va="center", fontsize=6.1, color=_F1_TEXT, linespacing=1.18, zorder=5)

    # Card 2: Input Alignment
    x1 = x0 + (cw + gap)
    bx, by, bw, bh = _f1_header_card(ax_a, x1, cy0, cw, ch, "2", "Input Alignment", title_fs=6.8)
    gx, gy, gw, gh = bx + 0.6, by + 9.2, bw - 1.2, bh - 10.2
    y_top, y_bot = gy + gh * 0.66, gy + gh * 0.28
    ax_a.plot([gx + 0.5, gx + gw - 0.5], [y_top, y_top], color="#94A3B8", lw=0.80, zorder=4)
    ax_a.plot([gx + 0.5, gx + gw - 0.5], [y_bot, y_bot], color=_F1_NAVY, lw=1.10, zorder=4)
    raw_pts = np.array([0.10, 0.24, 0.35, 0.57, 0.75, 0.89])
    uni_pts = np.linspace(0.10, 0.90, 7)
    ax_a.scatter(gx + raw_pts * gw, np.full_like(raw_pts, y_top), s=11, color=_F1_MUTED, zorder=5)
    ax_a.scatter(gx + uni_pts * gw, np.full_like(uni_pts, y_bot), s=11, marker="s", color=_F1_NAVY, zorder=5)
    for rx in [0.18, 0.50, 0.82]:
        _f1_straight_arrow(ax_a, gx + rx * gw, y_top - 1.4, gx + rx * gw, y_bot + 1.5, color=_F1_NAVY, lw=0.72, head_len=0.25, head_w=0.13)
    ax_a.text(gx + gw / 2, gy + gh * 0.96, "Field grid", ha="center", va="top", fontsize=6.0, color=_F1_MUTED, zorder=6)
    ax_a.text(gx + gw / 2, gy + gh * 0.01, "Train grid", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=_F1_NAVY, zorder=6)
    ax_a.text(x1 + cw / 2, by + 3.6, "Resample to training\ngrid (zero leak)", ha="center", va="center", fontsize=6.1, color=_F1_TEXT, linespacing=1.18, zorder=5)

    # Card 3: Backbone & TTA
    x2 = x0 + 2 * (cw + gap)
    bx, by, bw, bh = _f1_header_card(ax_a, x2, cy0, cw, ch, "3", "Backbone & TTA", title_fs=6.8)
    gx, gy, gw, gh = bx + 0.5, by + 9.2, bw - 1.0, bh - 10.2
    trap = patches.Polygon(
        [
            [gx + 0.4, gy + gh * 0.88],
            [gx + gw * 0.37, gy + gh * 0.76],
            [gx + gw * 0.37, gy + gh * 0.42],
            [gx + 0.4, gy + gh * 0.30],
        ],
        closed=True,
        facecolor="#E2E8F0",
        edgecolor="#475569",
        lw=0.80,
        zorder=4,
    )
    ax_a.add_patch(trap)
    ax_a.text(gx + gw * 0.19, gy + gh * 0.59, "NN", ha="center", va="center", fontsize=6.3, fontweight="bold", color=_F1_INK, zorder=5)
    _f1_box(ax_a, gx + gw * 0.52, gy + gh * 0.34, gw * 0.45, gh * 0.48, fc="#FFFFFF", ec="#64748B", lw=0.72, r=1.0, z=4)
    pz = np.array([0.76, 0.62, 0.62, 0.50, 0.50, 0.40])
    pv = np.array([0.58, 0.62, 0.72, 0.72, 0.84, 0.90])
    ax_a.plot(gx + pv * gw, gy + pz * gh, "-", color=_F1_NAVY, lw=1.05, zorder=5)
    ax_a.text(gx + gw * 0.74, gy + gh * 0.85, r"$\hat{V}_s(z)$", ha="center", va="bottom", fontsize=6.3, color=_F1_INK, zorder=6)
    _f1_straight_arrow(ax_a, gx + gw * 0.38, gy + gh * 0.58, gx + gw * 0.51, gy + gh * 0.58, color="#475569", lw=0.80, head_len=0.24, head_w=0.13)
    cy_bot = gy + gh * 0.16
    ax_a.plot([gx + gw * 0.74, gx + gw * 0.74, gx + gw * 0.19], [gy + gh * 0.34, cy_bot, cy_bot], color=_F1_NAVY, lw=0.82, zorder=5)
    _f1_straight_arrow(ax_a, gx + gw * 0.19, cy_bot, gx + gw * 0.19, gy + gh * 0.31, color=_F1_NAVY, lw=0.82, head_len=0.25, head_w=0.13)
    ax_a.text(gx + gw * 0.50, gy - 0.4, "Cycle loss", ha="center", va="bottom", fontsize=5.9, fontweight="bold", color=_F1_NAVY, zorder=6)
    ax_a.text(x2 + cw / 2, by + 3.6, "OpenSWI / DispFormer\n+ surrogate loop", ha="center", va="center", fontsize=6.1, color=_F1_TEXT, linespacing=1.18, zorder=5)

    # Card 4: Exact disba
    x3 = x0 + 3 * (cw + gap)
    bx, by, bw, bh = _f1_header_card(ax_a, x3, cy0, cw, ch, "4", "Exact disba", title_fs=6.8)
    gx, gy, gw, gh = bx + 0.6, by + 9.2, bw - 1.2, bh - 10.2
    ax_a.plot([gx, gx, gx + gw], [gy + gh, gy, gy], color=_F1_MUTED, lw=0.70, zorder=4)
    rx = np.linspace(0.08, 0.92, 16)
    y_m0 = 0.24 + 0.04 * rx
    y_jump = 0.24 + 0.58 * (1.0 / (1.0 + np.exp(-9.0 * (rx - 0.42))))
    ax_a.plot(gx + rx * gw, gy + y_jump * gh, "--", color=_F1_RUST, lw=0.95, zorder=4)
    ax_a.plot(gx + rx * gw, gy + y_m0 * gh, "-", color=_F1_SAGE, lw=1.25, zorder=5)
    ax_a.scatter(gx + rx[::3] * gw, gy + y_m0[::3] * gh, s=8, color=_F1_INK, zorder=6)
    ax_a.text(gx + gw * 0.04, gy + gh * 0.96, "Mode 1", ha="left", va="top", fontsize=6.0, color=_F1_RUST, zorder=6)
    ax_a.text(gx + gw * 0.96, gy + gh * 0.02, "Mode 0", ha="right", va="bottom", fontsize=6.0, fontweight="bold", color=_F1_SAGE, zorder=6)
    ax_a.text(
        x3 + cw / 2,
        by + 3.6,
        f"Sh.: ${_fmt_sci_tex(dc_sh)}$ km/s\nDp.: ${_fmt_sci_tex(dc_dp)}$ km/s",
        ha="center",
        va="center",
        fontsize=6.0,
        color=_F1_TEXT,
        linespacing=1.18,
        zorder=5,
    )

    # Card 5: Regime Audit
    x4 = x0 + 4 * (cw + gap)
    bx, by, bw, bh = _f1_header_card(ax_a, x4, cy0, cw, ch, "5", "Regime Audit", title_fs=6.8)
    gx, gy, gw, gh = bx + 0.6, by + 9.2, bw - 1.2, bh - 10.2
    y_zero = gy + gh * 0.48
    ax_a.add_patch(patches.Rectangle((gx, y_zero), gw, gy + gh - y_zero, facecolor=_F1_RUST_BG, alpha=0.90, lw=0, zorder=3))
    ax_a.add_patch(patches.Rectangle((gx, gy), gw, y_zero - gy, facecolor=_F1_SAGE_BG, alpha=0.90, lw=0, zorder=3))
    ax_a.plot([gx, gx, gx + gw], [gy + gh, gy, gy], color=_F1_MUTED, lw=0.70, zorder=4)
    ax_a.plot([gx, gx + gw], [y_zero, y_zero], ":", color="#475569", lw=0.70, zorder=4)
    sx = np.linspace(0.08, 0.92, 8)
    sy = np.array([0.85, 0.72, 0.50, 0.34, 0.22, 0.16, 0.45, 0.80])
    ax_a.plot(gx + sx * gw, gy + sy * gh, "-o", color=_F1_INK, lw=0.95, ms=2.1, zorder=5)
    ax_a.text(gx + gw * 0.52, gy + gh * 0.94, "Harm", ha="center", va="top", fontsize=6.0, fontweight="bold", color=_F1_RUST, zorder=6)
    ax_a.text(gx + gw * 0.52, gy + gh * 0.04, "Gain", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=_F1_SAGE, zorder=6)
    ax_a.text(
        x4 + cw / 2,
        by + 3.6,
        r"Bin by $r_0 = m_0/\hat{\sigma}_D$" + "\n" + r"Resolvable $V_s$ RMSE",
        ha="center",
        va="center",
        fontsize=6.0,
        color=_F1_TEXT,
        linespacing=1.18,
        zorder=5,
    )

    for k in range(4):
        xa = x0 + k * (cw + gap) + cw + 0.35
        xb = x0 + (k + 1) * (cw + gap) - 0.35
        _f1_straight_arrow(ax_a, xa, cy0 + ch * 0.50, xb, cy0 + ch * 0.50, color="#475569", lw=1.00)

    # Panel (b): Four-Step Operational Diagnostic and Routing Protocol
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_xlim(0.0, W)
    ax_b.set_ylim(0.0, HB)
    ax_b.set_aspect("equal")
    ax_b.axis("off")

    _f1_box(ax_b, 0.8, 0.8, W - 1.6, HB - 1.6, fc=_F1_BAND, ec=_F1_BORDER, lw=0.70, r=2.4, z=0)
    ax_b.text(
        4.0,
        HB - 3.6,
        "(b) Four-Step Operational Diagnostic and Routing Protocol",
        fontsize=8.5,
        fontweight="bold",
        color=_F1_INK,
        va="top",
        ha="left",
        zorder=5,
    )

    # Column 1: Step 1 & Step 2
    c1x, c1w = 4.0, 37.0

    bx, by, bw, bh = _f1_header_card(ax_b, c1x, 37.5, c1w, 27.0, "1", "Step 1: Align & ZS", hdr_h=7.8, badge_r=2.15, title_fs=6.8)
    sy_mid = by + bh * 0.58
    ax_b.plot([bx + 0.2, bx + 0.2, bx + 7.4], [sy_mid + 3.6, sy_mid - 3.4, sy_mid - 3.4], color=_F1_MUTED, lw=0.65, zorder=4)
    p1x = np.linspace(bx + 1.0, bx + 6.8, 5)
    p1y = sy_mid - 2.1 + 4.2 * (1 - np.exp(-0.55 * np.arange(5)))
    ax_b.plot(p1x, p1y, "-", color="#94A3B8", lw=0.75, zorder=4)
    ax_b.scatter(p1x[[0, 1, 3, 4]], p1y[[0, 1, 3, 4]], s=7, color=_F1_INK, zorder=5)
    ax_b.scatter([p1x[2]], [p1y[2]], s=10, marker="D", facecolors="white", edgecolors=_F1_RUST, linewidths=0.85, zorder=6)
    _f1_straight_arrow(ax_b, bx + 7.8, sy_mid, bx + 10.2, sy_mid, color="#475569", lw=0.80, head_len=0.23, head_w=0.12)
    _f1_box(ax_b, bx + 10.4, sy_mid - 3.1, 6.0, 6.2, fc=_F1_NAVY_BG, ec=_F1_NAVY, lw=0.75, r=1.0, z=4)
    ax_b.text(bx + 13.4, sy_mid, r"$\mathbf{g}_{\mathrm{train}}$", ha="center", va="center", fontsize=6.1, fontweight="bold", color=_F1_NAVY, zorder=5)
    _f1_straight_arrow(ax_b, bx + 16.6, sy_mid, bx + 19.0, sy_mid, color="#475569", lw=0.80, head_len=0.23, head_w=0.12)
    nn_poly = patches.Polygon(
        [[bx + 19.2, sy_mid + 3.3], [bx + 24.4, sy_mid + 2.1], [bx + 24.4, sy_mid - 2.1], [bx + 19.2, sy_mid - 3.3]],
        closed=True,
        facecolor="#E2E8F0",
        edgecolor="#475569",
        lw=0.72,
        zorder=4,
    )
    ax_b.add_patch(nn_poly)
    ax_b.text(bx + 21.7, sy_mid, r"$f_\theta$", ha="center", va="center", fontsize=6.2, fontweight="bold", color=_F1_INK, zorder=5)
    _f1_straight_arrow(ax_b, bx + 24.7, sy_mid, bx + 27.1, sy_mid, color="#475569", lw=0.80, head_len=0.23, head_w=0.12)
    ax_b.text(bx + 30.2, sy_mid, r"$\mathbf{v}_{s,i}^{(0)}$", ha="center", va="center", fontsize=6.4, fontweight="bold", color=_F1_INK, zorder=5)
    ax_b.text(c1x + c1w / 2, by + 1.8, "Seal 20% HO  |  Input-Aligned ZS", ha="center", va="center", fontsize=5.95, color=_F1_TEXT, zorder=5)

    _f1_straight_arrow(ax_b, c1x + c1w / 2, 37.2, c1x + c1w / 2, 32.0, color="#475569", lw=1.00)

    bx, by, bw, bh = _f1_header_card(ax_b, c1x, 4.5, c1w, 27.0, "2", "Step 2: Solver & Floor", hdr_h=7.8, badge_r=2.15, title_fs=6.8)
    my = by + 4.8
    mh = 8.4
    ax_b.text(bx + 7.4, my + mh + 2.4, r"disba $\Delta c \Rightarrow m_{0,i}$", ha="center", va="top", fontsize=5.75, fontweight="bold", color=_F1_INK, zorder=6)
    ax_b.plot([bx + 0.8, bx + 0.8, bx + 14.0], [my + mh, my, my], color=_F1_MUTED, lw=0.65, zorder=4)
    dx = np.linspace(bx + 1.6, bx + 13.4, 10)
    dy0 = my + 1.9 + 0.7 * np.linspace(0, 1, 10)
    dy1 = my + 1.9 + 5.2 * (1 / (1 + np.exp(-7 * (np.linspace(0, 1, 10) - 0.45))))
    ax_b.plot(dx, dy1, "--", color=_F1_RUST, lw=0.80, alpha=0.85, zorder=4)
    ax_b.plot(dx, dy0, "-", color=_F1_SAGE, lw=1.15, zorder=5)

    hx0 = bx + 17.6
    ax_b.text(hx0 + 7.2, my + mh + 2.4, r"Floor $\hat{\sigma}_D$", ha="center", va="top", fontsize=5.75, fontweight="bold", color=_F1_INK, zorder=6)
    ax_b.plot([hx0, hx0, hx0 + 14.4], [my + mh, my, my], color=_F1_MUTED, lw=0.65, zorder=4)
    bar_h = [2.2, 4.6, 6.4, 3.8, 1.8]
    for bi, bh_i in enumerate(bar_h):
        ax_b.add_patch(patches.Rectangle((hx0 + 1.1 + bi * 2.5, my + 0.15), 1.9, bh_i, facecolor="#CBD5E1", edgecolor="#64748B", lw=0.45, zorder=4))
    ax_b.plot([hx0 + 7.1, hx0 + 7.1], [my, my + mh - 0.3], "--", color=_F1_NAVY, lw=1.00, zorder=6)
    ax_b.text(c1x + c1w / 2, by + 1.6, "Exact misfit  +  Classical floor", ha="center", va="center", fontsize=5.95, color=_F1_TEXT, zorder=5)

    # Column 2: Step 3 Screen (explicit stacked Shallow S and Deep D regime bars)
    c2x, c2w, c2y, c2h = 47.5, 33.0, 9.5, 50.0
    bx, by, bw, bh = _f1_header_card(ax_b, c2x, c2y, c2w, c2h, "3", "Step 3: Screen", hdr_h=8.0, badge_r=2.15, title_fs=6.8)

    _f1_box(ax_b, bx + 1.0, by + 27.6, bw - 2.0, 9.6, fc=_F1_NAVY_BG, ec=_F1_NAVY, lw=0.85, r=1.6, z=4)
    ax_b.text(c2x + c2w / 2, by + 32.4, r"$r_{0,i} = m_{0,i} / \hat{\sigma}_D$", ha="center", va="center", fontsize=7.9, fontweight="bold", color=_F1_INK, zorder=5)
    ax_b.text(c2x + c2w / 2, by + 23.8, "Domain regime partition", ha="center", va="center", fontsize=5.95, color=_F1_TEXT, zorder=5)

    bar_x = bx + 4.4
    bar_w = bw - 5.2
    bar_hh = 5.4

    # Shallow (S) bar: A | B (no C outlier branch)
    ys = by + 15.0
    ax_b.text(bx + 1.9, ys + bar_hh / 2, "S", ha="center", va="center", fontsize=6.2, fontweight="bold", color=_F1_INK, zorder=5)
    ws_a = bar_w * 0.44
    ws_b = bar_w - ws_a
    ax_b.add_patch(patches.Rectangle((bar_x, ys), ws_a, bar_hh, facecolor="#64748B", lw=0, zorder=4))
    ax_b.add_patch(patches.Rectangle((bar_x + ws_a, ys), ws_b, bar_hh, facecolor=_F1_NAVY, lw=0, zorder=4))
    ax_b.text(bar_x + ws_a / 2, ys + bar_hh / 2, "A", ha="center", va="center", fontsize=6.3, fontweight="bold", color="white", zorder=5)
    ax_b.text(bar_x + ws_a + ws_b / 2, ys + bar_hh / 2, "B", ha="center", va="center", fontsize=6.3, fontweight="bold", color="white", zorder=5)

    # Deep (D) bar: A | B | C
    yd = by + 8.0
    ax_b.text(bx + 1.9, yd + bar_hh / 2, "D", ha="center", va="center", fontsize=6.2, fontweight="bold", color=_F1_INK, zorder=5)
    wd_a = bar_w * 0.28
    wd_b = bar_w * 0.38
    wd_c = bar_w - wd_a - wd_b
    ax_b.add_patch(patches.Rectangle((bar_x, yd), wd_a, bar_hh, facecolor="#64748B", lw=0, zorder=4))
    ax_b.add_patch(patches.Rectangle((bar_x + wd_a, yd), wd_b, bar_hh, facecolor=_F1_NAVY, lw=0, zorder=4))
    ax_b.add_patch(patches.Rectangle((bar_x + wd_a + wd_b, yd), wd_c, bar_hh, facecolor=_F1_RUST, lw=0, zorder=4))
    ax_b.text(bar_x + wd_a / 2, yd + bar_hh / 2, "A", ha="center", va="center", fontsize=6.3, fontweight="bold", color="white", zorder=5)
    ax_b.text(bar_x + wd_a + wd_b / 2, yd + bar_hh / 2, "B", ha="center", va="center", fontsize=6.3, fontweight="bold", color="white", zorder=5)
    ax_b.text(bar_x + wd_a + wd_b + wd_c / 2, yd + bar_hh / 2, "C", ha="center", va="center", fontsize=6.3, fontweight="bold", color="white", zorder=5)

    ax_b.text(bar_x, yd - 1.3, "Low $r_0$", ha="left", va="top", fontsize=5.35, color=_F1_MUTED)
    ax_b.text(bar_x + bar_w, yd - 1.3, "High $r_0$", ha="right", va="top", fontsize=5.35, color=_F1_MUTED)
    ax_b.text(c2x + c2w / 2, by + 1.8, "S: Shallow  |  D: Deep", ha="center", va="center", fontsize=5.75, style="italic", color=_F1_MUTED)

    # Straight horizontal arrows Col 1 -> Col 2
    _f1_straight_arrow(ax_b, c1x + c1w + 0.4, 51.0, c2x - 0.4, 51.0, color="#475569", lw=1.00)
    _f1_straight_arrow(ax_b, c1x + c1w + 0.4, 18.0, c2x - 0.4, 18.0, color="#475569", lw=1.00)

    # Column 3: Three Restrained Visual-First Regime Cards
    c3x, c3w = 101.5, 60.5
    rh = 17.8

    def _regime_card(
        y_bot: float,
        stripe_c: str,
        border_c: str,
        bg_c: str,
        title: str,
        tag: str,
        tag_solid: bool,
        footer: str,
    ) -> Tuple[float, float, float, float]:
        lw_card = 1.05 if tag_solid else 0.78
        _f1_box(ax_b, c3x, y_bot, c3w, rh, fc=bg_c, ec=border_c, lw=lw_card, r=1.8, z=2)
        ax_b.add_patch(
            patches.FancyBboxPatch(
                (c3x, y_bot),
                1.6,
                rh,
                boxstyle="round,pad=0.0,rounding_size=0.8",
                lw=0,
                facecolor=stripe_c,
                zorder=4,
            )
        )
        ix, iy, iw, ih = c3x + 3.4, y_bot + 2.1, 16.5, 13.6
        _f1_box(ax_b, ix, iy, iw, ih, fc="#FFFFFF", ec=_F1_BORDER, lw=0.60, r=1.1, z=4)

        tx0 = ix + iw + 2.4
        pill_w, pill_h = 15.0, 5.0
        px, py = c3x + c3w - pill_w - 1.8, y_bot + rh - pill_h - 1.9
        if tag_solid:
            _f1_box(ax_b, px, py, pill_w, pill_h, fc=stripe_c, ec="none", lw=0, r=1.1, z=5)
            tag_color = "white"
        else:
            _f1_box(ax_b, px, py, pill_w, pill_h, fc="#FFFFFF", ec=stripe_c, lw=0.75, r=1.1, z=5)
            tag_color = stripe_c
        ax_b.text(px + pill_w / 2, py + pill_h / 2 - 0.05, tag, ha="center", va="center", fontsize=5.8, fontweight="bold", color=tag_color, zorder=6)

        ax_b.text(tx0, y_bot + rh - 4.4, title, ha="left", va="center", fontsize=6.7, fontweight="bold", color=_F1_INK, zorder=5)
        ax_b.text(tx0, y_bot + 5.6, footer, ha="left", va="center", fontsize=5.95, color=_F1_TEXT, linespacing=1.34, zorder=5)
        return (ix + 1.4, iy + 1.4, iw - 2.8, ih - 2.8)

    # Regime A: Neutral Slate
    ix, iy, iw, ih = _regime_card(
        46.7,
        "#64748B",
        _F1_LINE,
        "#FFFFFF",
        "Regime A",
        "Retain ZS",
        False,
        "Near-floor: skip TTA\n" + r"$\mathbf{v}_{s,i}^{\mathrm{out}} = \mathbf{v}_{s,i}^{(0)}$",
    )
    ax_b.plot([ix, ix, ix + iw], [iy + ih, iy, iy], color="#94A3B8", lw=0.6, zorder=5)
    vz = np.array([0.88, 0.64, 0.64, 0.38, 0.38, 0.12])
    vv = np.array([0.22, 0.26, 0.48, 0.50, 0.74, 0.78])
    ax_b.plot(ix + vv * iw, iy + vz * ih, "-", color="#475569", lw=1.20, zorder=6)
    ax_b.text(ix + iw * 0.90, iy + ih * 0.88, "ZS", ha="right", va="top", fontsize=5.3, fontweight="bold", color="#475569", zorder=6)

    # Regime B: Highlighted Steel Navy (active TTA path)
    ix, iy, iw, ih = _regime_card(
        25.6,
        _F1_NAVY,
        _F1_NAVY,
        _F1_NAVY_BG,
        "Regime B",
        "Step 4: TTA",
        True,
        "Unfrozen Holdout ES\n+ exact disba rollback",
    )
    ax_b.plot([ix, ix, ix + iw], [iy + ih, iy, iy], color="#94A3B8", lw=0.6, zorder=5)
    ex_x = np.linspace(0.08, 0.92, 9)
    ex_y = np.array([0.86, 0.54, 0.34, 0.22, 0.18, 0.25, 0.38, 0.54, 0.68])
    ax_b.plot(ix + ex_x[:5] * iw, iy + ex_y[:5] * ih, "-", color=_F1_NAVY, lw=1.25, zorder=6)
    ax_b.plot(ix + ex_x[4:] * iw, iy + ex_y[4:] * ih, "--", color="#94A3B8", lw=0.90, zorder=5)
    ax_b.scatter([ix + ex_x[4] * iw], [iy + ex_y[4] * ih], s=22, marker="*", color=_F1_NAVY, edgecolors="white", linewidths=0.35, zorder=7)
    ax_b.text(ix + ex_x[4] * iw + 0.5, iy + ex_y[4] * ih + 3.2, r"$s^\star$", ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=_F1_NAVY, zorder=7)

    # Regime C: Quiet Mineral Terracotta
    ix, iy, iw, ih = _regime_card(
        4.5,
        _F1_RUST,
        "#D6A592",
        _F1_RUST_BG,
        "Regime C",
        "Audit / ZS",
        False,
        "Deep outliers (>10 km)\nSmoothness / Classical",
    )
    ax_b.plot([ix, ix, ix + iw], [iy + ih, iy, iy], color="#94A3B8", lw=0.6, zorder=5)
    oz = np.linspace(0.92, 0.08, 40)
    ov_sm = 0.24 + 0.50 * (1.0 - oz)
    osc_env = np.where(oz < 0.72, np.sin(5.0 * np.pi * (0.72 - oz) / 0.64) * np.sin(np.pi * (0.72 - oz) / 0.64), 0.0)
    ov_wig = ov_sm + 0.17 * osc_env
    ax_b.plot(ix + ov_sm * iw, iy + oz * ih, "--", color="#94A3B8", lw=0.85, zorder=5)
    ax_b.plot(ix + np.clip(ov_wig, 0.08, 0.94) * iw, iy + oz * ih, "-", color=_F1_RUST, lw=1.15, zorder=6)

    # Three straight horizontal routing arrows with flush-left aligned S:/D: labels
    ax_x0 = c2x + c2w + 0.5
    ax_x1 = c3x - 0.5
    lbl_x = ax_x0 + 2.2
    val_x = lbl_x + 3.4

    def _arrow_tag(y_line: float, rows: Sequence[Tuple[str, str]], color: str) -> None:
        base_y = y_line + 1.6
        step_y = 3.1
        for idx, (prefix, expr) in enumerate(reversed(rows)):
            yy = base_y + idx * step_y
            ax_b.text(lbl_x, yy, prefix, ha="left", va="bottom", fontsize=5.65, fontweight="bold", color=color)
            ax_b.text(val_x, yy, expr, ha="left", va="bottom", fontsize=5.65, color=color)

    _f1_straight_arrow(ax_b, ax_x0, 54.0, ax_x1, 54.0, color="#64748B", lw=1.00)
    _arrow_tag(54.0, [("S:", r"$r_0 < 2.0$"), ("D:", r"$r_0 < 1.25$")], _F1_TEXT)

    _f1_straight_arrow(ax_b, ax_x0, 34.5, ax_x1, 34.5, color=_F1_NAVY, lw=1.15)
    _arrow_tag(34.5, [("S:", r"$r_0 \geq 2.0$"), ("D:", r"$[1.25, 3.0]$")], _F1_NAVY)

    _f1_straight_arrow(ax_b, ax_x0, 14.5, ax_x1, 14.5, color=_F1_RUST, lw=1.00)
    _arrow_tag(14.5, [("D:", r"$r_0 > 3.0$")], _F1_RUST)

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Graphical Abstract (Exact 5:2 aspect ratio, >= 1328x531 px at 300+ DPI)
# ---------------------------------------------------------------------------


def make_graphical_abstract(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 360,
) -> Dict[str, Dict[str, object]]:
    """Generate Elsevier Graphical Abstract at exact 5:2 aspect ratio (2700x1080 px @ 360 DPI)."""
    apply_cg_style()
    rs = _load_json(summary_dir, "root_skipping_diagnostics.json")
    cb = _load_json(summary_dir, "cross_backbone_diagnostics.json")

    # Exact 5:2 canvas (7.5 x 3.0 in -> 2700 x 1080 px at 360 DPI, strictly >= 1328x531)
    fig = plt.figure(figsize=(7.5, 3.0), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Top title banner sized cleanly inside 7.5 in canvas with generous left/right margins
    fig.text(
        0.5,
        0.962,
        "When Can Dispersion Misfit Reduction Be Trusted? Forward-Solver & Noise-Floor TTA Diagnostics",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#111111",
    )

    gs = gridspec.GridSpec(
        1,
        3,
        left=0.065,
        right=0.985,
        bottom=0.20,
        top=0.83,
        wspace=0.31,
        width_ratios=[1.05, 1.32, 1.00],
    )

    # Panel 1: Hidden Evaluation Pitfall (disba root-skipping collapse) — D5-1 collision fixed
    ax1 = fig.add_subplot(gs[0])
    sh_rec = rs["recomputed_domains"]["shallow_benchmark"]
    labels = ["Out.-Space\nRef.", "Gated Adp.\n+ Out.-Ref.", "Ground\nTruth"]
    old_vals = [
        sh_rec["output_refined_aligned"]["old_tail_pct"],
        sh_rec["adapter_output_refined_aligned"]["old_tail_pct"],
        rs["gt_noise_floor_default_vs_converged"]["shallow_benchmark"]["ho_old_tail_pct"],
    ]
    new_vals = [
        sh_rec["output_refined_aligned"]["new_tail_pct"],
        sh_rec["adapter_output_refined_aligned"]["new_tail_pct"],
        rs["gt_noise_floor_default_vs_converged"]["shallow_benchmark"]["ho_new_tail_pct"],
    ]
    x = np.arange(len(labels))
    w = 0.34
    b1 = ax1.bar(x - w / 2, old_vals, w, color=PALETTE["vermillion"], label=r"Default $\Delta c=5\times 10^{-3}$")
    b2 = ax1.bar(x + w / 2, new_vals, w, color=PALETTE["teal"], label=r"Conv. $\Delta c=2.5\times 10^{-5}$")
    for i_b, (bar, v) in enumerate(zip(b1, old_vals)):
        dx = -0.09 if i_b == 2 else 0.0
        ax1.text(bar.get_x() + bar.get_width() / 2 + dx, v + 0.25, f"{v:.2f}%", ha="center", va="bottom", fontsize=6.3)
    for i_b, (bar, v) in enumerate(zip(b2, new_vals)):
        dx = 0.06 if i_b == 0 else (0.11 if i_b == 2 else 0.0)
        ax1.text(bar.get_x() + bar.get_width() / 2 + dx, v + 0.25, f"{v:.2f}%", ha="center", va="bottom", fontsize=6.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.8)
    ax1.set_ylabel("Tail Rate (>20% HO, %)", fontsize=7.4)
    ax1.tick_params(axis="y", labelsize=6.8)
    ax1.set_ylim(0, 11.8)
    ax1.set_title("1. Fix disba Root-Skipping", fontsize=8.0, fontweight="bold", color=PALETTE["vermillion"])
    ax1.legend(loc="upper right", fontsize=6.3, handlelength=1.1)

    # Panel 2: Noise-Floor-Normalized Signature Curve (Shallow Multi-Structure)
    ax2 = fig.add_subplot(gs[1])
    bins = cb["domains"]["shallow_multi_structure"]["by_backbone"]["openswi"]["all_judged_ckpts"]["pooled_signature_bins"]["bins"]
    bx = np.arange(len(bins))
    blabs = _fmt_bin_labels([b["bin_label"] for b in bins])
    y_p3 = [b["methods"]["output_refined_50"]["median_rel_change_pct"] for b in bins]
    y_ho = [b["methods"]["holdout_es"]["median_rel_change_pct"] for b in bins]
    y_id = [b["methods"]["ideal_es"]["median_rel_change_pct"] for b in bins]

    ax2.axhspan(0, 185, color=PALETTE["warn_tint"], alpha=0.55, zorder=0)
    ax2.axhspan(-94, 0, color=PALETTE["safe_tint"], alpha=0.55, zorder=0)
    ax2.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax2.plot(bx, y_p3, "-o", color=PALETTE["vermillion"], lw=1.4, ms=3.5, label="50-Step Ref.")
    ax2.plot(bx, y_ho, "-s", color=PALETTE["blue"], lw=1.4, ms=3.4, label="Holdout ES")
    ax2.plot(bx, y_id, "--^", color=PALETTE["teal"], lw=1.2, ms=3.2, label="Ideal ES")
    ax2.set_xticks(bx)
    ax2.set_xticklabels(blabs, fontsize=6.4, rotation=28, ha="right")
    ax2.tick_params(axis="y", labelsize=6.8)
    ax2.set_xlabel(r"Initial Misfit / GT Noise ($r_0^{\mathrm{GT}}$)", fontsize=7.2)
    ax2.set_ylabel("Median Δ $V_s$ RMSE (%)", fontsize=7.4)
    ax2.set_ylim(-94, 185)
    ax2.text(1.10, 138, "Noise-fitting\ndegradation", fontsize=6.5, color="#9A2D00", fontweight="bold")
    ax2.text(0.10, -82, "Shift recovery", fontsize=6.5, color="#00664B", fontweight="bold")
    ax2.set_title("2. Noise-Floor Regime Signature", fontsize=8.0, fontweight="bold", color=PALETTE["blue"])
    ax2.legend(loc="upper right", fontsize=6.3, handlelength=1.2)

    # Panel 3: Operational Diagnostic & Routing Guide
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.04)
    ax3.axis("off")
    ax3.set_title("3. Diagnostic Routing Guide", fontsize=8.0, fontweight="bold", color=PALETTE["teal"], pad=6)
    ax3.text(
        0.5,
        1.00,
        r"Operational ratio $r_0 = m_0 / \hat{\sigma}_D$",
        ha="center",
        va="top",
        fontsize=6.6,
        color="#333333",
        style="italic",
    )
    _draw_box(
        ax3,
        0.015,
        0.64,
        0.97,
        0.26,
        "Near Floor (S:$r_0<2$, D:$r_0<1.25$)",
        "Retain Input-Aligned Zero-Shot\n(mitigates near-floor harm)",
        PALETTE["safe_tint"],
        PALETTE["teal"],
        fontsize_title=6.5,
        fontsize_body=6.4,
    )
    _draw_box(
        ax3,
        0.015,
        0.33,
        0.97,
        0.26,
        "Shift (S:$r_0\\geq 2$, D:$1.25$–$3$)",
        "Output-Space Ref. + Holdout ES\n(converged full-kept disba rollback)",
        PALETTE["info_tint"],
        PALETTE["blue"],
        fontsize_title=6.4,
        fontsize_body=6.3,
    )
    _draw_box(
        ax3,
        0.015,
        0.02,
        0.97,
        0.26,
        "Deep Outliers (D:$r_0 > 3.0$)",
        "Audit fit & >10 km smoothness;\nretain ZS or Classical fallback",
        PALETTE["warn_tint"],
        PALETTE["vermillion"],
        fontsize_title=6.5,
        fontsize_body=6.4,
    )

    # Note: bbox_inches=None preserves the exact 5:2 canvas dimensions (2700 x 1080 px)
    return save_figure(fig, out_stem, formats=formats, dpi=dpi, bbox_inches=None)


# ---------------------------------------------------------------------------
# Figure 2: Three-Benchmark Performance Overview (RMSE vs Field Held-Out Misfit)
# ---------------------------------------------------------------------------


def make_fig02_benchmark_overview(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 2: same-dataset synthetic RMSE vs HO misfit (a-c) and field HO misfit (d-e)."""
    apply_cg_style()
    bo = _load_json(summary_dir, "benchmark_overview.json")
    rs = _load_json(summary_dir, "root_skipping_diagnostics.json")
    cal = _load_json(summary_dir, "calibration_and_domains.json")
    cnf = cal["calibrated_noise_floors"]
    method_order = bo["method_order"]

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(152.0)))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.20, 1.0], hspace=0.72, left=0.10, right=0.98, top=0.94, bottom=0.07)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.30)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.22)

    syn_panels = [
        ("longbeach_matched_shallow", "(a) LongBeach-Matched Shallow"),
        ("shallow_benchmark", "(b) Shallow Benchmark"),
        ("deep_benchmark", "(c) Deep Benchmark"),
    ]
    focal_methods = [
        "zero_shot_raw",
        "output_refined_raw",
        "zero_shot_aligned",
        "low_dof_adapter",
        "output_refined_aligned",
        "adapter_output_refined_aligned",
    ]
    other_raw = [m for m in method_order if m not in focal_methods]

    for idx, (syn_k, title) in enumerate(syn_panels):
        ax = fig.add_subplot(gs_top[0, idx])
        syn_blk = bo["synthetic_benchmarks"][syn_k]
        rec_blk = rs["recomputed_domains"][syn_k]
        gt_nf = bo["noise_floor_overfit"][syn_k]["gt_noise_floor_obs_median"] * 100.0

        x_aligned = rec_blk["zero_shot_aligned"]["ho_new_median"] * 100.0
        y_aligned = syn_blk["methods"]["zero_shot_aligned"]["judged_mean_vs_rmse_res"]

        xs_all, ys_all = [], []
        for mk in method_order:
            xv = rec_blk[mk]["ho_new_median"] * 100.0
            yv = syn_blk["methods"][mk]["judged_mean_vs_rmse_res"]
            xs_all.append(xv)
            ys_all.append(yv)

        x_min, x_max = min(min(xs_all), gt_nf), max(max(xs_all), gt_nf)
        y_min, y_max = min(ys_all), max(ys_all)
        x_pad = (x_max - x_min) * 0.14
        y_pad = (y_max - y_min) * 0.16
        ax.set_xlim(max(0.0, x_min - x_pad * 0.7), x_max + x_pad * 1.05)
        ax.set_ylim(max(0.0, y_min - y_pad * 0.9), y_max + y_pad * 1.05)

        ax.fill_between(
            [ax.get_xlim()[0], x_aligned],
            ax.get_ylim()[0],
            y_aligned,
            color=PALETTE["safe_tint"],
            alpha=0.55,
            zorder=1,
        )
        ax.axvline(x_aligned, color="#555555", lw=0.80, ls="--", zorder=2)
        ax.axhline(y_aligned, color="#555555", lw=0.80, ls="--", zorder=2)
        ax.axvline(gt_nf, color=PALETTE["mid_gray"], lw=0.95, ls=":", zorder=2)

        for mk in other_raw:
            xv = rec_blk[mk]["ho_new_median"] * 100.0
            yv = syn_blk["methods"][mk]["judged_mean_vs_rmse_res"]
            ax.scatter(xv, yv, facecolors="white", edgecolors="#8C8C8C", marker="v", s=30, linewidths=0.85, zorder=4)

        for mk in focal_methods:
            xv = rec_blk[mk]["ho_new_median"] * 100.0
            yv = syn_blk["methods"][mk]["judged_mean_vs_rmse_res"]
            color, marker, _, _ = METHOD_STYLE[mk]
            is_raw = mk.endswith("_raw")
            ms = 68 if mk == "adapter_output_refined_aligned" else (48 if not is_raw else 38)
            ax.scatter(
                xv,
                yv,
                facecolors="white" if is_raw else color,
                edgecolors=color if is_raw else ("#111111" if mk == "zero_shot_aligned" else "white"),
                marker=marker,
                s=ms,
                linewidths=1.05 if is_raw else 0.65,
                zorder=6 if not is_raw else 5,
            )

        ax.set_title(title, fontsize=8.2, fontweight="bold")
        ax.set_xlabel("Same-Dataset HO Misfit (%)", fontsize=7.6)
        ax.tick_params(labelsize=7.0)
        if idx == 0:
            ax.set_ylabel("Resolvable-Depth $V_s$ RMSE (km/s)", fontsize=7.8)

    handles = []
    for mk in focal_methods:
        color, marker, _, lbl = METHOD_STYLE[mk]
        is_raw = mk.endswith("_raw")
        handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                color="none",
                markerfacecolor="white" if is_raw else color,
                markeredgecolor=color if is_raw else ("#111111" if mk == "zero_shot_aligned" else "white"),
                markeredgewidth=1.05 if is_raw else 0.65,
                markersize=6.2 if mk == "adapter_output_refined_aligned" else 5.2,
                label=lbl,
            )
        )
    handles.append(
        Line2D([0], [0], marker="v", color="none", markerfacecolor="white", markeredgecolor="#8C8C8C",
               markeredgewidth=0.9, markersize=5.0, label="Other Raw TTA (3 variants)")
    )
    handles.append(
        Line2D([0], [0], color=PALETTE["mid_gray"], lw=1.0, ls=":", label=r"True noise floor ($\sigma_{\mathrm{GT}}$)")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.53, 0.515),
        fontsize=6.8,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    field_panels = [
        ("longbeach_report", "longbeach_tuning", "(d) LongBeach Report (3,705 stations)"),
        ("csrm_report", "csrm_validation", "(e) CSRM Report (8,842 grid points)"),
    ]
    bar_methods = [
        ("zero_shot_raw", "Zero-Shot (Raw)", "#8C8C8C"),
        ("zero_shot_aligned", "Input-Aligned Zero-Shot", "#1A1A1A"),
        ("low_dof_adapter", "Low-DOF Adapter", PALETTE["teal"]),
        ("output_refined_aligned", "Output-Space Refinement", PALETTE["vermillion"]),
        ("adapter_output_refined_aligned", "Gated Adapter + Output-Refined", PALETTE["blue"]),
        ("classical", "Classical Inversion (500 sampled)", PALETTE["purple"]),
    ]

    for f_idx, (fld_k, cal_k, f_title) in enumerate(field_panels):
        ax = fig.add_subplot(gs_bot[0, f_idx])
        fld_blk = bo["field_report"][fld_k]
        sig_hat = cnf[cal_k]["sigma_hat_d"] * 100.0
        vals, labels_f, cols_f = [], [], []
        for mk, mlbl, mcol in bar_methods:
            if mk == "classical":
                v = fld_blk["classical_inversion"]["ho_median"] * 100.0
            else:
                v = fld_blk["methods"][mk]["ho_median_mean4_converged_dc"] * 100.0
            vals.append(v)
            labels_f.append(mlbl)
            cols_f.append(mcol)

        y_pos = np.arange(len(vals))[::-1]
        bars = ax.barh(y_pos, vals, height=0.58, color=cols_f, alpha=0.88, zorder=3)
        ax.axvline(sig_hat, color="#444444", lw=1.0, ls=":", zorder=4)
        ax.text(
            sig_hat + max(vals) * 0.015,
            0.25,
            r"$\hat{\sigma}_D=$" + f"{sig_hat:.2f}%",
            fontsize=6.8,
            color="#333333",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        )
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.018, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", ha="left", fontsize=6.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_f if f_idx == 0 else [""] * len(labels_f), fontsize=7.0)
        ax.tick_params(axis="x", labelsize=7.0)
        ax.set_xlim(0, max(vals) * 1.28)
        ax.set_xlabel("Field Held-Out Dispersion Misfit Median (%)", fontsize=7.6)
        ax.set_title(f_title, fontsize=8.2, fontweight="bold")

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 3: disba Root-Skipping Pitfall, Convergence Audit & Tail Collapse
# ---------------------------------------------------------------------------


def make_fig03_disba_root_skipping(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 3: disba root-skipping counterfactual, convergence, and tail drop."""
    apply_cg_style()
    rs = _load_json(summary_dir, "root_skipping_diagnostics.json")
    cf = rs["counterfactual_interventions"]
    sens = rs["direct_dc_sensitivity"]
    conv = rs["convergence_audit"]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, mm_to_in(156.0)))
    plt.subplots_adjust(wspace=0.28, hspace=0.52, left=0.09, right=0.98, top=0.94, bottom=0.14)

    # Panel (a): Counterfactual & Direct dc Sensitivity on Shallow Tail Cases
    ax = axes[0, 0]
    cats = [
        "Original\n(5e-3)",
        "LVZ-Raised\n(5e-3)",
        r"Rest. $>z_{\mathrm{res}}$" + "\n(5e-3)",
        "Aligned ZS\n(5e-3)",
        "Original\n(2e-4)",
    ]
    vals_p3 = [
        cf["output_refined_aligned"]["tail"]["original"]["ho_median"] * 100.0,
        cf["output_refined_aligned"]["tail"]["lvz_raised"]["ho_median"] * 100.0,
        cf["output_refined_aligned"]["tail"]["restore_below_zres"]["ho_median"] * 100.0,
        cf["output_refined_aligned"]["tail"]["zero_shot_aligned"]["ho_median"] * 100.0,
        sens["output_refined_tail_original_ho_all_median"][-1] * 100.0,
    ]
    vals_p5 = [
        cf["adapter_output_refined_aligned"]["tail"]["original"]["ho_median"] * 100.0,
        cf["adapter_output_refined_aligned"]["tail"]["lvz_raised"]["ho_median"] * 100.0,
        cf["adapter_output_refined_aligned"]["tail"]["restore_below_zres"]["ho_median"] * 100.0,
        cf["adapter_output_refined_aligned"]["tail"]["zero_shot_aligned"]["ho_median"] * 100.0,
        sens["adapter_output_refined_tail_original_ho_all_median"][-1] * 100.0,
    ]
    x = np.arange(len(cats))
    w = 0.36
    b1 = ax.bar(x - w / 2, vals_p3, w, color=PALETTE["vermillion"], label="Output-Space Refinement")
    b2 = ax.bar(x + w / 2, vals_p5, w, color=PALETTE["blue"], label="Gated Adapter + Output-Refined")
    for bi, (bar, v) in enumerate(zip(b1, vals_p3)):
        dx = -0.07 if bi >= 1 else -0.04
        ax.text(bar.get_x() + bar.get_width() / 2 + dx, v + 1.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.1)
    for bi, (bar, v) in enumerate(zip(b2, vals_p5)):
        dx = 0.08 if bi >= 1 else 0.04
        ax.text(bar.get_x() + bar.get_width() / 2 + dx, v + 1.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.1)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=6.2)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_ylabel("Tail Median Held-Out Misfit (%)", fontsize=7.6)
    ax.set_ylim(0, 96)
    ax.set_title("(a) LVZ Counterfactual & Step-Size Collapse", fontsize=8.0, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.4)

    # Panel (b): Live LVZ Mode-Jumping Mechanism (sample #9391)
    ax = axes[0, 1]
    ex0 = rs["typical_lvz_examples"][0]
    T = np.asarray(ex0["periods_s"], dtype=float)
    obs = np.asarray(ex0["phase_obs_km_s"], dtype=float)
    is_ho = np.asarray(ex0["is_heldout"], dtype=bool)
    c_m0_old = np.asarray([np.nan if v is None else float(v) for v in ex0["curves"]["orig_mode0_dc005"]], dtype=float)
    c_m0_new = np.asarray([np.nan if v is None else float(v) for v in ex0["curves"].get("orig_mode0_dc2p5e5", ex0["curves"]["orig_mode0_dc0002"])], dtype=float)
    c_m1_old = np.asarray([np.nan if v is None else float(v) for v in ex0["curves"]["orig_mode1_dc005"]], dtype=float)

    ax.plot(T[~is_ho], obs[~is_ho], "o", color="#222222", ms=3.0, label="Obs. (kept)", zorder=5)
    ax.plot(T[is_ho], obs[is_ho], "D", mfc="white", mec="#222222", ms=3.5, mew=0.85, label="Obs. (held-out)", zorder=6)
    ax.plot(
        T, c_m0_old, "-", color=PALETTE["vermillion"], lw=1.4,
        label=r"Mode 0 ($\Delta c=5\times 10^{-3}\!\to\!$Mode 1)",
    )
    ax.plot(
        T, c_m0_new, "--", color=PALETTE["teal"], lw=1.4,
        label=r"Mode 0 ($\Delta c=2.5\times 10^{-5}$, 0.65%)",
    )
    ax.plot(
        T, c_m1_old, ":", color=PALETTE["purple"], lw=1.2,
        label=r"Mode 1 ($\Delta c=5\times 10^{-3}\!\to\!$Mode 2)",
    )
    ax.set_ylim(0.74, 2.55)
    ax.set_xlabel("Period (s)", fontsize=7.6)
    ax.set_ylabel("Rayleigh Phase Velocity (km/s)", fontsize=7.6)
    ax.tick_params(labelsize=7.0)
    ax.set_title(f"(b) Root-Skipping Example (#{ex0['sample_idx']}, {ex0['structure_type']})", fontsize=8.0, fontweight="bold")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.145),
        fontsize=5.6,
        handlelength=1.25,
        handletextpad=0.40,
        labelspacing=0.26,
        borderpad=0.32,
        framealpha=0.95,
    )

    # Panel (c): Per-Regime Convergence & Single-Forward Timing (unoccluded full deep curve)
    ax = axes[1, 0]
    sh_tm = sorted(conv["single_process_timing_ms"]["shallow_benchmark"].values(), key=lambda r: r["dc"], reverse=True)
    dp_tm = sorted(conv["single_process_timing_ms"]["deep_benchmark"].values(), key=lambda r: r["dc"], reverse=True)
    ax.plot(
        [r["dc"] for r in sh_tm],
        [r["median_ms"] for r in sh_tm],
        "-o",
        color=PALETTE["blue"],
        label="Shallow single-forward median",
    )
    ax.plot(
        [r["dc"] for r in dp_tm],
        [r["median_ms"] for r in dp_tm],
        "-s",
        color=PALETTE["orange"],
        label="Deep single-forward median",
    )
    sh_sel_t = conv["single_process_timing_ms"]["shallow_benchmark"]["2.5e-05"]["median_ms"]
    dp_sel_t = conv["single_process_timing_ms"]["deep_benchmark"]["0.005"]["median_ms"]
    ax.scatter(
        [2.5e-5],
        [sh_sel_t],
        s=95,
        marker="*",
        color=PALETTE["teal"],
        edgecolors="#111111",
        lw=0.6,
        zorder=6,
        label=r"Shallow ($\Delta c=2.5\times 10^{-5}$, " + f"{sh_sel_t:.1f} ms)",
    )
    ax.scatter(
        [5e-3],
        [dp_sel_t],
        s=95,
        marker="*",
        color=PALETTE["vermillion"],
        edgecolors="#111111",
        lw=0.6,
        zorder=6,
        label=r"Deep ($\Delta c=5.0\times 10^{-3}$, " + f"{dp_sel_t:.1f} ms)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_ylim(4.8, 5200.0)
    ax.set_xlabel(r"disba Step $\Delta c$ (km/s, finer $\rightarrow$)", fontsize=7.6)
    ax.set_ylabel("Median Forward Time (ms / station)", fontsize=7.6)
    ax.tick_params(labelsize=7.0)
    ax.set_title("(c) Per-Regime Convergence & Compute Cost", fontsize=8.0, fontweight="bold")
    ax.legend(loc="upper left", fontsize=6.2, framealpha=0.95, handlelength=1.3)

    # Panel (d): Shallow Benchmark Tail Failure Rate Collapse (>20% HO Misfit)
    ax = axes[1, 1]
    sh_d = rs["recomputed_domains"]["shallow_benchmark"]
    gt_sh = rs["gt_noise_floor_default_vs_converged"]["shallow_benchmark"]
    items = [
        ("Aligned ZS", sh_d["zero_shot_aligned"]["old_tail_pct"], sh_d["zero_shot_aligned"]["new_tail_pct"]),
        ("Out.-Ref. (Raw)", sh_d["output_refined_raw"]["old_tail_pct"], sh_d["output_refined_raw"]["new_tail_pct"]),
        ("Out.-Space Ref.", sh_d["output_refined_aligned"]["old_tail_pct"], sh_d["output_refined_aligned"]["new_tail_pct"]),
        ("Low-DOF Adp.", sh_d["low_dof_adapter"]["old_tail_pct"], sh_d["low_dof_adapter"]["new_tail_pct"]),
        ("Adp.+Out.-Ref.", sh_d["adapter_output_refined_aligned"]["old_tail_pct"], sh_d["adapter_output_refined_aligned"]["new_tail_pct"]),
        ("Ground Truth", gt_sh["ho_old_tail_pct"], gt_sh["ho_new_tail_pct"]),
    ]
    x = np.arange(len(items))
    w = 0.36
    v_old = [t[1] for t in items]
    v_new = [t[2] for t in items]
    b1 = ax.bar(x - w / 2, v_old, w, color=PALETTE["vermillion"], alpha=0.88, label=r"Default $\Delta c=5\times 10^{-3}$")
    b2 = ax.bar(x + w / 2, v_new, w, color=PALETTE["teal"], label=r"Converged $\Delta c=2.5\times 10^{-5}$")
    for bi, (bar, v) in enumerate(zip(b1, v_old)):
        dx = -0.09 if bi in (0, 3, 5) else -0.03
        ax.text(bar.get_x() + bar.get_width() / 2 + dx, v + 0.20, f"{v:.2f}", ha="center", va="bottom", fontsize=6.1)
    for bi, (bar, v) in enumerate(zip(b2, v_new)):
        dx = 0.10 if bi in (0, 3, 5) else 0.03
        ax.text(bar.get_x() + bar.get_width() / 2 + dx, v + 0.20, f"{v:.2f}", ha="center", va="bottom", fontsize=6.1)
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in items], fontsize=6.4, rotation=32, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_ylabel("Tail Rate (>20% HO, %)", fontsize=7.6)
    ax.set_ylim(0, max(v_old) * 1.30)
    ax.set_title("(d) Elimination of Artificial Tail Failures", fontsize=8.0, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.4)

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 4: OpenSWI Noise-Floor-Normalized Signature Curves + Holdout ES
# ---------------------------------------------------------------------------


def make_fig04_openswi_signature_curves(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 4: OpenSWI three-domain noise-floor signature curves with Holdout ES."""
    apply_cg_style()
    cb = _load_json(summary_dir, "cross_backbone_diagnostics.json")

    domains = [
        ("shallow_multi_structure", "(a) Shallow Multi-Structure"),
        ("shallow_prior_shift", "(b) Shallow Prior-Shift"),
        ("deep_lithospheric", "(c) Deep Lithospheric"),
    ]

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(98.0)))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.6, 1.0], hspace=0.08, wspace=0.24, left=0.09, right=0.985, top=0.86, bottom=0.16)

    for col, (dk, title) in enumerate(domains):
        d_blk = cb["domains"][dk]
        osw = d_blk["by_backbone"]["openswi"]["all_judged_ckpts"]
        bins = osw["pooled_signature_bins"]["bins"]
        bx = np.arange(len(bins))
        blabs = _fmt_bin_labels([b["bin_label"] for b in bins])
        fracs = [b["fraction_pct"] for b in bins]

        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col], sharex=ax_top)

        ax_top.axhline(0.0, color="#333333", lw=0.8, ls="--", zorder=2)

        series_specs = [
            ("output_refined_50", "Output-Space Refinement (50-step)", PALETTE["vermillion"], "o", "-"),
            ("holdout_es", "Holdout ES", PALETTE["blue"], "s", "-"),
            ("ideal_es", "Ideal ES (oracle bound)", PALETTE["teal"], "^", "--"),
        ]
        all_meds = []
        for mk, lbl, color, marker, ls in series_specs:
            med = np.array([b["methods"][mk]["median_rel_change_pct"] for b in bins], dtype=float)
            q25 = np.array([b["methods"][mk]["q25_rel_change_pct"] for b in bins], dtype=float)
            q75 = np.array([b["methods"][mk]["q75_rel_change_pct"] for b in bins], dtype=float)
            low_n = np.array([bool(b["low_n_flag"]) for b in bins], dtype=bool)
            all_meds.extend(med[np.isfinite(med)].tolist())

            ax_top.fill_between(bx, q25, q75, color=color, alpha=0.13, zorder=1)
            ax_top.plot(bx, med, ls=ls, color=color, lw=1.45, label=lbl, zorder=4)
            # Solid markers for well-populated bins, open markers for low-n (<200)
            ax_top.scatter(bx[~low_n], med[~low_n], color=color, marker=marker, s=24, zorder=5)
            if np.any(low_n):
                ax_top.scatter(bx[low_n], med[low_n], facecolors="white", edgecolors=color, marker=marker, s=24, lw=1.0, zorder=5)

        y_lo = min(-55.0, min(all_meds) - 18.0)
        y_hi = max(80.0, max(all_meds) * 1.28)
        if dk == "deep_lithospheric":
            y_lo, y_hi = -42.0, 160.0
        elif dk == "shallow_prior_shift":
            y_lo, y_hi = -92.0, 145.0
        else:
            y_lo, y_hi = -92.0, 185.0
        ax_top.set_ylim(y_lo, y_hi)
        ax_top.axhspan(0.0, y_hi, color=PALETTE["warn_tint"], alpha=0.35, zorder=0)
        ax_top.axhspan(y_lo, 0.0, color=PALETTE["safe_tint"], alpha=0.35, zorder=0)

        # Annotate overall mean Δ Vs RMSE for the 3 methods (top-left on Deep to clear rising right tail)
        o50_ov = osw["output_refined_50"]["mean_rel_change_vs_aligned_pct"]
        hes_ov = osw["holdout_es"]["mean_rel_change_vs_aligned_pct"]
        ies_ov = osw["ideal_es"]["mean_rel_change_vs_aligned_pct"]
        is_deep = dk == "deep_lithospheric"
        ax_top.text(
            0.04 if is_deep else 0.97,
            0.96,
            f"Overall ΔRMSE:\n50-step: {o50_ov:+.1f}%\nHoldout ES: {hes_ov:+.1f}%\nIdeal ES: {ies_ov:+.1f}%",
            transform=ax_top.transAxes,
            ha="left" if is_deep else "right",
            va="top",
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
        )

        ax_top.set_title(title, fontsize=8.4, fontweight="bold")
        ax_top.tick_params(axis="y", labelsize=7.0)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        if col == 0:
            ax_top.set_ylabel("Per-Sample Δ $V_s$ RMSE (%)\n(median & IQR)", fontsize=7.8)

        # Bottom histogram: sample share per bin (including <1% bins such as 0.5%)
        ax_bot.bar(bx, fracs, width=0.68, color=PALETTE["sky"], edgecolor=PALETTE["blue"], linewidth=0.5)
        for xi, fv in zip(bx, fracs):
            if fv >= 0.3:
                lbl_f = f"{fv:.1f}" if fv < 1.0 else f"{fv:.0f}"
                ax_bot.text(xi, fv + 1.4, lbl_f, ha="center", va="bottom", fontsize=6.4, color="#333333")
        ax_bot.set_ylim(0, max(fracs) * 1.42)
        ax_bot.set_xticks(bx)
        ax_bot.set_xticklabels(blabs, rotation=34, ha="right", fontsize=6.8)
        ax_bot.tick_params(axis="y", labelsize=6.8)
        ax_bot.set_xlabel(r"Initial Misfit / GT Noise ($r_0^{\mathrm{GT}}$)", fontsize=7.4)
        if col == 0:
            ax_bot.set_ylabel("Share (%)", fontsize=7.4)

    handles, labels = axes_handles = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.995),
        fontsize=7.4,
        frameon=True,
    )
    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 5: Cross-Backbone Signature Overlay (OpenSWI vs DispFormer)
# ---------------------------------------------------------------------------


def make_fig05_cross_backbone_signatures(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 5: OpenSWI vs DispFormer signature curves and bin-share comparison."""
    apply_cg_style()
    cb = _load_json(summary_dir, "cross_backbone_diagnostics.json")

    domains = [
        ("shallow_multi_structure", "(a) Shallow Multi-Structure"),
        ("shallow_prior_shift", "(b) Shallow Prior-Shift"),
        ("deep_lithospheric", "(c) Deep Lithospheric"),
    ]

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(112.0)))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.5, 1.15], hspace=0.10, wspace=0.24, left=0.09, right=0.985, top=0.85, bottom=0.15)

    for col, (dk, title) in enumerate(domains):
        d_blk = cb["domains"][dk]
        osw_bins = d_blk["by_backbone"]["openswi"]["all_judged_ckpts"]["pooled_signature_bins"]["bins"]
        dsp_bins = d_blk["by_backbone"]["dispformer"]["all_judged_ckpts"]["pooled_signature_bins"]["bins"]
        bx = np.arange(len(osw_bins))
        blabs = _fmt_bin_labels([b["bin_label"] for b in osw_bins])

        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col], sharex=ax_top)

        ax_top.axhline(0.0, color="#333333", lw=0.8, ls="--", zorder=2)

        def _plot_curve(bins_list: List[Dict[str, Any]], method_k: str, color: str, ls: str, marker: str, lbl: str) -> np.ndarray:
            med = np.array([b["methods"][method_k]["median_rel_change_pct"] for b in bins_list], dtype=float)
            low_n = np.array([bool(b["low_n_flag"]) for b in bins_list], dtype=bool)
            ax_top.plot(bx, med, ls=ls, color=color, lw=1.4, label=lbl, zorder=4)
            ax_top.scatter(bx[~low_n], med[~low_n], color=color, marker=marker, s=22, zorder=5)
            if np.any(low_n):
                ax_top.scatter(bx[low_n], med[low_n], facecolors="white", edgecolors=color, marker=marker, s=22, lw=1.0, zorder=5)
            return med

        _plot_curve(osw_bins, "output_refined_50", PALETTE["vermillion"], "-", "o", "OpenSWI: 50-step")
        _plot_curve(dsp_bins, "output_refined_50", PALETTE["blue"], "--", "s", "DispFormer: 50-step")
        _plot_curve(osw_bins, "holdout_es", PALETTE["teal"], "-", "^", "OpenSWI: Holdout ES")
        _plot_curve(dsp_bins, "holdout_es", PALETTE["purple"], ":", "v", "DispFormer: Holdout ES")

        if dk == "deep_lithospheric":
            # Highlight Retrained-Only cross-backbone consistency in [0, 1) cleanly via diamonds (values in legend/caption/Table S2)
            osw_ret = d_blk["by_backbone"]["openswi"]["retrained_only"]["pooled_signature_bins"]["bins"][0]
            dsp_ret = d_blk["by_backbone"]["dispformer"]["retrained_only"]["pooled_signature_bins"]["bins"][0]
            osw_ret_v = osw_ret["methods"]["output_refined_50"]["median_rel_change_pct"]
            dsp_ret_v = dsp_ret["methods"]["output_refined_50"]["median_rel_change_pct"]
            ax_top.scatter([0.08], [osw_ret_v], marker="D", s=34, color=PALETTE["vermillion"], edgecolors="#111111", lw=0.7, zorder=7)
            ax_top.scatter([-0.08], [dsp_ret_v], marker="D", s=34, color=PALETTE["blue"], edgecolors="#111111", lw=0.7, zorder=7)
            ax_top.set_ylim(-42.0, 160.0)
        elif dk == "shallow_prior_shift":
            ax_top.set_ylim(-92.0, 125.0)
        else:
            ax_top.set_ylim(-92.0, 180.0)

        yl0, yl1 = ax_top.get_ylim()
        ax_top.axhspan(0.0, yl1, color=PALETTE["warn_tint"], alpha=0.30, zorder=0)
        ax_top.axhspan(yl0, 0.0, color=PALETTE["safe_tint"], alpha=0.30, zorder=0)
        ax_top.set_title(title, fontsize=8.4, fontweight="bold")
        ax_top.tick_params(axis="y", labelsize=7.0)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        if col == 0:
            ax_top.set_ylabel("Median Δ $V_s$ RMSE (%)", fontsize=7.8)

        # Bottom paired bin-share bars (OpenSWI vs DispFormer)
        w = 0.38
        f_osw = [b["fraction_pct"] for b in osw_bins]
        f_dsp = [b["fraction_pct"] for b in dsp_bins]
        ax_bot.bar(bx - w / 2, f_osw, w, color=PALETTE["vermillion"], alpha=0.75, label="OpenSWI")
        ax_bot.bar(bx + w / 2, f_dsp, w, color=PALETTE["blue"], alpha=0.75, label="DispFormer")
        ax_bot.set_ylim(0, max(max(f_osw), max(f_dsp)) * 1.28)
        ax_bot.set_xticks(bx)
        ax_bot.set_xticklabels(blabs, rotation=34, ha="right", fontsize=6.8)
        ax_bot.tick_params(axis="y", labelsize=6.8)
        ax_bot.set_xlabel(r"Initial Misfit / GT Noise ($r_0^{\mathrm{GT}}$)", fontsize=7.4)
        if col == 0:
            ax_bot.set_ylabel("Bin Share (%)", fontsize=7.4)
            ax_bot.legend(loc="upper right", fontsize=6.4, handlelength=1.0)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["blue"], markeredgecolor="#111111", markersize=5.0, label="Retrained-only [0, 1)"))
    labels.append("Retrained-only [0, 1)")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.995),
        fontsize=6.8,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 6: Noise Amplitude Scan (0.5x - 4.0x) & Linear Noise-Floor Tracking
# ---------------------------------------------------------------------------


def make_fig06_noise_amplitude_scan(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 6: noise-amplitude scaling and calibrated sigma_hat_D linear tracking."""
    apply_cg_style()
    cb = _load_json(summary_dir, "cross_backbone_diagnostics.json")
    ns = cb["noise_amplitude_scan"]
    order = ["noise_0p5x", "noise_1p0x", "noise_2p0x", "noise_4p0x"]
    scales = np.array([ns[k]["scale"] for k in order], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, mm_to_in(86.0)))
    plt.subplots_adjust(wspace=0.28, left=0.095, right=0.98, top=0.88, bottom=0.17)

    # Panel (a): Δ Resolvable-Depth Vs RMSE (%) vs Noise Multiplier with 95% CIs
    curves = [
        ("step5", "Output-Space Refinement (5-step)", PALETTE["orange"], "s", "-"),
        ("step20", "Output-Space Refinement (20-step)", PALETTE["sky"], "D", "-"),
        ("step50", "Output-Space Refinement (50-step)", PALETTE["vermillion"], "o", "-"),
        ("holdout_es", "Holdout ES", PALETTE["blue"], "P", "-"),
        ("ideal_es", "Ideal ES (oracle bound)", PALETTE["teal"], "^", "--"),
    ]
    ax1.axhline(0.0, color="#333333", lw=0.8, ls="--")
    for ck, lbl, color, marker, ls in curves:
        pts, lo_err, hi_err = [], [], []
        has_ci = True
        for k in order:
            osw = ns[k]["openswi"]["all_judged_ckpts"]
            if ck == "ideal_es":
                ent = osw["ideal_es"]
            elif ck == "holdout_es":
                ent = osw.get("holdout_es")
            elif ck == "step50":
                ent = osw["output_refined_50"]
            else:
                st_num = ck.replace("step", "")
                ent = osw["output_refined_by_step"][st_num]
            if ent is None:
                has_ci = False
                break
            pt = float(ent["mean_rel_change_vs_aligned_pct"])
            pts.append(pt)
            if "cluster_bootstrap" in ent:
                ci = ent["cluster_bootstrap"]["ci95_rel_change_pct"]
                lo_err.append(max(0.0, pt - float(ci[0])))
                hi_err.append(max(0.0, float(ci[1]) - pt))
            else:
                has_ci = False
        if len(pts) != len(order):
            continue
        if has_ci:
            ax1.errorbar(
                scales,
                pts,
                yerr=[lo_err, hi_err],
                color=color,
                marker=marker,
                ls=ls,
                lw=1.45,
                capsize=2.8,
                capthick=0.9,
                label=f"{lbl} (95% CI)",
            )
        else:
            ax1.plot(scales, pts, color=color, marker=marker, ls=ls, lw=1.35, label=lbl)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(scales)
    ax1.set_xticklabels([f"{s:g}×" for s in scales])
    ax1.set_xlabel("Dispersion Noise Amplitude Multiplier")
    ax1.set_ylabel("Δ Resolvable-Depth $V_s$ RMSE (%)")
    ax1.set_title("(a) TTA Error Transition Across Noise Levels", fontsize=8.0, fontweight="bold")
    ax1.axhspan(0, 62, color=PALETTE["warn_tint"], alpha=0.35, zorder=0)
    ax1.axhspan(-45, 0, color=PALETTE["safe_tint"], alpha=0.35, zorder=0)
    ax1.set_ylim(-43, 62)
    ax1.tick_params(labelsize=7.2)
    ax1.legend(loc="upper left", fontsize=6.4)

    # Panel (b): Calibrated sigma_hat_D vs Ground-Truth Noise Floor linear tracking
    gt_vals = np.array([ns[k]["sigma_hat_d_vs_gt_tracking"]["gt_test_4000_median"] * 100.0 for k in order])
    sig_vals = np.array([ns[k]["sigma_hat_d_vs_gt_tracking"]["sigma_hat_D"] * 100.0 for k in order])
    aligned_vals = np.array([ns[k]["zero_shot_aligned_kept_misfit_median"] * 100.0 for k in order])
    ratios = np.array([ns[k]["sigma_hat_d_vs_gt_tracking"]["sigma_hat_D_over_gt_test"] for k in order])

    ax2.plot(scales, gt_vals, "-o", color="#1A1A1A", lw=1.5, label="Ground-Truth Noise Floor (%)")
    ax2.plot(scales, sig_vals, "--s", color=PALETTE["teal"], lw=1.5, label=r"Calibrated $\hat{\sigma}_D$ (Classical Inv.)")
    ax2.plot(scales, aligned_vals, ":^", color=PALETTE["vermillion"], lw=1.4, label="Input-Aligned Zero-Shot Kept Misfit (%)")

    for i_s, (s, gv, sv, r) in enumerate(zip(scales, gt_vals, sig_vals, ratios)):
        off = (14, -4) if i_s == 0 else (0, -12)
        ax2.annotate(
            f"{r:.2f}×",
            xy=(s, sv),
            xytext=off,
            textcoords="offset points",
            ha="left" if i_s == 0 else "center",
            fontsize=6.8,
            color="#005B42",
            fontweight="bold",
        )

    ax2.set_xscale("log", base=2)
    ax2.set_xticks(scales)
    ax2.set_xticklabels([f"{s:g}×" for s in scales])
    ax2.tick_params(labelsize=7.2)
    ax2.set_xlabel("Dispersion Noise Amplitude Multiplier")
    ax2.set_ylabel("Kept-Period Dispersion Misfit Median (%)")
    ax2.set_title(r"(b) Linear Tracking of $\hat{\sigma}_D$ vs GT Floor", fontsize=8.0, fontweight="bold")
    ax2.set_ylim(0, max(aligned_vals) * 1.22)
    ax2.legend(loc="upper left", fontsize=6.6)
    ax2.text(
        0.96,
        0.08,
        f"Ratio $\\hat{{\\sigma}}_D$ / GT:\n{ratios.min():.3f} – {ratios.max():.3f}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=PALETTE["safe_tint"], edgecolor=PALETTE["teal"], lw=0.75),
    )

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 7: Pre-Registered Safe-TTA Validation Failure Attribution (2x2)
# ---------------------------------------------------------------------------


def make_fig07_safe_tta_failure_attribution(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 7: four-panel validation failure attribution for Pre-Registered Safe-TTA."""
    apply_cg_style()
    st = _load_json(summary_dir, "safe_tta_validation_audit.json")

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, mm_to_in(150.0)))
    plt.subplots_adjust(wspace=0.30, hspace=0.46, left=0.095, right=0.975, top=0.94, bottom=0.11)

    # Panel (a): Shallow Benchmark (Val) Misfit/sigma_hat_D bins
    ax = axes[0, 0]
    sh_mb = st["misfit_over_sigma_bins"]["shallow_benchmark_val"]
    blabs = _fmt_bin_labels(sh_mb["bin_labels"])
    bx = np.arange(len(blabs))
    all_b = sh_mb["by_series"]["all_checkpoints"]["bins"]
    flt_b = sh_mb["by_series"]["flat_trained"]["bins"]
    ax.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax.plot(bx, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in all_b], "-o", color=PALETTE["blue"], label="Safe-TTA (α=1.25, all)")
    ax.plot(bx, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in flt_b], "--s", color=PALETTE["sky"], label="Safe-TTA (flat-trained)")
    ax.plot(bx, [b["methods"]["output_refined_step50"]["mean_rel_change_vs_aligned_pct"] for b in all_b], "-^", color=PALETTE["vermillion"], label="Output-Space Ref. (50)")
    ax.axvspan(0.5, 2.5, color="#FFF3CD", alpha=0.55, zorder=0)
    ax.text(1.5, 148, "Near-floor\ndegradation", ha="center", va="top", fontsize=6.6, color="#8A5A00", fontweight="bold")
    ax.set_xticks(bx)
    ax.set_xticklabels(blabs, rotation=28, ha="right", fontsize=6.8)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_xlabel(r"Input-Aligned Misfit / $\hat{\sigma}_D$", fontsize=7.6)
    ax.set_ylabel("Mean Δ $V_s$ RMSE (%)", fontsize=7.8)
    ax.set_ylim(-55, 160)
    ax.set_title("(a) Shallow Benchmark (Val): Near-Floor", fontsize=8.0, fontweight="bold")
    ax.legend(loc="lower left", fontsize=6.4)

    # Panel (b): Deep Benchmark (Val) Misfit/sigma_hat_D bins
    ax = axes[0, 1]
    dp_mb = st["misfit_over_sigma_bins"]["deep_benchmark_val"]
    blabs_d = _fmt_bin_labels(dp_mb["bin_labels"])
    bx_d = np.arange(len(blabs_d))
    dp_all = dp_mb["by_series"]["all_checkpoints"]["bins"]
    dp_ood = dp_mb["by_series"]["ood_excluded"]["bins"]
    dp_ret = dp_mb["by_series"]["all_type_retrained"]["bins"]
    ax.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax.plot(bx_d, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in dp_all], "-o", color=PALETTE["blue"], label="Safe-TTA (all 7 ckpts)")
    ax.plot(bx_d, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in dp_ood], "--s", color=PALETTE["orange"], label="Safe-TTA (OOD-excl.)")
    ax.plot(bx_d, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in dp_ret], ":d", color=PALETTE["teal"], label="Safe-TTA (all-type ret.)")
    ax.plot(bx_d, [b["methods"]["output_refined_step50"]["mean_rel_change_vs_aligned_pct"] for b in dp_all], "-^", color=PALETTE["vermillion"], label="Output-Space Ref. (50)")
    ax.axvspan(5.5, 6.45, color=PALETTE["warn_tint"], alpha=0.75, zorder=0)
    y_max_dp = max(
        max(b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in dp_ret),
        max(b["methods"]["output_refined_step50"]["mean_rel_change_vs_aligned_pct"] for b in dp_all),
    )
    ax.set_ylim(-26, max(130.0, y_max_dp * 1.30))
    ax.text(5.90, ax.get_ylim()[1] * 0.96, "High-misfit\ndegradation", ha="center", va="top", fontsize=6.4, color="#9A2D00", fontweight="bold")
    ax.set_xticks(bx_d)
    ax.set_xticklabels(blabs_d, rotation=28, ha="right", fontsize=6.8)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_xlabel(r"Input-Aligned Misfit / $\hat{\sigma}_D$", fontsize=7.6)
    ax.set_ylabel("Mean Δ $V_s$ RMSE (%)", fontsize=7.8)
    ax.set_title("(b) Deep Benchmark (Val): High-Misfit", fontsize=8.0, fontweight="bold")
    ax.legend(loc="upper left", fontsize=6.4)

    # Panel (c): Outcome-Group Decomposition at alpha=1.25
    ax = axes[1, 0]
    ds_specs = [
        ("longbeach_matched_val", "LB-Matched\n(Val)"),
        ("shallow_benchmark_val", "Shallow\n(Val)"),
        ("deep_benchmark_val", "Deep\n(Val)"),
    ]
    # Plot only active refinement groups (Step-0 Stop and Reverted contribute 0.0% by definition)
    grp_specs = [
        ("mid_stop", "Mid-Stop", PALETTE["orange"]),
        ("ran_50_steps", "Ran 50 Steps", PALETTE["vermillion"]),
    ]
    x = np.arange(len(ds_specs))
    w = 0.28
    ax.axhline(0.0, color="#333333", lw=0.8)
    for gi, (gk, glbl, gcol) in enumerate(grp_specs):
        vals = [
            st["outcome_group_decomposition"][dk]["by_alpha"]["1.25"]["series_mean"]["all_checkpoints"]["groups"][gk]["mean_rel_contribution_pct"]
            for dk, _ in ds_specs
        ]
        bars = ax.bar(x + (gi - 0.5) * w, vals, w, color=gcol, label=glbl)
        for bi, (bar, v) in enumerate(zip(bars, vals)):
            if abs(v) >= 0.4:
                dx = -0.07 if (gk == "mid_stop" and bi == 1) else 0.0
                ax.text(
                    bar.get_x() + bar.get_width() / 2 + dx,
                    v + (1.2 if v >= 0 else -1.6),
                    f"{v:+.1f}%",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=6.4,
                )
    for di, (dk, _) in enumerate(ds_specs):
        tot = st["outcome_group_decomposition"][dk]["by_alpha"]["1.25"]["series_mean"]["all_checkpoints"]["mean_total_rel_change_vs_aligned_pct"]
        ax.scatter([di], [tot], marker="D", s=28, color="#111111", zorder=6, label="Net Total" if di == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in ds_specs], fontsize=7.0)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_ylabel("Signed Contribution to Δ $V_s$ RMSE (%)", fontsize=7.6)
    ax.set_ylim(-58, 26)
    ax.set_title("(c) Outcome-Group Attribution (α = 1.25)", fontsize=8.0, fontweight="bold")
    ax.legend(loc="lower right", ncol=2, fontsize=6.4, columnspacing=0.6)

    # Panel (d): Deep >5*sigma_hat_D Attribution (EXPLICIT Ground-Truth Misfit label!)
    ax = axes[1, 1]
    dh = st["deep_high_misfit_attribution"]
    ck_order = [
        ("ood_excluded_seed0", "OOD-excl. s0"),
        ("ood_excluded_seed1", "OOD-excl. s1"),
        ("ood_excluded_seed2", "OOD-excl. s2"),
        ("all_type_seed0", "All-type s0"),
        ("all_type_seed1", "All-type s1"),
        ("all_type_seed2", "All-type s2"),
        ("pretrained_official", "Official"),
    ]
    cx = np.arange(len(ck_order))
    aligned_med = [dh["per_ckpt"][ck]["thresholds"]["gt_5sigma"]["aligned_misfit"]["median"] * 100.0 for ck, _ in ck_order]
    gt_med = [dh["per_ckpt"][ck]["thresholds"]["gt_5sigma"]["gt_profile_misfit_subset"]["median"] * 100.0 for ck, _ in ck_order]
    gt_all_med = dh["gt_profile_all_5000"]["combined_misfit"]["median"] * 100.0

    w = 0.36
    ax.bar(cx - w / 2, aligned_med, w, color=PALETTE["vermillion"], alpha=0.88, label="Aligned ZS Misfit (%)")
    ax.bar(cx + w / 2, gt_med, w, color=PALETTE["teal"], label="Ground-Truth Misfit (%)")
    ax.axhline(gt_all_med, color=PALETTE["teal"], lw=0.9, ls="--", label=f"Full-Set GT ({gt_all_med:.2f}%)")
    ax.set_xticks(cx)
    ax.set_xticklabels([lbl for _, lbl in ck_order], rotation=28, ha="right", fontsize=6.5, rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_ylabel("Kept Misfit Median (%)", fontsize=7.6)
    ax.set_ylim(0, max(aligned_med) * 1.38)
    ax.set_title(r"(d) Deep $>5\hat{\sigma}_D$: Initial vs. GT Misfit", fontsize=8.0, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.2, framealpha=0.95)

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 8: LongBeach Report 3D Depth Slices (4 depths x 5 methods)
# ---------------------------------------------------------------------------


def make_fig08_longbeach_slices(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 8: LongBeach (Report 70%) horizontal Vs slices at 4 depths."""
    apply_cg_style()
    z = np.load(os.path.join(summary_dir, "field_slices_longbeach.npz"))
    coords = z["coords_report"]
    coords_cls = z["coords_classical"]
    depths = z["slice_depths_km"]

    cols = [
        ("Input-Aligned\nZero-Shot\n($r_{\\mathrm{ref}}=0.551$)", coords, z["slices_zero_shot_aligned"], 3.0),
        ("Low-DOF\nAdapter\n($r_{\\mathrm{ref}}=0.632$)", coords, z["slices_low_dof_adapter"], 3.0),
        ("Gated Adapter +\nOutput-Refined\n($r_{\\mathrm{ref}}=0.710$)", coords, z["slices_adapter_output_refined_aligned"], 3.0),
        ("Classical Inv.\n(499 valid /\n500 sampled)", coords_cls, z["slices_classical"], 7.5),
        ("Reference Model\n(Fu et al.,\n2022)", coords, z["slices_reference"], 3.0),
    ]
    st0_xy = z["station_0__coords"]
    st1_xy = z["station_1__coords"]

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(154.0)))
    gs = gridspec.GridSpec(
        4,
        6,
        width_ratios=[1, 1, 1, 1, 1, 0.09],
        wspace=0.07,
        hspace=0.11,
        left=0.10,
        right=0.915,
        top=0.91,
        bottom=0.08,
    )

    cmap = plt.get_cmap("RdYlBu")
    for r in range(4):
        ref_vals = np.concatenate([z["slices_zero_shot_aligned"][:, r], z["slices_reference"][:, r]])
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        vmin, vmax = float(np.percentile(ref_vals, 2)), float(np.percentile(ref_vals, 98))
        sc_last = None
        for c, (cname, xy, mat, pt_s) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            vals = mat[:, r]
            ok = np.isfinite(vals)
            sc_last = ax.scatter(
                xy[ok, 0],
                xy[ok, 1],
                c=vals[ok],
                s=pt_s,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.0,
                rasterized=True,
            )
            if r == 0:
                ax.scatter([st0_xy[0], st1_xy[0]], [st0_xy[1], st1_xy[1]], marker="*", s=34, color="#111111", edgecolors="white", lw=0.5, zorder=6)
            ax.plot(
                [coords[:, 0].min(), coords[:, 0].max(), coords[:, 0].max(), coords[:, 0].min(), coords[:, 0].min()],
                [coords[:, 1].min(), coords[:, 1].min(), coords[:, 1].max(), coords[:, 1].max(), coords[:, 1].min()],
                "--", color="#666666", lw=0.45, alpha=0.7, zorder=4,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set_linewidth(0.55)
            ax.set_xlim(coords[:, 0].min() - 0.004, coords[:, 0].max() + 0.004)
            ax.set_ylim(coords[:, 1].min() - 0.004, coords[:, 1].max() + 0.004)
            if r == 0:
                ax.set_title(cname, fontsize=6.8, fontweight="bold", pad=3)
            if c == 0:
                ax.set_ylabel(f"z = {depths[r]:.2f} km\nLat (°N)", fontsize=7.0)
                ax.tick_params(axis="y", labelsize=6.5)
            else:
                ax.set_yticklabels([])
            if r == 3:
                ax.set_xticks([-118.20, -118.17, -118.14])
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
                ax.set_xlabel("Lon (°E)", fontsize=7.0)
                ax.tick_params(axis="x", labelsize=6.5, rotation=24)
            else:
                ax.set_xticks([-118.20, -118.17, -118.14])
                ax.set_xticklabels([])

        cax = fig.add_subplot(gs[r, 5])
        cb = fig.colorbar(sc_last, cax=cax)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("$V_s$ (km/s)", fontsize=6.9)

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 9: CSRM Depth Slices + Tibetan Plateau Station #10067 Diagnostics
# ---------------------------------------------------------------------------


def make_fig09_csrm_slices_and_station(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 9: CSRM (Report) depth slices and Tibetan Plateau station #10067 analysis."""
    apply_cg_style()
    z = np.load(os.path.join(summary_dir, "field_slices_csrm.npz"))
    coords = z["coords_report"]
    coords_cls = z["coords_classical"]
    depths = z["slice_depths_km"]

    cols = [
        ("Input-Aligned\nZero-Shot\n($r_{\\mathrm{ref}}=0.686$)", coords, z["slices_zero_shot_aligned"], 1.4),
        ("Low-DOF\nAdapter\n($r_{\\mathrm{ref}}=0.806$)", coords, z["slices_low_dof_adapter"], 1.4),
        ("Gated Adapter +\nOutput-Refined\n($r_{\\mathrm{ref}}=0.748$)", coords, z["slices_adapter_output_refined_aligned"], 1.4),
        ("Classical Inversion\n(500 sampled\ngrid pts)", coords_cls, z["slices_classical"], 5.5),
        ("Reference\nModel\n(CSRM)", coords, z["slices_reference"], 1.4),
    ]

    # Station 1 is #10067 (86.80E, 28.60N, Tibetan Plateau)
    st_id = int(z["station_1__global_id"][0])
    st_xy = z["station_1__coords"]

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(204.0)))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.65, 1.0], hspace=0.24, left=0.09, right=0.915, top=0.945, bottom=0.06)

    # Top block: 4 x 5 CSRM depth slices + right colorbar column
    gs_top = gridspec.GridSpecFromSubplotSpec(4, 6, subplot_spec=outer[0], width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.06, hspace=0.09)
    cmap = plt.get_cmap("RdYlBu")

    for r in range(4):
        ref_vals = np.concatenate([z["slices_zero_shot_aligned"][:, r], z["slices_reference"][:, r]])
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        vmin, vmax = float(np.percentile(ref_vals, 2)), float(np.percentile(ref_vals, 98))
        sc_last = None
        for c, (cname, xy, mat, pt_s) in enumerate(cols):
            ax = fig.add_subplot(gs_top[r, c])
            vals = mat[:, r]
            ok = np.isfinite(vals)
            sc_last = ax.scatter(
                xy[ok, 0],
                xy[ok, 1],
                c=vals[ok],
                s=pt_s,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.0,
                rasterized=True,
            )
            # Mark Station #10067 across all 5 columns on 35 km row & delineate study area
            if r == 1:
                ax.scatter([st_xy[0]], [st_xy[1]], marker="*", s=42, color="#111111", edgecolors="white", lw=0.6, zorder=6)
            ax.plot(
                [coords[:, 0].min(), coords[:, 0].max(), coords[:, 0].max(), coords[:, 0].min(), coords[:, 0].min()],
                [coords[:, 1].min(), coords[:, 1].min(), coords[:, 1].max(), coords[:, 1].max(), coords[:, 1].min()],
                "--", color="#666666", lw=0.45, alpha=0.65, zorder=4,
            )
            ax.grid(False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set_linewidth(0.55)
            ax.set_xlim(coords[:, 0].min() - 1.5, coords[:, 0].max() + 1.5)
            ax.set_ylim(coords[:, 1].min() - 1.5, coords[:, 1].max() + 1.5)
            if r == 0:
                ax.set_title(cname, fontsize=6.8, fontweight="bold", pad=3)
            if c == 0:
                ax.set_ylabel(f"({chr(97 + r)}) z = {depths[r]:.0f} km\nLat (°N)", fontsize=6.8)
                ax.tick_params(axis="y", labelsize=6.5)
            else:
                ax.set_yticklabels([])
            if r == 3:
                ax.set_xlabel("Lon (°E)", fontsize=6.8)
                ax.tick_params(axis="x", labelsize=6.5)
            else:
                ax.set_xticklabels([])

        cax = fig.add_subplot(gs_top[r, 5])
        cb = fig.colorbar(sc_last, cax=cax)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("$V_s$ (km/s)", fontsize=6.8)

    # Bottom block: Station #10067 dispersion fit (left) + 1D Vs(z) profile (right)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.28, width_ratios=[1.12, 1.0])

    # Panel (e): Station #10067 Phase & Group Dispersion
    ax_d = fig.add_subplot(gs_bot[0])
    T = z["station_1__periods"]
    ph_obs = z["station_1__phase_obs"]
    gp_obs = z["station_1__group_obs"]
    km = z["station_1__kept_mask"] > 0
    hm = z["station_1__ho_mask"] > 0

    ax_d.axvspan(10.0, 14.0, color="#FFF3CD", alpha=0.55, zorder=0, label="10–14 s bump")
    ax_d.plot(T[km[:, 0]], ph_obs[km[:, 0]], "o", color="#1A1A1A", ms=3.4, label="Ph. (kept)")
    ax_d.plot(T[hm[:, 0]], ph_obs[hm[:, 0]], "D", mfc="white", mec="#1A1A1A", ms=4.0, mew=0.9, label="Ph. (HO)")
    ax_d.plot(T[km[:, 1]], gp_obs[km[:, 1]], "^", color="#555555", ms=3.2, label="Gp. (kept)")
    ax_d.plot(T[hm[:, 1]], gp_obs[hm[:, 1]], "v", mfc="white", mec="#555555", ms=3.8, mew=0.9, label="Gp. (HO)")

    st_methods = [
        ("zero_shot_aligned", "Aligned ZS", PALETTE["black"], "--"),
        ("low_dof_adapter", "Low-DOF Adp.", PALETTE["teal"], "-."),
        ("adapter_output_refined_aligned", "Adp.+Out.-Ref.", PALETTE["blue"], "-"),
    ]
    for mk, lbl, col, ls in st_methods:
        fp = z[f"station_1__fwd_phase_{mk}"]
        fg = z[f"station_1__fwd_group_{mk}"]
        ho_p = float(z[f"station_1__ho_pct_{mk}"][0])
        ok_p = np.isfinite(fp) & (fp > 0)
        ok_g = np.isfinite(fg) & (fg > 0)
        ax_d.plot(T[ok_p], fp[ok_p], ls=ls, color=col, lw=1.35, label=f"{lbl} ({ho_p:.2f}%)")
        ax_d.plot(T[ok_g], fg[ok_g], ls=ls, color=col, lw=1.05, alpha=0.72)

    ax_d.set_xlabel("Period (s)", fontsize=7.6)
    ax_d.set_ylabel("Rayleigh Phase / Group Velocity (km/s)", fontsize=7.4)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_ylim(2.65, 4.42)
    ax_d.set_title(f"(e) Station #{st_id} ({st_xy[0]:.1f}°E, {st_xy[1]:.1f}°N): Dispersion", fontsize=7.8, fontweight="bold")
    ax_d.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=5.9, ncol=2, columnspacing=0.40, handlelength=1.1, framealpha=0.94)

    # Panel (f): Station #10067 1D Vs(z) Profile
    ax_v = fig.add_subplot(gs_bot[1])
    z_pred = z["pred_depths_km"]
    z_ref = z["ref_depths_km"]
    vs_ref = z["station_1__vs_ref"]

    ax_v.axhspan(10.0, 30.0, color=PALETTE["warn_tint"], alpha=0.65, zorder=0)
    ax_v.text(4.58, 20.0, "10–30 km\noscillation\nzone", ha="right", va="center", fontsize=6.4, color="#8C3800", fontweight="bold")
    ax_v.plot(vs_ref, z_ref, color=PALETTE["vermillion"], lw=1.7, label="Reference Model (CSRM)")
    for mk, lbl, col, ls in st_methods:
        vs_m = z[f"station_1__vs_{mk}"]
        ax_v.plot(vs_m, z_pred, ls=ls, color=col, lw=1.35, label=lbl)

    ax_v.set_ylim(120.0, 0.0)
    ax_v.set_xlim(2.6, 4.65)
    ax_v.set_xlabel("Shear-Wave Velocity $V_s$ (km/s)", fontsize=7.6)
    ax_v.set_ylabel("Depth (km)", fontsize=7.6)
    ax_v.tick_params(labelsize=6.8)
    ax_v.set_title(f"(f) Station #{st_id}: 1D $V_s(z)$ (Tibetan Plateau)", fontsize=7.8, fontweight="bold")
    ax_v.legend(loc="lower left", fontsize=6.3, handlelength=1.3)

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


def generate_all_main_figures(
    summary_dir: str,
    figures_dir: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Generate Graphical Abstract and all 9 main manuscript figures."""
    main_dir = os.path.join(figures_dir, "main")
    os.makedirs(main_dir, exist_ok=True)
    results: Dict[str, Dict[str, Dict[str, object]]] = {}

    results["graphical_abstract"] = make_graphical_abstract(
        summary_dir, os.path.join(figures_dir, "graphical_abstract"), formats=formats, dpi=360
    )
    builders = [
        ("fig01_workflow_and_protocol", make_fig01_workflow_and_protocol),
        ("fig02_benchmark_overview", make_fig02_benchmark_overview),
        ("fig03_disba_root_skipping", make_fig03_disba_root_skipping),
        ("fig04_openswi_signature_curves", make_fig04_openswi_signature_curves),
        ("fig05_cross_backbone_signatures", make_fig05_cross_backbone_signatures),
        ("fig06_noise_amplitude_scan", make_fig06_noise_amplitude_scan),
        ("fig07_safe_tta_failure_attribution", make_fig07_safe_tta_failure_attribution),
        ("fig08_longbeach_slices", make_fig08_longbeach_slices),
        ("fig09_csrm_slices_and_station", make_fig09_csrm_slices_and_station),
    ]
    for name, fn in builders:
        results[name] = fn(summary_dir, os.path.join(main_dir, name), formats=formats, dpi=dpi)
    return results
