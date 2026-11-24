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


def _draw_arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = "#333333",
    lw: float = 1.15,
    rad: float = 0.0,
) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.45,head_width=0.25",
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=4,
    )


# ---------------------------------------------------------------------------
# Figure 1: Diagnostic Evaluation Framework & Four-Step Safe-TTA Protocol
# (PaperBanana-inspired pure-vector scientific architecture diagram)
# ---------------------------------------------------------------------------


def _draw_shadow_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str = "#FFFFFF",
    edgecolor: str = "#CBD5E1",
    lw: float = 0.85,
    radius: float = 0.014,
    zorder: int = 2,
) -> None:
    shadow = patches.FancyBboxPatch(
        (x + 0.0035, y - 0.009),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=0.0,
        facecolor="#E2E8F0",
        alpha=0.65,
        zorder=zorder - 1,
    )
    ax.add_patch(shadow)
    card = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(card)


def _draw_module_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    step_num: str,
    title: str,
    footer: str,
    accent: str,
    tint: str,
) -> Tuple[float, float, float, float]:
    """Draw a sleek PaperBanana-style module card and return inner schematic bbox."""
    _draw_shadow_card(ax, x, y, w, h, facecolor="#FFFFFF", edgecolor=accent, lw=0.95, radius=0.015, zorder=2)

    hh = h * 0.23
    hy = y + h - hh
    header = patches.FancyBboxPatch(
        (x, hy),
        w,
        hh,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        linewidth=0.0,
        facecolor=tint,
        zorder=3,
    )
    ax.add_patch(header)
    ax.add_patch(patches.Rectangle((x, hy), w, hh * 0.45, linewidth=0.0, facecolor=tint, zorder=3))
    ax.plot([x + 0.002, x + w - 0.002], [hy, hy], color=accent, lw=0.75, alpha=0.55, zorder=4)

    cx = x + 0.016
    cy = hy + hh * 0.50
    badge = patches.Ellipse((cx, cy), width=0.022, height=hh * 0.62, facecolor=accent, edgecolor="none", zorder=5)
    ax.add_patch(badge)
    ax.text(cx, cy, step_num, ha="center", va="center", fontsize=5.6, fontweight="bold", color="white", zorder=6)

    ax.text(
        x + 0.030,
        cy,
        title,
        ha="left",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color="#0F172A",
        zorder=6,
    )

    ax.text(
        x + w / 2.0,
        y + h * 0.145,
        footer,
        ha="center",
        va="center",
        fontsize=5.35,
        color="#334155",
        linespacing=1.22,
        zorder=5,
    )

    return (x + w * 0.08, y + h * 0.29, w * 0.84, h * 0.43)


def _draw_accent_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    accent: str,
    tint: str,
    tag_text: str,
) -> None:
    """Draw a modern white card with bold left accent stripe and status pill."""
    _draw_shadow_card(ax, x, y, w, h, facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=0.8, radius=0.012, zorder=2)
    bg = patches.FancyBboxPatch(
        (x + 0.001, y + 0.001),
        w - 0.002,
        h - 0.002,
        boxstyle="round,pad=0.0,rounding_size=0.011",
        linewidth=0.0,
        facecolor=tint,
        alpha=0.52,
        zorder=2,
    )
    ax.add_patch(bg)
    stripe = patches.FancyBboxPatch(
        (x, y),
        0.010,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.006",
        linewidth=0.0,
        facecolor=accent,
        zorder=4,
    )
    ax.add_patch(stripe)

    pill_w = 0.098
    pill_h = h * 0.34
    px = x + w - pill_w - 0.010
    py = y + h - pill_h - h * 0.11
    pill = patches.FancyBboxPatch(
        (px, py),
        pill_w,
        pill_h,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        linewidth=0.0,
        facecolor=accent,
        zorder=5,
    )
    ax.add_patch(pill)
    ax.text(px + pill_w / 2.0, py + pill_h / 2.0, tag_text, ha="center", va="center", fontsize=5.0, fontweight="bold", color="white", zorder=6)

    ax.text(
        x + 0.020,
        y + h * 0.72,
        title,
        ha="left",
        va="center",
        fontsize=6.25,
        fontweight="bold",
        color="#0F172A",
        zorder=5,
    )
    ax.text(
        x + 0.020,
        y + h * 0.31,
        subtitle,
        ha="left",
        va="center",
        fontsize=5.5,
        color="#334155",
        linespacing=1.22,
        zorder=5,
    )


def make_fig01_workflow_and_protocol(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure 1: PaperBanana-style vector diagnostic pipeline and 4-step protocol."""
    apply_cg_style()
    rs = _load_json(summary_dir, "root_skipping_diagnostics.json")
    dc_sh = rs["step_sizes_km_s"]["shallow_converged"]
    dc_dp = rs["step_sizes_km_s"]["deep_converged"]

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(138.0)))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.02, 1.24], hspace=0.10, left=0.012, right=0.988, top=0.985, bottom=0.015)

    # Panel (a): Zero-Leakage Diagnostic Evaluation Pipeline
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.axis("off")

    band_a = patches.FancyBboxPatch(
        (0.005, 0.02),
        0.990,
        0.955,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0.85,
        edgecolor="#CBD5E1",
        facecolor="#F8FAFC",
        zorder=0,
    )
    ax_a.add_patch(band_a)
    ax_a.text(
        0.020,
        0.935,
        "(a) Zero-Leakage Diagnostic Evaluation Pipeline",
        fontsize=8.2,
        fontweight="bold",
        color="#0F172A",
        va="top",
        zorder=5,
    )

    cw = 0.174
    gap = 0.0215
    x0 = 0.018
    cy0, ch = 0.075, 0.765

    # Card 1: Dispersion Input
    gx, gy, gw, gh = _draw_module_card(
        ax_a, x0, cy0, cw, ch, "1", "Dispersion Input",
        "20% interior periods\nsealed before alignment", "#475569", "#F1F5F9"
    )
    ax_a.plot([gx, gx, gx + gw], [gy + gh, gy, gy], color="#64748B", lw=0.7, zorder=4)
    tx = np.linspace(0.08, 0.92, 9)
    ty = 0.18 + 0.68 * (1.0 - np.exp(-2.6 * tx))
    kept_idx = [0, 1, 3, 4, 6, 7, 8]
    ho_idx = [2, 5]
    ax_a.plot(gx + tx * gw, gy + ty * gh, "-", color="#94A3B8", lw=0.8, zorder=4)
    ax_a.scatter(gx + tx[kept_idx] * gw, gy + ty[kept_idx] * gh, s=11, color="#1E293B", zorder=5)
    ax_a.scatter(gx + tx[ho_idx] * gw, gy + ty[ho_idx] * gh, s=16, marker="D", facecolors="white", edgecolors=PALETTE["vermillion"], linewidths=0.9, zorder=6)
    ax_a.text(gx + gw * 0.96, gy + gh * 0.14, "Kept  /  HO", ha="right", va="bottom", fontsize=4.6, color="#475569", zorder=6)

    # Card 2: Input Alignment
    x1 = x0 + (cw + gap)
    gx, gy, gw, gh = _draw_module_card(
        ax_a, x1, cy0, cw, ch, "2", "Input Alignment",
        "Resample onto training\nperiod grid (zero leak)", PALETTE["blue"], "#EFF6FF"
    )
    y_top, y_bot = gy + gh * 0.72, gy + gh * 0.24
    ax_a.plot([gx + gw * 0.04, gx + gw * 0.96], [y_top, y_top], color="#94A3B8", lw=0.8, zorder=4)
    ax_a.plot([gx + gw * 0.04, gx + gw * 0.96], [y_bot, y_bot], color=PALETTE["blue"], lw=1.1, zorder=4)
    raw_pts = np.array([0.10, 0.24, 0.34, 0.56, 0.75, 0.88])
    uni_pts = np.linspace(0.10, 0.90, 7)
    ax_a.scatter(gx + raw_pts * gw, np.full_like(raw_pts, y_top), s=11, color="#64748B", zorder=5)
    ax_a.scatter(gx + uni_pts * gw, np.full_like(uni_pts, y_bot), s=12, marker="s", color=PALETTE["blue"], zorder=5)
    for rx in [0.18, 0.50, 0.82]:
        ax_a.annotate("", xy=(gx + rx * gw, y_bot + gh * 0.08), xytext=(gx + rx * gw, y_top - gh * 0.08),
                      arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.14", color=PALETTE["blue"], lw=0.7), zorder=5)
    ax_a.text(gx + gw * 0.5, gy + gh * 0.95, "Field periods", ha="center", va="top", fontsize=4.5, color="#64748B", zorder=6)
    ax_a.text(gx + gw * 0.5, gy + gh * 0.02, "Backbone grid", ha="center", va="bottom", fontsize=4.6, fontweight="bold", color=PALETTE["blue"], zorder=6)

    # Card 3: Backbone & TTA Loop
    x2 = x0 + 2 * (cw + gap)
    gx, gy, gw, gh = _draw_module_card(
        ax_a, x2, cy0, cw, ch, "3", "Backbone & TTA",
        "OpenSWI / DispFormer\n+ surrogate forward loop", "#D97706", "#FFFBEB"
    )
    trap = patches.Polygon(
        [[gx + gw * 0.03, gy + gh * 0.84], [gx + gw * 0.38, gy + gh * 0.72],
         [gx + gw * 0.38, gy + gh * 0.36], [gx + gw * 0.03, gy + gh * 0.24]],
        closed=True, facecolor="#FDE68A", edgecolor="#D97706", lw=0.8, zorder=4
    )
    ax_a.add_patch(trap)
    ax_a.text(gx + gw * 0.20, gy + gh * 0.54, "NN", ha="center", va="center", fontsize=5.2, fontweight="bold", color="#92400E", zorder=5)
    vs_box = patches.FancyBboxPatch((gx + gw * 0.52, gy + gh * 0.32), gw * 0.44, gh * 0.48,
                                    boxstyle="round,pad=0.0,rounding_size=0.008", facecolor="#FFFFFF", edgecolor="#D97706", lw=0.75, zorder=4)
    ax_a.add_patch(vs_box)
    pz = np.array([0.75, 0.58, 0.58, 0.45, 0.45, 0.35])
    pv = np.array([0.58, 0.62, 0.72, 0.72, 0.84, 0.90])
    ax_a.plot(gx + pv * gw, gy + pz * gh, "-", color="#B45309", lw=1.0, zorder=5)
    ax_a.text(gx + gw * 0.74, gy + gh * 0.85, r"$\hat{V}_s(z)$", ha="center", va="bottom", fontsize=4.8, color="#92400E", zorder=6)
    ax_a.annotate("", xy=(gx + gw * 0.51, gy + gh * 0.55), xytext=(gx + gw * 0.39, gy + gh * 0.55),
                  arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.14", color="#B45309", lw=0.8), zorder=5)
    ax_a.annotate("", xy=(gx + gw * 0.20, gy + gh * 0.22), xytext=(gx + gw * 0.74, gy + gh * 0.30),
                  arrowprops=dict(arrowstyle="-|>,head_length=0.28,head_width=0.15", color="#D97706", lw=0.85, connectionstyle="arc3,rad=-0.28"), zorder=5)
    ax_a.text(gx + gw * 0.48, gy + gh * 0.02, "Surrogate loss", ha="center", va="bottom", fontsize=4.5, color="#92400E", fontweight="bold", zorder=6)

    # Card 4: Exact disba Referee
    x3 = x0 + 3 * (cw + gap)
    gx, gy, gw, gh = _draw_module_card(
        ax_a, x3, cy0, cw, ch, "4", "Exact disba",
        f"Shallow dc={dc_sh:.1e}\nDeep dc={dc_dp:.0e} km/s", PALETTE["teal"], "#ECFDF5"
    )
    ax_a.plot([gx, gx, gx + gw], [gy + gh, gy, gy], color="#64748B", lw=0.7, zorder=4)
    rx = np.linspace(0.08, 0.92, 16)
    y_m0 = 0.22 + 0.04 * rx
    y_jump = 0.22 + 0.65 * (1.0 / (1.0 + np.exp(-9.0 * (rx - 0.42))))
    ax_a.plot(gx + rx * gw, gy + y_jump * gh, "--", color=PALETTE["vermillion"], lw=0.95, zorder=4)
    ax_a.plot(gx + rx * gw, gy + y_m0 * gh, "-", color=PALETTE["teal"], lw=1.25, zorder=5)
    ax_a.scatter(gx + rx[::3] * gw, gy + y_m0[::3] * gh, s=8, color="#1E293B", zorder=6)
    ax_a.text(gx + gw * 0.04, gy + gh * 0.94, "Mode 1 (coarse)", ha="left", va="top", fontsize=4.4, color=PALETTE["vermillion"], zorder=6)
    ax_a.text(gx + gw * 0.94, gy + gh * 0.04, "Mode 0 (conv.)", ha="right", va="bottom", fontsize=4.5, fontweight="bold", color=PALETTE["teal"], zorder=6)

    # Card 5: Regime Diagnostics
    x4 = x0 + 4 * (cw + gap)
    gx, gy, gw, gh = _draw_module_card(
        ax_a, x4, cy0, cw, ch, "5", "Regime Audit",
        r"Stratify by $r_0 = m_0/\hat{\sigma}_D$" + "\n" + r"Resolvable $V_s$ RMSE", PALETTE["vermillion"], "#FFF7ED"
    )
    y_zero = gy + gh * 0.48
    ax_a.add_patch(patches.Rectangle((gx, y_zero), gw, gy + gh - y_zero, facecolor="#FEE2E2", alpha=0.55, lw=0, zorder=3))
    ax_a.add_patch(patches.Rectangle((gx, gy), gw, y_zero - gy, facecolor="#D1FAE5", alpha=0.55, lw=0, zorder=3))
    ax_a.plot([gx, gx, gx + gw], [gy + gh, gy, gy], color="#64748B", lw=0.7, zorder=4)
    ax_a.plot([gx, gx + gw], [y_zero, y_zero], ":", color="#475569", lw=0.7, zorder=4)
    sx = np.linspace(0.08, 0.92, 8)
    sy = np.array([0.85, 0.72, 0.50, 0.34, 0.22, 0.16, 0.45, 0.80])
    ax_a.plot(gx + sx * gw, gy + sy * gh, "-o", color=PALETTE["vermillion"], lw=1.0, ms=2.2, zorder=5)
    ax_a.text(gx + gw * 0.52, gy + gh * 0.88, "Harm", ha="center", va="top", fontsize=4.4, color="#9A2D00", fontweight="bold", zorder=6)
    ax_a.text(gx + gw * 0.52, gy + gh * 0.05, "Gain", ha="center", va="bottom", fontsize=4.4, color="#00664B", fontweight="bold", zorder=6)

    for k in range(4):
        xa = x0 + k * (cw + gap) + cw + 0.002
        xb = x0 + (k + 1) * (cw + gap) - 0.002
        _draw_arrow(ax_a, xa, cy0 + ch * 0.50, xb, cy0 + ch * 0.50, color="#475569", lw=1.15)

    # Panel (b): Four-Step Safe-TTA Operational Protocol
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.axis("off")

    band_b = patches.FancyBboxPatch(
        (0.005, 0.015),
        0.990,
        0.965,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0.85,
        edgecolor="#CBD5E1",
        facecolor="#F8FAFC",
        zorder=0,
    )
    ax_b.add_patch(band_b)
    ax_b.text(
        0.020,
        0.945,
        "(b) Four-Step Operational Protocol for Safe Surface-Wave Test-Time Adaptation",
        fontsize=8.2,
        fontweight="bold",
        color="#0F172A",
        va="top",
        zorder=5,
    )

    lx, lw_col = 0.020, 0.220
    _draw_shadow_card(ax_b, lx, 0.495, lw_col, 0.355, facecolor="#FFFFFF", edgecolor=PALETTE["blue"], lw=0.95, radius=0.014, zorder=2)
    ax_b.add_patch(patches.FancyBboxPatch((lx, 0.775), lw_col, 0.075, boxstyle="round,pad=0.0,rounding_size=0.012", lw=0, facecolor="#EFF6FF", zorder=3))
    ax_b.add_patch(patches.Rectangle((lx, 0.775), lw_col, 0.030, lw=0, facecolor="#EFF6FF", zorder=3))
    ax_b.add_patch(patches.Ellipse((lx + 0.018, 0.812), width=0.024, height=0.046, facecolor=PALETTE["blue"], zorder=5))
    ax_b.text(lx + 0.018, 0.812, "1", ha="center", va="center", fontsize=5.8, fontweight="bold", color="white", zorder=6)
    ax_b.text(lx + 0.034, 0.812, "Step 1: Input Alignment", ha="left", va="center", fontsize=6.2, fontweight="bold", color="#0F172A", zorder=6)
    ax_b.text(lx + lw_col / 2.0, 0.630, "Resample observed curves\nto backbone training grid;\nrun Input-Aligned Zero-Shot.",
              ha="center", va="center", fontsize=5.55, color="#334155", linespacing=1.28, zorder=5)

    _draw_shadow_card(ax_b, lx, 0.065, lw_col, 0.355, facecolor="#FFFFFF", edgecolor=PALETTE["teal"], lw=0.95, radius=0.014, zorder=2)
    ax_b.add_patch(patches.FancyBboxPatch((lx, 0.345), lw_col, 0.075, boxstyle="round,pad=0.0,rounding_size=0.012", lw=0, facecolor="#ECFDF5", zorder=3))
    ax_b.add_patch(patches.Rectangle((lx, 0.345), lw_col, 0.030, lw=0, facecolor="#ECFDF5", zorder=3))
    ax_b.add_patch(patches.Ellipse((lx + 0.018, 0.382), width=0.024, height=0.046, facecolor=PALETTE["teal"], zorder=5))
    ax_b.text(lx + 0.018, 0.382, "2", ha="center", va="center", fontsize=5.8, fontweight="bold", color="white", zorder=6)
    ax_b.text(lx + 0.034, 0.382, "Step 2: Referee & Floor", ha="left", va="center", fontsize=6.2, fontweight="bold", color="#0F172A", zorder=6)
    ax_b.text(lx + lw_col / 2.0, 0.200, "Verify disba dc convergence;\ncalibrate domain floor " + r"$\hat{\sigma}_D$" + "\nvia Classical Inversion.",
              ha="center", va="center", fontsize=5.55, color="#334155", linespacing=1.28, zorder=5)

    _draw_arrow(ax_b, lx + lw_col / 2.0, 0.490, lx + lw_col / 2.0, 0.425, color="#475569", lw=1.1)

    mx, mw = 0.272, 0.205
    _draw_shadow_card(ax_b, mx, 0.175, mw, 0.560, facecolor="#FFFFFF", edgecolor="#D97706", lw=1.05, radius=0.015, zorder=2)
    ax_b.add_patch(patches.FancyBboxPatch((mx, 0.650), mw, 0.085, boxstyle="round,pad=0.0,rounding_size=0.013", lw=0, facecolor="#FFFBEB", zorder=3))
    ax_b.add_patch(patches.Rectangle((mx, 0.650), mw, 0.035, lw=0, facecolor="#FFFBEB", zorder=3))
    ax_b.add_patch(patches.Ellipse((mx + 0.018, 0.692), width=0.024, height=0.046, facecolor="#D97706", zorder=5))
    ax_b.text(mx + 0.018, 0.692, "3", ha="center", va="center", fontsize=5.8, fontweight="bold", color="white", zorder=6)
    ax_b.text(mx + 0.034, 0.692, "Step 3: Regime Screen", ha="left", va="center", fontsize=6.15, fontweight="bold", color="#0F172A", zorder=6)

    ax_b.text(mx + mw / 2.0, 0.535, "Compute initial exact\nkept-period misfit ratio:", ha="center", va="center", fontsize=5.5, color="#334155", linespacing=1.25, zorder=5)
    fpill = patches.FancyBboxPatch((mx + 0.022, 0.395), mw - 0.044, 0.088, boxstyle="round,pad=0.0,rounding_size=0.010",
                                   facecolor="#FEF3C7", edgecolor="#D97706", lw=0.8, zorder=4)
    ax_b.add_patch(fpill)
    ax_b.text(mx + mw / 2.0, 0.439, r"$r_0 = m_0 / \hat{\sigma}_D$", ha="center", va="center", fontsize=7.1, fontweight="bold", color="#92400E", zorder=5)

    bar_x = mx + 0.018
    bar_w = mw - 0.036
    bar_y, bar_h = 0.275, 0.048
    ax_b.add_patch(patches.Rectangle((bar_x, bar_y), bar_w * 0.32, bar_h, facecolor=PALETTE["teal"], alpha=0.85, lw=0, zorder=4))
    ax_b.add_patch(patches.Rectangle((bar_x + bar_w * 0.32, bar_y), bar_w * 0.36, bar_h, facecolor=PALETTE["blue"], alpha=0.85, lw=0, zorder=4))
    ax_b.add_patch(patches.Rectangle((bar_x + bar_w * 0.68, bar_y), bar_w * 0.32, bar_h, facecolor=PALETTE["vermillion"], alpha=0.85, lw=0, zorder=4))
    ax_b.text(bar_x, bar_y - 0.018, "0", ha="center", va="top", fontsize=4.8, color="#475569", zorder=5)
    ax_b.text(bar_x + bar_w * 0.32, bar_y - 0.018, "1.5–2", ha="center", va="top", fontsize=4.8, color="#475569", zorder=5)
    ax_b.text(bar_x + bar_w * 0.68, bar_y - 0.018, "5.0", ha="center", va="top", fontsize=4.8, color="#475569", zorder=5)
    ax_b.text(bar_x + bar_w, bar_y - 0.018, r"$\infty$", ha="center", va="top", fontsize=5.2, color="#475569", zorder=5)
    ax_b.text(bar_x + bar_w * 0.16, bar_y + bar_h * 0.5, "A", ha="center", va="center", fontsize=5.2, fontweight="bold", color="white", zorder=5)
    ax_b.text(bar_x + bar_w * 0.50, bar_y + bar_h * 0.5, "B", ha="center", va="center", fontsize=5.2, fontweight="bold", color="white", zorder=5)
    ax_b.text(bar_x + bar_w * 0.84, bar_y + bar_h * 0.5, "C", ha="center", va="center", fontsize=5.2, fontweight="bold", color="white", zorder=5)
    ax_b.text(mx + mw / 2.0, 0.212, "Separates noise vs. OOD shift", ha="center", va="center", fontsize=5.0, style="italic", color="#64748B", zorder=5)

    _draw_arrow(ax_b, lx + lw_col + 0.002, 0.660, mx - 0.003, 0.545, color="#475569", lw=1.1, rad=-0.08)
    _draw_arrow(ax_b, lx + lw_col + 0.002, 0.240, mx - 0.003, 0.355, color="#475569", lw=1.1, rad=0.08)

    rx, rw = 0.580, 0.400
    _draw_accent_card(
        ax_b, rx, 0.640, rw, 0.235,
        "Regime A: Near / Below Floor",
        "DO NOT REFINE — retain Input-Aligned Zero-Shot.\nPrevents +13% to +157% noise-fitting $V_s$ harm.",
        PALETTE["teal"], "#ECFDF5", "Safe Skip"
    )
    _draw_accent_card(
        ax_b, rx, 0.350, rw, 0.245,
        "Regime B: Moderate Shift",
        "Step 4: Output-Space Refinement + Holdout ES.\nInternal 20% stop split; freeze $z > z_{\\mathrm{res}}$; disba check.",
        PALETTE["blue"], "#EFF6FF", "Step 4: TTA"
    )
    _draw_accent_card(
        ax_b, rx, 0.060, rw, 0.245,
        "Regime C: Severe OOD / Outlier",
        "Audit exact disba fit & >10 km depth smoothness;\nflag deep oscillations or fall back to Classical Inv.",
        PALETTE["vermillion"], "#FFF7ED", "Audit / Fallback"
    )

    branches = [
        (0.585, 0.758, PALETTE["teal"], r"$r_0 < 1.5$–$2.0$", 0.528, 0.735, 0.10),
        (0.455, 0.472, PALETTE["blue"], r"$2.0 \leq r_0 \leq 5.0$", 0.528, 0.505, 0.00),
        (0.320, 0.182, PALETTE["vermillion"], r"$r_0 > 5.0$", 0.528, 0.205, -0.10),
    ]
    for y_start, y_end, col, lbl, tx_l, ty_l, rad in branches:
        _draw_arrow(ax_b, mx + mw + 0.003, y_start, rx - 0.004, y_end, color=col, lw=1.25, rad=rad)
        ax_b.text(
            tx_l,
            ty_l,
            lbl,
            ha="center",
            va="center",
            fontsize=5.1,
            fontweight="bold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.17", facecolor="white", edgecolor=col, lw=0.65, alpha=0.97),
            zorder=6,
        )

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
        0.955,
        "When Does Test-Time Adaptation Help Surface-Wave Inversion? Physics Referee & Noise-Floor Diagnostics",
        ha="center",
        va="top",
        fontsize=7.6,
        fontweight="bold",
        color="#111111",
    )

    gs = gridspec.GridSpec(
        1,
        3,
        left=0.060,
        right=0.980,
        bottom=0.17,
        top=0.83,
        wspace=0.28,
        width_ratios=[0.95, 1.35, 1.02],
    )

    # Panel 1: Hidden Evaluation Pitfall (disba root-skipping collapse)
    ax1 = fig.add_subplot(gs[0])
    sh_rec = rs["recomputed_domains"]["shallow_benchmark"]
    labels = ["Output-Ref.\n(50-step)", "Adapter +\nOutput-Ref.", "Ground\nTruth"]
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
    w = 0.36
    b1 = ax1.bar(x - w / 2, old_vals, w, color=PALETTE["vermillion"], label="Default dc=5e-3")
    b2 = ax1.bar(x + w / 2, new_vals, w, color=PALETTE["teal"], label="Converged dc=2.5e-5")
    for i_b, (bar, v) in enumerate(zip(b1, old_vals)):
        dx = -0.04 if i_b == 2 else 0.0
        ax1.text(bar.get_x() + bar.get_width() / 2 + dx, v + 0.25, f"{v:.2f}%", ha="center", va="bottom", fontsize=5.6)
    for i_b, (bar, v) in enumerate(zip(b2, new_vals)):
        dx = 0.06 if i_b == 2 else 0.0
        ax1.text(bar.get_x() + bar.get_width() / 2 + dx, v + 0.25, f"{v:.2f}%", ha="center", va="bottom", fontsize=5.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.4)
    ax1.set_ylabel("Tail Rate (>20% Misfit, %)", fontsize=7.0)
    ax1.set_ylim(0, 11.4)
    ax1.set_title("1. Fix disba Root-Skipping", fontsize=7.8, fontweight="bold", color=PALETTE["vermillion"])
    ax1.legend(loc="upper right", fontsize=5.9, handlelength=1.2)

    # Panel 2: Noise-Floor-Normalized Signature Curve (Shallow Multi-Structure)
    ax2 = fig.add_subplot(gs[1])
    bins = cb["domains"]["shallow_multi_structure"]["by_backbone"]["openswi"]["all_judged_ckpts"]["pooled_signature_bins"]["bins"]
    bx = np.arange(len(bins))
    blabs = [b["bin_label"] for b in bins]
    y_p3 = [b["methods"]["output_refined_50"]["median_rel_change_pct"] for b in bins]
    y_ho = [b["methods"]["holdout_es"]["median_rel_change_pct"] for b in bins]
    y_id = [b["methods"]["ideal_es"]["median_rel_change_pct"] for b in bins]

    ax2.axhspan(0, 185, color=PALETTE["warn_tint"], alpha=0.55, zorder=0)
    ax2.axhspan(-75, 0, color=PALETTE["safe_tint"], alpha=0.55, zorder=0)
    ax2.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax2.plot(bx, y_p3, "-o", color=PALETTE["vermillion"], lw=1.4, ms=3.5, label="Output-Refined (50-step)")
    ax2.plot(bx, y_ho, "-s", color=PALETTE["blue"], lw=1.4, ms=3.4, label="Holdout ES")
    ax2.plot(bx, y_id, "--^", color=PALETTE["teal"], lw=1.2, ms=3.2, label="Ideal ES")
    ax2.set_xticks(bx)
    ax2.set_xticklabels(blabs, fontsize=5.6, rotation=28, ha="right")
    ax2.set_xlabel("Initial Misfit / Ground-Truth Noise Floor", fontsize=6.8)
    ax2.set_ylabel("Median Δ Vs RMSE (%)", fontsize=7.0)
    ax2.set_ylim(-68, 182)
    ax2.text(1.12, 138, "Noise-fitting\ndegradation", fontsize=5.8, color="#9A2D00", fontweight="bold")
    ax2.text(0.10, -56, "OOD inversion gain", fontsize=5.9, color="#00664B", fontweight="bold")
    ax2.set_title("2. Noise-Floor Regime Signature", fontsize=7.8, fontweight="bold", color=PALETTE["blue"])
    ax2.legend(loc="upper right", fontsize=5.7, handlelength=1.4)

    # Panel 3: Operational Safe-TTA Decision Guide (dimensionless ratio r_0 = m_0 / sigma_hat_D)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.04)
    ax3.axis("off")
    ax3.set_title("3. Four-Step Safe Protocol", fontsize=7.8, fontweight="bold", color=PALETTE["teal"], pad=6)
    ax3.text(
        0.5,
        0.995,
        r"Dimensionless ratio $r_0 = m_0 / \hat{\sigma}_D$",
        ha="center",
        va="top",
        fontsize=5.9,
        color="#333333",
        style="italic",
    )
    _draw_box(
        ax3,
        0.015,
        0.64,
        0.97,
        0.26,
        "Below Floor ($r_0 < 1.5$–$2.0$)",
        "Keep Input-Aligned Zero-Shot\n(avoid noise-floor overfitting)",
        PALETTE["safe_tint"],
        PALETTE["teal"],
        fontsize_title=6.1,
        fontsize_body=5.6,
    )
    _draw_box(
        ax3,
        0.015,
        0.33,
        0.97,
        0.26,
        r"Moderate OOD ($2.0 \leq r_0 \leq 5.0$)",
        "Output-Space Refinement + Holdout ES\n(depth-resolvable regularization)",
        PALETTE["info_tint"],
        PALETTE["blue"],
        fontsize_title=6.1,
        fontsize_body=5.6,
    )
    _draw_box(
        ax3,
        0.015,
        0.02,
        0.97,
        0.26,
        r"Severe Tail ($r_0 > 5.0$)",
        "Audit exact disba & depth smoothness\nor fall back to Classical Inversion",
        PALETTE["warn_tint"],
        PALETTE["vermillion"],
        fontsize_title=6.1,
        fontsize_body=5.6,
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
    """Generate Figure 2: decluttered three-benchmark accuracy vs field held-out misfit."""
    apply_cg_style()
    bo = _load_json(summary_dir, "benchmark_overview.json")
    method_order = bo["method_order"]

    panels = [
        ("longbeach_matched_shallow", "longbeach_report", "(a) LongBeach-Matched Shallow", "LongBeach (Report 70%)"),
        ("shallow_benchmark", "longbeach_report", "(b) Shallow Benchmark", "LongBeach (Report 70%)"),
        ("deep_benchmark", "csrm_report", "(c) Deep Benchmark", "CSRM (Report)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, mm_to_in(82.0)))
    plt.subplots_adjust(wspace=0.28, bottom=0.29, top=0.89, left=0.075, right=0.985)

    for idx, (syn_k, fld_k, title, fld_label) in enumerate(panels):
        ax = axes[idx]
        syn_blk = bo["synthetic_benchmarks"][syn_k]
        fld_blk = bo["field_report"][fld_k]

        x_aligned = fld_blk["methods"]["zero_shot_aligned"]["ho_median_mean4_converged_dc"] * 100.0
        y_aligned = syn_blk["methods"]["zero_shot_aligned"]["judged_mean_vs_rmse_res"]
        x_cls = fld_blk["classical_inversion"]["ho_median"] * 100.0
        gt_nf = syn_blk["gt_noise_floor"]["obs_median_converged_dc"] * 100.0

        xs, ys = [], []
        for mk in method_order:
            xv = fld_blk["methods"][mk]["ho_median_mean4_converged_dc"] * 100.0
            yv = syn_blk["methods"][mk]["judged_mean_vs_rmse_res"]
            xs.append(xv)
            ys.append(yv)
            color, marker, _, _ = METHOD_STYLE[mk]
            ms = 74 if mk == "adapter_output_refined_aligned" else (54 if mk in ("zero_shot_aligned", "output_refined_aligned", "low_dof_adapter") else 38)
            ax.scatter(
                xv,
                yv,
                c=color,
                marker=marker,
                s=ms,
                edgecolors="white" if mk != "zero_shot_aligned" else "#111111",
                linewidths=0.65,
                zorder=5,
            )

        x_min = min(min(xs), x_cls, gt_nf)
        x_max = max(max(xs), x_cls)
        y_min, y_max = min(ys), max(ys)
        x_pad = (x_max - x_min) * 0.14
        y_pad = (y_max - y_min) * 0.16
        ax.set_xlim(max(0.0, x_min - x_pad * 0.7), x_max + x_pad * 1.05)
        ax.set_ylim(max(0.0, y_min - y_pad * 1.0), y_max + y_pad * 1.05)

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
        ax.axvline(x_cls, color=PALETTE["purple"], lw=1.05, ls="-.", zorder=2)
        ax.axvline(gt_nf, color=PALETTE["mid_gray"], lw=0.95, ls=":", zorder=2)

        ax.set_title(title, fontsize=8.0, fontweight="bold")
        ax.set_xlabel(f"{fld_label} HO Misfit Median (%)", fontsize=7.2)
        if idx == 0:
            ax.set_ylabel("Resolvable-Depth $V_s$ RMSE (km/s)", fontsize=7.5)

    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_STYLE[mk][1],
            color="none",
            markerfacecolor=METHOD_STYLE[mk][0],
            markeredgecolor="white" if mk != "zero_shot_aligned" else "#111111",
            markersize=6.6 if mk == "adapter_output_refined_aligned" else 5.4,
            label=METHOD_STYLE[mk][3],
        )
        for mk in method_order
    ]
    handles.extend([
        Line2D([0], [0], color="#555555", lw=0.9, ls="--", label="Zero-Shot (Aligned) reference"),
        Line2D([0], [0], color=PALETTE["purple"], lw=1.1, ls="-.", label="Classical Inversion median"),
        Line2D([0], [0], color=PALETTE["mid_gray"], lw=1.0, ls=":", label="Ground-Truth noise floor"),
        patches.Patch(facecolor=PALETTE["safe_tint"], edgecolor="#CCCCCC", alpha=0.75, label="Pareto gain quadrant"),
    ])
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.515, 0.008),
        fontsize=6.1,
        columnspacing=0.95,
        handletextpad=0.35,
    )
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

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, mm_to_in(132.0)))
    plt.subplots_adjust(wspace=0.28, hspace=0.38, left=0.08, right=0.98, top=0.94, bottom=0.09)

    # Panel (a): Counterfactual & Direct dc Sensitivity on Shallow Tail Cases
    ax = axes[0, 0]
    cats = ["Original\n(dc=5e-3)", "LVZ-Raised\n(dc=5e-3)", "Restore $>z_{\\mathrm{res}}$\n(dc=5e-3)", "ZS Aligned\n(dc=5e-3)", "Original\n(dc=2e-4)"]
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
    b1 = ax.bar(x - w / 2, vals_p3, w, color=PALETTE["vermillion"], label="Output-Refined (50-step)")
    b2 = ax.bar(x + w / 2, vals_p5, w, color=PALETTE["blue"], label="Gated Adapter + Output-Ref.")
    for bar, v in zip(b1, vals_p3):
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.03, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=5.1)
    for bar, v in zip(b2, vals_p5):
        ax.text(bar.get_x() + bar.get_width() / 2 + 0.03, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=5.1)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=6.1)
    ax.set_ylabel("Tail Median Held-Out Misfit (%)")
    ax.set_ylim(0, 90)
    ax.set_title("LVZ Counterfactual & Step-Size Collapse", fontsize=8.0)
    ax.legend(loc="upper right", fontsize=6.2)
    add_panel_label(ax, "(a)", x=-0.12, y=1.04)

    # Panel (b): Live LVZ Mode-Jumping Mechanism (sample #9391)
    ax = axes[0, 1]
    ex0 = rs["typical_lvz_examples"][0]
    T = np.asarray(ex0["periods_s"], dtype=float)
    obs = np.asarray(ex0["phase_obs_km_s"], dtype=float)
    is_ho = np.asarray(ex0["is_heldout"], dtype=bool)
    c_m0_old = np.asarray([np.nan if v is None else float(v) for v in ex0["curves"]["orig_mode0_dc005"]], dtype=float)
    c_m0_new = np.asarray([np.nan if v is None else float(v) for v in ex0["curves"]["orig_mode0_dc0002"]], dtype=float)
    c_m1_old = np.asarray([np.nan if v is None else float(v) for v in ex0["curves"]["orig_mode1_dc005"]], dtype=float)

    ax.plot(T[~is_ho], obs[~is_ho], "o", color="#222222", ms=3.0, label="Observed (kept)", zorder=5)
    ax.plot(T[is_ho], obs[is_ho], "D", mfc="white", mec="#222222", ms=3.5, mew=0.85, label="Observed (held-out)", zorder=6)
    ax.plot(T, c_m0_old, "-", color=PALETTE["vermillion"], lw=1.4, label=f"Mode 0 (dc=5e-3, HO={ex0['curves']['orig_ho_misfit_dc005']*100:.1f}%)")
    ax.plot(T, c_m0_new, "--", color=PALETTE["teal"], lw=1.4, label="Mode 0 (dc=2e-4, recovered)")
    ax.plot(T, c_m1_old, ":", color=PALETTE["purple"], lw=1.2, label="Mode 1 branch (higher mode)")
    ax.set_ylim(0.75, 2.55)
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Rayleigh Phase Velocity (km/s)")
    ax.set_title(f"Root-Skipping Example (#{ex0['sample_idx']}, {ex0['structure_type']})", fontsize=8.0)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.16), fontsize=5.6, handlelength=1.6, framealpha=0.95)
    add_panel_label(ax, "(b)", x=-0.12, y=1.04)

    # Panel (c): Per-Regime Convergence & Single-Forward Timing (clean legend only)
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
        label=f"Shallow selected (dc=2.5e-5, {sh_sel_t:.1f} ms)",
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
        label=f"Deep selected (dc=5e-3, {dp_sel_t:.1f} ms)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_ylim(5.5, 780.0)
    ax.set_xlabel("disba Phase-Velocity Step dc (km/s, finer →)")
    ax.set_ylabel("Median Forward Runtime (ms / station)")
    ax.set_title("Per-Regime Convergence & Compute Cost", fontsize=8.0)
    ax.legend(loc="upper left", fontsize=5.4, framealpha=0.95, handlelength=1.6)
    add_panel_label(ax, "(c)", x=-0.12, y=1.04)

    # Panel (d): Shallow Benchmark Tail Failure Rate Collapse (>20% HO Misfit)
    ax = axes[1, 1]
    sh_d = rs["recomputed_domains"]["shallow_benchmark"]
    gt_sh = rs["gt_noise_floor_default_vs_converged"]["shallow_benchmark"]
    items = [
        ("ZS (Aligned)", sh_d["zero_shot_aligned"]["old_tail_pct"], sh_d["zero_shot_aligned"]["new_tail_pct"]),
        ("Output-Ref.\n(Raw)", sh_d["output_refined_raw"]["old_tail_pct"], sh_d["output_refined_raw"]["new_tail_pct"]),
        ("Output-Ref.\n(50-step)", sh_d["output_refined_aligned"]["old_tail_pct"], sh_d["output_refined_aligned"]["new_tail_pct"]),
        ("Low-DOF\nAdapter", sh_d["low_dof_adapter"]["old_tail_pct"], sh_d["low_dof_adapter"]["new_tail_pct"]),
        ("Adapter +\nOutput-Ref.", sh_d["adapter_output_refined_aligned"]["old_tail_pct"], sh_d["adapter_output_refined_aligned"]["new_tail_pct"]),
        ("Ground\nTruth", gt_sh["ho_old_tail_pct"], gt_sh["ho_new_tail_pct"]),
    ]
    x = np.arange(len(items))
    w = 0.36
    v_old = [t[1] for t in items]
    v_new = [t[2] for t in items]
    b1 = ax.bar(x - w / 2, v_old, w, color=PALETTE["vermillion"], alpha=0.88, label="Default dc=5e-3")
    b2 = ax.bar(x + w / 2, v_new, w, color=PALETTE["teal"], label="Converged dc=2.5e-5")
    for bar, v in zip(b1, v_old):
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.03, v + 0.18, f"{v:.2f}", ha="center", va="bottom", fontsize=5.0)
    for bar, v in zip(b2, v_new):
        ax.text(bar.get_x() + bar.get_width() / 2 + 0.03, v + 0.18, f"{v:.2f}", ha="center", va="bottom", fontsize=5.0)
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in items], fontsize=5.7)
    ax.set_ylabel("Tail Rate (>20% HO, %)")
    ax.set_ylim(0, max(v_old) * 1.24)
    ax.set_title("Elimination of Artificial Shallow Tail Failures", fontsize=8.0)
    ax.legend(loc="upper right", fontsize=6.2)
    add_panel_label(ax, "(d)", x=-0.14, y=1.04)

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

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(86.0)))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.7, 0.9], hspace=0.08, wspace=0.22, left=0.075, right=0.985, top=0.87, bottom=0.14)

    for col, (dk, title) in enumerate(domains):
        d_blk = cb["domains"][dk]
        osw = d_blk["by_backbone"]["openswi"]["all_judged_ckpts"]
        bins = osw["pooled_signature_bins"]["bins"]
        bx = np.arange(len(bins))
        blabs = [b["bin_label"] for b in bins]
        fracs = [b["fraction_pct"] for b in bins]

        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col], sharex=ax_top)

        ax_top.axhline(0.0, color="#333333", lw=0.8, ls="--", zorder=2)

        series_specs = [
            ("output_refined_50", "Output-Refined (50-step)", PALETTE["vermillion"], "o", "-"),
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
            y_lo, y_hi = -65.0, 55.0
        elif dk == "shallow_prior_shift":
            y_lo, y_hi = -75.0, 145.0
        else:
            y_lo, y_hi = -72.0, 185.0
        ax_top.set_ylim(y_lo, y_hi)
        ax_top.axhspan(0.0, y_hi, color=PALETTE["warn_tint"], alpha=0.35, zorder=0)
        ax_top.axhspan(y_lo, 0.0, color=PALETTE["safe_tint"], alpha=0.35, zorder=0)

        # Annotate overall mean Δ Vs RMSE for the 3 methods (bottom-left on Deep to clear curve)
        o50_ov = osw["output_refined_50"]["mean_rel_change_vs_aligned_pct"]
        hes_ov = osw["holdout_es"]["mean_rel_change_vs_aligned_pct"]
        ies_ov = osw["ideal_es"]["mean_rel_change_vs_aligned_pct"]
        is_deep = dk == "deep_lithospheric"
        ax_top.text(
            0.04 if is_deep else 0.97,
            0.05 if is_deep else 0.96,
            f"Overall ΔRMSE:\n50-step: {o50_ov:+.1f}%\nHoldout ES: {hes_ov:+.1f}%\nIdeal ES: {ies_ov:+.1f}%",
            transform=ax_top.transAxes,
            ha="left" if is_deep else "right",
            va="bottom" if is_deep else "top",
            fontsize=5.6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
        )

        ax_top.set_title(title, fontsize=8.1, fontweight="bold")
        plt.setp(ax_top.get_xticklabels(), visible=False)
        if col == 0:
            ax_top.set_ylabel("Per-Sample Δ $V_s$ RMSE (%)\n(median & IQR)", fontsize=7.3)

        # Bottom histogram: sample share per bin
        ax_bot.bar(bx, fracs, width=0.68, color=PALETTE["sky"], edgecolor=PALETTE["blue"], linewidth=0.5)
        for xi, fv in zip(bx, fracs):
            if fv >= 1.0:
                ax_bot.text(xi, fv + 1.2, f"{fv:.0f}", ha="center", va="bottom", fontsize=5.2, color="#333333")
        ax_bot.set_ylim(0, max(fracs) * 1.32)
        ax_bot.set_xticks(bx)
        ax_bot.set_xticklabels(blabs, rotation=32, ha="right", fontsize=6.0)
        ax_bot.set_xlabel("Initial Misfit / GT Noise Floor", fontsize=6.8)
        if col == 0:
            ax_bot.set_ylabel("Share (%)", fontsize=6.8)

    handles, labels = axes_handles = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.995),
        fontsize=7.0,
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

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(102.0)))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.5, 1.15], hspace=0.10, wspace=0.22, left=0.075, right=0.985, top=0.86, bottom=0.13)

    for col, (dk, title) in enumerate(domains):
        d_blk = cb["domains"][dk]
        osw_bins = d_blk["by_backbone"]["openswi"]["all_judged_ckpts"]["pooled_signature_bins"]["bins"]
        dsp_bins = d_blk["by_backbone"]["dispformer"]["all_judged_ckpts"]["pooled_signature_bins"]["bins"]
        bx = np.arange(len(osw_bins))
        blabs = [b["bin_label"] for b in osw_bins]

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
        _plot_curve(osw_bins, "ideal_es", PALETTE["teal"], "-", "^", "OpenSWI: Ideal ES")
        _plot_curve(dsp_bins, "ideal_es", PALETTE["purple"], ":", "v", "DispFormer: Ideal ES")

        if dk == "deep_lithospheric":
            # Explicitly highlight Retrained-Only cross-backbone consistency in [0, 1)
            osw_ret = d_blk["by_backbone"]["openswi"]["retrained_only"]["pooled_signature_bins"]["bins"][0]
            dsp_ret = d_blk["by_backbone"]["dispformer"]["retrained_only"]["pooled_signature_bins"]["bins"][0]
            osw_ret_v = osw_ret["methods"]["output_refined_50"]["median_rel_change_pct"]
            dsp_ret_v = dsp_ret["methods"]["output_refined_50"]["median_rel_change_pct"]
            dsp_all_v = dsp_bins[0]["methods"]["output_refined_50"]["median_rel_change_pct"]
            ax_top.scatter([0.08], [osw_ret_v], marker="D", s=34, color=PALETTE["vermillion"], edgecolors="#111111", lw=0.7, zorder=7)
            ax_top.scatter([-0.08], [dsp_ret_v], marker="D", s=34, color=PALETTE["blue"], edgecolors="#111111", lw=0.7, zorder=7)
            ax_top.annotate(
                f"Retrained-only [0, 1):\nOpenSWI {osw_ret_v:+.2f}%\nDispFormer {dsp_ret_v:+.2f}%\n(All-ckpt DF: {dsp_all_v:+.2f}%)",
                xy=(0.0, dsp_ret_v),
                xytext=(0.14, 0.95),
                textcoords="axes fraction",
                ha="left",
                va="top",
                fontsize=5.5,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFF8E1", edgecolor=PALETTE["orange"], lw=0.8),
                arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], lw=0.85),
                zorder=8,
            )
            ax_top.set_ylim(-58.0, 68.0)
        elif dk == "shallow_prior_shift":
            ax_top.set_ylim(-70.0, 125.0)
        else:
            ax_top.set_ylim(-68.0, 180.0)

        yl0, yl1 = ax_top.get_ylim()
        ax_top.axhspan(0.0, yl1, color=PALETTE["warn_tint"], alpha=0.30, zorder=0)
        ax_top.axhspan(yl0, 0.0, color=PALETTE["safe_tint"], alpha=0.30, zorder=0)
        ax_top.set_title(title, fontsize=8.1, fontweight="bold")
        plt.setp(ax_top.get_xticklabels(), visible=False)
        if col == 0:
            ax_top.set_ylabel("Median Δ $V_s$ RMSE (%)", fontsize=7.4)

        # Bottom paired bin-share bars (OpenSWI vs DispFormer)
        w = 0.38
        f_osw = [b["fraction_pct"] for b in osw_bins]
        f_dsp = [b["fraction_pct"] for b in dsp_bins]
        ax_bot.bar(bx - w / 2, f_osw, w, color=PALETTE["vermillion"], alpha=0.75, label="OpenSWI share")
        ax_bot.bar(bx + w / 2, f_dsp, w, color=PALETTE["blue"], alpha=0.75, label="DispFormer share")
        ax_bot.set_ylim(0, max(max(f_osw), max(f_dsp)) * 1.25)
        ax_bot.set_xticks(bx)
        ax_bot.set_xticklabels(blabs, rotation=32, ha="right", fontsize=6.0)
        ax_bot.set_xlabel("Initial Misfit / GT Noise Floor", fontsize=6.8)
        if col == 0:
            ax_bot.set_ylabel("Bin Share (%)", fontsize=6.8)
            ax_bot.legend(loc="upper right", fontsize=5.5, handlelength=1.1)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["blue"], markeredgecolor="#111111", markersize=5.0, label="Retrained-only [0, 1)"))
    labels.append("Retrained-only [0, 1)")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.53, 0.995),
        fontsize=6.4,
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, mm_to_in(76.0)))
    plt.subplots_adjust(wspace=0.28, left=0.085, right=0.98, top=0.88, bottom=0.17)

    # Panel (a): Δ Resolvable-Depth Vs RMSE (%) vs Noise Multiplier with 95% CIs
    curves = [
        ("step5", "Output-Refined (5-step)", PALETTE["orange"], "s", "-"),
        ("step20", "Output-Refined (20-step)", PALETTE["blue"], "D", "-"),
        ("step50", "Output-Refined (50-step)", PALETTE["vermillion"], "o", "-"),
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
            elif ck == "step50":
                ent = osw["output_refined_50"]
            else:
                st_num = ck.replace("step", "")
                ent = osw["output_refined_by_step"][st_num]
            pt = float(ent["mean_rel_change_vs_aligned_pct"])
            pts.append(pt)
            if "cluster_bootstrap" in ent:
                ci = ent["cluster_bootstrap"]["ci95_rel_change_pct"]
                lo_err.append(max(0.0, pt - float(ci[0])))
                hi_err.append(max(0.0, float(ci[1]) - pt))
            else:
                has_ci = False
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
    ax1.set_title("TTA Error Transition Across Noise Levels", fontsize=8.2)
    ax1.axhspan(0, 62, color=PALETTE["warn_tint"], alpha=0.35, zorder=0)
    ax1.axhspan(-45, 0, color=PALETTE["safe_tint"], alpha=0.35, zorder=0)
    ax1.set_ylim(-43, 58)
    ax1.legend(loc="upper left", fontsize=6.4)
    add_panel_label(ax1, "(a)", x=-0.12, y=1.04)

    # Panel (b): Calibrated sigma_hat_D vs Ground-Truth Noise Floor linear tracking
    gt_vals = np.array([ns[k]["sigma_hat_d_vs_gt_tracking"]["gt_test_4000_median"] * 100.0 for k in order])
    sig_vals = np.array([ns[k]["sigma_hat_d_vs_gt_tracking"]["sigma_hat_D"] * 100.0 for k in order])
    aligned_vals = np.array([ns[k]["zero_shot_aligned_kept_misfit_median"] * 100.0 for k in order])
    ratios = np.array([ns[k]["sigma_hat_d_vs_gt_tracking"]["sigma_hat_D_over_gt_test"] for k in order])

    ax2.plot(scales, gt_vals, "-o", color="#1A1A1A", lw=1.5, label="Ground-Truth Noise Floor (%)")
    ax2.plot(scales, sig_vals, "--s", color=PALETTE["teal"], lw=1.5, label=r"Calibrated $\hat{\sigma}_D$ (Classical Inv.)")
    ax2.plot(scales, aligned_vals, ":^", color=PALETTE["vermillion"], lw=1.4, label="Zero-Shot (Aligned) Kept Misfit (%)")

    for i_s, (s, gv, sv, r) in enumerate(zip(scales, gt_vals, sig_vals, ratios)):
        off = (14, -4) if i_s == 0 else (0, -11)
        ax2.annotate(
            f"{r:.2f}×",
            xy=(s, sv),
            xytext=off,
            textcoords="offset points",
            ha="left" if i_s == 0 else "center",
            fontsize=6.2,
            color="#005B42",
            fontweight="bold",
        )

    ax2.set_xscale("log", base=2)
    ax2.set_xticks(scales)
    ax2.set_xticklabels([f"{s:g}×" for s in scales])
    ax2.set_xlabel("Dispersion Noise Amplitude Multiplier")
    ax2.set_ylabel("Kept-Period Dispersion Misfit Median (%)")
    ax2.set_title(r"Linear Tracking of $\hat{\sigma}_D$ vs Ground-Truth Floor", fontsize=8.2)
    ax2.set_ylim(0, max(aligned_vals) * 1.18)
    ax2.legend(loc="upper left", fontsize=6.4)
    ax2.text(
        0.96,
        0.08,
        f"Ratio $\\hat{{\\sigma}}_D$ / GT:\n{ratios.min():.3f} – {ratios.max():.3f}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=PALETTE["safe_tint"], edgecolor=PALETTE["teal"], lw=0.75),
    )
    add_panel_label(ax2, "(b)", x=-0.12, y=1.04)

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

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, mm_to_in(136.0)))
    plt.subplots_adjust(wspace=0.28, hspace=0.40, left=0.085, right=0.975, top=0.94, bottom=0.09)

    # Panel (a): Shallow Benchmark (Val) Misfit/sigma_hat_D bins
    ax = axes[0, 0]
    sh_mb = st["misfit_over_sigma_bins"]["shallow_benchmark_val"]
    blabs = sh_mb["bin_labels"]
    bx = np.arange(len(blabs))
    all_b = sh_mb["by_series"]["all_checkpoints"]["bins"]
    flt_b = sh_mb["by_series"]["flat_trained"]["bins"]
    ax.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax.plot(bx, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in all_b], "-o", color=PALETTE["blue"], label="Safe-TTA (α=1.25, all ckpts)")
    ax.plot(bx, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in flt_b], "--s", color=PALETTE["sky"], label="Safe-TTA (α=1.25, flat-trained)")
    ax.plot(bx, [b["methods"]["output_refined_step50"]["mean_rel_change_vs_aligned_pct"] for b in all_b], "-^", color=PALETTE["vermillion"], label="Output-Refined (50-step)")
    ax.axvspan(0.5, 2.5, color="#FFF3CD", alpha=0.55, zorder=0)
    ax.text(1.5, 146, "Near-floor\ndegradation", ha="center", va="top", fontsize=5.8, color="#8A5A00", fontweight="bold")
    ax.set_xticks(bx)
    ax.set_xticklabels(blabs, rotation=28, ha="right", fontsize=6.1)
    ax.set_xlabel(r"Zero-Shot (Aligned) Misfit / $\hat{\sigma}_D$")
    ax.set_ylabel("Mean Δ $V_s$ RMSE (%)")
    ax.set_ylim(-55, 155)
    ax.set_title("Shallow Benchmark (Val): Near-Floor Escape", fontsize=7.9)
    ax.legend(loc="lower left", fontsize=5.8)
    add_panel_label(ax, "(a)", x=-0.13, y=1.04)

    # Panel (b): Deep Benchmark (Val) Misfit/sigma_hat_D bins
    ax = axes[0, 1]
    dp_mb = st["misfit_over_sigma_bins"]["deep_benchmark_val"]
    blabs_d = dp_mb["bin_labels"]
    bx_d = np.arange(len(blabs_d))
    dp_all = dp_mb["by_series"]["all_checkpoints"]["bins"]
    dp_ood = dp_mb["by_series"]["ood_excluded"]["bins"]
    dp_ret = dp_mb["by_series"]["all_type_retrained"]["bins"]
    ax.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax.plot(bx_d, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in dp_all], "-o", color=PALETTE["blue"], label="Safe-TTA (α=1.25, all 7 ckpts)")
    ax.plot(bx_d, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in dp_ood], "--s", color=PALETTE["orange"], label="Safe-TTA (OOD-excluded)")
    ax.plot(bx_d, [b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in dp_ret], ":d", color=PALETTE["teal"], label="Safe-TTA (all-type retrained)")
    ax.plot(bx_d, [b["methods"]["output_refined_step50"]["mean_rel_change_vs_aligned_pct"] for b in dp_all], "-^", color=PALETTE["vermillion"], label="Output-Refined (50-step)")
    ax.axvspan(5.5, 6.45, color=PALETTE["warn_tint"], alpha=0.75, zorder=0)
    y_max_dp = max(
        max(b["methods"]["safe_tta_alpha_1p25"]["mean_rel_change_vs_aligned_pct"] for b in dp_ret),
        max(b["methods"]["output_refined_step50"]["mean_rel_change_vs_aligned_pct"] for b in dp_all),
    )
    ax.set_ylim(-26, max(124.0, y_max_dp * 1.26))
    ax.text(5.92, ax.get_ylim()[1] * 0.96, "High-misfit\ndegradation", ha="center", va="top", fontsize=5.6, color="#9A2D00", fontweight="bold")
    ax.set_xticks(bx_d)
    ax.set_xticklabels(blabs_d, rotation=28, ha="right", fontsize=6.1)
    ax.set_xlabel(r"Zero-Shot (Aligned) Misfit / $\hat{\sigma}_D$")
    ax.set_ylabel("Mean Δ $V_s$ RMSE (%)")
    ax.set_title("Deep Benchmark (Val): High-Misfit Inversion", fontsize=7.9)
    ax.legend(loc="upper left", fontsize=5.5)
    add_panel_label(ax, "(b)", x=-0.13, y=1.04)

    # Panel (c): Outcome-Group Decomposition at alpha=1.25
    ax = axes[1, 0]
    ds_specs = [
        ("longbeach_matched_val", "LongBeach-Matched\n(Val)"),
        ("shallow_benchmark_val", "Shallow Benchmark\n(Val)"),
        ("deep_benchmark_val", "Deep Benchmark\n(Val)"),
    ]
    grp_specs = [
        ("step0_stop", "Step-0 Stop", PALETTE["teal"]),
        ("mid_stop", "Mid-Stop", PALETTE["orange"]),
        ("ran_50_steps", "Ran 50 Steps", PALETTE["vermillion"]),
        ("reverted", "Reverted", PALETTE["mid_gray"]),
    ]
    x = np.arange(len(ds_specs))
    w = 0.19
    ax.axhline(0.0, color="#333333", lw=0.8)
    for gi, (gk, glbl, gcol) in enumerate(grp_specs):
        vals = [
            st["outcome_group_decomposition"][dk]["by_alpha"]["1.25"]["series_mean"]["all_checkpoints"]["groups"][gk]["mean_rel_contribution_pct"]
            for dk, _ in ds_specs
        ]
        bars = ax.bar(x + (gi - 1.5) * w, vals, w, color=gcol, label=glbl)
        for bi, (bar, v) in enumerate(zip(bars, vals)):
            if abs(v) >= 0.4:
                # Nudge +8.3% bar label slightly left on Shallow Benchmark (Val) to clear Net Total diamond
                dx = -0.11 if (gk == "mid_stop" and bi == 1) else 0.0
                ax.text(
                    bar.get_x() + bar.get_width() / 2 + dx,
                    v + (1.0 if v >= 0 else -1.4),
                    f"{v:+.1f}%",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=5.2,
                )
    for di, (dk, _) in enumerate(ds_specs):
        tot = st["outcome_group_decomposition"][dk]["by_alpha"]["1.25"]["series_mean"]["all_checkpoints"]["mean_total_rel_change_vs_aligned_pct"]
        ax.scatter([di], [tot], marker="D", s=26, color="#111111", zorder=6, label="Net Total" if di == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in ds_specs], fontsize=6.4)
    ax.set_ylabel("Signed Contribution to Δ $V_s$ RMSE (%)")
    ax.set_ylim(-58, 25)
    ax.set_title("Outcome-Group Error Attribution (α = 1.25)", fontsize=7.9)
    ax.legend(loc="lower right", ncol=2, fontsize=5.5, columnspacing=0.65)
    add_panel_label(ax, "(c)", x=-0.13, y=1.04)

    # Panel (d): Deep >5*sigma_hat_D Attribution (EXPLICIT Ground-Truth Misfit label!)
    ax = axes[1, 1]
    dh = st["deep_high_misfit_attribution"]
    ck_order = [
        ("ood_excluded_seed0", "OOD-s0"),
        ("ood_excluded_seed1", "OOD-s1"),
        ("ood_excluded_seed2", "OOD-s2"),
        ("all_type_seed0", "All-s0"),
        ("all_type_seed1", "All-s1"),
        ("all_type_seed2", "All-s2"),
        ("pretrained_official", "Official"),
    ]
    cx = np.arange(len(ck_order))
    aligned_med = [dh["per_ckpt"][ck]["thresholds"]["gt_5sigma"]["aligned_misfit"]["median"] * 100.0 for ck, _ in ck_order]
    gt_med = [dh["per_ckpt"][ck]["thresholds"]["gt_5sigma"]["gt_profile_misfit_subset"]["median"] * 100.0 for ck, _ in ck_order]
    gt_all_med = dh["gt_profile_all_5000"]["combined_misfit"]["median"] * 100.0
    deep_share = [dh["per_ckpt"][ck]["thresholds"]["gt_5sigma"]["by_alpha"]["1.25"]["depth_error_increment_split"]["share_of_increment_gt_10km_pct"] for ck, _ in ck_order]

    w = 0.36
    ax.bar(cx - w / 2, aligned_med, w, color=PALETTE["vermillion"], alpha=0.88, label="Zero-Shot (Aligned) Misfit (%)")
    ax.bar(cx + w / 2, gt_med, w, color=PALETTE["teal"], label="Ground-Truth Misfit (%)")
    ax.axhline(gt_all_med, color=PALETTE["teal"], lw=0.9, ls="--", label=f"Full-Set Ground-Truth ({gt_all_med:.2f}%)")
    ax.set_xticks(cx)
    ax.set_xticklabels([lbl for _, lbl in ck_order], rotation=24, ha="right", fontsize=6.1)
    ax.set_ylabel("Kept Dispersion Misfit Median (%)")
    ax.set_ylim(0, max(aligned_med) * 1.48)
    ax.set_title(r"Deep $>5\hat{\sigma}_D$ Subset: Model Failure, Not Bad Data", fontsize=7.9)
    ax.legend(loc="upper right", fontsize=5.3)

    ax.text(
        0.025,
        0.96,
        f">10 km ΔSqErr share:\n{min(deep_share):.0f}%–{max(deep_share):.0f}% (≤10 km improves)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.3,
        bbox=dict(boxstyle="round,pad=0.20", facecolor=PALETTE["info_tint"], edgecolor=PALETTE["blue"], lw=0.7),
    )
    add_panel_label(ax, "(d)", x=-0.13, y=1.04)

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
        ("Zero-Shot (Aligned)", coords, z["slices_zero_shot_aligned"], 3.0),
        ("Low-DOF Adapter", coords, z["slices_low_dof_adapter"], 3.0),
        ("Gated Adapter +\nOutput-Refined", coords, z["slices_adapter_output_refined_aligned"], 3.0),
        ("Classical Inversion\n(499 stations)", coords_cls, z["slices_classical"], 7.5),
        ("Reference Model\n(Lin et al., 2013)", coords, z["slices_reference"], 3.0),
    ]

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(146.0)))
    gs = gridspec.GridSpec(
        4,
        6,
        width_ratios=[1, 1, 1, 1, 1, 0.08],
        wspace=0.07,
        hspace=0.10,
        left=0.085,
        right=0.925,
        top=0.92,
        bottom=0.065,
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
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set_linewidth(0.55)
            ax.set_xlim(coords[:, 0].min() - 0.004, coords[:, 0].max() + 0.004)
            ax.set_ylim(coords[:, 1].min() - 0.004, coords[:, 1].max() + 0.004)
            if r == 0:
                ax.set_title(cname, fontsize=7.0, fontweight="bold", pad=4)
            if c == 0:
                ax.set_ylabel(f"z = {depths[r]:.2f} km\nLat (°N)", fontsize=6.6)
                ax.tick_params(axis="y", labelsize=5.8)
            else:
                ax.set_yticklabels([])
            if r == 3:
                ax.set_xticks([-118.20, -118.17, -118.14])
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
                ax.set_xlabel("Lon (°E)", fontsize=6.4)
                ax.tick_params(axis="x", labelsize=5.4, rotation=24)
            else:
                ax.set_xticks([-118.20, -118.17, -118.14])
                ax.set_xticklabels([])

        cax = fig.add_subplot(gs[r, 5])
        cb = fig.colorbar(sc_last, cax=cax)
        cb.ax.tick_params(labelsize=5.8)
        cb.set_label("$V_s$ (km/s)", fontsize=6.3)

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
        ("Zero-Shot (Aligned)", coords, z["slices_zero_shot_aligned"], 1.4),
        ("Low-DOF Adapter", coords, z["slices_low_dof_adapter"], 1.4),
        ("Gated Adapter +\nOutput-Refined", coords, z["slices_adapter_output_refined_aligned"], 1.4),
        ("Classical Inversion\n(500 grid pts)", coords_cls, z["slices_classical"], 5.5),
        ("Reference Model\n(CSRM)", coords, z["slices_reference"], 1.4),
    ]

    # Station 1 is #10067 (86.80E, 28.60N, Tibetan Plateau)
    st_id = int(z["station_1__global_id"][0])
    st_xy = z["station_1__coords"]

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(196.0)))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.65, 1.0], hspace=0.22, left=0.075, right=0.925, top=0.955, bottom=0.055)

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
            # Mark Station #10067 on 35 km row
            if r == 1 and c in (0, 2, 4):
                ax.scatter([st_xy[0]], [st_xy[1]], marker="*", s=42, color="#111111", edgecolors="white", lw=0.6, zorder=6)
            ax.grid(False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set_linewidth(0.55)
            ax.set_xlim(coords[:, 0].min() - 1.5, coords[:, 0].max() + 1.5)
            ax.set_ylim(coords[:, 1].min() - 1.5, coords[:, 1].max() + 1.5)
            if r == 0:
                ax.set_title(cname, fontsize=6.9, fontweight="bold", pad=3)
            if c == 0:
                ax.set_ylabel(f"({chr(97 + r)}) z = {depths[r]:.0f} km\nLat (°N)", fontsize=6.3)
                ax.tick_params(axis="y", labelsize=5.7)
            else:
                ax.set_yticklabels([])
            if r == 3:
                ax.set_xlabel("Lon (°E)", fontsize=6.2)
                ax.tick_params(axis="x", labelsize=5.6)
            else:
                ax.set_xticklabels([])

        cax = fig.add_subplot(gs_top[r, 5])
        cb = fig.colorbar(sc_last, cax=cax)
        cb.ax.tick_params(labelsize=5.7)
        cb.set_label("$V_s$ (km/s)", fontsize=6.2)

    # Bottom block: Station #10067 dispersion fit (left) + 1D Vs(z) profile (right)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.25, width_ratios=[1.15, 1.0])

    # Panel (e): Station #10067 Phase & Group Dispersion
    ax_d = fig.add_subplot(gs_bot[0])
    T = z["station_1__periods"]
    ph_obs = z["station_1__phase_obs"]
    gp_obs = z["station_1__group_obs"]
    km = z["station_1__kept_mask"] > 0
    hm = z["station_1__ho_mask"] > 0

    ax_d.axvspan(10.0, 14.0, color="#FFF3CD", alpha=0.55, zorder=0, label="10–14 s short-period bump")
    ax_d.plot(T[km[:, 0]], ph_obs[km[:, 0]], "o", color="#1A1A1A", ms=3.5, label="Obs. Phase (kept)")
    ax_d.plot(T[hm[:, 0]], ph_obs[hm[:, 0]], "D", mfc="white", mec="#1A1A1A", ms=4.2, mew=0.9, label="Obs. Phase (held-out)")
    ax_d.plot(T[km[:, 1]], gp_obs[km[:, 1]], "^", color="#555555", ms=3.3, label="Obs. Group (kept)")
    ax_d.plot(T[hm[:, 1]], gp_obs[hm[:, 1]], "v", mfc="white", mec="#555555", ms=4.0, mew=0.9, label="Obs. Group (held-out)")

    st_methods = [
        ("zero_shot_aligned", "ZS (Aligned)", PALETTE["black"], "--"),
        ("low_dof_adapter", "Low-DOF Adapter", PALETTE["teal"], "-."),
        ("adapter_output_refined_aligned", "Adapter + Output-Ref.", PALETTE["blue"], "-"),
    ]
    for mk, lbl, col, ls in st_methods:
        fp = z[f"station_1__fwd_phase_{mk}"]
        fg = z[f"station_1__fwd_group_{mk}"]
        ho_p = float(z[f"station_1__ho_pct_{mk}"][0])
        ok_p = np.isfinite(fp) & (fp > 0)
        ok_g = np.isfinite(fg) & (fg > 0)
        ax_d.plot(T[ok_p], fp[ok_p], ls=ls, color=col, lw=1.35, label=f"{lbl} (HO={ho_p:.2f}%)")
        ax_d.plot(T[ok_g], fg[ok_g], ls=ls, color=col, lw=1.05, alpha=0.72)

    ax_d.set_xlabel("Period (s)")
    ax_d.set_ylabel("Rayleigh Phase / Group Velocity (km/s)")
    ax_d.set_ylim(2.65, 4.28)
    ax_d.set_title(f"(e) Station #{st_id} ({st_xy[0]:.2f}°E, {st_xy[1]:.2f}°N): Dispersion Fit", fontsize=7.9)
    ax_d.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=5.0, ncol=2, columnspacing=0.55, handlelength=1.3)

    # Panel (f): Station #10067 1D Vs(z) Profile
    ax_v = fig.add_subplot(gs_bot[1])
    z_pred = z["pred_depths_km"]
    z_ref = z["ref_depths_km"]
    vs_ref = z["station_1__vs_ref"]

    ax_v.axhspan(10.0, 30.0, color=PALETTE["warn_tint"], alpha=0.65, zorder=0)
    ax_v.text(4.58, 20.0, "10–30 km\noscillation\nzone", ha="right", va="center", fontsize=5.9, color="#8C3800", fontweight="bold")
    ax_v.plot(vs_ref, z_ref, color=PALETTE["vermillion"], lw=1.7, label="Reference Model (CSRM)")
    for mk, lbl, col, ls in st_methods:
        vs_m = z[f"station_1__vs_{mk}"]
        ax_v.plot(vs_m, z_pred, ls=ls, color=col, lw=1.35, label=lbl)

    ax_v.set_ylim(120.0, 0.0)
    ax_v.set_xlim(2.6, 4.65)
    ax_v.set_xlabel("Shear-Wave Velocity $V_s$ (km/s)")
    ax_v.set_ylabel("Depth (km)")
    ax_v.set_title(f"(f) Station #{st_id} 1D $V_s(z)$ Profile (Tibetan Plateau)", fontsize=7.9)
    ax_v.legend(loc="lower left", fontsize=5.8, handlelength=1.5)

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
