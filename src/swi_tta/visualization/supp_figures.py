"""Supplementary publication figures (Figures S1-S5) for Computers & Geosciences.

All figures are generated deterministically on CPU strictly from the compact
summary archive in ``results_summary/``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from swi_tta.visualization.style import (
    DOUBLE_COL_IN,
    PALETTE,
    add_panel_label,
    apply_cg_style,
    mm_to_in,
    save_figure,
)


def _load_json(summary_dir: str, filename: str) -> Dict[str, Any]:
    with open(os.path.join(summary_dir, filename), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure S1: Iteration Curves, Optimal-Step Lag, Overfit & Spearman Audit
# ---------------------------------------------------------------------------


def make_figS01_iteration_curves_and_lag(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure S1: median iteration trajectories, step lag, and Spearman audit."""
    apply_cg_style()
    bo = _load_json(summary_dir, "benchmark_overview.json")

    ds_specs = [
        ("longbeach_matched_shallow", "(a) LB-Matched Shallow"),
        ("shallow_benchmark", "(b) Shallow Benchmark"),
        ("deep_benchmark", "(c) Deep Benchmark"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL_IN, mm_to_in(136.0)))
    plt.subplots_adjust(wspace=0.40, hspace=0.50, left=0.09, right=0.925, top=0.93, bottom=0.19)

    # Top row: Median iteration trajectories (RMSE left axis, HO misfit right axis)
    for col, (dk, title) in enumerate(ds_specs):
        ax = axes[0, col]
        ax_r = ax.twinx()
        ax_r.spines["top"].set_visible(False)
        ax_r.grid(False)

        c_out = bo["iteration_curves_median"][dk]["output_refined_aligned"]
        c_adp = bo["iteration_curves_median"][dk]["low_dof_adapter"]
        steps = np.asarray(c_out["steps"], dtype=float)
        rmse_out = np.asarray(c_out["vs_rmse_res"], dtype=float)
        ho_out = np.asarray(c_out["ho_misfit"], dtype=float) * 100.0
        rmse_adp = np.asarray(c_adp["vs_rmse_res"], dtype=float)
        ho_adp = np.asarray(c_adp["ho_misfit"], dtype=float) * 100.0

        l1, = ax.plot(steps, rmse_out, "-o", color=PALETTE["vermillion"], ms=3.0, lw=1.35, label="Out.-Space Ref.: RMSE")
        l2, = ax.plot(steps, rmse_adp, "--s", color=PALETTE["teal"], ms=2.8, lw=1.2, label="Low-DOF Adp.: RMSE")
        l3, = ax_r.plot(steps, ho_out, "-.^", color=PALETTE["blue"], ms=2.8, lw=1.2, label="Out.-Space Ref.: HO (%)")
        l4, = ax_r.plot(steps, ho_adp, ":d", color=PALETTE["sky"], ms=2.6, lw=1.1, label="Low-DOF Adp.: HO (%)")

        b_rmse = int(c_out["best_step_vs_rmse_res"])
        b_ho = int(c_out["best_step_ho_misfit"])
        lag = int(c_out["lag_best_ho_minus_best_rmse_steps"])
        ax.axvline(b_rmse, color=PALETTE["vermillion"], lw=0.85, ls=":")
        ax.axvline(b_ho, color=PALETTE["blue"], lw=0.85, ls=":")
        if col == 2:
            y_lo_c, y_hi_c = ax.get_ylim()
            ax.set_ylim(y_lo_c, y_hi_c + 0.40 * (y_hi_c - y_lo_c))
        ax.text(
            0.97,
            0.96,
            f"Out.-Space Ref.:\nBest RMSE: {b_rmse}\nBest HO: {b_ho}\nLag: {lag:+d} steps",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.3,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
        )

        ax.set_title(title, fontsize=8.0, fontweight="bold")
        ax.set_xlabel("Adaptation Step", fontsize=7.6)
        if col == 0:
            ax.set_ylabel("Median $V_s$ RMSE (km/s)", fontsize=7.6, color=PALETTE["vermillion"])
        if col == 2:
            ax_r.set_ylabel("Median HO Misfit (%)", fontsize=7.6, color=PALETTE["blue"])
        ax.tick_params(axis="x", labelsize=6.8)
        ax.tick_params(axis="y", labelcolor=PALETTE["vermillion"], labelsize=6.8)
        ax_r.tick_params(axis="y", labelcolor=PALETTE["blue"], labelsize=6.8)

        if col == 0:
            ax.legend(
                [l1, l2, l3, l4],
                [l.get_label() for l in (l1, l2, l3, l4)],
                loc="center right",
                bbox_to_anchor=(0.98, 0.50),
                fontsize=6.0,
                framealpha=0.92,
                handlelength=1.2,
            )

    # Bottom row: Spearman rho (all vs tail-excluded at converged dc) — single-line rotated ticks
    key_methods = [
        ("zero_shot_aligned", "Aligned ZS"),
        ("low_dof_adapter", "Low-DOF"),
        ("output_refined_raw", "Out.-Ref. (Raw)"),
        ("output_refined_aligned", "Out.-Space Ref."),
        ("adapter_output_refined_aligned", "Adp.+Out.-Ref."),
    ]
    for col, (dk, _) in enumerate(ds_specs):
        ax = axes[1, col]
        sp_d = bo["spearman_ho_vs_rmse"][dk]
        x = np.arange(len(key_methods))
        w = 0.36
        rho_all = [sp_d[mk]["converged_dc_all_rho"] for mk, _ in key_methods]
        rho_nt = [sp_d[mk]["converged_dc_no_tail_rho"] for mk, _ in key_methods]

        b1 = ax.bar(x - w / 2, rho_all, w, color=PALETTE["blue"], label=r"All (conv. $\Delta c$)")
        b2 = ax.bar(x + w / 2, rho_nt, w, color=PALETTE["orange"], label="Tail-excl. (≤20%)")
        for bar, v in zip(b1, rho_all):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0, rotation=90)
        for bar, v in zip(b2, rho_nt):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.0, rotation=90)

        # Annotate below-GT-floor fraction for Output-Refined (50-step)
        nf_out = bo["noise_floor_overfit"][dk]["methods"]["output_refined_aligned"]
        ax.text(
            0.03,
            0.97,
            f"<GT floor: {nf_out['below_own_gt_floor_fraction']*100:.1f}%\n"
            f"ΔRMSE: {nf_out['below_own_gt_floor_rel_rmse_change_pct']:+.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=PALETTE["warn_tint"], edgecolor=PALETTE["vermillion"], lw=0.6),
        )

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in key_methods], fontsize=6.4, rotation=36, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="y", labelsize=6.8)
        ax.set_ylim(0.0, 1.24)
        ax.set_title(f"({chr(100 + col)}) HO–RMSE Correlation", fontsize=8.0, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Spearman ρ (HO vs $V_s$ RMSE)", fontsize=7.4)
            ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.72), fontsize=6.1)

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure S2: Six Typical Shallow LVZ Counterfactual Examples
# ---------------------------------------------------------------------------


def make_figS02_lvz_counterfactual_examples(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure S2: 6 shallow LVZ examples (1D Vs profiles + Mode 0/1 curves)."""
    apply_cg_style()
    rs = _load_json(summary_dir, "root_skipping_diagnostics.json")
    depth_km = np.asarray(rs["typical_lvz_depth_km"], dtype=float)
    examples = rs["typical_lvz_examples"][:6]

    # 3 rows x 4 columns: left pair = Output-Space Refinement (0..2), right pair = Gated Adapter + Output-Refined (3..5)
    fig, axes = plt.subplots(3, 4, figsize=(DOUBLE_COL_IN, mm_to_in(182.0)))
    plt.subplots_adjust(wspace=0.40, hspace=0.52, left=0.085, right=0.985, top=0.825, bottom=0.065)

    for idx, ex in enumerate(examples):
        row = idx % 3
        col_base = 0 if idx < 3 else 2
        ax_v = axes[row, col_base]
        ax_c = axes[row, col_base + 1]

        # 1D Vs profile
        profs = ex["profiles"]
        ax_v.plot(profs["vs_true"], depth_km, "-", color="#1A1A1A", lw=1.35, label="Ground Truth")
        ax_v.plot(profs["vs_zero_shot_aligned"], depth_km, "--", color=PALETTE["mid_gray"], lw=1.0, label="ZS (Aligned)")
        ax_v.plot(profs["vs_refined"], depth_km, "-", color=PALETTE["vermillion"], lw=1.25, label="Refined (LVZ)")
        ax_v.plot(profs["vs_lvz_raised"], depth_km, ":", color=PALETTE["teal"], lw=1.25, label="LVZ-Raised")
        ax_v.axhline(ex["z_res_km"], color=PALETTE["blue"], lw=0.85, ls="-.")
        ax_v.text(
            0.96,
            0.05,
            f"$z_{{\\mathrm{{res}}}}={ex['z_res_km']:.2f}$ km\nmin $V_s={ex['vs_min_km_s']:.2f}$",
            transform=ax_v.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.2,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        )
        ax_v.set_ylim(depth_km[-1], 0.0)
        ax_v.set_title(f"#{ex['sample_idx']} ({ex['structure_type']})", fontsize=7.4, fontweight="bold")
        if col_base == 0:
            ax_v.set_ylabel("Depth (km)", fontsize=7.2)
        if row == 2:
            ax_v.set_xlabel("$V_s$ (km/s)", fontsize=7.2)
        ax_v.tick_params(labelsize=6.5)

        # Dispersion curves — M8 fix: plot true converged dc=2.5e-5 and separate step-refinement vs LVZ-raised HO
        T = np.asarray(ex["periods_s"], dtype=float)
        obs = np.asarray(ex["phase_obs_km_s"], dtype=float)
        is_ho = np.asarray(ex["is_heldout"], dtype=bool)
        c = ex["curves"]
        m0_old = np.asarray([np.nan if v is None else float(v) for v in c["orig_mode0_dc005"]], dtype=float)
        has_2p5 = "orig_mode0_dc2p5e5" in c
        m0_conv = np.asarray([np.nan if v is None else float(v) for v in (c["orig_mode0_dc2p5e5"] if has_2p5 else c["orig_mode0_dc0002"])], dtype=float)
        m1_old = np.asarray([np.nan if v is None else float(v) for v in c["orig_mode1_dc005"]], dtype=float)
        m0_int = np.asarray([np.nan if v is None else float(v) for v in c["lvz_raised_mode0_dc005"]], dtype=float)

        ho_old_pct = float(c["orig_ho_misfit_dc005"]) * 100.0
        ho_conv_pct = float(c.get("orig_ho_misfit_dc2p5e5", c.get("orig_ho_misfit_dc0002", np.nan))) * 100.0
        ho_lvz_pct = float(c["lvz_raised_ho_misfit_dc005"]) * 100.0
        conv_lbl = r"Mode 0 ($\Delta c=2.5\times 10^{-5}$)" if has_2p5 else r"Mode 0 ($\Delta c=2\times 10^{-4}$)"

        ax_c.plot(T[~is_ho], obs[~is_ho], "o", color="#222222", ms=2.1, label="Obs. (kept)")
        ax_c.plot(T[is_ho], obs[is_ho], "D", mfc="white", mec="#222222", ms=2.6, mew=0.7, label="Obs. (HO)")
        ax_c.plot(T, m0_old, "-", color=PALETTE["vermillion"], lw=1.2, label=r"Mode 0 ($\Delta c=5\times 10^{-3}$)")
        ax_c.plot(T, m0_conv, "--", color=PALETTE["teal"], lw=1.2, label=conv_lbl)
        ax_c.plot(T, m0_int, ":", color=PALETTE["blue"], lw=1.0, label=r"LVZ-Raised ($\Delta c=5\times 10^{-3}$)")
        ax_c.plot(T, m1_old, "-.", color=PALETTE["purple"], lw=0.85, alpha=0.8, label=r"Mode 1 ($\Delta c=5\times 10^{-3}$)")
        ax_c.set_title(
            f"Step $\\Delta c$: {ho_old_pct:.1f}%→{ho_conv_pct:.2f}%\nLVZ-raised: →{ho_lvz_pct:.1f}%",
            fontsize=6.5,
            pad=3,
        )
        if row == 2:
            ax_c.set_xlabel("Period (s)", fontsize=7.2)
        ax_c.set_ylabel("Phase $c$ (km/s)", fontsize=7.0)
        ax_c.tick_params(labelsize=6.5)

    # Column group titles & legends clearly above row 0
    fig.text(0.28, 0.985, "Output-Space Refinement (3 Tail Cases)", ha="center", va="top", fontsize=8.2, fontweight="bold", color=PALETTE["vermillion"])
    fig.text(0.76, 0.985, "Gated Adapter + Output-Refined (3 Tail Cases)", ha="center", va="top", fontsize=8.2, fontweight="bold", color=PALETTE["blue"])

    h_v, l_v = axes[0, 0].get_legend_handles_labels()
    h_c, l_c = axes[0, 1].get_legend_handles_labels()
    fig.legend(h_v, l_v, loc="upper left", bbox_to_anchor=(0.04, 0.955), ncol=2, fontsize=6.3, columnspacing=0.65)
    fig.legend(h_c, l_c, loc="upper right", bbox_to_anchor=(0.99, 0.955), ncol=2, fontsize=6.2, columnspacing=0.60)

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure S3: Default vs Converged dc Held-Out Misfit Scatter
# ---------------------------------------------------------------------------


def make_figS03_old_vs_new_dc_scatter(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure S3: log-log scatter of default vs converged dc held-out misfit."""
    apply_cg_style()
    rs = _load_json(summary_dir, "root_skipping_diagnostics.json")
    sc = np.load(os.path.join(summary_dir, rs["scatter_npz_file"]))

    rows = [
        ("shallow_benchmark", "Shallow Benchmark (dc: 5e-3 → 2.5e-5)"),
        ("deep_benchmark", "Deep Benchmark (dc: 5e-3 unchanged)"),
    ]
    methods = [
        ("zero_shot_aligned", "Aligned Zero-Shot"),
        ("output_refined_aligned", "Out.-Space Ref. (50)"),
        ("adapter_output_refined_aligned", "Gated Adp. + Out.-Ref."),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL_IN, mm_to_in(122.0)))
    plt.subplots_adjust(wspace=0.28, hspace=0.42, left=0.10, right=0.985, top=0.86, bottom=0.10)

    for r, (dk, row_lbl) in enumerate(rows):
        for c, (mk, mlbl) in enumerate(methods):
            ax = axes[r, c]
            old_v = np.clip(sc[f"{dk}__{mk}__old_ho_pct"], 0.1, 200.0)
            new_v = np.clip(sc[f"{dk}__{mk}__new_ho_pct"], 0.1, 200.0)
            dropped = (old_v > 20.0) & (new_v <= 20.0)

            ax.plot([0.1, 200.0], [0.1, 200.0], "--", color="#666666", lw=0.8, zorder=1)
            ax.axvline(20.0, color=PALETTE["vermillion"], lw=0.75, ls=":", zorder=1)
            ax.axhline(20.0, color=PALETTE["teal"], lw=0.75, ls=":", zorder=1)
            ax.scatter(old_v[~dropped], new_v[~dropped], s=5.0, color=PALETTE["blue"], alpha=0.45, linewidths=0, rasterized=True, label="Stable")
            if np.any(dropped):
                ax.scatter(old_v[dropped], new_v[dropped], s=7.0, color=PALETTE["vermillion"], alpha=0.75, linewidths=0, rasterized=True, label="Crossed down ≤20%")

            st = rs["recomputed_domains"][dk][mk]
            ax.text(
                0.04,
                0.95,
                f"Tail: {st['old_tail_pct']:.2f}%→{st['new_tail_pct']:.2f}%\n↓20%: {st['n_down_cross_20']} | ↑20%: {st['n_up_cross_20']}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.4,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.15, 180.0)
            ax.set_ylim(0.15, 180.0)
            ax.tick_params(labelsize=6.8)
            p_lbl = f"({chr(97 + r * 3 + c)}) "
            ax.set_title(f"{p_lbl}{mlbl}" if r == 0 else f"{p_lbl}{mlbl} (Deep)", fontsize=7.6)
            if r == 1:
                ax.set_xlabel(r"Default $\Delta c=5\times 10^{-3}$ HO (%)", fontsize=7.4)
            if c == 0:
                ax.set_ylabel(f"{'Shallow' if r == 0 else 'Deep'}\n" + r"Converged $\Delta c$ HO (%)", fontsize=7.4)

    fig.text(0.53, 0.975, r"Default ($\Delta c=5\times 10^{-3}$) vs. Converged-Step Held-Out Misfit Across Regimes", ha="center", va="top", fontsize=8.6, fontweight="bold")
    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure S4: Pre-Registered Safe-TTA Alpha Sensitivity Audit (2x2)
# ---------------------------------------------------------------------------


def make_figS04_safe_tta_alpha_sensitivity(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure S4: Safe-TTA sensitivity across alpha in {1.0, 1.25, 1.5, 2.0}."""
    apply_cg_style()
    st = _load_json(summary_dir, "safe_tta_validation_audit.json")
    alpha_keys = ["1", "1.25", "1.5", "2"]
    alphas = np.array([1.0, 1.25, 1.5, 2.0])

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, mm_to_in(138.0)))
    plt.subplots_adjust(wspace=0.30, hspace=0.40, left=0.095, right=0.95, top=0.94, bottom=0.09)

    val_specs = [
        ("longbeach_matched_val", "LongBeach-Matched (Val)", PALETTE["teal"], "o"),
        ("shallow_benchmark_val", "Shallow Benchmark (Val)", PALETTE["vermillion"], "s"),
        ("deep_benchmark_val", "Deep Benchmark (Val)", PALETTE["blue"], "^"),
    ]

    # Panel (a): Validation Δ Vs RMSE (%) with 95% cluster-bootstrap CIs
    ax = axes[0, 0]
    ax.axhline(0.0, color="#333333", lw=0.75, ls="--")
    ax.axhline(2.0, color=PALETTE["vermillion"], lw=0.9, ls=":", label="Pre-specified +2.0% threshold")
    for dk, lbl, col, mk in val_specs:
        pts, lo, hi = [], [], []
        for ak in alpha_keys:
            cb = st["paired_cluster_bootstrap_and_retrained_only"][dk]["by_alpha"][ak]["all_judged_ckpts"]["cluster_bootstrap"]
            pt = float(cb["point_rel_change_pct"])
            ci = cb["ci95_rel_change_pct"]
            pts.append(pt)
            lo.append(max(0.0, pt - float(ci[0])))
            hi.append(max(0.0, float(ci[1]) - pt))
        ax.errorbar(alphas, pts, yerr=[lo, hi], color=col, marker=mk, lw=1.4, capsize=2.8, label=lbl)
    ax.set_xticks(alphas)
    ax.set_xlabel(r"Discrepancy Threshold Multiplier $\alpha$")
    ax.set_ylabel("Δ Resolvable-Depth $V_s$ RMSE (%)")
    ax.set_title("(a) Validation Criterion Across α", fontsize=8.0, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.37), fontsize=6.4)

    # Panel (b): LongBeach Tuning (30%) HO Misfit & Tail Rate vs alpha
    ax = axes[0, 1]
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.grid(False)
    lb = st["alpha_sensitivity"]["longbeach_tuning"]
    ho_m = [lb["by_alpha"][ak]["mean_ho_median_across_ckpts"] * 100.0 for ak in alpha_keys]
    tl_m = [lb["by_alpha"][ak]["mean_tail_rate_gt_20pct_across_ckpts"] * 100.0 for ak in alpha_keys]
    l1, = ax.plot(alphas, ho_m, "-o", color=PALETTE["blue"], lw=1.45, label="Safe-TTA HO Median (%)")
    l2 = ax.axhline(lb["zero_shot_aligned_ho_median_mean"] * 100.0, color="#333333", ls="--", lw=0.95, label="Input-Aligned Zero-Shot HO")
    l3 = ax.axhline(lb["output_refined_50_ho_median_mean"] * 100.0, color=PALETTE["orange"], ls=":", lw=1.05, label="Output-Space Refinement (50-step) HO")
    l4, = ax_r.plot(alphas, tl_m, "--s", color=PALETTE["vermillion"], lw=1.35, label="Tail Rate (>20% HO, %)")
    ax_r.set_ylim(-0.15, 2.0)
    ax.set_xticks(alphas)
    ax.set_xlabel(r"Discrepancy Threshold Multiplier $\alpha$")
    ax.set_ylabel("LongBeach (Tuning) HO Median (%)", color=PALETTE["blue"])
    ax_r.set_ylabel("Tail Rate (>20% HO, %)", color=PALETTE["vermillion"])
    ax.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax_r.tick_params(axis="y", labelcolor=PALETTE["vermillion"])
    ax.set_title("(b) LongBeach (30%): Fit vs Tail Risk", fontsize=8.0, fontweight="bold")
    ax.legend([l1, l2, l3, l4], [h.get_label() for h in (l1, l2, l3, l4)], loc="center right", bbox_to_anchor=(0.98, 0.58), fontsize=6.3)

    # Panel (c): Discrepancy Stop Rate (%) vs alpha
    all_specs = val_specs + [("longbeach_tuning", "LongBeach (Tuning 30%)", PALETTE["purple"], "D")]
    ax = axes[1, 0]
    for dk, lbl, col, mk in all_specs:
        sr = [st["alpha_sensitivity"][dk]["by_alpha"][ak]["mean_stop_rate_across_ckpts"] * 100.0 for ak in alpha_keys]
        ax.plot(alphas, sr, f"-{mk}", color=col, lw=1.4, label=lbl)
    ax.set_xticks(alphas)
    ax.set_xlabel(r"Discrepancy Threshold Multiplier $\alpha$")
    ax.set_ylabel("Discrepancy Stop Rate (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("(c) Discrepancy Early-Stop Rate", fontsize=8.0, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.4)

    # Panel (d): Exact-Verification Revert Rate (%) vs alpha
    ax = axes[1, 1]
    for dk, lbl, col, mk in all_specs:
        rr = [st["alpha_sensitivity"][dk]["by_alpha"][ak]["mean_revert_rate_across_ckpts"] * 100.0 for ak in alpha_keys]
        ax.plot(alphas, rr, f"-{mk}", color=col, lw=1.4, label=lbl)
    ax.set_xticks(alphas)
    ax.set_xlabel(r"Discrepancy Threshold Multiplier $\alpha$")
    ax.set_ylabel("Exact-Verification Revert Rate (%)")
    ax.set_ylim(-0.6, 15.5)
    ax.set_title("(d) Exact-Verification Revert Rate", fontsize=8.0, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.4)

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure S5: Field Spatial Misfit Maps, Resampling Alignment & Classical Conv.
# ---------------------------------------------------------------------------


def make_figS05_field_spatial_misfit_and_data(
    summary_dir: str,
    out_stem: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, object]]:
    """Generate Figure S5: field spatial ΔHO maps, LongBeach station #2059, KS alignment, and Classical convergence."""
    apply_cg_style()
    cal = _load_json(summary_dir, "calibration_and_domains.json")
    z_lb = np.load(os.path.join(summary_dir, "field_slices_longbeach.npz"))
    z_cs = np.load(os.path.join(summary_dir, "field_slices_csrm.npz"))

    fig = plt.figure(figsize=(DOUBLE_COL_IN, mm_to_in(152.0)))
    gs = gridspec.GridSpec(2, 3, wspace=0.72, hspace=0.46, left=0.09, right=0.98, top=0.935, bottom=0.095)

    # Panel (a): LongBeach spatial ΔHO (Gated Adapter + Output-Refined minus Zero-Shot Aligned)
    ax = fig.add_subplot(gs[0, 0])
    d_lb = z_lb["ho_misfit_pct_adapter_output_refined_aligned"] - z_lb["ho_misfit_pct_zero_shot_aligned"]
    xy_lb = z_lb["coords_report"]
    ok = np.isfinite(d_lb)
    sc = ax.scatter(xy_lb[ok, 0], xy_lb[ok, 1], c=d_lb[ok], s=3.0, cmap="RdBu_r", vmin=-8.0, vmax=8.0, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([-118.20, -118.17, -118.14])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    cb = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.03)
    cb.ax.tick_params(labelsize=6.2)
    cb.set_label("ΔHO (pp)", fontsize=6.4, labelpad=1.5)
    ax.set_xlabel("Lon (°E)", fontsize=7.0)
    ax.set_ylabel("Lat (°N)", fontsize=7.0)
    ax.tick_params(labelsize=6.2)
    ax.set_title("(a) LongBeach: ΔHO (pp)", fontsize=7.8, fontweight="bold")

    # Panel (b): CSRM spatial ΔHO
    ax = fig.add_subplot(gs[0, 1])
    d_cs = z_cs["ho_misfit_pct_adapter_output_refined_aligned"] - z_cs["ho_misfit_pct_zero_shot_aligned"]
    xy_cs = z_cs["coords_report"]
    ok = np.isfinite(d_cs)
    sc = ax.scatter(xy_cs[ok, 0], xy_cs[ok, 1], c=d_cs[ok], s=1.5, cmap="RdBu_r", vmin=-4.0, vmax=4.0, linewidths=0, rasterized=True)
    ax.grid(False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.03)
    cb.ax.tick_params(labelsize=6.2)
    cb.set_label("ΔHO (pp)", fontsize=6.4, labelpad=1.5)
    ax.set_xlabel("Lon (°E)", fontsize=7.0)
    ax.set_ylabel("Lat (°N)", fontsize=7.0)
    ax.tick_params(labelsize=6.2)
    ax.set_title("(b) CSRM: ΔHO (pp)", fontsize=7.8, fontweight="bold")

    # Panel (c): LongBeach typical high-improvement station (#2059)
    ax = fig.add_subplot(gs[0, 2])
    st_id = int(z_lb["station_1__global_id"][0])
    T = z_lb["station_1__periods"]
    ph = z_lb["station_1__phase_obs"]
    km = z_lb["station_1__kept_mask"][:, 0] > 0
    hm = z_lb["station_1__ho_mask"][:, 0] > 0
    ax.plot(T[km], ph[km], "o", color="#222222", ms=2.5, label="Obs. (kept)")
    ax.plot(T[hm], ph[hm], "D", mfc="white", mec="#222222", ms=3.1, mew=0.8, label="Obs. (HO)")
    for mk, lbl, col, ls in (
        ("zero_shot_aligned", "Aligned ZS", PALETTE["black"], "--"),
        ("low_dof_adapter", "Low-DOF", PALETTE["teal"], "-."),
        ("adapter_output_refined_aligned", "Adp.+Out.", PALETTE["blue"], "-"),
    ):
        fp = z_lb[f"station_1__fwd_phase_{mk}"]
        ho_p = float(z_lb[f"station_1__ho_pct_{mk}"][0])
        ok_p = np.isfinite(fp) & (fp > 0)
        ax.plot(T[ok_p], fp[ok_p], ls=ls, color=col, lw=1.25, label=f"{lbl} ({ho_p:.2f}%)")
    ax.set_xlabel("Period (s)", fontsize=7.0)
    ax.set_ylabel("Phase Velocity (km/s)", fontsize=7.0)
    ax.tick_params(labelsize=6.4)
    ax.set_ylim(0.42, 1.32)
    ax.set_title(f"(c) LongBeach #{st_id} Fit", fontsize=7.8, fontweight="bold")
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.1)

    # Panel (d): LongBeach-Matched Shallow Resampling Boxplots (5 period anchors)
    ax = fig.add_subplot(gs[1, 0])
    aln = cal["longbeach_matched_alignment"]
    anchors = aln["anchors_s"]
    bx_s = aln["boxplot_stats"]
    x = np.arange(len(anchors))
    offsets = [(-0.24, "openswi_base", "OpenSWI Base", PALETTE["mid_gray"]), (0.0, "longbeach_matched", "LB-Matched", PALETTE["teal"]), (0.24, "longbeach_report", "LB Report", PALETTE["blue"])]
    for dx, k, lbl, col in offsets:
        stats_list = [
            {"med": s["med"], "q1": s["q1"], "q3": s["q3"], "whislo": s["whislo"], "whishi": s["whishi"]}
            for s in bx_s[k]
        ]
        bp = ax.bxp(
            stats_list,
            positions=x + dx,
            widths=0.20,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor=col, alpha=0.65, edgecolor="#333333", lw=0.6),
            medianprops=dict(color="#111111", lw=1.0),
            whiskerprops=dict(color="#444444", lw=0.6),
            capprops=dict(color="#444444", lw=0.6),
        )
        bp["boxes"][0].set_label(lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.2f}" for a in anchors], fontsize=6.4)
    ax.tick_params(axis="y", labelsize=6.4)
    ax.set_xlabel("Anchor Period (s)", fontsize=7.0)
    ax.set_ylabel("Phase Velocity (km/s)", fontsize=7.0)
    ax.set_title("(d) Velocity Resampling", fontsize=7.8, fontweight="bold")
    ax.legend(loc="upper left", fontsize=6.0)

    # Panel (e): Per-Period KS Drop
    ax = fig.add_subplot(gs[1, 1])
    ks_bins = aln["per_period_ks_bins"]
    t_c = [b["period_center_s"] for b in ks_bins]
    ks_pre = [b["ks_before_resampling"] for b in ks_bins]
    ks_post = [b["ks_after_resampling"] for b in ks_bins]
    ax.plot(t_c, ks_pre, "-o", color=PALETTE["vermillion"], ms=3.2, label=f"Before ({aln['per_period_ks_median_before']:.3f})")
    ax.plot(t_c, ks_post, "-s", color=PALETTE["teal"], ms=3.2, label=f"After ({aln['per_period_ks_median_after']:.3f})")
    ax.axhline(aln["ks_threshold"], color="#444444", lw=0.85, ls="--", label="Target ≤ 0.20")
    ax.set_xlabel("Period Center (s)", fontsize=7.0)
    ax.set_ylabel("Two-Sample KS", fontsize=7.0)
    ax.tick_params(labelsize=6.4)
    ax.set_ylim(0, max(ks_pre) * 1.20)
    ax.set_title("(e) Per-Period KS Audit", fontsize=7.8, fontweight="bold")
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.44), fontsize=6.0)

    # Panel (f): Classical Inversion Convergence Curves (LongBeach & CSRM)
    ax = fig.add_subplot(gs[1, 2])
    conv = cal["classical_inversion_convergence"]
    for dk, lbl, col, mk in (
        ("longbeach", "LongBeach", PALETTE["blue"], "o"),
        ("csrm", "CSRM", PALETTE["vermillion"], "s"),
    ):
        tiers = conv[dk]["tiers"]
        nfev = [t["nfev_mean"] for t in tiers]
        ho = [t["ho_median"] * 100.0 for t in tiers]
        ax.plot(nfev, ho, f"-{mk}", color=col, lw=1.35, ms=3.5, label=lbl)
        ax.scatter([conv[dk]["unconverged_nfev_mean"]], [conv[dk]["unconverged_ho_median"] * 100.0], marker="x", s=34, color=col, zorder=6)
        ax.scatter([conv[dk]["selected_budget_nfev_mean"]], [conv[dk]["formal_500sta_ho_median"] * 100.0], marker="*", s=72, color=col, edgecolors="#111111", lw=0.5, zorder=7)
    ax.set_xscale("log")
    ax.set_xlabel("Mean Forward Calls (nfev)", fontsize=7.0)
    ax.set_ylabel("HO Misfit Median (%)", fontsize=7.0)
    ax.tick_params(labelsize=6.4)
    ax.set_title("(f) Classical Inv. Conv.", fontsize=7.8, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.1)
    ax.text(
        0.96,
        0.54,
        "× Initial budget\n★ 500-sta final",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.0,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
    )

    return save_figure(fig, out_stem, formats=formats, dpi=dpi)


def generate_all_supp_figures(
    summary_dir: str,
    figures_dir: str,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Generate all 5 supplementary figures."""
    supp_dir = os.path.join(figures_dir, "supplementary")
    os.makedirs(supp_dir, exist_ok=True)
    builders = [
        ("figS01_iteration_curves_and_lag", make_figS01_iteration_curves_and_lag),
        ("figS02_lvz_counterfactual_examples", make_figS02_lvz_counterfactual_examples),
        ("figS03_old_vs_new_dc_scatter", make_figS03_old_vs_new_dc_scatter),
        ("figS04_safe_tta_alpha_sensitivity", make_figS04_safe_tta_alpha_sensitivity),
        ("figS05_field_spatial_misfit_and_data", make_figS05_field_spatial_misfit_and_data),
    ]
    results: Dict[str, Dict[str, Dict[str, object]]] = {}
    for name, fn in builders:
        results[name] = fn(summary_dir, os.path.join(supp_dir, name), formats=formats, dpi=dpi)
    return results
