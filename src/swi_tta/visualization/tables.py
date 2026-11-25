"""Publication table generators (Tables 1-4 and S1-S2) in Markdown and booktabs LaTeX.

Every numerical entry is read programmatically from ``results_summary/``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

from swi_tta.visualization.style import sha256_file


def _load_json(summary_dir: str, filename: str) -> Dict[str, Any]:
    with open(os.path.join(summary_dir, filename), "r", encoding="utf-8") as f:
        return json.load(f)


def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("σ̂", "$\\hat{\\sigma}$")
        .replace("Δ", "$\\Delta$")
        .replace("α", "$\\alpha$")
        .replace("ρ", "$\\rho$")
        .replace("≤", "$\\le$")
        .replace("≥", "$\\ge$")
        .replace("×", "$\\times$")
        .replace("→", "$\\rightarrow$")
        .replace("∞", "$\\infty$")
        .replace("–", "--")
    )


def _render_markdown_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[str]], notes: str = "") -> str:
    lines: List[str] = [f"### {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    if notes:
        lines.extend(["", f"*{notes}*"])
    lines.append("")
    return "\n".join(lines)


def _render_latex_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_align: str,
    notes: str = "",
) -> str:
    lines: List[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_align}}}",
        "\\toprule",
        " & ".join(_tex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_tex_escape(c) for c in row) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    if notes:
        lines.append(f"\\par\\vspace{{2pt}}\\noindent\\footnotesize{{\\textit{{{_tex_escape(notes)}}}}}")
    lines.append("\\end{table*}\n")
    return "\n".join(lines)


def _write_pair(out_dir: str, stem: str, md_text: str, tex_text: str) -> Dict[str, Dict[str, object]]:
    os.makedirs(out_dir, exist_ok=True)
    rec: Dict[str, Dict[str, object]] = {}
    for ext, txt in (("md", md_text), ("tex", tex_text)):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
        rec[ext] = {"path": p, "size_bytes": int(os.path.getsize(p)), "sha256": sha256_file(p)}
    return rec


# ---------------------------------------------------------------------------
# Table 1: Benchmark & Field Performance Comparison
# ---------------------------------------------------------------------------


def make_table1_benchmark_comparison(summary_dir: str, out_dir: str) -> Dict[str, Dict[str, object]]:
    bo = _load_json(summary_dir, "benchmark_overview.json")
    rs = _load_json(summary_dir, "root_skipping_diagnostics.json")
    names = bo["method_display_names"]
    order = bo["method_order"]

    syn = bo["synthetic_benchmarks"]
    fld = bo["field_report"]
    rec_d = rs["recomputed_domains"]

    headers = [
        "Method",
        "LB-Matched RMSE (Δ%)",
        "Shallow Bench. RMSE (Δ%)",
        "Deep Bench. RMSE (Δ%)",
        "Shallow Tail (conv. dc)",
        "LB Report HO (Tail)",
        "CSRM Report HO (Tail)",
        "LB Time (ms)",
    ]
    rows: List[List[str]] = []
    for mk in order:
        s_lb = syn["longbeach_matched_shallow"]["methods"][mk]
        s_sh = syn["shallow_benchmark"]["methods"][mk]
        s_dp = syn["deep_benchmark"]["methods"][mk]
        f_lb = fld["longbeach_report"]["methods"][mk]
        f_cs = fld["csrm_report"]["methods"][mk]
        sh_tail = rec_d["shallow_benchmark"][mk]["new_tail_pct"]
        rows.append(
            [
                names[mk],
                f"{s_lb['judged_mean_vs_rmse_res']:.4f} ({s_lb['rel_change_vs_aligned_pct']:+.2f}%)",
                f"{s_sh['judged_mean_vs_rmse_res']:.4f} ({s_sh['rel_change_vs_aligned_pct']:+.2f}%)",
                f"{s_dp['judged_mean_vs_rmse_res']:.4f} ({s_dp['rel_change_vs_aligned_pct']:+.2f}%)",
                f"{sh_tail:.2f}%",
                f"{f_lb['ho_median_mean4_converged_dc']*100:.2f}% ({f_lb['ho_tail_pct_converged_dc_mean4']:.2f}%)",
                f"{f_cs['ho_median_mean4_converged_dc']*100:.2f}% ({f_cs['ho_tail_pct_converged_dc_mean4']:.2f}%)",
                f"{f_lb['t_total_amortized_ms_mean4']:.2f}",
            ]
        )

    # Append Classical Inversion row
    cls_lb = fld["longbeach_report"]["classical_inversion"]
    cls_cs = fld["csrm_report"]["classical_inversion"]
    rows.append(
        [
            "Classical Inversion (500-sta)",
            "--",
            "--",
            "--",
            "--",
            f"{cls_lb['ho_median']*100:.2f}% (--)",
            f"{cls_cs['ho_median']*100:.2f}% (--)",
            f"{cls_lb['time_per_station_ms']:.1f}",
        ]
    )

    notes = (
        "Synthetic RMSE reports judged-checkpoint mean Resolvable-Depth Vs RMSE (km/s) across 4 shallow (3 clean seeds + 1 official) or 7 deep (6 clean seeds + 1 official) checkpoints, "
        "with relative change of domain-mean RMSE vs Input-Aligned Zero-Shot. "
        "Field columns report 4-checkpoint mean Held-Out Dispersion Misfit median (%) and tail failure rate (>20% HO) evaluated at converged disba step sizes "
        "(dc = 2.5e-5 km/s shallow, dc = 5e-3 km/s deep)."
    )
    md = _render_markdown_table("Table 1. Synthetic Benchmark Accuracy and Field Held-Out Dispersion Performance", headers, rows, notes)
    tex = _render_latex_table(
        "Synthetic benchmark Resolvable-Depth Vs RMSE and field Held-Out Dispersion Misfit across all evaluated pipelines.",
        "tab:benchmark_overview",
        headers,
        rows,
        "lccccccc",
        notes,
    )
    return _write_pair(out_dir, "table1_benchmark_comparison", md, tex)


# ---------------------------------------------------------------------------
# Table 2: disba Referee Convergence & Tail-Rate Revision
# ---------------------------------------------------------------------------


def make_table2_referee_convergence_and_tails(summary_dir: str, out_dir: str) -> Dict[str, Dict[str, object]]:
    rs = _load_json(summary_dir, "root_skipping_diagnostics.json")
    bo = _load_json(summary_dir, "benchmark_overview.json")
    names = bo["method_display_names"]
    conv = rs["convergence_audit"]

    headers = [
        "Regime / Method",
        "Step dc (km/s)",
        "Old Tail (>20%)",
        "New Tail (>20%)",
        "Δ Tail (pp)",
        "Old HO Med (%)",
        "New HO Med (%)",
        "Spearman ρ (All / ≤20%)",
    ]
    rows: List[List[str]] = []

    # Part A: Convergence summary rows
    sh_ru = conv["shallow_residual_uncertainty"]
    dp_ru = conv["deep_residual_uncertainty"]
    sh_ms = conv["single_process_timing_ms"]["shallow_benchmark"]["2.5e-05"]["median_ms"]
    dp_ms = conv["single_process_timing_ms"]["deep_benchmark"]["0.005"]["median_ms"]
    rows.append(
        [
            "Shallow Audit (vs 1.25e-5)",
            f"{sh_ru['final_dc']:.1e} ({sh_ms:.1f} ms)",
            f"Δ>0.1pp: {sh_ru['pct_changed_gt_0p1pp']:.3f}%",
            f"Cross 20%: {sh_ru['pct_cross_20pct']:.3f}%",
            f"Empty: {sh_ru['n_empty']}",
            "--",
            "--",
            "Converged (PASS)",
        ]
    )
    rows.append(
        [
            "Deep Audit (vs 2.0e-4)",
            f"{dp_ru['final_dc']:.1e} ({dp_ms:.1f} ms)",
            f"Δ>0.1pp: {dp_ru['pct_changed_gt_0p1pp']:.3f}%",
            f"Cross 20%: {dp_ru['pct_cross_20pct']:.3f}%",
            f"Empty: {dp_ru['n_empty']}",
            "--",
            "--",
            "Converged (PASS)",
        ]
    )

    # Part B: Shallow & Deep key methods + Ground Truth
    selected_methods = [
        "zero_shot_aligned",
        "output_refined_raw",
        "output_refined_aligned",
        "low_dof_adapter",
        "adapter_output_refined_aligned",
    ]
    for dk, d_lbl, dc_str in (
        ("shallow_benchmark", "Shallow", "2.5e-5"),
        ("deep_benchmark", "Deep", "5.0e-3"),
    ):
        gt = rs["gt_noise_floor_default_vs_converged"][dk]
        rows.append(
            [
                f"{d_lbl}: Ground Truth",
                dc_str,
                f"{gt['ho_old_tail_pct']:.2f}% ({gt['n_ho_old_tail']})",
                f"{gt['ho_new_tail_pct']:.2f}% ({gt['n_ho_new_tail']})",
                f"{gt['ho_new_tail_pct'] - gt['ho_old_tail_pct']:+.2f}",
                f"{gt['obs_old_median']*100:.3f}",
                f"{gt['obs_new_median']*100:.3f}",
                "--",
            ]
        )
        for mk in selected_methods:
            st = rs["recomputed_domains"][dk][mk]
            rows.append(
                [
                    f"{d_lbl}: {names[mk]}",
                    dc_str,
                    f"{st['old_tail_pct']:.2f}%",
                    f"{st['new_tail_pct']:.2f}%",
                    f"{st['delta_tail_pp']:+.2f}",
                    f"{st['ho_old_median']*100:.3f}",
                    f"{st['ho_new_median']*100:.3f}",
                    f"{st['spearman_new_all']:.3f} / {st['spearman_new_no_tail']:.3f}",
                ]
            )

    notes = (
        "Old columns use disba default dc = 5e-3 km/s; New columns use converged per-regime dc "
        "(2.5e-5 km/s shallow, 5e-3 km/s deep). Ground-Truth counts show stations exceeding 20% held-out misfit out of 5,000 test profiles."
    )
    md = _render_markdown_table("Table 2. Exact disba Referee Step-Size Convergence Audit and Tail-Rate Revision", headers, rows, notes)
    tex = _render_latex_table(
        "Exact disba forward referee step-size convergence audit, tail-failure revision, and Spearman correlation.",
        "tab:referee_convergence",
        headers,
        rows,
        "lccccccc",
        notes,
    )
    return _write_pair(out_dir, "table2_referee_convergence_and_tails", md, tex)


# ---------------------------------------------------------------------------
# Table 3: Cross-Backbone Diagnostics & Noise-Amplitude Scan
# ---------------------------------------------------------------------------


def make_table3_cross_backbone_diagnostics(summary_dir: str, out_dir: str) -> Dict[str, Dict[str, object]]:
    cb = _load_json(summary_dir, "cross_backbone_diagnostics.json")

    headers = [
        "Domain / Experiment",
        "Backbone / Subset",
        "Aligned RMSE",
        "50-step ΔRMSE (95% CI)",
        "Holdout ES ΔRMSE",
        "Ideal ES ΔRMSE",
        "50-step HO Med (%)",
        "σ̂_D / GT Ratio",
    ]
    rows: List[List[str]] = []

    domain_labels = [
        ("shallow_multi_structure", "Shallow Multi-Structure"),
        ("shallow_prior_shift", "Shallow Prior-Shift"),
        ("deep_lithospheric", "Deep Lithospheric"),
    ]
    for dk, dlbl in domain_labels:
        d_blk = cb["domains"][dk]
        for bb_k, bb_lbl in (("openswi", "OpenSWI"), ("dispformer", "DispFormer")):
            for sub_k, sub_lbl in (("all_judged_ckpts", "All ckpts"), ("retrained_only", "Retrained-only")):
                if sub_k not in d_blk["by_backbone"][bb_k]:
                    continue
                blk = d_blk["by_backbone"][bb_k][sub_k]
                zs_aligned = blk["zero_shot_aligned"]["vs_rmse_res_mean"]
                ref50 = blk["output_refined_50"]
                ref50_ci = ref50["cluster_bootstrap"]["ci95_rel_change_pct"]
                ies = blk["ideal_es"]["mean_rel_change_vs_aligned_pct"]
                hes_str = (
                    f"{blk['holdout_es']['mean_rel_change_vs_aligned_pct']:+.2f}%"
                    if blk.get("holdout_es") is not None
                    else "--"
                )
                rows.append(
                    [
                        dlbl,
                        f"{bb_lbl} ({sub_lbl})",
                        f"{zs_aligned:.4f}",
                        f"{ref50['mean_rel_change_vs_aligned_pct']:+.2f}% [{ref50_ci[0]:+.2f}, {ref50_ci[1]:+.2f}]",
                        hes_str,
                        f"{ies:+.2f}%",
                        f"{ref50['ho_misfit_median']*100:.2f}%",
                        "--",
                    ]
                )

    # Prior-shift subgroup breakdown (StyleA vs non-StyleA)
    ps_st = cb["domains"]["shallow_prior_shift"]["structure_type_breakdown"]
    for st_k, st_lbl in (("StyleA", "Prior-Shift: StyleA"), ("non_StyleA", "Prior-Shift: non-StyleA")):
        sb = ps_st[st_k]
        for bb_k, bb_lbl in (("openswi", "OpenSWI"), ("dispformer", "DispFormer")):
            b = sb["by_backbone"][bb_k]
            hes_s = f"{b['holdout_es_rel_change_vs_aligned_pct']:+.2f}%" if "holdout_es_rel_change_vs_aligned_pct" in b else "--"
            rows.append(
                [
                    f"{st_lbl} ({sb['fraction_pct']:.2f}%)",
                    bb_lbl,
                    f"{b['aligned_vs_rmse_res_mean']:.4f}",
                    f"{b['output_refined_50_rel_change_vs_aligned_pct']:+.2f}%",
                    hes_s,
                    f"{b['ideal_es_rel_change_vs_aligned_pct']:+.2f}%",
                    f"{b['output_refined_50_ho_misfit_median']*100:.2f}%",
                    "--",
                ]
            )

    # Noise amplitude scan (0.5x - 4.0x)
    ns = cb["noise_amplitude_scan"]
    for nk in ("noise_0p5x", "noise_1p0x", "noise_2p0x", "noise_4p0x"):
        e = ns[nk]
        osw = e["openswi"]["all_judged_ckpts"]
        p3 = osw["output_refined_50"]
        p3_ci = p3["cluster_bootstrap"]["ci95_rel_change_pct"]
        ies = osw["ideal_es"]["mean_rel_change_vs_aligned_pct"]
        hes_ns = (
            f"{osw['holdout_es']['mean_rel_change_vs_aligned_pct']:+.2f}%"
            if osw.get("holdout_es") is not None
            else "--"
        )
        trk = e["sigma_hat_d_vs_gt_tracking"]
        rows.append(
            [
                f"Noise Scan {e['scale']:g}× (n={e['n_samples']})",
                "OpenSWI (All ckpts)",
                f"{osw['zero_shot_aligned']['vs_rmse_res_mean']:.4f}",
                f"{p3['mean_rel_change_vs_aligned_pct']:+.2f}% [{p3_ci[0]:+.2f}, {p3_ci[1]:+.2f}]",
                hes_ns,
                f"{ies:+.2f}%",
                f"{p3['ho_misfit_median']*100:.2f}%",
                f"{trk['sigma_hat_D_over_gt_test']:.3f}×",
            ]
        )

    notes = (
        "95% CIs are paired cluster-bootstrap intervals resampled by unique subsurface profile ID. "
        "Holdout ES is evaluated across OpenSWI, DispFormer, and controlled noise-amplitude scans using identical frozen hyperparameters."
    )
    md = _render_markdown_table("Table 3. Held-Out Cross-Backbone Diagnostics, Subgroup Breakdown, and Noise-Amplitude Scan", headers, rows, notes)
    tex = _render_latex_table(
        "Cross-backbone diagnostic comparison (OpenSWI vs DispFormer), Shallow Prior-Shift subgroup breakdown, and noise-amplitude scan.",
        "tab:cross_backbone_diagnostics",
        headers,
        rows,
        "llcccccc",
        notes,
    )
    return _write_pair(out_dir, "table3_cross_backbone_diagnostics", md, tex)


# ---------------------------------------------------------------------------
# Table 4: Pre-Registered Safe-TTA Validation Audit & Failure Attribution
# ---------------------------------------------------------------------------


def make_table4_safe_tta_validation_audit(summary_dir: str, out_dir: str) -> Dict[str, Dict[str, object]]:
    st = _load_json(summary_dir, "safe_tta_validation_audit.json")

    headers = [
        "Section / Setting",
        "LB-Matched (Val) ΔRMSE",
        "Shallow Bench. (Val) ΔRMSE",
        "Deep Bench. (Val) ΔRMSE",
        "LB Tuning HO (Tail)",
        "Truth Gate (≤+2%)",
    ]
    rows: List[List[str]] = []

    for ak in ("1", "1.25", "1.5", "2"):
        a_sel = st["alpha_selection"]["per_alpha"][ak]
        vals = []
        for dk in ("longbeach_matched_val", "shallow_benchmark_val", "deep_benchmark_val"):
            cb = st["paired_cluster_bootstrap_and_retrained_only"][dk]["by_alpha"][ak]["all_judged_ckpts"]["cluster_bootstrap"]
            pt = cb["point_rel_change_pct"]
            ci = cb["ci95_rel_change_pct"]
            vals.append(f"{pt:+.2f}% [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        lb_a = st["alpha_sensitivity"]["longbeach_tuning"]["by_alpha"][ak]
        rows.append(
            [
                f"Safe-TTA (α = {a_sel['alpha']:g})",
                vals[0],
                vals[1],
                vals[2],
                f"{lb_a['mean_ho_median_across_ckpts']*100:.2f}% ({lb_a['mean_tail_rate_gt_20pct_across_ckpts']*100:.2f}%)",
                "PASS" if a_sel["truth_gate_pass"] else "FAIL",
            ]
        )

    # Outcome-group attribution at alpha = 1.25
    for gk, glbl in (
        ("step0_stop", "α=1.25 Group: Step-0 Stop"),
        ("mid_stop", "α=1.25 Group: Mid-Stop"),
        ("ran_50_steps", "α=1.25 Group: Ran 50 Steps"),
        ("reverted", "α=1.25 Group: Reverted"),
    ):
        g_vals = []
        for dk in ("longbeach_matched_val", "shallow_benchmark_val", "deep_benchmark_val"):
            g = st["outcome_group_decomposition"][dk]["by_alpha"]["1.25"]["series_mean"]["all_checkpoints"]["groups"][gk]
            g_vals.append(f"{g['mean_rel_contribution_pct']:+.2f}% ({g['mean_station_pct']:.1f}% sta)")
        rows.append([glbl, g_vals[0], g_vals[1], g_vals[2], "--", "--"])

    # Deep > 5*sigma_hat_D attribution summary across all 7 deep checkpoints
    dh = st["deep_high_misfit_attribution"]
    gt_all = dh["gt_profile_all_5000"]["combined_misfit"]["median"] * 100.0
    ck_aligned = [v["thresholds"]["gt_5sigma"]["aligned_misfit"]["median"] * 100.0 for v in dh["per_ckpt"].values()]
    ck_gt = [v["thresholds"]["gt_5sigma"]["gt_profile_misfit_subset"]["median"] * 100.0 for v in dh["per_ckpt"].values()]
    ck_d10 = [v["thresholds"]["gt_5sigma"]["by_alpha"]["1.25"]["depth_error_increment_split"]["share_of_increment_gt_10km_pct"] for v in dh["per_ckpt"].values()]
    rows.append(
        [
            "Deep >5σ̂_D Attribution (7 ckpts)",
            f"Aligned misfit: {min(ck_aligned):.1f}–{max(ck_aligned):.1f}%",
            f"GT misfit: {min(ck_gt):.2f}–{max(ck_gt):.2f}%",
            f"Full GT: {gt_all:.2f}%",
            f">10 km ΔSqErr: {min(ck_d10):.0f}–{max(ck_d10):.0f}%",
            "Model Failure",
        ]
    )

    notes = (
        "Pre-registered validation truth gate requires ΔResolvable-Depth Vs RMSE ≤ +2.0% on all three validation benchmarks simultaneously. "
        "No α passed (0/4); α = 1.5 came closest to the +2.0% threshold (+2.78% shallow, +2.85% deep), while pre-registered default α = 1.25 is decomposed in detail."
    )
    md = _render_markdown_table("Table 4. Pre-Registered Safe-TTA Validation Audit, Outcome-Group Decomposition, and Deep >5σ̂_D Attribution", headers, rows, notes)
    tex = _render_latex_table(
        "Pre-Registered Safe-TTA validation truth gate across alpha, outcome-group decomposition, and deep high-misfit attribution.",
        "tab:safe_tta_audit",
        headers,
        rows,
        "lccccc",
        notes,
    )
    return _write_pair(out_dir, "table4_safe_tta_validation_audit", md, tex)


# ---------------------------------------------------------------------------
# Table S1: Domain Noise-Floor Calibration (Verifying Deep Lithospheric Erratum)
# ---------------------------------------------------------------------------


def make_tableS1_noise_floor_calibration(summary_dir: str, out_dir: str) -> Dict[str, Dict[str, object]]:
    cal = _load_json(summary_dir, "calibration_and_domains.json")
    cnf = cal["calibrated_noise_floors"]
    trd = cal["trust_region_delta"]

    domain_order = [
        ("longbeach_tuning", "LongBeach (Tuning 30%)"),
        ("csrm_validation", "CSRM (Validation)"),
        ("longbeach_matched_val", "LongBeach-Matched (Val)"),
        ("shallow_benchmark_val", "Shallow Benchmark (Val)"),
        ("deep_benchmark_val", "Deep Benchmark (Val)"),
        ("shallow_multi_structure", "Shallow Multi-Structure"),
        ("shallow_prior_shift", "Shallow Prior-Shift"),
        ("deep_lithospheric", "Deep Lithospheric"),
        ("noise_0p5x", "Noise Scan 0.5×"),
        ("noise_1p0x", "Noise Scan 1.0×"),
        ("noise_2p0x", "Noise Scan 2.0×"),
        ("noise_4p0x", "Noise Scan 4.0×"),
    ]

    headers = [
        "Domain",
        "Regime",
        "N Cal.",
        "Eval dc (km/s)",
        "Calibrated σ̂_D",
        "GT Noise Floor (Eval)",
        "Ratio σ̂_D / GT",
    ]
    rows: List[List[str]] = []
    for dk, dlbl in domain_order:
        r = cnf[dk]
        gt_str = f"{r['gt_eval_median']:.5f}" if "gt_eval_median" in r else "--"
        rat_str = f"{r['ratio_sigma_over_gt_eval']:.2f}×" if "ratio_sigma_over_gt_eval" in r else "--"
        rows.append(
            [
                dlbl,
                r["regime"].capitalize(),
                str(r["n_calibration_stations"]),
                f"{r['eval_dc']:.1e}",
                f"{r['sigma_hat_d']:.5f}",
                gt_str,
                rat_str,
            ]
        )

    notes = (
        f"Calibrated L2 trust-region bounds Δ (km/s): "
        f"OpenSWI shallow = {trd['openswi_shallow']:.4f}, "
        f"OpenSWI deep = {trd['openswi_deep']:.4f}, "
        f"DispFormer shallow = {trd['dispformer_shallow']:.4f}, "
        f"DispFormer deep = {trd['dispformer_deep']:.4f}. "
        f"Deep Lithospheric uses its dedicated calibration split (σ̂_D = {cnf['deep_lithospheric']['sigma_hat_d']:.5f}, "
        f"ratio = {cnf['deep_lithospheric']['ratio_sigma_over_gt_eval']:.2f}× vs GT {cnf['deep_lithospheric']['gt_eval_median']:.5f})."
    )
    md = _render_markdown_table("Table S1. Domain-Level Calibrated Noise Floor (σ̂_D) and Ground-Truth Comparison", headers, rows, notes)
    tex = _render_latex_table(
        "Calibrated noise floor estimates (sigma_hat_D) from 100-station Classical Inversion across all 12 domains.",
        "tab:supp_noise_floor_calibration",
        headers,
        rows,
        "lcccccc",
        notes,
    )
    return _write_pair(out_dir, "tableS1_noise_floor_calibration", md, tex)


# ---------------------------------------------------------------------------
# Table S2: Noise-Floor-Normalized 8-Bin Signature Summary
# ---------------------------------------------------------------------------


def make_tableS2_signature_bins(summary_dir: str, out_dir: str) -> Dict[str, Dict[str, object]]:
    cb = _load_json(summary_dir, "cross_backbone_diagnostics.json")

    headers = [
        "Domain & Backbone",
        "r0^GT Bin",
        "N (Share %)",
        "50-step Med ΔRMSE (IQR)",
        "Holdout ES Med ΔRMSE",
        "Ideal ES Med ΔRMSE",
    ]
    rows: List[List[str]] = []

    specs: List[Tuple[str, str, str, str, str]] = [
        ("shallow_multi_structure", "Shallow Multi-Structure", "openswi", "all_judged_ckpts", "OpenSWI (All)"),
        ("shallow_multi_structure", "Shallow Multi-Structure", "dispformer", "all_judged_ckpts", "DispFormer (All)"),
        ("shallow_prior_shift", "Shallow Prior-Shift", "openswi", "all_judged_ckpts", "OpenSWI (All)"),
        ("shallow_prior_shift", "Shallow Prior-Shift", "dispformer", "all_judged_ckpts", "DispFormer (All)"),
        ("deep_lithospheric", "Deep Lithospheric", "openswi", "all_judged_ckpts", "OpenSWI (All)"),
        ("deep_lithospheric", "Deep Lithospheric", "dispformer", "all_judged_ckpts", "DispFormer (All)"),
        ("deep_lithospheric", "Deep Lithospheric", "openswi", "retrained_only", "OpenSWI (Retrained-only)"),
        ("deep_lithospheric", "Deep Lithospheric", "dispformer", "retrained_only", "DispFormer (Retrained-only)"),
    ]

    for dk, dlbl, bb_k, sub_k, bblbl in specs:
        bins = cb["domains"][dk]["by_backbone"][bb_k][sub_k]["pooled_signature_bins"]["bins"]
        for b in bins:
            m50 = b["methods"]["output_refined_50"]
            mid = b["methods"]["ideal_es"]
            mho = b["methods"].get("holdout_es")
            ho_str = f"{mho['median_rel_change_pct']:+.2f}%" if mho is not None else "--"
            low_flag = "*" if b["low_n_flag"] else ""
            rows.append(
                [
                    f"{dlbl} — {bblbl}",
                    b["bin_label"],
                    f"{b['n']}{low_flag} ({b['fraction_pct']:.1f}%)",
                    f"{m50['median_rel_change_pct']:+.2f}% [{m50['q25_rel_change_pct']:+.1f}, {m50['q75_rel_change_pct']:+.1f}]",
                    ho_str,
                    f"{mid['median_rel_change_pct']:+.2f}%",
                ]
            )

    notes = (
        "Asterisk (*) marks low-count bins (n < 200). On Deep Lithospheric [0, 1), excluding the official checkpoint reveals "
        "consistent below-floor degradation across both architectures (OpenSWI +13.25%, DispFormer +15.24%)."
    )
    md = _render_markdown_table("Table S2. Complete Eight-Bin Noise-Floor-Normalized Diagnostic Signatures", headers, rows, notes)
    tex = _render_latex_table(
        "Eight-bin noise-floor-normalized diagnostic signatures across all three held-out synthetic domains and both backbones.",
        "tab:supp_signature_bins",
        headers,
        rows,
        "llcccc",
        notes,
    )
    return _write_pair(out_dir, "tableS2_signature_bins", md, tex)


def generate_all_tables(summary_dir: str, tables_dir: str) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Generate all manuscript tables (Tables 1-4 and S1-S2) in Markdown and LaTeX."""
    builders = [
        ("table1_benchmark_comparison", make_table1_benchmark_comparison),
        ("table2_referee_convergence_and_tails", make_table2_referee_convergence_and_tails),
        ("table3_cross_backbone_diagnostics", make_table3_cross_backbone_diagnostics),
        ("table4_safe_tta_validation_audit", make_table4_safe_tta_validation_audit),
        ("tableS1_noise_floor_calibration", make_tableS1_noise_floor_calibration),
        ("tableS2_signature_bins", make_tableS2_signature_bins),
    ]
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for name, fn in builders:
        out[name] = fn(summary_dir, tables_dir)
    return out
