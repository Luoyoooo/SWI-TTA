### Table S3 (Part A). Closed-Loop Operational Four-Step Protocol Across All 14 Cohorts

| Cohort & Backbone | Rule | 50-Step | Holdout ES | Protocol | Routed (Rev.) | disba Saved |
| --- | --- | --- | --- | --- | --- | --- |
| LB-Matched (Val, OpenSWI) | r_0 ≥ 2.0 + HES | -59.14% | -58.00% | -57.46% | 93.6% (0.00%) | -11.2% |
| Shallow Bench. (Val, OpenSWI) | r_0 ≥ 2.0 + HES | -3.04% | -10.99% | -23.16% | 36.7% (0.11%) | +50.9% |
| Deep Bench. (Val, OpenSWI) | [1.25, 3.0] + HES | +8.01% | -3.85% | -3.51% | 12.7% (0.15%) | +77.2% |
| LB-Matched (Test, OpenSWI) | r_0 ≥ 2.0 + HES | -52.01% | -50.51% | -50.18% | 94.5% (0.00%) | -12.1% |
| Shallow Bench. (Test, OpenSWI) | r_0 ≥ 2.0 + HES | -6.74% | -- | -25.81% | 37.6% (0.12%) | +50.0% |
| Deep Bench. (Test, OpenSWI) | [1.25, 3.0] + HES | +21.86% | -3.29% | -3.30% | 13.0% (0.13%) | +76.9% |
| Sh. Multi-Struct. (Diag., OpenSWI) (All/Ret.) | r_0 ≥ 2.0 + HES | -4.48% / +4.73% | -12.15% / -2.54% | -22.86% / -13.93% | 42.9% (0.17%) | +44.2% |
| Sh. Prior-Shift (Diag., OpenSWI) (All/Ret.) | r_0 ≥ 2.0 + HES | -39.07% / -35.74% | -39.34% / -34.66% | -40.48% / -35.80% | 73.0% (0.04%) | +11.3% |
| Deep Lith. (Diag., OpenSWI) (All/Ret.) | [1.25, 3.0] + HES | +10.59% / +10.80% | -1.50% / -1.98% | -3.63% / -4.18% | 24.2% (0.05%) | +64.6% |
| Sh. Multi-Struct. (Diag., DispFormer) | r_0 ≥ 2.0 + HES | -13.87% | -15.66% | -15.49% | 63.1% (0.05%) | +22.2% |
| Sh. Prior-Shift (Diag., DispFormer) | r_0 ≥ 2.0 + HES | -19.81% | -18.81% | -17.48% | 71.7% (0.01%) | +12.7% |
| Deep Lith. (Diag., DispFormer) (All/Ret.) | [1.25, 3.0] + HES | -1.00% / -0.34% | -7.42% / -9.01% | -5.45% / -6.98% | 33.7% (0.07%) | +54.2% |
| LongBeach Report (OpenSWI) | r_0 ≥ 2.0 + HES | -- | -- | HO 3.75→0.82% (r_ref 0.551→0.713) | 98.1% (0.04%) | -16.1% |
| CSRM Report (OpenSWI) | [1.25, 3.0] + HES | -- | -- | HO 1.23→1.10% (r_ref 0.686→0.707) | 21.6% (0.02%) | +67.5% |
### Table S3 (Part B). Retrained-Only Sensitivity Counterpart to Table 1

| Pipeline | LB-Matched | Shallow Bench. | Deep Bench. | Deep Repeat | vs. All |
| --- | --- | --- | --- | --- | --- |
| Zero-Shot (Raw) | 0.1257 (-2.46%) | 0.0614 (+99.21%) | 0.1837 (+40.52%) | 0.00000 km/s | Same |
| Input-Aligned Zero-Shot | 0.1288 (+0.00%) | 0.0308 (+0.00%) | 0.1307 (+0.00%) | 0.00000 km/s | Base |
| Ungated Adapter (Raw) | 0.0721 (-44.00%) | 0.0399 (+29.67%) | 0.2101 (+60.69%) | 0.00903 km/s | Same |
| Gated Adapter (Raw) | 0.0744 (-42.24%) | 0.0408 (+32.41%) | 0.1589 (+21.58%) | 0.00921 km/s | Same |
| Soft-Gated Output-Refined (Raw) | 0.0571 (-55.68%) | 0.0591 (+91.86%) | 0.1724 (+31.92%) | 0.00004 km/s | Same |
| Output-Refined (Raw) | 0.0550 (-57.29%) | 0.0510 (+65.50%) | 0.1820 (+39.22%) | 0.00000 km/s | Same |
| Adapter + Output-Refined (Raw) | 0.0530 (-58.86%) | 0.0520 (+68.84%) | 0.1816 (+38.95%) | 0.00943 km/s | Same |
| Output-Space Refinement | 0.0594 (-53.90%) | 0.0326 (+5.87%) | 0.1622 (+24.05%) | 0.00000 km/s | Sh. flips + |
| Low-DOF Adapter | 0.0705 (-45.29%) | 0.0327 (+6.12%) | 0.1401 (+7.18%) | 0.00743 km/s | Sh. flips + |
| Gated Adapter + Output-Refined | 0.0510 (-60.43%) | 0.0315 (+2.35%) | 0.1702 (+30.18%) | 0.00619 km/s | Sh. flips + |
### Table S3 (Part C). Post-Hoc Deep r_0 > 5 Smoothing Ablation (2,062 Pairs) & Dunkin Verification

| Setting (lr, λ_prior) | ΔRMSE (w/o RB) | ΔRMSE (w/ RB) | Med. Sta. Δ | Kept / HO Med | >10 km ΔSSE Share | Roughness |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (0.02, 0.05) | +58.58% | +57.96% | +98.30% | 4.39% / 3.35% | 99.0% | 0.95× |
| Strong (0.02, 0.50) | +49.50% | +47.76% | +95.66% | 4.33% / 3.20% | 89.3% | 0.46× |
| Stronger (0.02, 2.00) | +38.87% | +36.58% | +86.01% | 4.49% / 3.00% | 56.8% | 0.31× |
| Heavy (0.02, 5.00) | +36.60% | +33.81% | +80.34% | 5.12% / 3.16% | 26.1% | 0.26× |
| Small step + smooth (0.005, 0.50) | +13.20% | +12.95% | +28.65% | 5.40% / 3.04% | 239.0% | 0.72× |
| Small step + heavy (0.005, 2.00) | +10.00% | +9.55% | +25.05% | 6.07% / 2.98% | 41.5% | 0.60× |

*Part A reports live end-to-end execution of swi_tta.adaptation.safe_tta.run_operational_protocol (validation-locked r_0 routing + unfrozen Holdout ES + converged full-kept disba rollback; every routed fraction cross-verified against operational_protocol_records.npz). Part C separates ΔRMSE without rollback (w/o RB) and with exact disba rollback (w/ RB); >10 km ΔSSE share = ΔSSE_{z>10} / ΔSSE_res exceeds 100% (e.g., 239.0%) when shallow layers (z ≤ 10 km) improve while z > 10 km degrades. Independent 6×6 Dunkin compound-matrix check on sample #9391: 132/132 periods solved, max |Δc| = 7.79e-07 km/s (0.000087% relative difference), confirming 0.65% vs 63.84% HO misfit.*
