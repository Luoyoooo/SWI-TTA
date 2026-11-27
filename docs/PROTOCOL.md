# Four-Step Operational Protocol for Safe Surface-Wave Test-Time Adaptation

This protocol translates our diagnostic findings into an actionable engineering checklist for applying test-time adaptation (TTA) to pretrained deep-learning surface-wave dispersion inversion models (e.g., OpenSWI, DispFormer).

---

## Step 1: Exact Input Alignment with Strict Zero-Leakage Masking

1. **Seal off evaluation periods first**: If held-out periods are reserved for independent quality control, apply the held-out mask **before** any preprocessing or interpolation (`swi_tta.data.alignment.prepare_aligned_input`).
2. **Resample onto the backbone training period grid**: Interpolate each station's kept observed dispersion curves onto the pretrained model's exact training period grid without out-of-band extrapolation (masking out-of-range tokens as `-1.0`).
3. **Establish the Input-Aligned Zero-Shot baseline**: In both shallow urban (LongBeach) and deep lithospheric (CSRM) arrays, input alignment alone accounts for the majority of error reduction relative to naive raw-input zero-shot inference. Always treat **Input-Aligned Zero-Shot**—never raw unaligned zero-shot—as the operational baseline.

---

## Step 2: Verify Exact Forward-Solver Convergence and Calibrate the Domain Noise Floor

1. **Eliminate forward-solver root-skipping**: When using `disba` as an exact forward solver, **do not rely on the default phase-velocity step `dc = 0.005 km/s` for shallow near-surface profiles**. Profiles containing deep low-velocity zones (LVZs) can cause coarse root searches to skip the fundamental mode (Mode 0) and lock onto Mode 1, producing artificial >20% misfit tails (e.g., 63.84% vs 0.65% on a representative shallow LVZ profile).
   - **Recommended converged defaults**: `dc = 2.5e-5 km/s` for shallow profiles (`z <= 1 km`); `dc = 5.0e-3 km/s` for deep lithospheric profiles (`z <= 300 km`).
   - Run `scripts/run_root_convergence.py` on a representative sample whenever entering a new velocity regime.
2. **Calibrate the unlabeled domain noise floor ($\hat{\sigma}_D$)**: Run converged classical global inversion (e.g., 3 random starts, 20,000 forward evaluations) on a small unlabeled calibration subset (100 stations) and record the median exact kept-period misfit evaluated at the converged `dc`. Across controlled noise-amplitude scans (`0.5×` to `4.0×`), $\hat{\sigma}_D$ tracks the true data noise floor linearly (`0.929–0.953×` in shallow regimes).

---

## Step 3: Regime-Aware Station Screening by Initial Misfit-to-Noise-Floor Ratio ($r_0$)

Compute each station's exact kept-period misfit under Input-Aligned Zero-Shot, $m_0$, and its operational noise-floor-normalized ratio $r_0 = m_0 / \hat{\sigma}_D$. Because simplified global search tracks the shallow noise floor closely ($\hat{\sigma}_D / \sigma_{\text{GT}} \approx 0.91\text{--}1.02\times$) but overestimates deep lithospheric noise floors by $1.82\text{--}1.92\times$, operational routing bounds are regime-aware (validated strictly on synthetic validation sets without test-set leakage):

| Operational Ratio $r_0 = m_0 / \hat{\sigma}_D$ | Physical Regime | Recommended Action | Diagnostic Rationale |
| :--- | :--- | :--- | :--- |
| **Shallow: $r_0 < 2.0$** / **Deep: $r_0 < 1.25$** | Near or below the observational noise floor | **Do not refine** — retain Input-Aligned Zero-Shot | Forward-loss refinement near or below the noise floor fits observation noise into the inversion null space, degrading Resolvable-Depth $V_s$ RMSE by $+13\%$ to $+157\%$ across OpenSWI and DispFormer. |
| **Shallow: $r_0 \ge 2.0$** / **Deep: $1.25 \le r_0 \le 3.0$** | Moderate-to-strong structural domain shift (safe refinement window) | **Apply Step 4: Output-Space Refinement + Holdout ES** | Physical structural misfit dominates noise; shallow stations with $r_0 > 5$ (including LongBeach, kept median $3.86\%$, $r_0 \approx 6.3\times$) deliver the largest $V_s$ gains ($-40\%$ to $-75\%$) while preserving the $174\times$ GPU speedup over classical inversion. |
| **Deep only: $r_0 > 3.0$** | Severe deep out-of-distribution failure ($r_0^{\text{GT}} \gtrsim 5.5\times$) | **Audit exact fit & depth smoothness; retain zero-shot or fall back to Classical Inversion** | In deep lithospheric arrays, stations with $r_0 > 3.0$ (especially $r_0 > 5.0$) reflect severe initial backbone misprediction; aggressive gradient updates fit dispersion by injecting unphysical $>10\text{ km}$ crustal oscillations and pushing profiles off the surrogate training manifold. |

---

## Step 4: Unfrozen Output Refinement with Internal Holdout ES and Exact Physics Rollback

Executed programmatically via `swi_tta.adaptation.safe_tta.run_operational_protocol` (CLI: `scripts/run_operational_protocol.py`):

1. **Avoid hard freezing strictly at $z > z_{\text{res}} = \lambda_{\max}/3$**: Fundamental-mode Rayleigh sensitivity kernels peak near $\lambda/3$ but retain substantial weight across $(\lambda_{\max}/3, 2\lambda_{\max}/3]$. Hard-freezing $z > \lambda_{\max}/3$ locks in $86\%\text{--}87\%$ of shallow layers and degrades resolvable-depth $V_s$ RMSE by $13\text{--}22$ percentage points (e.g., Shallow Benchmark validation drops from $-24.80\%$ unfrozen to $-5.96\%$ frozen at $\lambda_{\max}/3$, and Deep Benchmark validation drops from $-2.56\%$ to $-0.55\%$). Accordingly, Step 4 leaves refinement unfrozen by default (`enable_zres_freeze=False`; optional sensitivity-tail freezing at `freeze_depth_factor * z_res` remains configurable in code).
2. **Use Holdout Early Stopping (Holdout ES)**: Split each routed station's kept periods internally (80% optimization / 20% internal stopping, strictly disjoint from any external evaluation holdout) and select the iteration minimizing exact `disba` stopping-split misfit.
3. **Final exact verification & automatic rollback**: Evaluate the adapted profile on all kept periods with the converged exact `disba` forward solver (`dc = 2.5e-5` shallow, `dc = 5e-3` deep) and revert to Input-Aligned Zero-Shot if exact forward dispersion is empty, incomplete, non-finite, or fails to lower kept-period misfit.
4. **Closed-loop audit across 14 cohorts**: Unlike single-threshold Morozov discrepancy stopping ($m_i \le \alpha \hat{\sigma}_D$, $\alpha \in \{1.0, 1.25, 1.5, 2.0\}$, which fails simultaneous validation $0/4$), executing `run_operational_protocol` across all 14 validation, primary synthetic, cross-backbone diagnostic, and field cohorts (archived in `results_summary/operational_protocol_audit.json` and `operational_protocol_records.npz`) mitigates near-floor overfitting while cutting exact `disba` forward evaluations relative to unrouted Holdout ES.
