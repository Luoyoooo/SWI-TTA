# Four-Step Operational Protocol for Safe Surface-Wave Test-Time Adaptation

This protocol translates our diagnostic findings into an actionable engineering checklist for applying test-time adaptation (TTA) to pretrained deep-learning surface-wave dispersion inversion models (e.g., OpenSWI, DispFormer).

---

## Step 1: Exact Input Alignment with Strict Zero-Leakage Masking

1. **Seal off evaluation periods first**: If held-out periods are reserved for independent quality control, apply the held-out mask **before** any preprocessing or interpolation (`swi_tta.data.alignment.prepare_aligned_input`).
2. **Resample onto the backbone training period grid**: Interpolate each station's kept observed dispersion curves onto the pretrained model's exact training period grid without out-of-band extrapolation (masking out-of-range tokens as `-1.0`).
3. **Establish the Input-Aligned Zero-Shot baseline**: In both shallow urban (LongBeach) and deep lithospheric (CSRM) arrays, input alignment alone accounts for the majority of error reduction relative to naive raw-input zero-shot inference. Always treat **Input-Aligned Zero-Shot**—never raw unaligned zero-shot—as the operational baseline.

---

## Step 2: Verify Exact Physics Referee Convergence and Calibrate the Domain Noise Floor

1. **Eliminate forward-solver root-skipping**: When using `disba` as an exact physical referee, **do not rely on the default phase-velocity step `dc = 0.005 km/s` for shallow near-surface profiles**. Profiles containing deep low-velocity zones (LVZs) can cause coarse root searches to skip the fundamental mode (Mode 0) and lock onto Mode 1, producing artificial >20% misfit tails (e.g., 63.84% vs 0.65% on a representative shallow LVZ profile).
   - **Recommended converged defaults**: `dc = 2.5e-5 km/s` for shallow profiles (`z <= 1 km`); `dc = 5.0e-3 km/s` for deep lithospheric profiles (`z <= 300 km`).
   - Run `scripts/run_root_convergence.py` on a representative sample whenever entering a new velocity regime.
2. **Calibrate the unlabeled domain noise floor ($\hat{\sigma}_D$)**: Run converged classical global inversion (e.g., 3 random starts, 20,000 forward evaluations) on a small unlabeled calibration subset (100 stations) and record the median exact kept-period misfit evaluated at the converged `dc`. Across controlled noise-amplitude scans (`0.5×` to `4.0×`), $\hat{\sigma}_D$ tracks the true data noise floor linearly (`0.929–0.953×` in shallow regimes).

---

## Step 3: Screen Stations by Initial Misfit-to-Noise-Floor Ratio ($r_0$)

Compute each station's exact kept-period misfit under Input-Aligned Zero-Shot, $m_0$, and its noise-floor-normalized ratio $r_0 = m_0 / \hat{\sigma}_D$.

| Initial Ratio $r_0 = m_0 / \hat{\sigma}_D$ | Physical Regime | Recommended Action | Diagnostic Rationale |
| :--- | :--- | :--- | :--- |
| **$r_0 < 1.5\text{--}2.0$** | Near or below the observational noise floor | **Do not refine** — keep Input-Aligned Zero-Shot | Forward-loss refinement below the noise floor fits observation noise rather than subsurface structure, degrading Resolvable-Depth $V_s$ RMSE by +40% to +157% across both OpenSWI and DispFormer. |
| **$2.0 \le r_0 \le 5.0$** | Moderate-to-strong structural domain shift | **Apply Output-Space Refinement with Holdout ES** | Surrogate-gradient refinement consistently improves $V_s$ accuracy when the initial physical misfit clearly exceeds data noise, while internal held-out early stopping curbs late-step over-refinement. |
| **$r_0 > 5.0$** | Extreme out-of-distribution failure or severe structural mismatch | **Audit exact fit & depth smoothness; fall back to Classical Inversion if needed** | In deep lithospheric arrays, stations with $r_0 > 5$ often reflect severe backbone misprediction rather than corrupted observations. Aggressive gradient steps can fit dispersion at the expense of deep crustal oscillations (`>10 km`); inspect 1D profiles and fall back to classical inversion when smoothness or exact verification degrades. |

---

## Step 4: Constrain Refinement to Resolvable Depths and Verify with Exact Physics

1. **Freeze unresolvable depths ($z > z_{\text{res}}$)**: Restrict velocity updates to depths constrained by the longest observed period, $z_{\text{res}} = \min(c(T_{\max}) T_{\max} / 3, z_{\max})$.
2. **Use Holdout Early Stopping (Holdout ES)**: Split each station's kept periods internally (e.g., 80% optimization / 20% internal stopping, strictly disjoint from any external evaluation holdout) and select the iteration minimizing internal held-out surrogate misfit.
3. **Final exact verification**: Evaluate the adapted profile with the converged exact `disba` referee (`dc = 2.5e-5` shallow, `dc = 5e-3` deep) and revert to Input-Aligned Zero-Shot if exact forward dispersion is incomplete or fails to improve kept-period misfit.
4. **Why naive discrepancy stopping alone is insufficient**: Our pre-registered validation audit demonstrates that stopping solely when kept-period misfit drops below $\alpha \hat{\sigma}_D$ (with $\alpha \in \{1.0, 1.25, 1.5, 2.0\}$) fails to guarantee non-degradation across domains because stations in the near-floor transition band ($1 < r_0 \le 2$) degrade before hitting the threshold while severe deep-regime outliers ($r_0 > 5$) degrade at depths $> 10\text{ km}$ even while kept-period misfit decreases. Combining higher screening thresholds ($r_0 \ge 2$), Holdout ES, and depth-smoothness / classical-inversion fallback addresses both failure modes transparently.
