# SWI-TTA User Guide

This user guide documents the inputs, outputs, command-line options, Python API, and expected behavior of **SWI-TTA** (Python package `swi_tta`, v1.0.0), supporting *Noise-floor regimes and forward-solver pitfalls in test-time adaptation for deep surface-wave dispersion inversion* (*Computers & Geosciences*).

---

## 1. Computational Requirements and Installation

### Hardware Requirements
- **Pure-CPU mode** (verification, unit tests, live LVZ root-skipping audit, tutorials, and complete reproduction of all 15 figures and 7 tables): multi-core CPU with 4 GB RAM (`< 30 s` total wall time).
- **GPU mode** (optional; full-archive neural inference and test-time adaptation over raw OpenSWI archives): CUDA-capable GPU with >= 8 GB VRAM.

### Installation

```bash
git clone https://github.com/Luoyoooo/SWI-TTA.git
cd SWI-TTA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Verify the environment on CPU (`< 1 s`):

```bash
python scripts/verify_install.py
pytest -v
```

---

## 2. Physical Conventions and Input Data Schema

### Velocity and Depth Grid Conventions
| Regime | Depth Layers (`D`) | Depth Range | Layer Spacing (`dz`) | Velocity Bounds (`km/s`) | Period Band (`s`) | Converged `disba` Step (`dc`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Shallow** | `70` | `0.00–2.76 km` | `0.04 km` | `[0.08, 4.45]` | `0.20–10.0` (field `0.26–1.67`) | **`2.5e-5 km/s`** |
| **Deep** | `301` | `0.0–300.0 km` | `1.00 km` | `[0.15, 4.98]` | `1.0–100.0` (field `8.0–70.0`) | **`5.0e-3 km/s`** |

- Shear-wave velocity `vs` (`km/s`) is deterministically mapped to compressional velocity `vp` (`km/s`) and bulk density `rho` (`g/cm^3`) via Brocher (2005) polynomials for `z <= 120 km` and a constant mantle ratio `vp / vs = 1.79` for `z > 120 km`, with safety floor `vs >= 0.05 km/s` and minimum layer thickness `1e-4 km`.
- **Resolvable depth** is computed per station from the longest valid period as `z_res = (c_max * T_max) / 3.0`, where `c_max` is the phase velocity at `T_max` (clamped within the model depth grid).

### Domain NPZ Input Schema (`--domain-npz`)
Scripts that execute live adaptation (`run_operational_protocol.py`, `run_diagnostics.py`, `run_mechanism_audit.py`) accept a compressed NumPy `.npz` file containing:

| Key | Shape | Dtype | Units / Description | Required |
| :--- | :--- | :--- | :--- | :---: |
| `period` | `(N, K)` | `float32/64` | Observed periods per station (`s`); invalid slots `<= 0` | Yes |
| `phase_vel` | `(N, K)` | `float32/64` | Fundamental-mode Rayleigh phase velocity (`km/s`) | Yes |
| `group_vel` | `(N, K)` | `float32/64` | Rayleigh group velocity (`km/s`); zeros if shallow | Yes |
| `kept_mask` | `(N, K)` or `(N, K, 2)` | `bool/float` | Binary mask (`1` = kept for adaptation/screening; `0` = sealed) | Yes |
| `obs_mask` | `(N, K)` or `(N, K, 2)` | `bool/float` | Binary mask of all valid observations (defaults to `kept_mask` if omitted) | Optional |
| `depth` | `(D,)` | `float32/64` | Strictly increasing 1D depth grid (`km`) | Yes |
| `grid_periods` | `(G,)` | `float32/64` | Pretrained backbone training period grid (`s`) | Yes |
| `vs_true` | `(N, D)` | `float32/64` | Ground-truth `Vs(z)` (`km/s`), synthetic domains only | Optional |
| `z_res` | `(N,)` | `float32/64` | Precomputed resolvable depth (`km`), synthetic evaluation only | Optional |

### Custom Convergence Audit NPZ Schema (`run_root_convergence.py --input-npz`)
| Key | Shape | Units / Description |
| :--- | :--- | :--- |
| `vs_profiles` | `(N, D)` | Candidate 1D `Vs(z)` profiles (`km/s`) |
| `depth_km` | `(D,)` | 1D depth grid (`km`) |
| `periods_s` | `(N, K)` or `(K,)` | Evaluation periods (`s`) |
| `phase_obs` | `(N, K)` | Observed fundamental-mode phase velocities (`km/s`) |
| `ho_mask` | `(N, K)` | Binary mask of held-out (or target) evaluation periods |

---

## 3. Command-Line Interface (CLI) Reference

All scripts live in `scripts/` and run from the repository root.

### 3.1 `scripts/verify_install.py`
Pure-CPU smoke verification checking package imports, `results_summary/manifest_sha256.json` integrity, live shallow LVZ root-skipping (`#9391`: `63.84%` at `dc=0.005` vs `0.65%` at `dc=2.5e-5`), resolvable-depth metric invariance, and zero-leakage alignment.
- **Options**: None.
- **Expected output**: 4 `[PASS]` lines and exit code `0` in `< 1 s`.

### 3.2 `scripts/run_root_convergence.py`
Audits `disba` root-finding convergence across phase-velocity step sizes (`dc`).
- **Options**:
  - `--config PATH` (default: `configs/forward_referee.yaml`): YAML listing candidate `dc` grids.
  - `--regime {shallow,deep}` (default: `shallow`): Velocity regime.
  - `--demo-lvz`: Run fast CPU convergence sweep on the 6 archived shallow LVZ examples.
  - `--input-npz PATH`: Run convergence audit on custom user profiles.
  - `--out-json PATH`: Optional path to write JSON diagnostics.
- **Expected output**: Per-step held-out misfit medians, tail rates (`>20%`), and completeness counts printed as JSON.

### 3.3 `scripts/run_operational_protocol.py`
Executes or verifies the four-step operational TTA protocol (`swi_tta.adaptation.safe_tta.run_operational_protocol`).
- **Options**:
  - `--summary-check`: Verify the archived 14-cohort closed-loop audit (`results_summary/operational_protocol_audit.json`) on CPU (`< 1 s`).
  - `--backbone-ckpt PATH`: Path to OpenSWI backbone checkpoint (live execution).
  - `--surrogate-ckpts PATH [PATH ...]`: One or more forward surrogate ensemble checkpoints.
  - `--domain-npz PATH`: Input domain `.npz` file (schema in Section 2).
  - `--regime {shallow,deep}` (default: `shallow`): Determines default routing bounds (`r_0 >= 2.0` shallow; `1.25 <= r_0 <= 3.0` deep) and converged `dc`.
  - `--sigma-hat-d FLOAT`: Calibrated unlabeled domain noise floor (`required` for live execution).
  - `--r0-low FLOAT`, `--r0-high FLOAT`: Override regime routing thresholds.
  - `--enable-zres-freeze` (default: `False`): Optionally freeze layers deeper than `freeze_depth_factor * z_res`. Disabled by default because Rayleigh sensitivity persists across `(lambda_max/3, 2*lambda_max/3]`.
  - `--freeze-depth-factor FLOAT` (default: `1.0`): Depth multiplier applied only when `--enable-zres-freeze` is set.
  - `--device {cuda,cpu}` (default: `cuda`): Automatically falls back to `cpu` if CUDA is unavailable.
  - `--workers INT` (default: `4`): Worker processes for parallel `disba` evaluation (capped at `20`).
  - `--out-per-station-npz PATH`: Optional `.npz` path saving per-station audit records (`m_init_kept`, `m_final_kept`, `r0`, `route_action`, `selected_step`, `z_res_kept`, `max_frozen_delta`, `reverted`, `revert_reason`).
- **Expected output**: JSON summary reporting routed fraction, revert rate, revert reason breakdown, exact `disba` call count, savings vs unrouted Holdout ES, and (when `vs_true` is supplied) `Resolvable-Depth Vs RMSE` change.

### 3.4 `scripts/run_diagnostics.py` & `scripts/run_mechanism_audit.py`
Inspect archived cross-backbone noise-floor signatures (`--summary-check`) or evaluate single-threshold discrepancy stopping (`--alpha`, `--sigma-hat-d`) on custom domains.

### 3.5 `scripts/reproduce_all_figures.py` & `scripts/reproduce_tables.py`
Deterministically regenerate all 15 publication figures (`.pdf`, 600-DPI `.png`, `.svg`) and all 7 tables (`.md`, booktabs `.tex`) purely on CPU from `results_summary/`.
- **Options**:
  - `--summary-dir PATH` (default: `results_summary/`)
  - `--out-dir PATH` (default: `figures/` or `tables/`)
  - `--dpi INT` (`reproduce_all_figures.py`, default: `600`)
- **Expected output**: Byte-reproducible figures/tables plus updated `manifest_sha256.json` in `< 25 s`.

### 3.6 `scripts/download_data.py`
Downloads official upstream OpenSWI archives (CC BY 4.0) from HuggingFace (`LiuFeng2317/OpenSWI`) and verifies SHA-256 checksums against `data/checksums_sha256.txt`.
- **Options**: `--subset {all,real,shallow,deep}` (default: `real`), `--data-root PATH`, `--manifest PATH`, `--verify-only`.

---

## 4. Core Python API Reference

### 4.1 Forward Solver & Convergence Verification (`swi_tta.physics.disba_referee`)
- `vs_to_vp_rho(vs, depth=None, mantle_depth_km=120.0, mantle_vp_vs_ratio=1.79) -> (vp, rho)`
- `resolve_regime_dc(domain, dc=None, dc_shallow=None, dc_deep=None, dc_cfg=None) -> float`: Resolves the explicit converged step size for `"shallow"` or `"deep"` (raises `ValueError` if none is supplied, preventing silent fallback to coarse defaults; canonical constants `DEFAULT_SHALLOW_DC_KM_S = 2.5e-5` and `DEFAULT_DEEP_DC_KM_S = 5.0e-3` are exported).
- `forward_dispersion_diag(vs, depth, periods, mode=0, compute_group=False, dc=...) -> dict`:
  - **Returns**: `{"phase_vel": ndarray, "group_vel": ndarray, "dc_used": float, "n_requested": int, "n_returned": int, "is_complete": bool, "is_empty": bool}`.

### 4.2 Zero-Leakage Splits & Input Alignment (`swi_tta.data`)
- `create_spatial_block_split(coords, ratio_a=0.30, grid_nx=10, grid_ny=10, seed=42)`
- `create_period_holdout_split(obs_mask, holdout_ratio=0.20, seed=42)`
- `prepare_aligned_input(periods, phase_vel, group_vel, obs_mask, holdout_mask, train_grid, remove_group=True) -> (x_aligned, aligned_mask)`:
  - **Guarantee**: Held-out period entries are masked **before** interpolation onto `train_grid` without out-of-band extrapolation. Perturbing held-out entries to arbitrary values changes `x_aligned` by `0.0` (verified in `tests/test_zero_leakage.py`).

### 4.3 Operational Four-Step Protocol (`swi_tta.adaptation.safe_tta`)
```python
vs_out, info = run_operational_protocol(
    vs_init_aligned,          # (N, D) float32/64 Input-Aligned Zero-Shot profiles (km/s)
    ens,                      # Forward surrogate ensemble (torch.nn.Module)
    depths,                   # (D,) depth grid (km)
    periods, phase_vel, group_vel, obs_mask, kept_mask,
    domain="shallow",         # "shallow" or "deep"
    sigma_hat_d=0.00932,      # Calibrated domain noise floor (fraction, e.g., 0.932% = 0.00932)
    dc_shallow=2.5e-5,        # Converged shallow root-search step (pass dc_deep=5.0e-3 for deep)
    enable_zres_freeze=False, # Scheme A default: unfrozen refinement
    enable_holdout_es=True,   # Internal 80/20 kept-period Holdout ES
    enable_exact_verification=True,
    disba_workers=4,
    device="cpu",
)
```
- **Outputs**:
  - `vs_out`: `(N, D)` adapted velocity profiles (`km/s`). Unrouted (`retain_zs`, `flag_outlier`) and reverted stations are **bit-identical** to `vs_init_aligned`.
  - `info`: Dictionary containing per-station arrays (`m_init_kept`, `m_final_kept`, `r0`, `route_action`, `selected_step`, `z_res_kept`, `max_frozen_delta`, `reverted`, `revert_reason`) and aggregate counters (`n_routed`, `routed_fraction`, `route_counts`, `revert_rate_all`, `n_disba_calls`, `disba_call_savings_vs_unrouted_hes_pct`).

### 4.4 Evaluation Metrics (`swi_tta.evaluation.metrics`)
- `vs_rmse_res(vs_pred, vs_true, depth, z_res) -> dict`: Computes Resolvable-Depth `Vs` RMSE strictly over layers `z <= z_res_i` (perturbations at `z > z_res_i` produce zero change).
- `disba_misfit_ho(pred_phase, obs_phase, obs_mask, ho_mask) -> float`: Relative L1 dispersion misfit on masked held-out periods.

---

## 5. Expected Behavior and Troubleshooting

1. **Exact Rollback Guarantees**: When `enable_exact_verification=True`, any routed station whose adapted profile yields an empty, incomplete, non-finite, or non-improving converged `disba` kept-period misfit automatically reverts to `vs_init_aligned` (`revert_reason` records `"final_incomplete_or_empty"` or `"final_not_improved"`).
2. **Negative `disba` Call Savings on Strong-Shift Arrays**: When nearly all stations exceed the routing threshold (e.g., LongBeach Report, `98.1%` routed), the protocol executes `13` exact calls per routed station (`1` initial screen + `11` internal Holdout ES snapshots + `1` final full-kept verification) vs `11` for unrouted Holdout ES (`-16.1%` overhead). On near-floor regimes (e.g., CSRM Report, Deep Benchmark), routing skips `78%–87%` of stations and saves **`+50.0%` to `+77.2%`** of `disba` calls.
3. **Optional Depth Freezing**: Setting `enable_zres_freeze=True` with `freeze_depth_factor=1.0` strictly enforces `max_frozen_delta == 0.0` for `z > z_res`, but is **not recommended** as default due to Rayleigh kernel sensitivity extending across `(lambda_max/3, 2*lambda_max/3]`. If deep-tail regularization is desired, set `freeze_depth_factor=2.0`.
4. **Slow `disba` Evaluation on Shallow Profiles**: Fine stepping (`dc = 2.5e-5 km/s`) is intentionally more expensive per station than `5e-3 km/s` to guarantee fundamental-mode tracking across LVZs. Increase `--workers` up to your physical CPU core count (capped internally at `20` with single-threaded BLAS/Numba workers to avoid oversubscription).
