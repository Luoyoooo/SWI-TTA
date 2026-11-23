# `swi-tta-diagnostics`: Physics-Referee and Noise-Floor Regime Diagnostics for Surface-Wave Test-Time Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-0072B2.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-009E73.svg)](pyproject.toml)
[![Journal: Computers & Geosciences](https://img.shields.io/badge/Target%20Journal-Computers%20%26%20Geosciences-D55E00.svg)](docs/CODE_AVAILABILITY_BLOCK.md)
[![CPU Reproducible](https://img.shields.io/badge/Figures%20%26%20Tables-CPU%20Reproducible-56B4E9.svg)](docs/FIGURE_TABLE_MAP.md)

Official software companion for **diagnostic evaluation, exact physics-referee convergence auditing, and noise-floor regime characterization of test-time adaptation (TTA) in deep-learning surface-wave dispersion inversion**.

---

## Research Overview and Key Quantitative Findings

Pretrained neural surface-wave inversion models (such as OpenSWI and DispFormer) can exhibit severe performance degradation when applied to field arrays with altered period grids, velocity distributions, or geological structures. Although forward-loss test-time adaptation (TTA) consistently lowers observed and held-out dispersion misfits, our systematic diagnostic study reveals two governing physical mechanisms that determine when TTA genuinely improves shear-wave velocity ($V_s$) recovery versus when it degrades accuracy:

1. **Hidden Evaluation Pitfall (`disba` Root-Skipping in Shallow Low-Velocity Zones)**: Evaluating shallow profiles containing deep low-velocity zones (LVZs) under `disba`'s default phase-velocity search step (`dc = 0.005 km/s`) causes fundamental-mode root skipping onto Mode 1, inflating >20% tail-failure rates artificially (`Output-Space Refinement`: **9.35% → 1.36%**; `Gated Adapter + Output-Refined`: **8.30% → 0.01%**; `Ground Truth`: **0.16% → 0.00%**). Refining to the converged step **`dc = 2.5e-5 km/s`** (shallow) while retaining **`dc = 5.0e-3 km/s`** (deep) eliminates artificial root-skipping within `< 0.1%` residual uncertainty.
2. **Noise-Floor-Normalized Regime Signature Across Backbones**: When stations are stratified by initial exact misfit normalized by the observational noise floor, both **OpenSWI** and **DispFormer** follow a unified regime law:
   - **Below / near the noise floor (`< 1.5–2×`)**: Forward-loss refinement fits observation noise and degrades Resolvable-Depth $V_s$ RMSE (median `+156.7%` in `[0, 1)` on Shallow Multi-Structure; `+13.25%` OpenSWI and `+15.24%` DispFormer on Deep Lithospheric when excluding the official checkpoint).
   - **Above the noise floor (`>= 2–3×`)**: Forward-loss refinement yields consistent $V_s$ gains (`-39.07%` overall on Shallow Prior-Shift), and **Holdout ES** captures substantial accuracy improvements (`-12.15%` vs `-4.48%` on Shallow Multi-Structure; `-1.50%` vs `+10.59%` on Deep Lithospheric) without oracle stopping.
   - **Controlled noise-amplitude scaling (`0.5×` to `4.0×`)**: Unlabeled classical-inversion noise-floor calibration ($\hat{\sigma}_D$) tracks the ground-truth noise floor linearly (`0.929–0.953×`), while 50-step TTA transitions monotonically from `-9.56%` (`0.5×`) gain to `+46.93%` (`4.0×`) degradation.
3. **Why Naive Discrepancy Stopping Alone Fails**: A pre-registered validation audit of single-threshold discrepancy stopping (`Pre-Registered Safe-TTA`, $\alpha \in \{1.0, 1.25, 1.5, 2.0\}$) fails the `<= +2.0%` multi-domain truth gate (`0/4` passing) due to near-floor escape in shallow benchmarks and deep crustal (`>10 km`) over-correction on high-misfit (`> 5\hat{\sigma}_D`) model failures.

### Quantitative Summary Across Benchmarks and Diagnostics

| Evaluation Domain | Key Diagnostic Metric | Reference Value |
| :--- | :--- | :--- |
| **Shallow Benchmark (`disba` Referee Audit)** | Tail rate (`>20%` HO): Default `dc=5e-3` → Converged `dc=2.5e-5` | `Output-Refined (50-step)`: **9.35% → 1.36%**; `Gated Adapter + Output-Ref.`: **8.30% → 0.01%** |
| **LongBeach-Matched Shallow** | Resolvable-Depth $V_s$ RMSE (`Input-Aligned Zero-Shot` → `Gated Adapter + Output-Ref.`) | **0.1179 → 0.0498 km/s** (**-57.79%**) |
| **LongBeach (Report 70%, 3,705 stations)** | Held-Out Dispersion Misfit median (`Input-Aligned Zero-Shot` → `Gated Adapter + Output-Ref.`) | **3.75% → 0.83%** (`Classical Inversion`: **0.66%**) |
| **CSRM (Report, 8,842 grid points)** | Held-Out Dispersion Misfit median (`Input-Aligned Zero-Shot` → `Low-DOF Adapter`) | **1.23% → 0.96%** (`Classical Inversion`: **1.43%**) |
| **Shallow Multi-Structure (Held-Out)** | OpenSWI Δ $V_s$ RMSE: `50-step` / `Holdout ES` / `Ideal ES` | **-4.48%** / **-12.15%** / **-36.47%** |
| **Shallow Prior-Shift (81.54% StyleA)** | OpenSWI Δ $V_s$ RMSE: `50-step` / `Holdout ES` / `Ideal ES` | **-39.07%** / **-39.34%** / **-49.18%** |
| **Deep Lithospheric (Held-Out)** | Calibrated $\hat{\sigma}_D$ vs GT floor; Retrained-only `[0, 1)` ΔRMSE | $\hat{\sigma}_D =$ **0.02417** (**1.82×** vs `0.01329`); OpenSWI **+13.25%**, DispFormer **+15.24%** |
| **Noise Amplitude Scan (`0.5×`–`4.0×`)** | $\hat{\sigma}_D / \text{GT}$ tracking ratio; `50-step` ΔRMSE shift | **0.929–0.953×**; **-9.56% (`0.5×`) → +46.93% (`4.0×`)** |

---

## Repository Structure

```text
swi-tta-diagnostics/
├── README.md                         # Project guide, reproduction map, and C&G code availability
├── LICENSE                           # MIT License
├── CITATION.cff                      # Citation File Format (CFF 1.2.0)
├── pyproject.toml                    # Editable package build (package name: swi_tta)
├── environment.yml                   # Conda environment specification
├── requirements.txt                  # Core Python dependencies
├── requirements-lock.txt             # Exact tested package versions
├── configs/                          # Portable relative-path YAML configurations
│   ├── forward_referee.yaml          # Converged disba step sizes & empirical relations
│   ├── calibration.yaml              # Calibrated noise floors (sigma_hat_D) & trust-region Delta
│   ├── benchmark_domains.yaml        # Benchmark domains, split manifests & period grids
│   ├── methods.yaml                  # Adaptation hyperparameters & public naming catalog
│   └── figures.yaml                  # Elsevier / C&G typography, palette & export settings
├── src/swi_tta/                      # Core installable Python library
│   ├── physics/                      # Exact disba forward referee & root-search diagnostics
│   ├── data/                         # OpenSWI loaders, deterministic splits & zero-leakage alignment
│   ├── models/                       # OpenSWI / DispFormer backbones, forward surrogates & adapters
│   ├── adaptation/                   # Zero-Shot, BN-Only TTA, Output Refinement, Holdout ES, Safe-TTA
│   ├── evaluation/                   # Resolvable-depth Vs RMSE, held-out misfit & field evaluation
│   └── visualization/                # Publication figures (Figs 1-9, S1-S5, GA) & LaTeX/MD tables
├── scripts/                          # Reproducible CLI entry points
│   ├── verify_install.py             # <=30 s pure-CPU smoke test + live LVZ root-skipping demo
│   ├── reproduce_all_figures.py      # One-command CPU regeneration of all 15 publication figures
│   ├── reproduce_tables.py           # One-command CPU regeneration of Tables 1-4 and S1-S2
│   ├── download_data.py              # Official HuggingFace OpenSWI downloader + SHA-256 verifier
│   ├── run_root_convergence.py       # Per-regime disba step-size convergence audit CLI
│   ├── run_diagnostics.py            # Noise-floor signature diagnostics CLI (GPU / summary)
│   └── run_mechanism_audit.py        # Safe-TTA validation & attribution audit CLI
├── data/                             # Small deterministic split manifests & upstream checksums
│   ├── DATA_AVAILABILITY.md          # OpenSWI CC BY 4.0 access & dataset documentation
│   ├── checksums_sha256.txt          # SHA-256 verification list for all 66 upstream OpenSWI files
│   └── splits/                       # Spatial-block & stratified held-out period JSON manifests
├── results_summary/                  # Compact 5.00 MB CPU reproduction archive + manifest_sha256.json
├── figures/                          # Pre-rendered Graphical Abstract, main/ (Figs 1-9), supplementary/ (S1-S5)
├── tables/                           # Generated Markdown & booktabs LaTeX tables (Tables 1-4, S1-S2)
├── docs/                             # Operational protocol, figure/table map & C&G submission block
└── tests/                            # Unit, invariant, zero-leakage, reproduction & hygiene pytest suite
```

---

## Installation and 30-Second CPU Verification

### Option 1: Editable `pip` Installation (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Option 2: Conda Environment

```bash
conda env create -f environment.yml
conda activate swi-tta-diagnostics
pip install --no-deps -e .
```

### Pure-CPU Smoke Verification (`< 1 s` Typical Runtime)

Verify the installation, summary archive SHA-256 digests, zero-leakage alignment, and run a **live CPU reproduction of the shallow LVZ `disba` root-skipping pitfall** (`dc = 0.005` yielding `63.84%` held-out misfit vs converged `dc = 2.5e-5` recovering `0.65%`):

```bash
python scripts/verify_install.py
pytest -v
```

---

## One-Command Manuscript Figure and Table Reproduction (Pure CPU)

All manuscript figures (Graphical Abstract, Figures 1–9, Figures S1–S5 in `.pdf`, 600-DPI `.png`, and editable `.svg`) and tables (Tables 1–4 and Tables S1–S2 in `.md` and `booktabs` `.tex`) reproduce deterministically on CPU directly from `results_summary/` with zero GPU or raw dataset downloads:

```bash
python scripts/reproduce_all_figures.py
python scripts/reproduce_tables.py
```

| Manuscript Item | Output Location | Reproduction Command |
| :--- | :--- | :--- |
| **Graphical Abstract (5:2, 2700×1080 px)** | `figures/graphical_abstract.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figure 1** (Workflow & 4-Step Safe-TTA Protocol) | `figures/main/fig01_workflow_and_protocol.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figure 2** (Three-Benchmark Overview) | `figures/main/fig02_benchmark_overview.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figure 3** (`disba` Root-Skipping & Tail Collapse) | `figures/main/fig03_disba_root_skipping.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figure 4** (OpenSWI 3-Domain Signature Curves) | `figures/main/fig04_openswi_signature_curves.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figure 5** (OpenSWI vs DispFormer Cross-Backbone) | `figures/main/fig05_cross_backbone_signatures.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figure 6** (Noise Amplitude Scan `0.5×–4.0×`) | `figures/main/fig06_noise_amplitude_scan.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figure 7** (Pre-Registered Safe-TTA Attribution) | `figures/main/fig07_safe_tta_failure_attribution.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figure 8** (LongBeach Report Depth Slices) | `figures/main/fig08_longbeach_slices.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figure 9** (CSRM Slices & Station `#10067` Profile) | `figures/main/fig09_csrm_slices_and_station.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Figures S1–S5** (Supplementary Diagnostics) | `figures/supplementary/figS01..S05.{pdf,png,svg}` | `python scripts/reproduce_all_figures.py` |
| **Tables 1–4 & Tables S1–S2** (Markdown & LaTeX) | `tables/table{1..4,S1,S2}.{md,tex}` | `python scripts/reproduce_tables.py` |

See [`docs/FIGURE_TABLE_MAP.md`](docs/FIGURE_TABLE_MAP.md) for the complete item-to-input provenance table and [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the practitioner four-step Safe-TTA decision protocol.

---

## Optional Full-Data & GPU Workflows

Users wishing to download the full upstream OpenSWI archives (CC BY 4.0) and run end-to-end convergence or adaptation audits on new checkpoints can use:

```bash
# 1. Fetch official OpenSWI field/synthetic data from HuggingFace with SHA-256 verification
python scripts/download_data.py --subset real

# 2. Run per-regime disba step-size convergence audit (CPU demo or custom NPZ)
python scripts/run_root_convergence.py --demo-lvz

# 3. Inspect or execute cross-backbone diagnostics & Safe-TTA mechanism audits
python scripts/run_diagnostics.py --summary-check
python scripts/run_mechanism_audit.py --summary-check
```

---

## Computer Code Availability

- **Name of the code/library**: `swi-tta-diagnostics` (Python package: `swi_tta`, v1.0.0)
- **Contact**: Corresponding author (`corresponding.author@institution.edu`; GitHub issue tracker at the repository URL below)
- **Hardware requirements**: Multi-core CPU with 4 GB RAM for installation verification, unit testing, and complete reproduction of all manuscript figures and tables (< 30 s). Optional CUDA-capable GPU (>= 8 GB VRAM) for full-scale neural adaptation on raw OpenSWI archives.
- **Program language**: Python (3.10–3.11)
- **Software required**: `numpy`, `scipy`, `torch`, `disba` (>= 0.7.0), `matplotlib`, `pyyaml`, `pillow`, `pytest` (exact versions locked in `requirements-lock.txt` and `environment.yml`).
- **Program size**: 0.41 MB for source code, scripts, configurations, documentation, and tests (`src/`, `scripts/`, `configs/`, `docs/`, `tests/`, root metadata); 5.00 MB for the compact reproduction archive (`results_summary/`); 2.86 MB for split manifests and checksums (`data/`); 25.4 MB for pre-rendered publication figures (`figures/`) and tables (`tables/`); 33.7 MB total repository working tree.
- **The source codes are available for downloading at the link**: <https://github.com/Luoyoooo/swi-tta-diagnostics>

---

## Citation

If you use this diagnostic framework, converged `disba` referee protocol, or benchmark splits in your research, please cite this repository (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{mao2026switta,
  title   = {When Does Test-Time Adaptation Help or Harm Deep Surface-Wave Dispersion Inversion? Noise-Floor Regimes, Forward-Solver Pitfalls, and Cross-Backbone Diagnostics},
  author  = {Mao, Rui},
  journal = {Computers \& Geosciences},
  year    = {2026},
  url     = {https://github.com/Luoyoooo/swi-tta-diagnostics},
  note    = {Software archive: swi-tta-diagnostics (v1.0.0)}
}
```

## Acknowledgements

We gratefully acknowledge the authors of **OpenSWI** (Liu et al., 2025) for open-access surface-wave datasets and baselines, the developers of **DispFormer** and **`disba`** (Luu, 2021) for fast surface-wave dispersion modeling, and the teams behind the **LongBeach** (Lin et al., 2013) and **CSRM** reference velocity models.
