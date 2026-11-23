# Data Availability Statement

## 1. Public Upstream Datasets (OpenSWI)

All synthetic and field surface-wave dispersion datasets used in this study originate from the open-access **OpenSWI** benchmark suite (Liu et al., 2025; CC BY 4.0 license):

- **HuggingFace Repository (Primary)**: <https://huggingface.co/datasets/LiuFeng2317/OpenSWI>
- **Zenodo Archive (Archival Mirror)**: <https://zenodo.org/records/14271499>

The repository contains three subsets used in this work:

1. **OpenSWI-shallow (`0.2-10s-Base`)**: Five shallow geological structure families (`FlatVelA`, `CurveVelA`, `FlatFaultA`, `CurveFaultA`, `StyleA`) spanning 0.2–10 s Rayleigh phase velocity and 0–0.70 km depth (70 layers, 0.01 km spacing).
2. **OpenSWI-deep (`1s-100s-Base`)**: Ten regional lithospheric velocity models (`Alaska`, `CSEM_Australasia`, `CSEM_Eastmed`, `CSEM_Europe`, `CSEM_Japan`, `CSEM_North_Atlantic`, `CSEM_South_Atlantic`, `CSEM_lberia`, `Central_and_Western_US_Shen2013`, `Continental_China_Shen2016`, `EUCrust`, `LITHO1`, `US-upper-mantle`, `USTCLitho1`) spanning 1–100 s Rayleigh phase and group velocity and 0–300 km depth (301 layers, 1.0 km spacing).
3. **OpenSWI-real (Field Benchmarks)**:
   - **LongBeach (Southern California, shallow urban array)**: 5,297 stations with observed Rayleigh phase-velocity dispersion and the published 3D reference model of Lin et al. (2013). After spatial-block partitioning (`data/splits/longbeach_blocks.json`) and filtering for stations with at least 10 valid periods, the **LongBeach (Tuning 30%)** split contains 1,589 stations and the **LongBeach (Report 70%)** split contains 3,705 stations.
   - **CSRM (Continental China, deep lithospheric array)**: 12,705 geographic grid points in the distributed OpenSWI archive (compared to 12,901 reported in the original CSRM publication; all evaluations use the 12,705 archived points) with Rayleigh phase and group dispersion and the published CSRM reference model. After spatial-block partitioning (`data/splits/csrm_blocks.json`) and filtering, the **CSRM (Validation)** split contains 3,863 grid points and the **CSRM (Report)** split contains 8,842 grid points.

## 2. Deterministic Split Manifests and Checksums Included in This Repository

To ensure exact split reproducibility without bundling multi-gigabyte raw binary arrays in Git, this repository includes:

- `data/checksums_sha256.txt`: SHA-256 digests for all 66 upstream OpenSWI files.
- `data/splits/longbeach_blocks.json`: Deterministic 30% / 70% spatial-block split of LongBeach stations.
- `data/splits/csrm_blocks.json`: Deterministic 30% / 70% spatial-block split of CSRM grid points.
- `data/splits/heldout_periods.json`: Deterministic per-station 20% interior held-out period masks (endpoints preserved) used strictly for post-adaptation evaluation.

Download and verify upstream files via:

```bash
python scripts/download_data.py --subset real
python scripts/download_data.py --subset all --verify-only
```

## 3. Compact Reproduction Archive (`results_summary/`)

All manuscript figures (Graphical Abstract, Figures 1–9, Figures S1–S5) and tables (Tables 1–4, Tables S1–S2) can be reproduced purely on CPU in under one minute without downloading the raw OpenSWI archives or pretrained GPU checkpoints by reading the self-contained summary files in `results_summary/` (5.00 MB total across 9 files, verified by `results_summary/manifest_sha256.json`).
