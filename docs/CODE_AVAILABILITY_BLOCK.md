# Computer Code Availability (Computers & Geosciences Standard Block)

Copy the block below directly into the *Computer Code Availability* section of the *Computers & Geosciences* manuscript.

---

**Computer Code Availability**

- **Name of the code/library**: `SWI-TTA` (Python package: `swi_tta`, v1.0.0; MIT License)
- **Contact**: Corresponding author (to be finalized prior to publication; GitHub issue tracker at the repository URL below)
- **Hardware requirements**: Standard multi-core CPU with 4 GB RAM is sufficient for installation verification, the live `disba` low-velocity-zone root-skipping demonstration, unit tests, interactive tutorials, and complete reproduction of all manuscript figures and tables (`< 30 s` total wall time). An NVIDIA GPU (`>= 8 GB` VRAM) is required only if re-running full neural-network test-time adaptation over the multi-gigabyte upstream OpenSWI archives.
- **Program language**: Python (`3.10–3.11`)
- **Software required**: `numpy`, `scipy`, `torch`, `disba` (`>= 0.7.0`), `numba`, `matplotlib`, `pyyaml`, `pillow`, `requests`, `tqdm`, and `pytest` (English `README.md`, `docs/USER_GUIDE.md`, `tutorials/`, and pinned dependencies in `environment.yml`, `requirements.txt`, and `requirements-lock.txt`).
- **Program size**: `0.56 MB` for source code, scripts, configuration files, user guide, tutorials, static project website, and test suite (`src/`, `scripts/`, `configs/`, `docs/`, `tutorials/`, `website/`, `tests/`, and root metadata); `14.76 MB` for the self-contained summary and per-station protocol audit archive (`results_summary/`, 11 files verified by SHA-256); `2.86 MB` for deterministic spatial/period split manifests and upstream checksums (`data/`); `25.25 MB` for pre-rendered publication figures (`figures/`, 15 figures in PDF, 600-DPI PNG, and SVG) and tables (`tables/`, 51 KB); `43.42 MB` total repository working tree.
- **The source codes are available for downloading at the link**: <https://github.com/Luoyoooo/SWI-TTA> (Project page: <https://luoyoooo.github.io/SWI-TTA/>; archival Zenodo DOI to be minted upon release)
