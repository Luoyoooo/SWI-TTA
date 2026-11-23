# Computer Code Availability (Computers & Geosciences Standard Block)

Copy the block below directly into the *Computer Code Availability* section of the *Computers & Geosciences* manuscript.

---

**Computer Code Availability**

- **Name of the code/library**: `swi-tta-diagnostics` (Python package: `swi_tta`, v1.0.0)
- **Contact**: Corresponding author (`corresponding.author@institution.edu`; GitHub issue tracker at the repository URL below)
- **Hardware requirements**: Standard multi-core CPU with 4 GB RAM is sufficient for installation verification, the live `disba` low-velocity-zone root-skipping demonstration, unit tests, and complete reproduction of all manuscript figures and tables (< 30 s total wall time). An NVIDIA GPU (>= 8 GB VRAM) is required only if re-running full neural-network test-time adaptation over the multi-gigabyte upstream OpenSWI archives.
- **Program language**: Python (3.10–3.11)
- **Software required**: `numpy`, `scipy`, `torch`, `disba` (>= 0.7.0), `matplotlib`, `pyyaml`, `pillow`, `pytest` (pinned dependencies provided in `environment.yml`, `requirements.txt`, and `requirements-lock.txt`).
- **Program size**: 0.41 MB for source code, scripts, configuration files, documentation, and test suite (`src/`, `scripts/`, `configs/`, `docs/`, `tests/`, and root metadata); 5.00 MB for the self-contained summary archive (`results_summary/`, 9 files verified by SHA-256); 2.86 MB for deterministic spatial/period split manifests and upstream checksums (`data/`); 25.3 MB for pre-rendered publication figures (`figures/`, 15 figures in PDF, 600-DPI PNG, and SVG) and tables (`tables/`, 38 KB); 33.6 MB total repository working tree.
- **The source codes are available for downloading at the link**: <https://github.com/cageo/swi-tta-diagnostics> *(archival Zenodo DOI to be minted upon manuscript acceptance)*
