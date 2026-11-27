# SWI-TTA Tutorials

Three self-contained, **pure-CPU** tutorials illustrating typical use cases of **SWI-TTA** (no GPU or multi-gigabyte dataset download required; total runtime `< 30 s`):

| Tutorial | Topic | Runtime (CPU) |
| :--- | :--- | :---: |
| [`01_lvz_root_skipping_audit.ipynb`](01_lvz_root_skipping_audit.ipynb) | Live reproduction of `disba` Mode-0 to Mode-1 root skipping on shallow LVZ profile `#9391` (`63.84% -> 0.65%`) and step-size convergence sweeping across all 6 archived LVZ profiles. | `< 3 s` |
| [`02_zero_leakage_and_protocol.ipynb`](02_zero_leakage_and_protocol.ipynb) | Zero-leakage held-out period masking, Resolvable-Depth $V_s$ RMSE, live CPU execution of the four-step operational protocol (`run_operational_protocol`) with exact `disba` verification/rollback, and inspection of the 14-cohort audit archive. | `< 5 s` |
| [`03_reproduce_paper_results.ipynb`](03_reproduce_paper_results.ipynb) | Verifying SHA-256 summary digests, inspecting calibrated noise floors (`Table S1`), and regenerating all 7 manuscript tables and 15 publication figures deterministically on CPU. | `< 25 s` |

## Running the Tutorials

Install the package in editable mode from the repository root, then launch Jupyter (or execute cells directly in VS Code):

```bash
pip install -e ".[dev]"
pip install jupyterlab  # optional, if not already installed
jupyter lab tutorials/
```
