# Figure and Table Reproduction Mapping

All manuscript figures and tables are reproduced purely on CPU via two single-command scripts reading strictly from the compact summary archive in `results_summary/`.

## One-Command Reproduction

```bash
# Reproduce Graphical Abstract, Figures 1-9, and Figures S1-S5 (.pdf, .png @ 600 DPI, .svg)
python scripts/reproduce_all_figures.py

# Reproduce Tables 1-4 and Supplementary Tables S1-S3 (.md and booktabs .tex)
python scripts/reproduce_tables.py
```

## Main Figures and Graphical Abstract

| Manuscript Item | Output Stem | Generator Function | Primary `results_summary/` Inputs |
| :--- | :--- | :--- | :--- |
| **Graphical Abstract** | `figures/graphical_abstract.{pdf,png,svg}` | `swi_tta.visualization.main_figures.make_graphical_abstract` | `root_skipping_diagnostics.json`, `cross_backbone_diagnostics.json` |
| **Figure 1** | `figures/main/fig01_workflow_and_protocol.{pdf,png,svg}` | `make_fig01_workflow_and_protocol` | `root_skipping_diagnostics.json` |
| **Figure 2** | `figures/main/fig02_benchmark_overview.{pdf,png,svg}` | `make_fig02_benchmark_overview` | `benchmark_overview.json`, `root_skipping_diagnostics.json` |
| **Figure 3** | `figures/main/fig03_disba_root_skipping.{pdf,png,svg}` | `make_fig03_disba_root_skipping` | `root_skipping_diagnostics.json` |
| **Figure 4** | `figures/main/fig04_openswi_signature_curves.{pdf,png,svg}` | `make_fig04_openswi_signature_curves` | `cross_backbone_diagnostics.json` |
| **Figure 5** | `figures/main/fig05_cross_backbone_signatures.{pdf,png,svg}` | `make_fig05_cross_backbone_signatures` | `cross_backbone_diagnostics.json` |
| **Figure 6** | `figures/main/fig06_noise_amplitude_scan.{pdf,png,svg}` | `make_fig06_noise_amplitude_scan` | `cross_backbone_diagnostics.json` |
| **Figure 7** | `figures/main/fig07_safe_tta_failure_attribution.{pdf,png,svg}` | `make_fig07_safe_tta_failure_attribution` | `safe_tta_validation_audit.json` |
| **Figure 8** | `figures/main/fig08_longbeach_slices.{pdf,png,svg}` | `make_fig08_longbeach_slices` | `field_slices_longbeach.npz` |
| **Figure 9** | `figures/main/fig09_csrm_slices_and_station.{pdf,png,svg}` | `make_fig09_csrm_slices_and_station` | `field_slices_csrm.npz` |

## Supplementary Figures

| Manuscript Item | Output Stem | Generator Function | Primary `results_summary/` Inputs |
| :--- | :--- | :--- | :--- |
| **Figure S1** | `figures/supplementary/figS01_iteration_curves_and_lag.{pdf,png,svg}` | `make_figS01_iteration_curves_and_lag` | `benchmark_overview.json` |
| **Figure S2** | `figures/supplementary/figS02_lvz_counterfactual_examples.{pdf,png,svg}` | `make_figS02_lvz_counterfactual_examples` | `root_skipping_diagnostics.json` |
| **Figure S3** | `figures/supplementary/figS03_old_vs_new_dc_scatter.{pdf,png,svg}` | `make_figS03_old_vs_new_dc_scatter` | `root_skipping_diagnostics.json`, `root_skipping_scatter.npz` |
| **Figure S4** | `figures/supplementary/figS04_safe_tta_alpha_sensitivity.{pdf,png,svg}` | `make_figS04_safe_tta_alpha_sensitivity` | `safe_tta_validation_audit.json` |
| **Figure S5** | `figures/supplementary/figS05_field_spatial_misfit_and_data.{pdf,png,svg}` | `make_figS05_field_spatial_misfit_and_data` | `field_slices_longbeach.npz`, `field_slices_csrm.npz`, `calibration_and_domains.json` |

## Manuscript and Supplementary Tables

| Manuscript Item | Output Stem | Generator Function | Primary `results_summary/` Inputs |
| :--- | :--- | :--- | :--- |
| **Table 1** | `tables/table1_benchmark_comparison.{md,tex}` | `make_table1_benchmark_comparison` | `benchmark_overview.json`, `root_skipping_diagnostics.json` |
| **Table 2** | `tables/table2_referee_convergence_and_tails.{md,tex}` | `make_table2_referee_convergence_and_tails` | `root_skipping_diagnostics.json`, `benchmark_overview.json` |
| **Table 3** | `tables/table3_cross_backbone_diagnostics.{md,tex}` | `make_table3_cross_backbone_diagnostics` | `cross_backbone_diagnostics.json` |
| **Table 4** | `tables/table4_safe_tta_validation_audit.{md,tex}` | `make_table4_safe_tta_validation_audit` | `safe_tta_validation_audit.json` |
| **Table S1** | `tables/tableS1_noise_floor_calibration.{md,tex}` | `make_tableS1_noise_floor_calibration` | `calibration_and_domains.json` |
| **Table S2** | `tables/tableS2_signature_bins.{md,tex}` | `make_tableS2_signature_bins` | `cross_backbone_diagnostics.json` |
| **Table S3** | `tables/tableS3_protocol_and_retrained_audit.{md,tex}` | `make_tableS3_protocol_and_retrained_audit` | `operational_protocol_audit.json`, `operational_protocol_records.npz` |
