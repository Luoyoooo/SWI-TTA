### Table 3. Held-Out Cross-Backbone Diagnostics, Subgroup Breakdown, and Noise-Amplitude Scan

| Domain / Experiment | Backbone / Subset | Aligned RMSE | 50-step ΔRMSE (95% CI) | Holdout ES ΔRMSE | Ideal ES ΔRMSE | 50-step HO Med (%) | σ̂_D / GT Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shallow Multi-Structure | OpenSWI (All ckpts) | 0.0360 | -4.48% [-5.49, -3.41] | -12.15% | -36.47% | 1.04% | -- |
| Shallow Multi-Structure | OpenSWI (Retrained-only) | 0.0320 | +4.73% [+3.56, +5.96] | -2.54% | -28.64% | 1.05% | -- |
| Shallow Multi-Structure | DispFormer (All ckpts) | 0.0354 | -13.87% [-14.57, -13.16] | -15.66% | -30.81% | 0.78% | -- |
| Shallow Prior-Shift | OpenSWI (All ckpts) | 0.0812 | -39.07% [-39.72, -38.43] | -39.34% | -49.18% | 1.24% | -- |
| Shallow Prior-Shift | OpenSWI (Retrained-only) | 0.0742 | -35.74% [-36.43, -35.05] | -34.66% | -44.24% | 1.08% | -- |
| Shallow Prior-Shift | DispFormer (All ckpts) | 0.0485 | -19.81% [-20.61, -19.00] | -18.81% | -33.46% | 0.77% | -- |
| Deep Lithospheric | OpenSWI (All ckpts) | 0.1477 | +10.59% [+9.42, +11.76] | -1.50% | -13.66% | 1.62% | -- |
| Deep Lithospheric | OpenSWI (Retrained-only) | 0.1472 | +10.80% [+9.55, +12.04] | -1.98% | -14.37% | 1.65% | -- |
| Deep Lithospheric | DispFormer (All ckpts) | 0.1680 | -1.00% [-1.76, -0.22] | -7.42% | -16.45% | 1.61% | -- |
| Deep Lithospheric | DispFormer (Retrained-only) | 0.1480 | -0.34% [-1.22, +0.58] | -9.01% | -18.06% | 1.64% | -- |
| Prior-Shift: StyleA (81.54%) | OpenSWI | 0.0886 | -46.98% | -45.99% | -53.27% | 1.36% | -- |
| Prior-Shift: StyleA (81.54%) | DispFormer | 0.0442 | -24.36% | -22.58% | -37.72% | 0.76% | -- |
| Prior-Shift: non-StyleA (18.46%) | OpenSWI | 0.0482 | +24.21% | +13.64% | -16.93% | 0.90% | -- |
| Prior-Shift: non-StyleA (18.46%) | DispFormer | 0.0676 | -6.63% | -7.92% | -21.13% | 0.83% | -- |
| Noise Scan 0.5× (n=4000) | OpenSWI (All ckpts) | 0.0366 | -9.56% [-11.97, -7.16] | -15.04% | -37.62% | 0.77% | 0.929× |
| Noise Scan 1× (n=4000) | OpenSWI (All ckpts) | 0.0362 | -3.37% [-5.90, -0.93] | -11.24% | -36.08% | 1.06% | 0.953× |
| Noise Scan 2× (n=4000) | OpenSWI (All ckpts) | 0.0372 | +12.38% [+9.58, +15.28] | -2.76% | -32.51% | 1.50% | 0.953× |
| Noise Scan 4× (n=4000) | OpenSWI (All ckpts) | 0.0411 | +46.93% [+42.92, +51.24] | +12.51% | -26.57% | 2.57% | 0.947× |

*95% CIs are paired cluster-bootstrap intervals resampled by unique subsurface profile ID. Holdout ES is evaluated across OpenSWI, DispFormer, and controlled noise-amplitude scans using identical frozen hyperparameters.*
