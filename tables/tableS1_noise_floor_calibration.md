### Table S1. Domain-Level Calibrated Noise Floor (σ̂_D) and Ground-Truth Comparison

| Domain | Regime | N Cal. | Eval dc (km/s) | Calibrated σ̂_D | GT Noise Floor (Eval) | Ratio σ̂_D / GT |
| --- | --- | --- | --- | --- | --- | --- |
| LongBeach (Tuning 30%) | Shallow | 300 | 2.5e-05 | 0.00612 | -- | -- |
| CSRM (Validation) | Deep | 300 | 5.0e-03 | 0.01356 | -- | -- |
| LongBeach-Matched (Val) | Shallow | 100 | 2.5e-05 | 0.00924 | 0.00963 | 0.96× |
| Shallow Benchmark (Val) | Shallow | 100 | 2.5e-05 | 0.00932 | 0.00983 | 0.95× |
| Deep Benchmark (Val) | Deep | 100 | 5.0e-03 | 0.02584 | 0.01347 | 1.92× |
| Shallow Multi-Structure | Shallow | 100 | 2.5e-05 | 0.00881 | 0.00967 | 0.91× |
| Shallow Prior-Shift | Shallow | 100 | 2.5e-05 | 0.00992 | 0.00977 | 1.02× |
| Deep Lithospheric | Deep | 100 | 5.0e-03 | 0.02417 | 0.01329 | 1.82× |
| Noise Scan 0.5× | Shallow | 100 | 2.5e-05 | 0.00456 | 0.00491 | 0.93× |
| Noise Scan 1.0× | Shallow | 100 | 2.5e-05 | 0.00927 | 0.00972 | 0.95× |
| Noise Scan 2.0× | Shallow | 100 | 2.5e-05 | 0.01845 | 0.01935 | 0.95× |
| Noise Scan 4.0× | Shallow | 100 | 2.5e-05 | 0.03759 | 0.03969 | 0.95× |

*Calibrated L2 trust-region bounds Δ (km/s): OpenSWI shallow = 0.2445, OpenSWI deep = 0.0821, DispFormer shallow = 0.1713, DispFormer deep = 0.0784. Deep Lithospheric uses its dedicated calibration split (σ̂_D = 0.02417, ratio = 1.82× vs GT 0.01329).*
