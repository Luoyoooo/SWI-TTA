### Table 4. Pre-Registered Safe-TTA Validation Audit, Outcome-Group Decomposition, and Deep >5σ̂_D Attribution

| Section / Setting | LB-Matched (Val) ΔRMSE | Shallow Bench. (Val) ΔRMSE | Deep Bench. (Val) ΔRMSE | LB Tuning HO (Tail) | Truth Gate (≤+2%) |
| --- | --- | --- | --- | --- | --- |
| Safe-TTA (α = 1) | -50.63% [-54.50, -46.08] | +19.60% [+17.10, +22.18] | +2.07% [+1.16, +3.02] | 1.01% (0.00%) | FAIL |
| Safe-TTA (α = 1.25) | -50.17% [-54.11, -45.54] | +12.43% [+10.22, +14.71] | +2.38% [+1.53, +3.29] | 1.01% (0.00%) | FAIL |
| Safe-TTA (α = 1.5) | -49.23% [-53.34, -44.62] | +2.78% [+0.96, +4.60] | +2.85% [+2.03, +3.72] | 1.02% (0.00%) | FAIL |
| Safe-TTA (α = 2) | -45.91% [-49.87, -41.43] | -12.26% [-13.52, -10.98] | +3.54% [+2.79, +4.34] | 1.15% (0.00%) | FAIL |
| α=1.25 Group: Step-0 Stop | +0.00% (1.4% sta) | +0.00% (19.5% sta) | +0.00% (76.6% sta) | -- | -- |
| α=1.25 Group: Mid-Stop | -10.71% (28.4% sta) | +8.31% (32.5% sta) | -2.89% (15.8% sta) | -- | -- |
| α=1.25 Group: Ran 50 Steps | -39.46% (70.2% sta) | +4.12% (38.9% sta) | +5.27% (7.3% sta) | -- | -- |
| α=1.25 Group: Reverted | +0.00% (0.0% sta) | +0.00% (9.1% sta) | +0.00% (0.3% sta) | -- | -- |
| Deep >5σ̂_D Attribution (7 ckpts) | Aligned misfit: 19.0–38.2% | GT misfit: 1.33–1.40% | Full GT: 1.35% | >10 km ΔSqErr: 80–153% | Model Failure |

*Pre-registered validation truth gate requires ΔResolvable-Depth Vs RMSE ≤ +2.0% on all three validation benchmarks simultaneously. No α passed (0/4); α = 1.5 came closest to the +2.0% threshold (+2.78% shallow, +2.85% deep), while pre-registered default α = 1.25 is decomposed in detail.*
