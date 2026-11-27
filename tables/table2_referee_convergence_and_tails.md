### Table 2. Exact disba Referee Step-Size Convergence Audit and Tail-Rate Revision

| Regime / Method | Step dc (km/s) | Old Tail (>20%) | New Tail (>20%) | Δ Tail (pp) | Old HO Med (%) | New HO Med (%) | Spearman ρ (All / ≤20%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shallow Audit (vs 1.25e-5) | 2.5e-05 (20.7 ms) | Δ>0.1pp: 0.024% | Cross 20%: 0.000% | Empty: 0 | -- | -- | Converged (PASS) |
| Deep Audit (vs 2.0e-4) | 5.0e-03 (22.0 ms) | Δ>0.1pp: 0.000% | Cross 20%: 0.000% | Empty: 0 | -- | -- | Converged (PASS) |
| Shallow: Ground Truth | 2.5e-5 | 0.16% (34) | 0.00% (0) | -0.16 | 0.975 | 0.974 | -- |
| Shallow: Input-Aligned Zero-Shot | 2.5e-5 | 4.11% | 4.07% | -0.04 | 1.568 | 1.568 | 0.583 / 0.533 |
| Shallow: Output-Refined (Raw) | 2.5e-5 | 11.18% | 7.80% | -3.38 | 1.455 | 1.412 | 0.422 / 0.328 |
| Shallow: Output-Space Refinement | 2.5e-5 | 9.35% | 1.36% | -7.99 | 1.138 | 1.041 | 0.056 / 0.038 |
| Shallow: Low-DOF Adapter | 2.5e-5 | 0.13% | 0.07% | -0.06 | 1.212 | 1.212 | 0.436 / 0.435 |
| Shallow: Gated Adapter + Output-Refined | 2.5e-5 | 8.30% | 0.01% | -8.29 | 0.938 | 0.857 | 0.057 / 0.057 |
| Deep: Ground Truth | 5.0e-3 | 0.00% (0) | 0.00% (0) | +0.00 | 1.357 | 1.357 | -- |
| Deep: Input-Aligned Zero-Shot | 5.0e-3 | 2.32% | 2.32% | +0.00 | 1.681 | 1.681 | 0.356 / 0.358 |
| Deep: Output-Refined (Raw) | 5.0e-3 | 1.78% | 1.78% | +0.00 | 1.634 | 1.634 | 0.425 / 0.396 |
| Deep: Output-Space Refinement | 5.0e-3 | 1.35% | 1.35% | +0.00 | 1.668 | 1.668 | 0.415 / 0.396 |
| Deep: Low-DOF Adapter | 5.0e-3 | 2.59% | 2.59% | +0.00 | 1.564 | 1.564 | 0.318 / 0.274 |
| Deep: Gated Adapter + Output-Refined | 5.0e-3 | 0.84% | 0.84% | +0.00 | 1.588 | 1.588 | 0.374 / 0.358 |

*Old columns use disba default dc = 5e-3 km/s; New columns use converged per-regime dc (2.5e-5 km/s shallow, 5e-3 km/s deep). Ground-Truth counts show stations exceeding 20% held-out misfit out of 21,469 shallow and 10,626 deep test profiles.*
