#!/usr/bin/env python3
"""Run per-regime exact disba step-size convergence audit.

Evaluates a sequence of candidate phase-velocity search steps ``dc`` (km/s)
on shallow or deep velocity-model ensembles to determine the coarsest step
meeting strict convergence tolerances (< 1.0% stations shifting > 0.1 pp and
<= 0.1% stations crossing the 20% tail-failure threshold).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swi_tta.evaluation.metrics import disba_misfit_ho
from swi_tta.physics.disba_referee import forward_dispersion_diag


def audit_ensemble(
    vs_profiles: np.ndarray,
    depth_km: np.ndarray,
    periods_list: List[np.ndarray],
    phase_obs_list: List[np.ndarray],
    ho_mask_list: List[np.ndarray],
    *,
    regime: str,
    dc_grid: List[float],
    compute_group: bool = False,
    group_obs_list: List[np.ndarray] | None = None,
) -> Dict[str, Any]:
    """Run step-size halving audit across a list of candidate dc values."""
    n = len(vs_profiles)
    misfits_by_dc: Dict[str, np.ndarray] = {}
    timings_ms: Dict[str, Dict[str, float]] = {}

    for dc in dc_grid:
        key = f"{dc:g}"
        vals = np.full(n, np.nan, dtype=np.float64)
        dts = np.zeros(n, dtype=np.float64)
        for i in range(n):
            t0 = time.perf_counter()
            per_i = periods_list[i]
            ph_i = phase_obs_list[i]
            res = forward_dispersion_diag(
                vs_profiles[i],
                depth_km,
                per_i,
                mode=0,
                compute_group=compute_group,
                dc=dc,
            )
            dts[i] = (time.perf_counter() - t0) * 1000.0
            g_obs_i = group_obs_list[i] if group_obs_list is not None else np.zeros_like(ph_i)
            d_pred = np.stack([res["phase_vel"], res["group_vel"]], axis=-1)
            d_obs = np.stack([ph_i, g_obs_i], axis=-1)
            obs_m = (d_obs > 0.0).astype(np.float64)
            vals[i] = disba_misfit_ho(d_pred, d_obs, obs_m, ho_mask_list[i])
        misfits_by_dc[key] = vals
        timings_ms[key] = {
            "median_ms": float(np.median(dts)),
            "mean_ms": float(np.mean(dts)),
            "p95_ms": float(np.percentile(dts, 95)),
        }

    comparisons: Dict[str, Dict[str, float | bool]] = {}
    for dc_hi, dc_lo in zip(dc_grid[:-1], dc_grid[1:]):
        a = misfits_by_dc[f"{dc_hi:g}"]
        b = misfits_by_dc[f"{dc_lo:g}"]
        ok = np.isfinite(a) & np.isfinite(b)
        diff_pp = np.abs(a[ok] - b[ok]) * 100.0
        cross = ((a[ok] > 0.20) != (b[ok] > 0.20)).mean() * 100.0 if np.any(ok) else 0.0
        gt_01 = (diff_pp > 0.1).mean() * 100.0 if np.any(ok) else 0.0
        comparisons[f"{dc_hi:g}_vs_{dc_lo:g}"] = {
            "n_finite": int(np.sum(ok)),
            "pct_changed_gt_0p1pp": float(gt_01),
            "pct_cross_20pct": float(cross),
            "max_abs_diff": float(np.max(np.abs(a[ok] - b[ok]))) if np.any(ok) else 0.0,
            "converged": bool(gt_01 < 1.0 and cross <= 0.1),
        }

    return {"regime": regime, "n_profiles": n, "timings_ms": timings_ms, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run disba root-finding convergence audit.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "forward_referee.yaml")
    parser.add_argument("--regime", choices=("shallow", "deep"), default="shallow")
    parser.add_argument("--input-npz", type=Path, default=None, help="Optional NPZ with vs_profiles, depth_km, periods_s, phase_obs, ho_mask.")
    parser.add_argument("--demo-lvz", action="store_true", help="Run fast CPU convergence demo on the 6 archived LVZ examples.")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ref_cfg = cfg.get("physics_referee", {})
    rs_cfg = ref_cfg.get("root_search_step_km_s", {})
    if args.regime == "shallow":
        dc_grid = [
            float(rs_cfg.get("default", 0.005)),
            0.001,
            0.0002,
            0.00005,
            float(rs_cfg.get("shallow_converged", 0.000025)),
            0.0000125,
        ]
    else:
        dc_grid = [float(rs_cfg.get("deep_converged", 0.005)), 0.001, 0.0002]

    if args.demo_lvz or args.input_npz is None:
        rs = json.loads((REPO_ROOT / "results_summary" / "root_skipping_diagnostics.json").read_text(encoding="utf-8"))
        depth_km = np.asarray(rs["typical_lvz_depth_km"], dtype=np.float64)
        exs = rs["typical_lvz_examples"]
        vs_profiles = np.asarray([e["profiles"]["vs_refined"] for e in exs], dtype=np.float64)
        periods_list = [np.asarray(e["periods_s"], dtype=np.float64) for e in exs]
        phase_list = [np.asarray(e["phase_obs_km_s"], dtype=np.float64) for e in exs]
        ho_list = []
        for e in exs:
            hm = np.zeros((len(e["periods_s"]), 2), dtype=np.float64)
            hm[np.asarray(e["is_heldout"], dtype=bool), 0] = 1.0
            ho_list.append(hm)
        report = audit_ensemble(vs_profiles, depth_km, periods_list, phase_list, ho_list, regime="shallow", dc_grid=dc_grid[:4])
    else:
        with np.load(args.input_npz) as z:
            n = len(z["vs_profiles"])
            per_arr = z["periods_s"]
            periods_list = [per_arr[i] if per_arr.ndim == 2 else per_arr for i in range(n)]
            phase_list = [z["phase_obs"][i] for i in range(n)]
            ho_list = [z["ho_mask"][i] for i in range(n)]
            grp_list = [z["group_obs"][i] for i in range(n)] if "group_obs" in z else None
            report = audit_ensemble(
                z["vs_profiles"],
                z["depth_km"],
                periods_list,
                phase_list,
                ho_list,
                regime=args.regime,
                dc_grid=dc_grid,
                compute_group=(args.regime == "deep"),
                group_obs_list=grp_list,
            )

    text = json.dumps(report, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
        print(f"Saved convergence audit to {args.out_json}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
