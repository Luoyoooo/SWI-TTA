#!/usr/bin/env python3
"""Run or audit the four-step operational surface-wave TTA protocol.

Supports two modes:
1. ``--summary-check`` (CPU, <1 s): inspects the archived closed-loop protocol
   summary in ``results_summary/operational_protocol_audit.json``, verifying
   that frozen-layer changes are strictly zero and reporting routed fractions,
   revert rates, and exact ``disba`` call savings.
2. Live GPU/CPU execution on a domain NPZ + backbone/surrogate checkpoints:
   executes ``swi_tta.adaptation.safe_tta.run_operational_protocol`` end-to-end
   and optionally writes minimal per-station audit records (initial/final kept
   misfit, selected step, max frozen delta, route action, revert reason) to NPZ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swi_tta.adaptation.baselines import run_zeroshot_inference
from swi_tta.adaptation.safe_tta import DEFAULT_ROUTING_BOUNDS, run_operational_protocol
from swi_tta.evaluation.metrics import vs_rmse_res
from swi_tta.models.backbones import load_openswi_checkpoint
from swi_tta.models.surrogate import load_surrogate_ensemble


def run_domain_operational_protocol(
    *,
    backbone_ckpt: Path,
    surrogate_ckpt_paths: List[Path],
    domain_npz: Path,
    regime: str,
    sigma_hat_d: float,
    r0_low: Optional[float] = None,
    r0_high: Optional[float] = None,
    enable_zres_freeze: bool = False,
    freeze_depth_factor: float = 1.0,
    enable_holdout_es: bool = True,
    enable_exact_verification: bool = True,
    device: str = "cuda",
    n_steps: int = 50,
    lr: float = 0.02,
    lambda_prior: float = 0.05,
    disba_workers: int = 4,
    out_per_station_npz: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the four-step operational protocol on a single domain NPZ."""
    dev_str = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model, _ = load_openswi_checkpoint(str(backbone_ckpt), domain=regime, device=dev_str)
    ens = load_surrogate_ensemble([str(p) for p in surrogate_ckpt_paths], device=dev_str)

    with np.load(domain_npz) as z:
        periods = z["period"]
        phase = z["phase_vel"]
        group = z["group_vel"]
        obs_mask = z["obs_mask"] if "obs_mask" in z else z["kept_mask"]
        kept_mask = z["kept_mask"]
        vs_true = z["vs_true"] if "vs_true" in z else None
        depth = z["depth"]
        z_res = z["z_res"] if "z_res" in z else None
        grid_t = z["grid_periods"]

    remove_group = regime == "shallow"
    slice_dim = int(depth.shape[0])
    vs_zs, _ = run_zeroshot_inference(
        model,
        periods,
        phase,
        group,
        obs_mask,
        kept_mask,
        slice_dim=slice_dim,
        input_mode="aligned",
        train_grid=grid_t,
        device=dev_str,
        remove_group=remove_group,
    )

    def_lo, def_hi = DEFAULT_ROUTING_BOUNDS[regime]
    vs_out, info = run_operational_protocol(
        vs_zs,
        ens,
        depth,
        periods,
        phase,
        group,
        obs_mask,
        kept_mask,
        domain=regime,
        sigma_hat_d=float(sigma_hat_d),
        r0_low=def_lo if r0_low is None else float(r0_low),
        r0_high=def_hi if r0_high is None else float(r0_high),
        enable_zres_freeze=enable_zres_freeze,
        freeze_depth_factor=freeze_depth_factor,
        enable_holdout_es=enable_holdout_es,
        enable_exact_verification=enable_exact_verification,
        dc_shallow=2.5e-5,
        dc_deep=5.0e-3,
        num_steps=n_steps,
        lr=lr,
        lambda_prior=lambda_prior,
        device=dev_str,
        remove_group=remove_group,
        disba_workers=disba_workers,
    )

    if out_per_station_npz is not None:
        out_per_station_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_per_station_npz,
            m_init_kept=info["m_init_kept"],
            m_final_kept=info["m_final_kept"],
            r0=info["r0"],
            route_action=info["route_action"],
            selected_step=info["selected_step"],
            z_res_kept=info["z_res_kept"],
            max_frozen_delta=info["max_frozen_delta"],
            reverted=info["reverted"],
            revert_reason=info["revert_reason"],
        )

    res: Dict[str, Any] = {
        "regime": regime,
        "sigma_hat_d": float(sigma_hat_d),
        "r0_low": info["r0_low"],
        "r0_high": info["r0_high"],
        "switches": info["switches"],
        "n_stations": info["n_stations"],
        "n_routed": info["n_routed"],
        "routed_fraction": info["routed_fraction"],
        "route_counts": info["route_counts"],
        "n_reverted": info["n_reverted"],
        "revert_rate_all": info["revert_rate_all"],
        "revert_rate_routed": info["revert_rate_routed"],
        "revert_reason_counts": info["revert_reason_counts"],
        "max_frozen_delta_overall": info["max_frozen_delta_overall"],
        "n_disba_calls": info["n_disba_calls"],
        "unrouted_holdout_es_disba_calls": info["unrouted_holdout_es_disba_calls"],
        "disba_call_savings_vs_unrouted_hes_pct": info["disba_call_savings_vs_unrouted_hes_pct"],
    }
    if vs_true is not None and z_res is not None:
        rmse_zs = float(vs_rmse_res(vs_zs, vs_true, depth, z_res)["overall"])
        rmse_out = float(vs_rmse_res(vs_out, vs_true, depth, z_res)["overall"])
        res["zero_shot_aligned_rmse_mean"] = rmse_zs
        res["protocol_rmse_mean"] = rmse_out
        res["protocol_rel_change_pct"] = float((rmse_out - rmse_zs) / rmse_zs * 100.0)
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or verify the four-step operational surface-wave TTA protocol.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "methods.yaml")
    parser.add_argument("--backbone-ckpt", type=Path, default=None)
    parser.add_argument("--surrogate-ckpts", type=Path, nargs="+", default=None)
    parser.add_argument("--domain-npz", type=Path, default=None)
    parser.add_argument("--regime", choices=("shallow", "deep"), default="shallow")
    parser.add_argument("--sigma-hat-d", type=float, default=None)
    parser.add_argument("--r0-low", type=float, default=None)
    parser.add_argument("--r0-high", type=float, default=None)
    parser.add_argument("--enable-zres-freeze", action="store_true", help="Optionally freeze layers deeper than freeze_depth_factor * z_res.")
    parser.add_argument("--freeze-depth-factor", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-per-station-npz", type=Path, default=None)
    parser.add_argument("--summary-check", action="store_true", help="Verify archived operational protocol summary on CPU.")
    args = parser.parse_args()

    if args.summary_check or args.backbone_ckpt is None or not args.surrogate_ckpts or args.domain_npz is None:
        audit_path = REPO_ROOT / "results_summary" / "operational_protocol_audit.json"
        if not audit_path.exists():
            print(f"[Protocol] Summary archive not found at {audit_path}", file=sys.stderr)
            return 1
        doc = json.loads(audit_path.read_text(encoding="utf-8"))
        summary = {
            "schema_version": doc.get("schema_version"),
            "default_switches": doc.get("protocol_definition", {}).get("default_switches"),
            "invariants": doc.get("invariants"),
            "n_cohorts": len(doc["cohorts"]),
            "cohorts": {
                c["cohort_id"]: {
                    "label": c["display_label"],
                    "routed_pct": round(c["routed_pct"], 2),
                    "reverted_pct_of_all": round(c["reverted_pct_of_all"], 2),
                    "protocol_delta_rmse_pct": (
                        round(c["protocol_delta_rmse_pct"], 2) if c.get("protocol_delta_rmse_pct") is not None else None
                    ),
                    "disba_call_savings_pct": round(c["disba_call_savings_vs_unrouted_hes_pct"], 2),
                }
                for c in doc["cohorts"]
            },
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.sigma_hat_d is None:
        raise SystemExit("--sigma-hat-d is required for live execution.")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    m_cfg = cfg.get("methods", {}).get("output_refined_aligned", {})
    out = run_domain_operational_protocol(
        backbone_ckpt=args.backbone_ckpt,
        surrogate_ckpt_paths=list(args.surrogate_ckpts),
        domain_npz=args.domain_npz,
        regime=args.regime,
        sigma_hat_d=float(args.sigma_hat_d),
        r0_low=args.r0_low,
        r0_high=args.r0_high,
        enable_zres_freeze=bool(args.enable_zres_freeze),
        freeze_depth_factor=float(args.freeze_depth_factor),
        device=args.device,
        n_steps=int(m_cfg.get("num_steps", 50)),
        lr=float(m_cfg.get("lr", 0.02)),
        lambda_prior=float(m_cfg.get("lambda_prior", 0.05)),
        disba_workers=int(args.workers),
        out_per_station_npz=args.out_per_station_npz,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
