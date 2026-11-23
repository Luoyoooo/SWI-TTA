#!/usr/bin/env python3
"""Run Pre-Registered Safe-TTA validation and attribution audit.

Supports full evaluation with trained backbone/surrogate checkpoints and a
``--summary-check`` mode that inspects the archived calibration and validation
truth-gate audit from ``results_summary/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swi_tta.adaptation.baselines import run_zeroshot_inference
from swi_tta.adaptation.safe_tta import run_safe_tta
from swi_tta.models.backbones import load_openswi_checkpoint
from swi_tta.models.surrogate import load_surrogate_ensemble


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pre-Registered Safe-TTA mechanism and validation audit.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "calibration.yaml")
    parser.add_argument("--backbone-ckpt", type=Path, default=None)
    parser.add_argument("--surrogate-ckpts", type=Path, nargs="+", default=None)
    parser.add_argument("--domain-npz", type=Path, default=None)
    parser.add_argument("--regime", choices=("shallow", "deep"), default="shallow")
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--sigma-hat-d", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-check", action="store_true", help="Inspect archived Safe-TTA validation truth gate.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if args.summary_check or args.backbone_ckpt is None or not args.surrogate_ckpts or args.domain_npz is None:
        st = json.loads((REPO_ROOT / "results_summary" / "safe_tta_validation_audit.json").read_text(encoding="utf-8"))
        cal = json.loads((REPO_ROOT / "results_summary" / "calibration_and_domains.json").read_text(encoding="utf-8"))
        out: Dict[str, Any] = {
            "alphas": st["alphas"],
            "passing_alphas": st["alpha_selection"]["passing_alphas"],
            "validation_checkpoint": st["alpha_selection"]["validation_checkpoint"],
            "deep_lithospheric_calibration": cal["calibrated_noise_floors"]["deep_lithospheric"],
        }
        print(json.dumps(out, indent=2))
        return 0

    dev_str = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model, _ = load_openswi_checkpoint(str(args.backbone_ckpt), domain=args.regime, device=dev_str)
    ens = load_surrogate_ensemble([str(p) for p in args.surrogate_ckpts], device=dev_str)

    with np.load(args.domain_npz) as z:
        periods = z["period"]
        phase = z["phase_vel"]
        group = z["group_vel"]
        obs_mask = z["obs_mask"] if "obs_mask" in z else z["kept_mask"]
        kept_mask = z["kept_mask"]
        depth = z["depth"]
        grid_t = z["grid_periods"]

    nfc = cfg["noise_floor_calibration"]
    tr = cfg["trust_region_delta_p95"]
    dc_exact = float(
        nfc["classical_inversion_budget"]["eval_dc_shallow_km_s" if args.regime == "shallow" else "eval_dc_deep_km_s"]
    )
    sigma_d = float(args.sigma_hat_d) if args.sigma_hat_d is not None else float(
        nfc["sigma_hat_d"]["shallow_benchmark_val" if args.regime == "shallow" else "deep_benchmark_val"]
    )
    delta_tr = float(tr["openswi_shallow" if args.regime == "shallow" else "openswi_deep"])

    vs_init, _ = run_zeroshot_inference(
        model,
        periods,
        phase,
        group,
        obs_mask,
        kept_mask,
        slice_dim=len(depth),
        input_mode="aligned",
        train_grid=grid_t,
        device=dev_str,
        remove_group=(args.regime == "shallow"),
    )
    vs_out, diag = run_safe_tta(
        vs_init,
        ens,
        depth,
        periods,
        phase,
        group,
        obs_mask,
        kept_mask,
        domain=args.regime,
        alpha=args.alpha,
        sigma_hat_d=sigma_d,
        delta_trust=delta_tr,
        dc=dc_exact,
        device=dev_str,
    )
    print(
        json.dumps(
            {
                "regime": args.regime,
                "alpha": args.alpha,
                "sigma_hat_d": sigma_d,
                "n_stations": int(len(vs_out)),
                "stop_rate": float(diag["stop_rate"]),
                "revert_rate": float(diag["revert_rate"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
