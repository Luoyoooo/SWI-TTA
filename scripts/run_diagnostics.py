#!/usr/bin/env python3
"""Run noise-floor-normalized signature diagnostics on trained backbones.

Supports both full GPU evaluation on OpenSWI / DispFormer checkpoints and a
``--summary-check`` mode that validates the archived diagnostic summary without
requiring raw datasets or GPU weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swi_tta.adaptation.baselines import run_output_space_refinement, run_zeroshot_inference
from swi_tta.adaptation.safe_tta import run_holdout_early_stopping
from swi_tta.evaluation.metrics import vs_rmse_res
from swi_tta.models.backbones import load_openswi_checkpoint
from swi_tta.models.surrogate import load_surrogate_ensemble


def run_single_domain_diagnostics(
    *,
    backbone_ckpt: Path,
    surrogate_ckpt_paths: List[Path],
    domain_npz: Path,
    regime: str,
    device: str = "cuda",
    n_steps: int = 50,
    lr: float = 0.02,
    lambda_prior: float = 0.05,
) -> Dict[str, Any]:
    """Run Input-Aligned Zero-Shot, Output-Space Refinement, and Holdout ES on a domain NPZ."""
    dev_str = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model, _ = load_openswi_checkpoint(str(backbone_ckpt), domain=regime, device=dev_str)
    ens = load_surrogate_ensemble([str(p) for p in surrogate_ckpt_paths], device=dev_str)

    with np.load(domain_npz) as z:
        periods = z["period"]
        phase = z["phase_vel"]
        group = z["group_vel"]
        obs_mask = z["obs_mask"] if "obs_mask" in z else z["kept_mask"]
        kept_mask = z["kept_mask"]
        vs_true = z["vs_true"]
        depth = z["depth"]
        z_res = z["z_res"]
        grid_t = z["grid_periods"]

    remove_group = regime == "shallow"
    vs_zs, _ = run_zeroshot_inference(
        model,
        periods,
        phase,
        group,
        obs_mask,
        kept_mask,
        slice_dim=vs_true.shape[1],
        input_mode="aligned",
        train_grid=grid_t,
        device=dev_str,
        remove_group=remove_group,
    )
    vs_ref, _ = run_output_space_refinement(
        model,
        ens,
        periods,
        phase,
        group,
        obs_mask,
        kept_mask,
        domain=regime,
        slice_dim=vs_true.shape[1],
        input_mode="aligned",
        train_grid=grid_t,
        num_steps=n_steps,
        lr=lr,
        lambda_prior=lambda_prior,
        device=dev_str,
        remove_group=remove_group,
    )
    vs_hes, _ = run_holdout_early_stopping(
        vs_zs,
        ens,
        depth,
        periods,
        phase,
        group,
        obs_mask,
        kept_mask,
        domain=regime,
        num_steps=n_steps,
        lr=lr,
        lambda_prior=lambda_prior,
        device=dev_str,
        remove_group=remove_group,
    )

    rmse_zs = vs_rmse_res(vs_zs, vs_true, depth, z_res)["overall"]
    rmse_ref = vs_rmse_res(vs_ref, vs_true, depth, z_res)["overall"]
    rmse_hes = vs_rmse_res(vs_hes, vs_true, depth, z_res)["overall"]

    return {
        "regime": regime,
        "n_samples": int(len(vs_true)),
        "zero_shot_aligned_rmse_mean": float(rmse_zs),
        "output_refined_50_rmse_mean": float(rmse_ref),
        "output_refined_50_rel_change_pct": float((rmse_ref - rmse_zs) / rmse_zs * 100.0),
        "holdout_es_rmse_mean": float(rmse_hes),
        "holdout_es_rel_change_pct": float((rmse_hes - rmse_zs) / rmse_zs * 100.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cross-backbone noise-floor signature diagnostics.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "methods.yaml")
    parser.add_argument("--backbone-ckpt", type=Path, default=None)
    parser.add_argument("--surrogate-ckpts", type=Path, nargs="+", default=None)
    parser.add_argument("--domain-npz", type=Path, default=None)
    parser.add_argument("--regime", choices=("shallow", "deep"), default="shallow")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-check", action="store_true", help="Print key metrics from archived results_summary.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.summary_check or args.backbone_ckpt is None or not args.surrogate_ckpts or args.domain_npz is None:
        cb = json.loads((REPO_ROOT / "results_summary" / "cross_backbone_diagnostics.json").read_text(encoding="utf-8"))
        summary: Dict[str, Any] = {}
        for dk, d_blk in cb["domains"].items():
            summary[dk] = {
                bb: {
                    "all_ckpts_50step_rel_pct": d_blk["by_backbone"][bb]["all_judged_ckpts"]["output_refined_50"]["mean_rel_change_vs_aligned_pct"],
                    "all_ckpts_ideal_es_rel_pct": d_blk["by_backbone"][bb]["all_judged_ckpts"]["ideal_es"]["mean_rel_change_vs_aligned_pct"],
                }
                for bb in ("openswi", "dispformer")
            }
        print(json.dumps(summary, indent=2))
        return 0

    m_cfg = cfg.get("methods", {}).get("output_refined_aligned", {})
    out = run_single_domain_diagnostics(
        backbone_ckpt=args.backbone_ckpt,
        surrogate_ckpt_paths=list(args.surrogate_ckpts),
        domain_npz=args.domain_npz,
        regime=args.regime,
        device=args.device,
        n_steps=int(m_cfg.get("num_steps", 50)),
        lr=float(m_cfg.get("lr", 0.02)),
        lambda_prior=float(m_cfg.get("lambda_prior", 0.05)),
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
