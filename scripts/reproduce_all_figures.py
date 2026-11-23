#!/usr/bin/env python3
"""Reproduce Graphical Abstract, Figures 1-9, and Figures S1-S5 purely on CPU.

Reads solely from ``results_summary/`` and ``configs/figures.yaml`` and writes
``figures/manifest_sha256.json`` with deterministic SHA-256 digests and sizes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swi_tta.visualization.main_figures import generate_all_main_figures
from swi_tta.visualization.supp_figures import generate_all_supp_figures


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce all publication figures purely on CPU.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "figures.yaml")
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    js = cfg.get("journal_standard", cfg)
    summary_dir = str(args.summary_dir or (REPO_ROOT / "results_summary"))
    out_dir = str(args.out_dir or (REPO_ROOT / "figures"))
    formats: List[str] = list(js.get("formats", ["pdf", "png", "svg"]))
    dpi = int(args.dpi or js.get("png_dpi", 600))

    t0 = time.perf_counter()
    print(f"[Figures] Generating main figures + Graphical Abstract from {summary_dir} -> {out_dir} (DPI={dpi})...")
    main_res = generate_all_main_figures(summary_dir, out_dir, formats=formats, dpi=dpi)
    print(f"[Figures] Generating supplementary figures (S1-S5)...")
    supp_res = generate_all_supp_figures(summary_dir, out_dir, formats=formats, dpi=dpi)

    all_records: Dict[str, Dict[str, Any]] = {}
    all_records.update(main_res)
    all_records.update(supp_res)

    # Convert absolute paths inside manifest to portable paths relative to out_dir
    manifest_entries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    total_bytes = 0
    for fig_name, fmt_map in sorted(all_records.items()):
        manifest_entries[fig_name] = {}
        for ext, meta in sorted(fmt_map.items()):
            rel_p = os.path.relpath(str(meta["path"]), out_dir).replace("\\", "/")
            sz = int(meta["size_bytes"])
            total_bytes += sz
            manifest_entries[fig_name][ext] = {
                "relative_path": rel_p,
                "size_bytes": sz,
                "sha256": str(meta["sha256"]),
            }

    manifest = {
        "schema_version": "1.0",
        "dpi_png": dpi,
        "formats": formats,
        "n_figures": len(manifest_entries),
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024.0 * 1024.0), 4),
        "figures": manifest_entries,
    }
    manifest_path = Path(out_dir) / "manifest_sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(
        f"[Figures] Done! Generated {len(manifest_entries)} figures "
        f"({total_bytes / (1024*1024):.2f} MB across {len(formats)} formats) in {elapsed:.2f} s."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
