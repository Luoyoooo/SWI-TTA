#!/usr/bin/env python3
"""Reproduce Tables 1-4 and Supplementary Tables S1-S3 in Markdown and LaTeX purely on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swi_tta.visualization.tables import generate_all_tables


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce all manuscript tables (Markdown & booktabs LaTeX) on CPU.")
    parser.add_argument("--summary-dir", type=Path, default=REPO_ROOT / "results_summary")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "tables")
    args = parser.parse_args()

    t0 = time.perf_counter()
    records = generate_all_tables(str(args.summary_dir), str(args.out_dir))

    manifest_tables: Dict[str, Dict[str, Dict[str, Any]]] = {}
    total_bytes = 0
    for tname, fmt_map in sorted(records.items()):
        manifest_tables[tname] = {}
        for ext, meta in sorted(fmt_map.items()):
            rel_p = os.path.relpath(str(meta["path"]), str(args.out_dir)).replace("\\", "/")
            sz = int(meta["size_bytes"])
            total_bytes += sz
            manifest_tables[tname][ext] = {
                "relative_path": rel_p,
                "size_bytes": sz,
                "sha256": str(meta["sha256"]),
            }

    manifest = {
        "schema_version": "1.0",
        "n_tables": len(manifest_tables),
        "total_size_bytes": total_bytes,
        "tables": manifest_tables,
    }
    (args.out_dir / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"[Tables] Done! Generated {len(manifest_tables)} tables (Markdown + LaTeX) in {elapsed:.2f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
