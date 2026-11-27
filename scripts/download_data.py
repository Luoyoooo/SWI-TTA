#!/usr/bin/env python3
"""Download official OpenSWI datasets from HuggingFace and verify SHA-256 checksums.

Only the official upstream repository (``https://huggingface.co/datasets/LiuFeng2317/OpenSWI``)
is accessed. No third-party mirrors or platform-specific helper scripts are used.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

HF_DATASET_BASE_URL = "https://huggingface.co/datasets/LiuFeng2317/OpenSWI/resolve/main"
REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksum_manifest(manifest_path: Path) -> List[Tuple[str, str]]:
    """Return list of (expected_sha256, relative_path) tuples from checksums_sha256.txt."""
    entries: List[Tuple[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                entries.append((parts[0].lower(), parts[1].strip()))
    return entries


def _hf_remote_subpath(local_rel: str) -> str:
    """Map local ``data/raw/<subpath>`` to HuggingFace dataset path."""
    prefix = "data/raw/"
    if local_rel.startswith(prefix):
        return local_rel[len(prefix) :]
    return local_rel


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "SWI-TTA/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp_path, dest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download official OpenSWI datasets from HuggingFace and verify SHA-256 digests."
    )
    parser.add_argument(
        "--subset",
        choices=("all", "real", "shallow", "deep"),
        default="real",
        help="Dataset subset to fetch or verify (default: real).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root where data/raw/ will be placed (default: repo root).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data" / "checksums_sha256.txt",
        help="Path to SHA-256 checksum manifest.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; only verify SHA-256 checksums of existing local files.",
    )
    args = parser.parse_args()

    entries = parse_checksum_manifest(args.manifest)
    subset_filters: Dict[str, Tuple[str, ...]] = {
        "real": ("data/raw/OpenSWI-real/",),
        "shallow": ("data/raw/OpenSWI-shallow/",),
        "deep": ("data/raw/OpenSWI-deep/",),
        "all": ("data/raw/OpenSWI-real/", "data/raw/OpenSWI-shallow/", "data/raw/OpenSWI-deep/"),
    }
    prefixes = subset_filters[args.subset]
    selected = [(digest, rel) for digest, rel in entries if rel.startswith(prefixes)]

    n_ok = 0
    n_fail = 0
    for expected_sha, rel_path in selected:
        local_path = args.data_root / rel_path
        if local_path.exists():
            actual = sha256_file(local_path)
            if actual == expected_sha:
                print(f"[OK]   {rel_path}")
                n_ok += 1
                continue
            print(f"[WARN] Checksum mismatch for {rel_path}: got {actual[:12]}..., expected {expected_sha[:12]}...")
            if args.verify_only:
                n_fail += 1
                continue

        if args.verify_only:
            print(f"[MISS] {rel_path}")
            n_fail += 1
            continue

        remote_rel = _hf_remote_subpath(rel_path)
        url = f"{HF_DATASET_BASE_URL}/{remote_rel}"
        print(f"[GET]  {url} -> {local_path}")
        try:
            download_file(url, local_path)
            actual = sha256_file(local_path)
            if actual != expected_sha:
                print(f"[FAIL] SHA-256 mismatch after download: {rel_path}")
                n_fail += 1
            else:
                print(f"[OK]   {rel_path}")
                n_ok += 1
        except Exception as exc:
            print(f"[ERR]  Failed downloading {url}: {exc}")
            n_fail += 1

    print(f"Summary: {n_ok} verified, {n_fail} failed/missing out of {len(selected)} files.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
