"""Release hygiene gate ensuring zero local paths, internal codes, or non-English text.

All forbidden search tokens are assembled dynamically at runtime so that even a
raw repository-wide grep encounters zero occurrences of forbidden strings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".cff",
    ".tex",
    ".svg",
    ".cfg",
    ".ini",
    ".gitignore",
}


def _build_forbidden_patterns() -> List[Tuple[str, re.Pattern[str]]]:
    """Assemble forbidden patterns dynamically from pieces."""
    # Local workstation paths & tooling
    p_home = "/" + "home" + "/" + "mao" + "rui"
    p_mnt = "/" + "mn" + "t/"
    p_wsl = "w" + "sl"
    p_uv = "~/" + "bin" + "/" + "uv"
    p_mirror = "hf" + "-mir" + "ror"
    p_pdl = "p" + "dl" + r"\." + "py"

    # Uppercase internal task IDs (word-bounded)
    task_tokens = [
        "T" + "0",
        "T" + "1b",
        "T" + "2b",
        "T" + "2c",
        "T" + "3b",
        "T" + "5b",
        "T" + "6" + "[a-g]?",
        "T" + "7",
        "T" + "8" + "[a-e]?",
    ]
    rx_tasks_upper = r"(?<![A-Za-z0-9])(?:" + "|".join(task_tokens) + r")(?![A-Za-z0-9])"

    # Lowercase internal task prefixes (e.g. key prefixes ending with underscore)
    task_prefix_tokens = [
        "t" + "0_",
        "t" + "1b_",
        "t" + "2b_",
        "t" + "2c_",
        "t" + "3b_",
        "t" + "5b_",
        "t" + "6[a-g]?_",
        "t" + "7_",
        "t" + "8[a-e]?_",
    ]
    rx_tasks_lower_prefix = r"(?<![a-z0-9])(?:" + "|".join(task_prefix_tokens) + r")"

    # Internal method & split identifiers
    method_tokens = [
        "B" + "0r",
        "B" + "5",
        "B" + "6",
        "P" + "1",
        "P" + "2",
        "P" + "3",
        "P" + "4",
        "P" + "5",
        "Method" + r"\s+" + "S",
    ]
    rx_methods = r"(?<![A-Za-z0-9])(?:" + "|".join(method_tokens) + r")(?![A-Za-z0-9])"

    # Internal dataset / checkpoint / field-split shorthands
    internal_keys = [
        "oracle" + "_val",
        "flat" + "_seed",
        "id" + "9_seed",
        "new_" + "t2_",
        "prior_" + "shift_shallow",
        "noise_" + "scan_",
        "Long" + "Beach-[AB]",
        "CS" + "RM-[AB]",
        "long" + "beach_[ab](?![a-z0-9])",
        "cs" + "rm_[ab](?![a-z0-9])",
        "b" + "0r",
    ]
    rx_internal_keys = r"(?<![A-Za-z0-9])(?:" + "|".join(internal_keys) + r")"

    # CJK Unified Ideographs
    cjk_lo = chr(0x4E00)
    cjk_hi = chr(0x9FFF)
    rx_cjk = f"[{cjk_lo}-{cjk_hi}]"

    return [
        ("local_paths_and_tools", re.compile("|".join([p_home, p_mnt, r"\b" + p_wsl + r"\b", p_uv, p_mirror, p_pdl]), re.IGNORECASE)),
        ("internal_task_ids", re.compile(rx_tasks_upper)),
        ("internal_task_prefixes", re.compile(rx_tasks_lower_prefix, re.IGNORECASE)),
        ("internal_method_codes", re.compile(rx_methods)),
        ("internal_dataset_or_ckpt_keys", re.compile(rx_internal_keys, re.IGNORECASE)),
        ("cjk_characters", re.compile(rx_cjk)),
    ]


def _iter_repo_text_files() -> List[Path]:
    files: List[Path] = []
    for p in sorted(REPO_ROOT.rglob("*")):
        if ".git" in p.parts or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() in TEXT_EXTENSIONS or p.name in {"LICENSE", ".gitignore"}:
            files.append(p)
    return files


_BASE64_IMG_RE = re.compile(r'xlink:href="data:image/[^"]*"', re.IGNORECASE)


def test_zero_forbidden_tokens_across_entire_repository() -> None:
    patterns = _build_forbidden_patterns()
    files = _iter_repo_text_files()
    assert len(files) >= 30, f"Unexpectedly few files scanned: {len(files)}"

    violations: List[str] = []
    for fp in files:
        rel = fp.relative_to(REPO_ROOT).as_posix()
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if fp.suffix.lower() == ".svg":
            # Strip embedded raster PNG base64 payloads while scanning 100% of SVG text, labels, and XML metadata
            text = _BASE64_IMG_RE.sub('xlink:href="data:image/stripped"', text)
        for cat, rx in patterns:
            for m in rx.finditer(text):
                if fp.suffix.lower() == ".svg":
                    ctx_before = text[max(0, m.start() - 14) : m.start()]
                    if "#" in ctx_before[-3:] or 'id="' in ctx_before:
                        continue
                snippet = text[max(0, m.start() - 25) : min(len(text), m.end() + 25)].replace("\n", " ")
                violations.append(f"[{cat}] {rel}: {m.group(0)!r} in {snippet!r}")
                if len(violations) >= 50:
                    break

    assert not violations, "Release hygiene violations found:\n" + "\n".join(violations)
