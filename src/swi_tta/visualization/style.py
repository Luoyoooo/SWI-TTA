"""Elsevier / Computers & Geosciences publication styling and deterministic export."""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Exact Elsevier / Computers & Geosciences print widths (mm -> inches)
# Set DOUBLE_COL_MM to 166 mm (~471 pt, ~0.96x scaling on cas-sc 455 pt textwidth)
# so native Matplotlib lettering renders at >= 6.4-8.0 pt post-scaling without shrinking.
MM_PER_INCH: float = 25.4
SINGLE_COL_MM: float = 90.0
ONE_HALF_COL_MM: float = 140.0
DOUBLE_COL_MM: float = 166.0

SINGLE_COL_IN: float = SINGLE_COL_MM / MM_PER_INCH
ONE_HALF_COL_IN: float = ONE_HALF_COL_MM / MM_PER_INCH
DOUBLE_COL_IN: float = DOUBLE_COL_MM / MM_PER_INCH

# Unified Okabe-Ito colorblind-safe palette (plus neutral tones)
PALETTE: Dict[str, str] = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "teal": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#1A1A1A",
    "dark_gray": "#4D4D4D",
    "mid_gray": "#8C8C8C",
    "light_gray": "#D9D9D9",
    "bg_tint": "#F7F9FB",
    "safe_tint": "#E8F5EE",
    "warn_tint": "#FDECE6",
    "info_tint": "#EAF3FA",
}

# Canonical method styling (color, marker, linestyle, short display label)
METHOD_STYLE: Dict[str, Tuple[str, str, str, str]] = {
    "zero_shot_raw": ("#7F7F7F", "o", ":", "Zero-Shot (Raw)"),
    "zero_shot_aligned": ("#1A1A1A", "s", "-", "Input-Aligned Zero-Shot"),
    "ungated_adapter_raw": ("#8C564B", "v", "--", "Ungated Adapter (Raw)"),
    "gated_adapter_raw": ("#E69F00", "^", "--", "Gated Adapter (Raw)"),
    "soft_gated_output_refined_raw": ("#CC79A7", "P", "-.", "Soft-Gated Output-Refined (Raw)"),
    "output_refined_raw": ("#56B4E9", "D", "--", "Output-Refined (Raw)"),
    "adapter_output_refined_raw": ("#9467BD", "X", "-.", "Adapter + Output-Refined (Raw)"),
    "output_refined_aligned": ("#D55E00", "o", "-", "Output-Space Refinement"),
    "low_dof_adapter": ("#009E73", "^", "-", "Low-DOF Adapter"),
    "adapter_output_refined_aligned": ("#0072B2", "*", "-", "Gated Adapter + Output-Refined"),
}


def mm_to_in(mm: float) -> float:
    """Convert millimetres to inches."""
    return float(mm) / MM_PER_INCH


def apply_cg_style() -> None:
    """Apply Elsevier / Computers & Geosciences publication matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "axes.labelcolor": "#1A1A1A",
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "legend.title_fontsize": 7.6,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linestyle": ":",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "swi-tta-diagnostics-cg",
        }
    )


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.10,
    y: float = 1.05,
    fontsize: float = 9.0,
) -> None:
    """Place a bold lowercase panel identifier, e.g. '(a)', outside the top-left corner."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def sha256_file(path: str) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_figure(
    fig: Figure,
    out_stem: str,
    *,
    formats: Iterable[str] = ("pdf", "png", "svg"),
    dpi: int = 600,
    bbox_inches: Optional[str] = "tight",
    pad_inches: float = 0.04,
) -> Dict[str, Dict[str, object]]:
    """Save a figure deterministically in multiple formats and return file metadata.

    Timestamps are stripped from PDF and SVG metadata so repeated runs yield
    bit-identical SHA-256 digests on the same environment.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    records: Dict[str, Dict[str, object]] = {}
    for fmt in formats:
        ext = fmt.lower().lstrip(".")
        out_path = f"{out_stem}.{ext}"
        kwargs: Dict[str, object] = {
            "format": ext,
            "facecolor": "white",
        }
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
            kwargs["pad_inches"] = pad_inches
        if ext == "png":
            kwargs["dpi"] = int(dpi)
            kwargs["pil_kwargs"] = {"compress_level": 6}
        elif ext == "pdf":
            kwargs["metadata"] = {
                "Creator": "swi-tta-diagnostics",
                "Producer": "matplotlib",
                "CreationDate": None,
            }
        elif ext == "svg":
            kwargs["metadata"] = {
                "Creator": "swi-tta-diagnostics",
                "Date": None,
            }
        if bbox_inches is None:
            with plt.rc_context({"savefig.bbox": "standard"}):
                fig.savefig(out_path, **kwargs)
        else:
            fig.savefig(out_path, **kwargs)
        records[ext] = {
            "path": out_path,
            "size_bytes": int(os.path.getsize(out_path)),
            "sha256": sha256_file(out_path),
        }
    plt.close(fig)
    return records


def format_pct(val: float, signed: bool = False, decimals: int = 2) -> str:
    """Format a percentage value concisely."""
    fmt = f"{{:{'+' if signed else ''}.{decimals}f}}%"
    return fmt.format(val)
