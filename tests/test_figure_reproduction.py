"""End-to-end verification of compact summary integrity, tables, and deterministic figures."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from swi_tta.visualization.main_figures import make_fig06_noise_amplitude_scan
from swi_tta.visualization.style import sha256_file
from swi_tta.visualization.tables import generate_all_tables

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_DIR = REPO_ROOT / "results_summary"
FIGURES_DIR = REPO_ROOT / "figures"
TABLES_DIR = REPO_ROOT / "tables"


def test_results_summary_manifest_and_size_caps() -> None:
    manifest = json.loads((SUMMARY_DIR / "manifest_sha256.json").read_text(encoding="utf-8"))
    total_bytes = 0
    for fname, meta in manifest["files"].items():
        fp = SUMMARY_DIR / fname
        assert fp.exists(), f"Missing summary file: {fname}"
        sz = fp.stat().st_size
        assert sz <= 15 * 1024 * 1024, f"{fname} exceeds 15 MB cap"
        assert sz == int(meta["size_bytes"])
        assert sha256_file(str(fp)) == meta["sha256"]
        total_bytes += sz
    assert total_bytes <= 35 * 1024 * 1024, "results_summary exceeds 35 MB cap"


def test_deep_lithospheric_calibration_erratum_fixed() -> None:
    cal = json.loads((SUMMARY_DIR / "calibration_and_domains.json").read_text(encoding="utf-8"))
    dl = cal["calibrated_noise_floors"]["deep_lithospheric"]
    assert dl["sigma_hat_d"] == pytest.approx(0.02416589723283842, abs=1e-12)
    assert dl["gt_eval_median"] == pytest.approx(0.013289561546567415, abs=1e-12)
    assert dl["ratio_sigma_over_gt_eval"] == pytest.approx(1.8184119278999296, abs=1e-10)
    assert f"{dl['sigma_hat_d']:.5f}" == "0.02417"
    assert f"{dl['ratio_sigma_over_gt_eval']:.2f}" == "1.82"


def test_table_generation_and_erratum_values(tmp_path: Path) -> None:
    records = generate_all_tables(str(SUMMARY_DIR), str(tmp_path))
    expected_tables = {
        "table1_benchmark_comparison",
        "table2_referee_convergence_and_tails",
        "table3_cross_backbone_diagnostics",
        "table4_safe_tta_validation_audit",
        "tableS1_noise_floor_calibration",
        "tableS2_signature_bins",
        "tableS3_protocol_and_retrained_audit",
    }
    assert set(records.keys()) == expected_tables
    t2_md = (tmp_path / "table2_referee_convergence_and_tails.md").read_text(encoding="utf-8")
    assert "21,469" in t2_md and "10,626" in t2_md
    s1_md = (tmp_path / "tableS1_noise_floor_calibration.md").read_text(encoding="utf-8")
    assert "0.02417" in s1_md and "1.82×" in s1_md
    s2_md = (tmp_path / "tableS2_signature_bins.md").read_text(encoding="utf-8")
    assert "+15.24%" in s2_md and "+13.25%" in s2_md
    s3_md = (tmp_path / "tableS3_protocol_and_retrained_audit.md").read_text(encoding="utf-8")
    assert "LongBeach Report" in s3_md and "CSRM Report" in s3_md
    assert "ΔRMSE (w/o RB)" in s3_md and "ΔRMSE (w/ RB)" in s3_md


def test_generated_figures_manifest_and_graphical_abstract_dimensions() -> None:
    manifest = json.loads((FIGURES_DIR / "manifest_sha256.json").read_text(encoding="utf-8"))
    assert manifest["n_figures"] == 15
    for fig_name, fmt_map in manifest["figures"].items():
        for ext in ("pdf", "png", "svg"):
            assert ext in fmt_map, f"Missing format {ext} for {fig_name}"
            fp = FIGURES_DIR / fmt_map[ext]["relative_path"]
            assert fp.exists(), f"Missing figure file: {fp}"
            assert sha256_file(str(fp)) == fmt_map[ext]["sha256"]

    ga_png = FIGURES_DIR / "graphical_abstract.png"
    with Image.open(ga_png) as img:
        w, h = img.size
    assert w >= 1328 and h >= 531
    assert abs((w / h) - 2.5) < 0.02, f"Graphical abstract aspect {w/h:.3f} != 2.5"


def test_deterministic_figure_reproduction(tmp_path: Path) -> None:
    r1 = make_fig06_noise_amplitude_scan(str(SUMMARY_DIR), str(tmp_path / "run1"), formats=("pdf", "png", "svg"), dpi=150)
    r2 = make_fig06_noise_amplitude_scan(str(SUMMARY_DIR), str(tmp_path / "run2"), formats=("pdf", "png", "svg"), dpi=150)
    for ext in ("pdf", "png", "svg"):
        assert r1[ext]["sha256"] == r2[ext]["sha256"], f"Non-deterministic {ext} output"
