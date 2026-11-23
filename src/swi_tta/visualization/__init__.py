"""Publication-grade figure and table generation utilities for Computers & Geosciences."""

from swi_tta.visualization.main_figures import generate_all_main_figures, make_graphical_abstract
from swi_tta.visualization.style import (
    DOUBLE_COL_IN,
    METHOD_STYLE,
    PALETTE,
    SINGLE_COL_IN,
    apply_cg_style,
    save_figure,
    sha256_file,
)
from swi_tta.visualization.supp_figures import generate_all_supp_figures
from swi_tta.visualization.tables import generate_all_tables

__all__ = [
    "DOUBLE_COL_IN",
    "METHOD_STYLE",
    "PALETTE",
    "SINGLE_COL_IN",
    "apply_cg_style",
    "generate_all_main_figures",
    "generate_all_supp_figures",
    "generate_all_tables",
    "make_graphical_abstract",
    "save_figure",
    "sha256_file",
]
