"""Shared figure style (Nature-like sizing; validated categorical palette)."""
from __future__ import annotations

import matplotlib as mpl

MM = 1 / 25.4
SINGLE = 89 * MM  # Nature single-column width (inches)
DOUBLE = 183 * MM  # Nature double-column width (inches)

# Categorical slots in fixed order (validated with the dataviz palette checker:
# all adjacent pairs pass CVD and normal-vision separation on a light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")  # blue - gray midpoint - red

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Food groups -> categorical slot (identity follows the group in every figure).
GROUP_COLORS = {
    "Poultry & pork": SERIES[0],
    "Fish": SERIES[1],
    "Eggs": SERIES[2],
    "Dairy": SERIES[3],
    "Legumes & soy": SERIES[4],
    "Mycoprotein & plant mince": SERIES[5],
    "Nuts, seeds & grains": SERIES[6],
    "Fortified foods & supplements": SERIES[7],
}


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": INK,
    })


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left", color=INK)
