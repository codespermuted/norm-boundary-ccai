"""Shared publication figure style: one entity -> one color across ALL figures.

Palette: validated categorical slots (dataviz reference palette, light mode,
worst adjacent CVD dE 24.2). Region fills are white-blended tints of the same
hues; text labels always in ink, never in series color.
"""

import matplotlib as mpl

# entity -> hue (fixed across Fig 1-5; do not reassign per figure)
METHOD_COLORS = {
    "in": "#2a78d6",      # slot 1 blue  — instance normalization (incumbent)
    "cn": "#1baf7a",      # slot 2 aqua  — conditional normalization (proposed)
    "raw": "#eda100",     # slot 3 yellow — global z-score only
    "cn_oracle": "#4a3aa7",  # slot 5 violet — oracle variant where shown
}
METHOD_LABELS = {"raw": "RAW", "in": "IN (RevIN)", "cn": "CN (proposed)"}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d4"


def tint(hex_color: str, frac: float = 0.62) -> str:
    """Blend toward white by frac (region fills readable under ink labels)."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * frac) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def apply_paper_style() -> None:
    mpl.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.labelcolor": INK,
        "text.color": INK,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # editable text in PDF (journal requirement)
        "ps.fonttype": 42,
    })
