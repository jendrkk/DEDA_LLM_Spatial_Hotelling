"""Shared styling for the run-report visualisation pipeline.

Centralises the LaTeX/transparent matplotlib configuration (mirroring
``report/plot_calibration.py``), the per-chain-type static colours, the
per-chain-type sequential colormaps used for price/profit-coloured markers, and
the chain-type marker glyphs.  Importing this module has no side effects; call
:func:`apply_style` once per pipeline run.

Conventions (locked by the project):
    static colours  D = royalblue,  S = firebrick,  B = forestgreen
    marker cmaps    D = winter,     S = autumn,     B = summer
    markers         D = v (down-triangle), S = o (circle), B = s (square)
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ── Canonical chain-type ordering (controls legend order everywhere) ──────────
CHAIN_TYPES = ("discount", "standard", "bio")

# ── Static per-chain-type colours (also used for the global/Nash/mono refs) ──
CHAIN_COLOURS: Dict[str, str] = {
    "discount": "royalblue",
    "standard": "firebrick",
    "bio": "forestgreen",
    "global": "black",
}

# ── Per-chain-type sequential colormaps for price/profit-coloured markers ────
CHAIN_CMAPS: Dict[str, str] = {
    "discount": "winter",
    "standard": "autumn",
    "bio": "summer",
}

# ── Chain-type marker glyphs (distinct shape per type) ───────────────────────
CHAIN_MARKERS: Dict[str, str] = {
    "discount": "v",
    "standard": "o",
    "bio": "s",
}

# ── LaTeX display labels ─────────────────────────────────────────────────────
CHAIN_LABELS: Dict[str, str] = {
    "discount": r"Discount $(D)$",
    "standard": r"Standard $(S)$",
    "bio": r"Bio $(B)$",
    "global": r"Global",
}

# Greek subscript per chain type for math labels (p_D^N etc.)
CHAIN_SUB: Dict[str, str] = {
    "discount": "D",
    "standard": "S",
    "bio": "B",
    "global": r"\mathrm{all}",
}


def apply_style(use_latex: bool = True, transparent: bool = True) -> bool:
    """Configure global matplotlib rcParams for the pipeline.

    Parameters
    ----------
    use_latex : render all text with a LaTeX toolchain (``text.usetex``).
        Falls back to mathtext (``usetex=False``) automatically if LaTeX is
        unavailable, so figures still render with serif math.
    transparent : set figure / axes / savefig facecolors to ``"none"`` so every
        saved figure has a transparent background.

    Returns
    -------
    bool — whether a LaTeX toolchain was successfully enabled.
    """
    import matplotlib

    latex_ok = False
    if use_latex:
        try:
            matplotlib.rcParams.update({
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            })
            latex_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("LaTeX setup failed (%s); falling back to mathtext.", exc)
            matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif"})
    else:
        matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif"})

    face = "none" if transparent else "white"
    matplotlib.rcParams.update({
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "figure.facecolor": face,
        "axes.facecolor": face,
        "savefig.facecolor": face,
        "savefig.edgecolor": "none",
    })
    return latex_ok


def latex_or_plain(s_latex: str, s_plain: str) -> str:
    """Return *s_latex* when usetex is active, else *s_plain*.

    Lets callers keep heavy LaTeX (e.g. ``\\bar p``) for the usetex path while
    degrading to ASCII-safe mathtext when LaTeX is off.
    """
    import matplotlib
    return s_latex if matplotlib.rcParams.get("text.usetex", False) else s_plain
