#!/usr/bin/env python3
"""
Calibration sensitivity plots for presentation.

Simplified 3-firm aggregate logit model (no spatial structure, effort=0).
Sweeps calibrated parameters and shows how equilibrium outcomes change.

Output: report/figures/calibration/
  alpha_sensitivity.png   — alpha_L and alpha_H sensitivity
  quality_sensitivity.png — q_S and q_B sensitivity
  cost_sensitivity.png    — marginal cost spread sensitivity
  mu_sensitivity.png      — logit scale mu sensitivity
  a0_sensitivity.png      — outside option a_0 sensitivity

Run from repo root:
    conda activate py314
    python report/plot_calibration.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR   = REPO_ROOT / "report" / "figures" / "calibration"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib / LaTeX ─────────────────────────────────────────────────────
try:
    matplotlib.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    _USETEX = True
except Exception:
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif"})
    _USETEX = False

matplotlib.rcParams.update({
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.9,
    "figure.facecolor": "none",
    "axes.facecolor":   "none",
    "savefig.facecolor": "none",
})

DPI        = 200
FIGSIZE_21 = (12.0, 5.0)   # two-panel
FIGSIZE_31 = (15.0, 5.0)   # three-panel

# ── Wong (2011) colorblind-safe palette ───────────────────────────────────
_C = dict(
    blue      = "#0072B2",
    orange    = "#E69F00",
    green     = "#009E73",
    pink      = "#CC79A7",
    skyblue   = "#56B4E9",
    vermillion= "#D55E00",
    yellow    = "#F0E442",
    black     = "#000000",
)
_CC   = {"D": _C["green"], "S": _C["blue"],    "B": _C["pink"]}   # chain colours
_LBL  = {
    "D"  : r"Discount $s_D$",   "S": r"Standard $s_S$",
    "B"  : r"Bio $s_B$",        "out": r"Outside $s_0$",
    "pD" : r"Discount $p_D^N$", "pS": r"Standard $p_S^N$",
    "pB" : r"Bio $p_B^N$",
    "LD" : r"Discount $\mathcal{L}_D$",
    "LS" : r"Standard $\mathcal{L}_S$",
    "LB" : r"Bio $\mathcal{L}_B$",
}

# ── Calibrated base parameter vector ──────────────────────────────────────
BASE = dict(
    alpha_L = 0.4866,
    alpha_H = 1.2165,
    q_S     = 6.0,
    q_B     = 18.0,
    c_D     = 26.52,
    c_S     = 31.20,
    c_B     = 40.56,
    mu      = 9.119,
    a_0     = -18.275,
    pi_H    = 0.50,
)

# IIA store-count scaling
_N = np.array([196, 207, 91], dtype=np.float64)   # [N_D, N_S, N_B]

# ─────────────────────────────────────────────────────────────────────────────
# Core model functions
# ─────────────────────────────────────────────────────────────────────────────

def _logit_shares(prices: np.ndarray, p: dict
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate 3-firm logit shares with IIA store-count scaling.

    s_tau = sum_h pi_h * N_tau * exp(V_{h,tau}/mu) /
            [exp(a0/mu) + sum_k N_k * exp(V_{h,k}/mu)]

    Returns: (s_agg, s_L, s_H)  each shape (3,)
    """
    aL  = float(p["alpha_L"])
    aH  = float(p["alpha_H"])
    q   = np.array([0.0, float(p["q_S"]), float(p["q_B"])])
    mu  = float(p["mu"])
    a0  = float(p["a_0"])
    piH = float(p["pi_H"])
    piL = 1.0 - piH

    s_agg = np.zeros(3)
    s_L   = np.zeros(3)
    s_H   = np.zeros(3)

    for alpha_h, pi_h, is_H in [(aL, piL, False), (aH, piH, True)]:
        V     = alpha_h * q - prices          # (3,) raw utility
        V_sc  = V / mu
        a0_sc = a0 / mu
        vmax  = max(float(np.max(V_sc)), a0_sc)
        eV    = np.exp(V_sc  - vmax)
        ea0   = np.exp(a0_sc - vmax)
        denom = np.dot(_N, eV) + ea0
        s_h   = _N * eV / denom
        s_agg += pi_h * s_h
        if is_H:
            s_H = s_h
        else:
            s_L = s_h

    return s_agg, s_L, s_H


def nash_prices(p: dict, max_iter: int = 2000, tol: float = 1e-7) -> np.ndarray:
    """
    Bertrand-Nash equilibrium via damped FOC iteration.
    FOC: p_tau - c_tau = mu / (1 - s_tau/N_tau)
    Update: p_new = c + mu / (1 - s/N); p <- 0.5*p + 0.5*p_new
    """
    c  = np.array([p["c_D"], p["c_S"], p["c_B"]])
    mu = float(p["mu"])
    prices = c + mu + 3.0
    for _ in range(max_iter):
        s, _, _ = _logit_shares(prices, p)
        s_ind   = np.clip(s / _N, 1e-10, 1.0 - 1e-10)
        p_new   = c + mu / (1.0 - s_ind)
        delta   = float(np.max(np.abs(p_new - prices)))
        prices  = 0.5 * prices + 0.5 * p_new
        if delta < tol:
            break
    return prices


def lerner(prices: np.ndarray, p: dict) -> np.ndarray:
    c = np.array([p["c_D"], p["c_S"], p["c_B"]])
    return (prices - c) / np.maximum(prices, 1e-9)


def hhi(s: np.ndarray) -> float:
    return float(np.sum(s ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _calib_vline(ax: plt.Axes, x: float) -> None:
    ax.axvline(x, color="black", ls="--", lw=1.1, alpha=0.68,
               label=r"\textit{calibrated}" if _USETEX else "calibrated",
               zorder=10)


def _save(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", transparent=True,
                pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {out}")


def _setup_axes(axes):
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor("none")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: alpha_L and alpha_H sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_sensitivity() -> None:
    """
    3 panels:
      (a) 2D contour heatmap (alpha_L, alpha_H) -> bio market share s_B
      (b) Chain-type shares vs alpha_H (alpha_L fixed at calibrated)
      (c) Nash prices vs alpha_H (alpha_L fixed at calibrated)
    """
    # ── 2D sweep ──
    aL_range = np.linspace(0.20, 0.85, 50)
    aH_range = np.linspace(0.60, 2.85, 50)
    bio_2d   = np.full((len(aH_range), len(aL_range)), np.nan)
    for i, aH in enumerate(aH_range):
        for j, aL in enumerate(aL_range):
            if aH <= aL + 0.05:   # enforce alpha_H > alpha_L
                continue
            pp = dict(BASE, alpha_L=aL, alpha_H=aH)
            pN = nash_prices(pp)
            s, _, _ = _logit_shares(pN, pp)
            bio_2d[i, j] = s[2]

    # ── line sweep over alpha_H ──
    aH_vals = np.linspace(0.55, 2.85, 130)
    res = {k: [] for k in ("D","S","B","out","pD","pS","pB")}
    for aH in aH_vals:
        pp = dict(BASE, alpha_H=aH)
        pN = nash_prices(pp)
        s, _, _ = _logit_shares(pN, pp)
        res["D"].append(s[0]); res["S"].append(s[1]); res["B"].append(s[2])
        res["out"].append(1.0 - s.sum())
        res["pD"].append(pN[0]); res["pS"].append(pN[1]); res["pB"].append(pN[2])

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_31)
    _setup_axes(axes)

    # Panel (a)
    im = axes[0].contourf(aL_range, aH_range, bio_2d, levels=22, cmap="viridis")
    axes[0].contour(aL_range, aH_range, bio_2d, levels=10,
                    colors="white", linewidths=0.35, alpha=0.45)
    fig.colorbar(im, ax=axes[0], shrink=0.80, pad=0.02,
                 label=r"$s_B$ (bio market share)")
    axes[0].axvline(BASE["alpha_L"], color="red", ls="--", lw=1.4,
                    label=r"$\alpha_L^*$")
    axes[0].axhline(BASE["alpha_H"], color="red", ls="-.", lw=1.4,
                    label=r"$\alpha_H^*$")
    axes[0].set_xlabel(r"$\alpha_L$"); axes[0].set_ylabel(r"$\alpha_H$")
    axes[0].set_title(r"Bio share $s_B(\alpha_L,\,\alpha_H)$")
    axes[0].legend(fontsize=8, framealpha=0.85, facecolor="white",
                   edgecolor="none")

    # Panel (b)
    for k, col, lab in [("D",_CC["D"],_LBL["D"]),("S",_CC["S"],_LBL["S"]),
                         ("B",_CC["B"],_LBL["B"]),
                         ("out",_C["skyblue"],_LBL["out"])]:
        axes[1].plot(aH_vals, res[k], color=col, label=lab)
    _calib_vline(axes[1], BASE["alpha_H"])
    axes[1].set_xlabel(r"$\alpha_H$"); axes[1].set_ylabel("Market share")
    axes[1].set_title(r"Shares vs $\alpha_H$ ($\alpha_L$ fixed)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8, framealpha=0.85, facecolor="white",
                   edgecolor="none")

    # Panel (c)
    for k, col, lab in [("pD",_CC["D"],_LBL["pD"]),
                         ("pS",_CC["S"],_LBL["pS"]),
                         ("pB",_CC["B"],_LBL["pB"])]:
        axes[2].plot(aH_vals, res[k], color=col, label=lab)
    _calib_vline(axes[2], BASE["alpha_H"])
    axes[2].set_xlabel(r"$\alpha_H$")
    axes[2].set_ylabel(r"Nash price $p_\tau^N$ (EUR)")
    axes[2].set_title(r"Nash prices vs $\alpha_H$")
    axes[2].legend(fontsize=8, framealpha=0.85, facecolor="white",
                   edgecolor="none")

    fig.tight_layout(pad=1.0)
    _save(fig, "alpha_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: q_S and q_B quality sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_quality_sensitivity() -> None:
    """
    2 panels:
      (a) 2D heatmap (q_S, q_B) -> Nash bio price p_B^N
      (b) Chain-type shares vs q_B (q_S fixed at calibrated)
    """
    qS_range = np.linspace(1.5, 13.0, 45)
    qB_range = np.linspace(7.0, 35.0, 45)
    nash_pB  = np.full((len(qB_range), len(qS_range)), np.nan)
    for i, qB in enumerate(qB_range):
        for j, qS in enumerate(qS_range):
            if qB <= qS + 1.0:
                continue
            pp = dict(BASE, q_S=qS, q_B=qB)
            pN = nash_prices(pp)
            nash_pB[i, j] = pN[2]

    qB_vals = np.linspace(6.5, 38.0, 130)
    res = {k: [] for k in ("D","S","B","out")}
    for qB in qB_vals:
        if qB <= BASE["q_S"] + 0.5:
            for k in res: res[k].append(np.nan)
            continue
        pp = dict(BASE, q_B=qB)
        pN = nash_prices(pp)
        s, _, _ = _logit_shares(pN, pp)
        res["D"].append(s[0]); res["S"].append(s[1]); res["B"].append(s[2])
        res["out"].append(1.0 - s.sum())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_21)
    _setup_axes(axes)

    # Panel (a)
    im = axes[0].contourf(qS_range, qB_range, nash_pB, levels=22, cmap="plasma")
    axes[0].contour(qS_range, qB_range, nash_pB, levels=10,
                    colors="white", linewidths=0.35, alpha=0.40)
    fig.colorbar(im, ax=axes[0], shrink=0.80, pad=0.02,
                 label=r"$p_B^N$ (EUR)")
    axes[0].axvline(BASE["q_S"], color="white", ls="--", lw=1.5,
                    label=r"$q_S^*$")
    axes[0].axhline(BASE["q_B"], color="white", ls="-.", lw=1.5,
                    label=r"$q_B^*$")
    axes[0].set_xlabel(r"$q_S$"); axes[0].set_ylabel(r"$q_B$")
    axes[0].set_title(r"Bio Nash price $p_B^N(q_S,\,q_B)$")
    axes[0].legend(fontsize=8, framealpha=0.85, facecolor="white",
                   edgecolor="none", labelcolor="black")

    # Panel (b)
    for k, col, lab in [("D",_CC["D"],_LBL["D"]),("S",_CC["S"],_LBL["S"]),
                         ("B",_CC["B"],_LBL["B"]),
                         ("out",_C["skyblue"],_LBL["out"])]:
        axes[1].plot(qB_vals, res[k], color=col, label=lab)
    _calib_vline(axes[1], BASE["q_B"])
    axes[1].set_xlabel(r"$q_B$"); axes[1].set_ylabel("Market share")
    axes[1].set_title(r"Shares vs $q_B$ ($q_S$ fixed)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8, framealpha=0.85, facecolor="white",
                   edgecolor="none")

    fig.tight_layout(pad=1.0)
    _save(fig, "quality_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Marginal cost spread sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_cost_sensitivity() -> None:
    """
    2 panels:
      (a) Lerner index vs cost spread Δ = c_B - c_D
          c_S rescaled proportionally; c_D fixed at calibrated
      (b) Market shares vs cost spread
    """
    base_spread = BASE["c_B"] - BASE["c_D"]   # ~14 EUR
    spread_vals = np.linspace(1.5, 30.0, 110)

    lD, lS, lB = [], [], []
    sD, sS, sB, so = [], [], [], []

    for Δ in spread_vals:
        scale = Δ / base_spread
        pp = dict(BASE,
                  c_S=BASE["c_D"] + (BASE["c_S"] - BASE["c_D"]) * scale,
                  c_B=BASE["c_D"] + Δ)
        pN = nash_prices(pp)
        L  = lerner(pN, pp)
        s, _, _ = _logit_shares(pN, pp)
        lD.append(L[0]); lS.append(L[1]); lB.append(L[2])
        sD.append(s[0]); sS.append(s[1]); sB.append(s[2])
        so.append(1.0 - s.sum())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_21)
    _setup_axes(axes)

    axes[0].plot(spread_vals, lD, color=_CC["D"], label=_LBL["LD"])
    axes[0].plot(spread_vals, lS, color=_CC["S"], label=_LBL["LS"])
    axes[0].plot(spread_vals, lB, color=_CC["B"], label=_LBL["LB"])
    _calib_vline(axes[0], base_spread)
    axes[0].set_xlabel(r"Cost spread $c_B - c_D$ (EUR)")
    axes[0].set_ylabel(r"Lerner index $(p_\tau - c_\tau)/p_\tau$")
    axes[0].set_title("Lerner index vs cost spread")
    axes[0].legend(fontsize=9, framealpha=0.85, facecolor="white",
                   edgecolor="none")

    axes[1].plot(spread_vals, sD, color=_CC["D"], label=_LBL["D"])
    axes[1].plot(spread_vals, sS, color=_CC["S"], label=_LBL["S"])
    axes[1].plot(spread_vals, sB, color=_CC["B"], label=_LBL["B"])
    axes[1].plot(spread_vals, so, color=_C["skyblue"], ls="--",
                 label=_LBL["out"])
    _calib_vline(axes[1], base_spread)
    axes[1].set_xlabel(r"Cost spread $c_B - c_D$ (EUR)")
    axes[1].set_ylabel("Market share")
    axes[1].set_title("Market shares vs cost spread")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=9, framealpha=0.85, facecolor="white",
                   edgecolor="none")

    fig.tight_layout(pad=1.0)
    _save(fig, "cost_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: mu (logit scale) sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_mu_sensitivity() -> None:
    """
    2 panels:
      (a) Nash prices vs mu
      (b) Outside share (left axis) and HHI (right axis) vs mu
    """
    mu_vals = np.linspace(2.0, 22.0, 130)
    pD, pS, pB, so, hhi_vals = [], [], [], [], []

    for mu in mu_vals:
        pp = dict(BASE, mu=mu)
        pN = nash_prices(pp)
        s, _, _ = _logit_shares(pN, pp)
        pD.append(pN[0]); pS.append(pN[1]); pB.append(pN[2])
        so.append(1.0 - s.sum())
        hhi_vals.append(hhi(s))

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_21)
    _setup_axes(axes)

    # Panel (a)
    axes[0].plot(mu_vals, pD, color=_CC["D"], label=_LBL["pD"])
    axes[0].plot(mu_vals, pS, color=_CC["S"], label=_LBL["pS"])
    axes[0].plot(mu_vals, pB, color=_CC["B"], label=_LBL["pB"])
    _calib_vline(axes[0], BASE["mu"])
    axes[0].set_xlabel(r"$\mu$ (logit scale)")
    axes[0].set_ylabel(r"Nash price $p_\tau^N$ (EUR)")
    axes[0].set_title(r"Nash prices vs $\mu$")
    axes[0].legend(fontsize=9, framealpha=0.85, facecolor="white",
                   edgecolor="none")

    # Panel (b): dual y-axis
    ax_l = axes[1]
    ax_r = axes[1].twinx()
    ax_r.set_facecolor("none")
    l1, = ax_l.plot(mu_vals, so,       color=_C["skyblue"],
                    label=r"Outside $s_0$")
    l2, = ax_r.plot(mu_vals, hhi_vals, color=_C["vermillion"],
                    ls="--", label="HHI")
    _calib_vline(ax_l, BASE["mu"])
    ax_l.set_xlabel(r"$\mu$ (logit scale)")
    ax_l.set_ylabel(r"Outside share $s_0$", color=_C["skyblue"])
    ax_r.set_ylabel("HHI", color=_C["vermillion"])
    axes[1].set_title(r"Outside share and concentration vs $\mu$")
    calib_patch = plt.Line2D([0],[0], ls="--", lw=1.1, color="black",
                              alpha=0.68,
                              label=r"\textit{calibrated}" if _USETEX
                              else "calibrated")
    ax_l.legend(handles=[l1, l2, calib_patch], fontsize=9,
                framealpha=0.85, facecolor="white", edgecolor="none")

    fig.tight_layout(pad=1.0)
    _save(fig, "mu_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: a_0 (outside option) sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_a0_sensitivity() -> None:
    """
    2 panels:
      (a) Market shares (including outside) vs a_0
      (b) Nash prices vs a_0
    """
    a0_vals = np.linspace(-36.0, -3.0, 130)
    sD, sS, sB, so = [], [], [], []
    pD, pS, pB     = [], [], []

    for a0 in a0_vals:
        pp = dict(BASE, a_0=a0)
        pN = nash_prices(pp)
        s, _, _ = _logit_shares(pN, pp)
        sD.append(s[0]); sS.append(s[1]); sB.append(s[2])
        so.append(1.0 - s.sum())
        pD.append(pN[0]); pS.append(pN[1]); pB.append(pN[2])

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_21)
    _setup_axes(axes)

    axes[0].plot(a0_vals, sD, color=_CC["D"], label=_LBL["D"])
    axes[0].plot(a0_vals, sS, color=_CC["S"], label=_LBL["S"])
    axes[0].plot(a0_vals, sB, color=_CC["B"], label=_LBL["B"])
    axes[0].plot(a0_vals, so, color=_C["skyblue"], ls="--",
                 label=_LBL["out"])
    _calib_vline(axes[0], BASE["a_0"])
    axes[0].set_xlabel(r"$a_0$ (outside option utility)")
    axes[0].set_ylabel("Market share")
    axes[0].set_title(r"Market shares vs $a_0$")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=9, framealpha=0.85, facecolor="white",
                   edgecolor="none")

    axes[1].plot(a0_vals, pD, color=_CC["D"], label=_LBL["pD"])
    axes[1].plot(a0_vals, pS, color=_CC["S"], label=_LBL["pS"])
    axes[1].plot(a0_vals, pB, color=_CC["B"], label=_LBL["pB"])
    _calib_vline(axes[1], BASE["a_0"])
    axes[1].set_xlabel(r"$a_0$ (outside option utility)")
    axes[1].set_ylabel(r"Nash price $p_\tau^N$ (EUR)")
    axes[1].set_title(r"Nash prices vs $a_0$")
    axes[1].legend(fontsize=9, framealpha=0.85, facecolor="white",
                   edgecolor="none")

    fig.tight_layout(pad=1.0)
    _save(fig, "a0_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point + sanity check
# ─────────────────────────────────────────────────────────────────────────────

def _sanity_check() -> None:
    pN = nash_prices(BASE)
    s, _, _ = _logit_shares(pN, BASE)
    L = lerner(pN, BASE)
    print(f"  Nash prices : D={pN[0]:.2f}  S={pN[1]:.2f}  B={pN[2]:.2f}")
    print(f"  Shares      : D={s[0]:.3f}  S={s[1]:.3f}  B={s[2]:.3f}"
          f"  out={1-s.sum():.3f}")
    print(f"  Lerner      : D={L[0]:.3f}  S={L[1]:.3f}  B={L[2]:.3f}")
    print(f"  Target px   : D=34.00  S=40.00  B=52.00")
    print()


def main() -> None:
    print(f"Calibration sensitivity plots -> {FIG_DIR}")
    print()
    print("Sanity check at calibrated parameters:")
    _sanity_check()
    print("Generating …")
    plot_alpha_sensitivity()
    plot_quality_sensitivity()
    plot_cost_sensitivity()
    plot_mu_sensitivity()
    plot_a0_sensitivity()
    print(f"\nDone. All figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
