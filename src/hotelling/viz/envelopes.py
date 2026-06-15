"""Strategic-run envelope visualisations (CEO layer).

Reads envelopes.parquet + metadata.json from a strategic run dir and plots:
- plot_envelope_bands : p_bar +/- delta_p band per chain over epochs
- plot_delta_by_chain : Calvano Delta per chain type (bar)
- plot_epsilon_trajectory : per-chain exploration epsilon over epochs

Public API: plot_envelope_bands, plot_delta_by_chain, plot_epsilon_trajectory
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _load(run_dir: Path | str):
    import pandas as pd
    run_dir = Path(run_dir)
    env = pd.read_parquet(run_dir / "envelopes.parquet")
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return env, meta, run_dir


def plot_envelope_bands(run_dir: Path | str, group: Optional[str] = None,
                        save_path: Optional[Path] = None) -> Any:
    """Plot each chain's p_bar with a shaded [p_bar-delta_p, p_bar+delta_p] band over epochs.

    If ``group`` is None, uses the first group key present per chain (e.g. 'default'
    or 'HEAVY'); pass a specific group label to focus a multi-group run.
    """
    import matplotlib.pyplot as plt
    env, meta, run_dir = _load(run_dir)
    if group is not None:
        env = env[env["group"] == group]
    fig, ax = plt.subplots(figsize=(11, 6))
    for chain, g in env.groupby("chain"):
        g = g.sort_values("epoch")
        if group is None:
            first = g["group"].iloc[0]
            g = g[g["group"] == first]
        ax.plot(g["epoch"], g["p_bar"], marker="o", linewidth=1.5, label=str(chain))
        ax.fill_between(g["epoch"], g["p_bar"] - g["delta_p"], g["p_bar"] + g["delta_p"],
                        alpha=0.12)
    ax.set_xlabel("CEO epoch"); ax.set_ylabel("Price envelope p\u0304 \u00b1 \u0394p (\u20ac)")
    ttl = "CEO price envelopes over epochs" + (f" — group {group}" if group else "")
    ax.set_title(ttl); ax.legend(fontsize=7, ncol=2, title="Chain")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_delta_by_chain(run_dir: Path | str, save_path: Optional[Path] = None) -> Any:
    """Bar chart of Calvano Delta per chain type from metadata.json."""
    import matplotlib.pyplot as plt
    _, meta, run_dir = _load(run_dir)
    d = meta.get("deltas_by_chain", {})
    keys = [k for k in ("global", "discount", "standard", "bio") if k in d and d[k] is not None]
    vals = [float(d[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(keys, vals, color=["#444", "#2a9d8f", "#e9c46a", "#264653"][: len(keys)])
    ax.axhline(0.0, color="k", linewidth=0.8); ax.axhline(1.0, color="r", linewidth=0.8,
               linestyle="--", label="monopoly (\u0394=1)")
    ax.set_ylabel("Calvano \u0394"); ax.set_title("Collusion index by chain type")
    ax.bar_label(bars, fmt="%.2f", fontsize=8); ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_epsilon_trajectory(run_dir: Path | str, save_path: Optional[Path] = None) -> Any:
    """Plot each chain's chosen exploration epsilon over epochs (first group per chain)."""
    import matplotlib.pyplot as plt
    env, _, run_dir = _load(run_dir)
    fig, ax = plt.subplots(figsize=(11, 5))
    for chain, g in env.groupby("chain"):
        g = g.sort_values("epoch")
        first = g["group"].iloc[0]
        g = g[g["group"] == first]
        ax.plot(g["epoch"], g["epsilon"], marker=".", label=str(chain))
    ax.set_xlabel("CEO epoch"); ax.set_ylabel("exploration \u03b5")
    ax.set_title("CEO-set exploration over epochs"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
