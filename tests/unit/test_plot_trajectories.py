"""Unit tests for plot_price_trajectories_by_chain."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from hotelling.viz.static import plot_price_trajectories_by_chain


def _write_full_run(tmp_path) -> None:
    steps = list(range(1000, 10001, 1000))
    n = len(steps)
    agg = pd.DataFrame({
        "step": steps,
        "mean_price": [40.0 + i * 0.5 for i in range(n)],
        "mean_effort": [0.0] * n,
        "mean_price_discount": [38.0 + i * 0.4 for i in range(n)],
        "mean_price_standard": [40.0 + i * 0.5 for i in range(n)],
        "mean_price_bio": [45.0 + i * 0.6 for i in range(n)],
    })
    agg.to_parquet(tmp_path / "aggregate.parquet", index=False)
    meta = {
        "chain_price_table": {
            "global": {"n": 494, "learned": 45.0, "nash": 40.0, "mono": 51.0},
            "discount": {"n": 196, "learned": 44.0, "nash": 36.0, "mono": 46.0},
            "standard": {"n": 207, "learned": 45.0, "nash": 40.0, "mono": 52.0},
            "bio": {"n": 91, "learned": 46.0, "nash": 50.0, "mono": 62.0},
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def test_plot_trajectories_full_columns(tmp_path) -> None:
    _write_full_run(tmp_path)
    ax = plot_price_trajectories_by_chain(tmp_path)
    assert ax is not None
    assert len(ax.lines) >= 4


def test_plot_trajectories_legacy_aggregate_only(tmp_path) -> None:
    steps = [1000, 2000, 3000]
    agg = pd.DataFrame({
        "step": steps,
        "mean_price": [0.7, 0.71, 0.72],
        "mean_effort": [0.0, 0.0, 0.0],
    })
    agg.to_parquet(tmp_path / "aggregate.parquet", index=False)
    ax = plot_price_trajectories_by_chain(tmp_path, show_benchmarks=False)
    assert ax is not None
    assert len(ax.lines) == 1
