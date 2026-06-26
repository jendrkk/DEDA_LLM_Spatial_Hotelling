"""Run-report visualisation pipeline.

A single, highly-configurable suite that turns a finished baseline or strategic
run directory into a full set of figures and animations (price/profit/Δ
trajectories, store-marker price/profit animations, per-cell local collusion
choropleths, local-HHI concentration maps, and graph-state loop deltas).

Entry point:
    >>> from hotelling.viz.run_report import run_pipeline, VizConfig
    >>> cfg = VizConfig.load("configs/viz/run_report.yaml")
    >>> run_pipeline("results/runs/20260625_211444_811193ee", cfg)

The pipeline is self-contained: it reconstructs the City/DenseLog with full env
parity, computes its own benchmarks and metrics, and writes nothing outside the
run directory's ``figures/run_report`` subfolder.
"""
from __future__ import annotations

from .config import VizConfig
from .loading import RunBundle
from .pipeline import resolve_run_dir, run_pipeline

__all__ = ["run_pipeline", "resolve_run_dir", "VizConfig", "RunBundle"]
