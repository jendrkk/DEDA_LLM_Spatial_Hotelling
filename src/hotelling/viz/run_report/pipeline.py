"""Pipeline orchestrator for the run-report visualisation suite.

``run_pipeline(run_id_or_dir, viz_cfg)`` resolves the run directory, rebuilds a
:class:`~hotelling.viz.run_report.loading.RunBundle`, applies the global style,
and dispatches every enabled artefact under ``produce.*``.  Each artefact is
isolated in its own try/except so a single failure (e.g. a missing optional
dependency for one map) never aborts the rest of the report.  Returns the list
of written file paths.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .config import VizConfig
from .loading import RunBundle
from .style import apply_style

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_run_dir(run_id_or_dir: str | Path) -> Path:
    """Resolve a run id / partial path / absolute path to a directory with config.yaml."""
    arg = Path(run_id_or_dir)
    candidates = [
        arg,
        _REPO_ROOT / arg,
        _REPO_ROOT / "results" / arg,
        _REPO_ROOT / "results" / "runs" / arg,
        _REPO_ROOT / "strategic_runs" / arg,
        _REPO_ROOT / "strategic_runs" / "runs" / arg,
        _REPO_ROOT / "results" / "strategic_runs" / "runs" / arg,
    ]
    for c in candidates:
        if (c / "config.yaml").exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Could not resolve run '{run_id_or_dir}' to a directory containing config.yaml. "
        f"Tried: {', '.join(str(c) for c in candidates)}")


def run_pipeline(run_id_or_dir: str | Path, viz_cfg: Optional[VizConfig] = None) -> List[Path]:
    cfg = viz_cfg or VizConfig()
    run_dir = resolve_run_dir(run_id_or_dir)
    apply_style(use_latex=cfg.global_.use_latex, transparent=cfg.global_.transparent)

    logger.info("Building run bundle for %s ...", run_dir)
    bundle = RunBundle.from_run_dir(run_dir, cfg)
    out_dir = run_dir / cfg.global_.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run: N=%d stores, M=%d cells, %d recorded rows, lean=%s, strategic=%s, state=%s.",
                bundle.N, bundle.M, len(bundle.recorded_steps), bundle.lean,
                bundle.is_strategic, bundle.meta.get("state_mode", "?"))

    from . import trajectories as traj
    from . import store_maps as smaps
    from . import cell_maps as cmaps
    from . import graph_loops as gloops

    written: List[Path] = []

    def _do(flag: bool, label: str, fn):
        if not flag:
            return
        try:
            res = fn()
            if res is None:
                return
            if isinstance(res, (list, tuple)):
                written.extend(p for p in res if p is not None)
            else:
                written.append(res)
            logger.info("  [ok] %s", label)
        except Exception as exc:  # noqa: BLE001
            logger.exception("  [FAIL] %s: %s", label, exc)

    p = cfg.produce
    # trajectories / time series
    _do(p.price_trajectory, "01 price trajectory", lambda: traj.plot_price_trajectory(bundle, cfg, out_dir))
    _do(p.profit_trajectory, "02 profit trajectory", lambda: traj.plot_profit_trajectory(bundle, cfg, out_dir))
    _do(p.delta_trajectory, "03 delta trajectory", lambda: traj.plot_delta_trajectory(bundle, cfg, out_dir))
    _do(p.delta_over_time_extra, "03b delta panels", lambda: traj.plot_delta_over_time(bundle, cfg, out_dir))
    _do(p.market_shares_trajectory, "market shares", lambda: traj.plot_market_shares(bundle, cfg, out_dir))
    _do(p.global_hhi_trajectory, "global HHI", lambda: traj.plot_global_hhi(bundle, cfg, out_dir))
    _do(p.price_dispersion_trajectory, "price dispersion", lambda: traj.plot_price_dispersion(bundle, cfg, out_dir))
    _do(p.welfare_trajectory, "welfare", lambda: traj.plot_welfare(bundle, cfg, out_dir))
    _do(p.final_price_distribution, "final price distribution", lambda: traj.plot_final_price_distribution(bundle, cfg, out_dir))

    # store-marker animations
    _do(p.store_price_animation, "04 store price animation", lambda: smaps.animate_store_price(bundle, cfg, out_dir))
    _do(p.store_profit_animation, "05 store profit animation", lambda: smaps.animate_store_profit(bundle, cfg, out_dir))

    # cell choropleth animations (single shared pass)
    want_price = p.cell_price_delta_animation
    want_profit = p.cell_profit_delta_animation
    want_hhi = p.local_hhi_animation
    if want_price or want_profit or want_hhi:
        _do(True, "06/07/08 cell animations",
            lambda: cmaps.render_cell_animations(bundle, cfg, out_dir, want_price, want_profit, want_hhi))

    # graph-state loops (#9)
    _do(p.graph_loop_deltas, "09 graph loop deltas", lambda: gloops.render_graph_loops(bundle, cfg, out_dir))

    # strategic CEO envelope plots
    if bundle.is_strategic and p.envelope_plots:
        from hotelling.viz import envelopes as env
        import matplotlib.pyplot as plt

        def _envelope(fn, fname):
            path = out_dir / fname
            fig = fn(run_dir, save_path=path)
            plt.close(fig)
            return path

        _do(True, "envelope bands",
            lambda: _envelope(env.plot_envelope_bands, "strategic_envelope_bands.png"))
        _do(True, "delta by chain",
            lambda: _envelope(env.plot_delta_by_chain, "strategic_delta_by_chain.png"))
        _do(True, "epsilon trajectory",
            lambda: _envelope(env.plot_epsilon_trajectory, "strategic_epsilon_trajectory.png"))

    logger.info("Run report complete: %d artefacts in %s", len(written), out_dir)
    print(f"[run_report] wrote {len(written)} artefacts to {out_dir}")
    return written
