"""Typed configuration for the run-report visualisation pipeline.

A single :class:`VizConfig` holds every tunable parameter, grouped into nested
dataclasses that mirror the top-level blocks of ``configs/viz/run_report.yaml``.
``VizConfig.load`` deep-merges a user YAML over the defaults, so any subset of
keys may be supplied (or none at all — the pipeline runs with defaults).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Nested blocks ────────────────────────────────────────────────────────────

@dataclass
class GlobalCfg:
    transparent: bool = True
    use_latex: bool = True
    dpi: int = 200
    output_subdir: str = "figures/run_report"
    overwrite: bool = True


@dataclass
class ProduceCfg:
    price_trajectory: bool = True
    profit_trajectory: bool = True
    delta_trajectory: bool = True
    store_price_animation: bool = True
    store_profit_animation: bool = True
    cell_price_delta_animation: bool = True
    cell_profit_delta_animation: bool = True
    local_hhi_animation: bool = True
    graph_loop_deltas: bool = True
    delta_over_time_extra: bool = True
    market_shares_trajectory: bool = True
    welfare_trajectory: bool = True
    global_hhi_trajectory: bool = True
    price_dispersion_trajectory: bool = True
    final_price_distribution: bool = True
    envelope_plots: bool = True


@dataclass
class ColoursCfg:
    discount: str = "royalblue"
    standard: str = "firebrick"
    bio: str = "forestgreen"
    global_: str = "black"
    nash: str = "#444444"
    mono: str = "#b22222"

    def for_type(self, ct: str) -> str:
        return {"discount": self.discount, "standard": self.standard,
                "bio": self.bio, "global": self.global_}.get(ct, "black")


@dataclass
class ChainCmapsCfg:
    discount: str = "winter"
    standard: str = "autumn"
    bio: str = "summer"

    def for_type(self, ct: str) -> str:
        return {"discount": self.discount, "standard": self.standard,
                "bio": self.bio}.get(ct, "viridis")


@dataclass
class FramesCfg:
    n_frames: int = 60
    mode: str = "even"          # even | all
    max_frames: int = 240


@dataclass
class AnalysisCfg:
    """Row-subsampling for trajectory / time-series plots.

    Trajectories do not need every recorded DenseLog row: a moving-average
    smoothed series is visually identical when sampled at a coarser stride, and
    on lean runs each retained row costs one full spatial market-clearing
    reconstruction.  This block controls that subsampling.  It does NOT affect
    the animations (#4–#9), which sample independently via ``frames.n_frames``.

    auto_stride : derive the stride so there is ~one analysis point per
        ``target_steps_per_point`` simulation steps, given the run's native
        recording stride.
    target_steps_per_point : target spacing (in simulation steps) between
        retained analysis points when ``auto_stride`` is true. Default 100.
    stride : take every ``stride``-th recorded row when ``auto_stride`` is
        false. ``1`` == use every recorded row (legacy behaviour).
    max_points : hard upper bound on retained analysis points (safety cap that
        overrides the stride with an even ``linspace`` when exceeded).
    """
    auto_stride: bool = True
    target_steps_per_point: int = 100
    stride: int = 1
    max_points: int = 20000


@dataclass
class AnimationCfg:
    fps: int = 8
    format: str = "mov"              # mov | webp | apng | gif | mp4
    transparent_format: str = "mov"  # fallback when transparent=True and format lacks alpha
    loop: int = 0
    lossless_webp: bool = True
    # ── .mov / ffmpeg settings ────────────────────────────────────────────────
    anim_dpi: Optional[int] = None   # None → use global_.dpi. Set 100–120 for .mov to
                                     # keep file sizes manageable. 100 DPI → 1100×1000 px
                                     # for an 11×10 figure → ~35–45 MB / 60 ProRes frames.
    mov_codec: str = "prores_ks"     # QuickTime codec: prores_ks (ProRes 4444, alpha,
                                     # Keynote-native) | png (lossless, larger) | qtrle.
    mov_bits_per_mb: int = 8000      # prores_ks target bits per 16×16 macroblock.
                                     # 8000 = standard ProRes 4444 quality.
                                     # Reduce to 2000–4000 for smaller files at imperceptible
                                     # quality loss on screen content (choropleths, scatter).
    ffmpeg_path: str = "ffmpeg"      # path to ffmpeg binary; can be absolute,
                                     # e.g. /opt/homebrew/bin/ffmpeg (brew install ffmpeg).


@dataclass
class TrajectoryCfg:
    ma_window_steps: int = 1000
    ma_kind: str = "trailing"   # trailing | centered
    raw_alpha: float = 0.25
    show_benchmarks: bool = True
    max_lean_points: int = 4000
    delta_clip: List[float] = field(default_factory=lambda: [-0.5, 1.5])
    figsize: List[float] = field(default_factory=lambda: [11.0, 5.2])


@dataclass
class StoreMarkersCfg:
    marker_size: float = 70.0
    edge_width: float = 0.4
    edge_colour: str = "black"
    basemap_alpha: float = 1.0
    price_scale_mode: str = "action_grid"     # action_grid | learned
    price_scale_xi: Optional[float] = None
    price_scale_anchor: str = "chain_mean"    # chain_mean | store_minmax
    profit_scale_mode: str = "bench"          # bench | learned
    profit_scale_margin: float = 0.05


@dataclass
class CellsCfg:
    alpha: float = 0.5
    delta_cmap: str = "RdYlGn_r"
    delta_vmin: float = 0.0
    delta_vmax: float = 1.0
    min_cell_demand: float = 1.0e-6
    variants: List[str] = field(default_factory=lambda: ["discount", "standard", "bio", "all"])


@dataclass
class HHICfg:
    radius_m: float = 500.0
    group_by: str = "chain"     # chain | chain_type
    include_outside: bool = False
    normalised: bool = True
    cmap: str = "magma"
    vmin: Optional[float] = None
    vmax: Optional[float] = None


@dataclass
class GraphLoopsCfg:
    which_delta: str = "both"   # price | profit | both
    loop_cmap: str = "RdYlGn_r"
    draw_candidate_edges: bool = False
    annotate: bool = True
    min_loop_size: int = 3
    per_loop_timeseries: bool = True


@dataclass
class WindowCfg:
    tail_fraction: float = 0.1
    tail_steps: Optional[int] = None


@dataclass
class BasemapCfg:
    provider: str = "OpenStreetMap.Mapnik"
    zoom: Any = "auto"


@dataclass
class LegendCfg:
    outside: bool = True
    loc: str = "center left"
    bbox: List[float] = field(default_factory=lambda: [1.02, 0.5])
    fontsize: float = 8.0
    title_fontsize: float = 9.0

    def kwargs(self) -> Dict[str, Any]:
        """matplotlib legend kwargs placing the legend outside the axes."""
        if self.outside:
            return dict(loc=self.loc, bbox_to_anchor=tuple(self.bbox),
                        fontsize=self.fontsize, title_fontsize=self.title_fontsize,
                        borderaxespad=0.0, framealpha=0.85)
        return dict(loc="best", fontsize=self.fontsize, title_fontsize=self.title_fontsize,
                    framealpha=0.85)


# ── Root config ──────────────────────────────────────────────────────────────

@dataclass
class VizConfig:
    global_: GlobalCfg = field(default_factory=GlobalCfg)
    produce: ProduceCfg = field(default_factory=ProduceCfg)
    colours: ColoursCfg = field(default_factory=ColoursCfg)
    chain_cmaps: ChainCmapsCfg = field(default_factory=ChainCmapsCfg)
    frames: FramesCfg = field(default_factory=FramesCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    animation: AnimationCfg = field(default_factory=AnimationCfg)
    trajectory: TrajectoryCfg = field(default_factory=TrajectoryCfg)
    store_markers: StoreMarkersCfg = field(default_factory=StoreMarkersCfg)
    cells: CellsCfg = field(default_factory=CellsCfg)
    hhi: HHICfg = field(default_factory=HHICfg)
    graph_loops: GraphLoopsCfg = field(default_factory=GraphLoopsCfg)
    window: WindowCfg = field(default_factory=WindowCfg)
    basemap: BasemapCfg = field(default_factory=BasemapCfg)
    legend: LegendCfg = field(default_factory=LegendCfg)

    # YAML key 'global' -> attribute 'global_' (Python keyword); same for colours.global.
    _ALIASES = {"global": "global_"}

    @classmethod
    def load(cls, path: "str | Path | None" = None) -> "VizConfig":
        """Build a config from defaults, deep-merging a YAML file if given."""
        cfg = cls()
        if path is None:
            return cfg
        import yaml
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Viz config not found: {p}")
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        _merge_into(cfg, data)
        return cfg

    def animation_format(self) -> str:
        """Effective animation container, honouring the global transparency switch.

        GIF carries only 1-bit transparency, MP4 none; when a transparent
        background is requested these are upgraded to ``transparent_format``.
        MOV (ProRes 4444 via prores_ks) carries true per-pixel RGBA alpha
        natively and is never auto-upgraded.
        """
        fmt = self.animation.format.lower()
        if self.global_.transparent and fmt in ("gif", "mp4"):
            return self.animation.transparent_format.lower()
        return fmt

    def animation_dpi(self) -> int:
        """DPI for rasterising animation frames.

        Returns ``animation.anim_dpi`` when set, otherwise falls back to
        ``global_.dpi``. Decoupling animation DPI from static-figure DPI
        lets .mov files stay under the 50 MB / clip budget while PNG/PDF
        exports remain at full resolution (global_.dpi = 200 by default).

        Typical values: 100–120 for .mov, None (→ global_.dpi) for webp/apng.
        """
        if self.animation.anim_dpi is not None:
            return int(self.animation.anim_dpi)
        return int(self.global_.dpi)


# ── Deep-merge machinery ─────────────────────────────────────────────────────

def _merge_into(obj: Any, data: Dict[str, Any]) -> None:
    """Recursively overwrite dataclass fields on *obj* from nested dict *data*."""
    alias = getattr(type(obj), "_ALIASES", {})
    field_names = {f.name for f in fields(obj)}
    for key, val in data.items():
        attr = alias.get(key, key)
        # block-level alias 'global' -> 'global_'; field-level 'global' inside colours
        if attr not in field_names and (attr + "_") in field_names:
            attr = attr + "_"
        if attr not in field_names:
            logger.debug("Ignoring unknown viz-config key %r (block %s).",
                         key, type(obj).__name__)
            continue
        cur = getattr(obj, attr)
        if is_dataclass(cur) and isinstance(val, dict):
            _merge_into(cur, val)
        else:
            setattr(obj, attr, val)
