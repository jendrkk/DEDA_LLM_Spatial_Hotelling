#!/usr/bin/env python
"""CLI entry point for the run-report visualisation pipeline.

Examples
--------
    conda activate py314
    python scripts/visualize_run.py results/runs/20260625_211444_811193ee
    python scripts/visualize_run.py 20260625_211444_811193ee --config configs/viz/run_report.yaml
    python scripts/visualize_run.py strategic_runs/runs/<id> --only price_trajectory store_price_animation
    python scripts/visualize_run.py <run> --no-transparent --format gif --n-frames 120

A run may be given as an absolute path, a path relative to the repo root, or a
bare run id (the resolver searches results/runs and strategic_runs/runs).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# make 'hotelling' importable when run from anywhere
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hotelling.viz.run_report import VizConfig, run_pipeline  # noqa: E402
from hotelling.viz.run_report.config import ProduceCfg  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "viz" / "run_report.yaml"


def _apply_filters(cfg: VizConfig, only, skip) -> None:
    flags = [f.name for f in ProduceCfg.__dataclass_fields__.values()]
    if only:
        for name in flags:
            setattr(cfg.produce, name, name in set(only))
        unknown = set(only) - set(flags)
        if unknown:
            logging.warning("Unknown --only keys ignored: %s", ", ".join(sorted(unknown)))
    if skip:
        for name in set(skip):
            if name in flags:
                setattr(cfg.produce, name, False)
            else:
                logging.warning("Unknown --skip key ignored: %s", name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the full visualisation report for a run.")
    ap.add_argument("run", help="run id, repo-relative path, or absolute run directory")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG),
                    help=f"viz config YAML (default: {_DEFAULT_CONFIG})")
    ap.add_argument("--only", nargs="+", metavar="ARTEFACT",
                    help="produce ONLY these produce.* artefacts")
    ap.add_argument("--skip", nargs="+", metavar="ARTEFACT",
                    help="skip these produce.* artefacts")
    ap.add_argument("--transparent", dest="transparent", action="store_true", default=None,
                    help="force transparent backgrounds")
    ap.add_argument("--no-transparent", dest="transparent", action="store_false",
                    help="force opaque (white) backgrounds")
    ap.add_argument("--use-latex", dest="use_latex", action="store_true", default=None)
    ap.add_argument("--no-latex", dest="use_latex", action="store_false")
    ap.add_argument("--format", choices=["webp", "apng", "gif", "mp4"],
                    help="override animation container format")
    ap.add_argument("--n-frames", type=int, help="override number of animation frames")
    ap.add_argument("--dpi", type=int, help="override figure DPI")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg_path = args.config if Path(args.config).exists() else None
    if cfg_path is None and args.config:
        logging.warning("Config %s not found; using built-in defaults.", args.config)
    cfg = VizConfig.load(cfg_path)

    if args.transparent is not None:
        cfg.global_.transparent = args.transparent
    if args.use_latex is not None:
        cfg.global_.use_latex = args.use_latex
    if args.format is not None:
        cfg.animation.format = args.format
    if args.n_frames is not None:
        cfg.frames.n_frames = args.n_frames
    if args.dpi is not None:
        cfg.global_.dpi = args.dpi
    _apply_filters(cfg, args.only, args.skip)

    written = run_pipeline(args.run, cfg)
    for pth in written:
        print(f"  {pth}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
