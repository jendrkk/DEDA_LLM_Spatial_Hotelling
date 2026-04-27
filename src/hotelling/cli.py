"""Typer CLI for the hotelling package.

Responsibility: provide a command-line interface for running simulations,
launching parameter sweeps, and exporting results without writing Python code.

Public API: app (Typer CLI application)

Key dependencies: typer, pathlib, hotelling.simulation.runner

Usage::

    hotelling train --config configs/config.yaml
    hotelling sweep --config configs/sweep/alpha_beta.yaml --jobs 4
    hotelling export --run-dir results/runs/ --out results/summary.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print(
        "typer not installed. Run: pip install 'hotelling[cli]'",
        file=sys.stderr,
    )
    sys.exit(1)

app = typer.Typer(
    name="hotelling",
    help="LLM-Driven 2-D Spatial Hotelling Simulation Toolkit",
    add_completion=False,
)


@app.command()
def train(
    config: Path = typer.Option(
        Path("configs/config.yaml"),
        "--config",
        "-c",
        help="Path to the Hydra config YAML file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Override random seed."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Override results output directory."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Run a single simulation training session from a config file."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        typer.echo("pyyaml not installed. Run: pip install pyyaml", err=True)
        raise typer.Exit(1)

    with open(config) as fh:
        cfg = yaml.safe_load(fh)

    if seed is not None:
        cfg["seed"] = seed
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    if verbose:
        cfg.setdefault("logging", {})["level"] = "DEBUG"

    from hotelling.simulation.runner import run_single_session  # noqa: PLC0415

    result = run_single_session(cfg)
    typer.echo(f"Run {result.get('run_id', '?')} complete in {result.get('elapsed_s', 0):.1f}s")
    typer.echo(
        f"  profit_gain={result.get('profit_gain', float('nan')):.4f}  "
        f"n_steps={result.get('n_steps', 0)}"
    )


@app.command()
def sweep(
    config: Path = typer.Option(
        Path("configs/sweep/alpha_beta.yaml"),
        "--config",
        "-c",
        help="Path to the sweep config YAML file.",
        exists=True,
    ),
    jobs: int = typer.Option(-1, "--jobs", "-j", help="Number of parallel workers (-1=all CPUs)."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Directory to write run Parquet files."
    ),
) -> None:
    """Run a parameter sweep defined in a configs/sweep/ YAML file."""
    config_dir = config.parent.parent
    sweep_name = config.stem

    from hotelling.simulation.runner import run_sweep  # noqa: PLC0415

    results = run_sweep(
        config_dir=config_dir,
        sweep_config_name=sweep_name,
        n_jobs=jobs,
        output_dir=output_dir,
    )
    typer.echo(f"Sweep complete: {len(results)} runs finished.")


@app.command()
def export(
    run_dir: Path = typer.Argument(..., help="Directory containing *.parquet run files."),
    out: Path = typer.Option(
        Path("results/summary.parquet"),
        "--out",
        "-o",
        help="Output path for the merged summary Parquet file.",
    ),
) -> None:
    """Merge all per-run Parquet files in run_dir into a single summary file."""
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        typer.echo("pandas not installed. Run: pip install pandas pyarrow", err=True)
        raise typer.Exit(1)

    parquet_files = list(run_dir.glob("*.parquet"))
    if not parquet_files:
        typer.echo(f"No .parquet files found in {run_dir}", err=True)
        raise typer.Exit(1)

    dfs = [pd.read_parquet(p) for p in parquet_files]
    merged = pd.concat(dfs, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    typer.echo(f"Exported {len(parquet_files)} runs ({len(merged):,} rows) to {out}")


def main() -> None:
    """Entry point for the hotelling CLI."""
    app()


if __name__ == "__main__":
    main()
