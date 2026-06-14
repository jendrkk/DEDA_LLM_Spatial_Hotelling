#!/usr/bin/env python
"""Post-hoc store price heterogeneity diagnostic for an existing Phase-0 run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from hotelling.analysis.heterogeneity import store_price_heterogeneity


def _json_ready(result: dict) -> dict:
    payload = {
        key: value
        for key, value in result.items()
        if key != "per_store"
    }
    return payload


def _print_summary(result: dict) -> None:
    print(f"Stores (N):                 {result['n_stores']}")
    print(f"Price mean / std / CV:      {result['price_mean']:.4f} / "
          f"{result['price_std']:.4f} / {result['price_cv']:.4f}")
    print(f"Between / within variance:  {result['price_between_var']:.6f} / "
          f"{result['price_within_var']:.6f}")
    print(f"Between / total ratio:      {result['price_between_total_ratio']:.4f}")
    print("Regression coefficients:")
    for name, value in result["regression_coefficients"].items():
        print(f"  {name:22s} {value: .6f}")
    print(f"Regression R^2:             {result['regression_r2']:.4f}")
    if not result["social_index_available"]:
        print("Social index:               unavailable (omitted from regression)")


def _maybe_plot(result: dict, run_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    per_store = result["per_store"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for chain_type, group in per_store.groupby("chain_type"):
        ax.scatter(
            group["n_local_competitors"],
            group["converged_price"],
            label=chain_type,
            alpha=0.7,
            s=24,
        )
    ax.set_xlabel("Local competitors within radius")
    ax.set_ylabel("Converged price")
    ax.set_title("Store price heterogeneity")
    ax.legend(title="chain_type")
    fig.tight_layout()

    plot_path = run_dir / "heterogeneity.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose store-level price heterogeneity from a Phase-0 run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to results/runs/<timestamp_id>/",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=800.0,
        help="Competitor search radius in metres (default: 800).",
    )
    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.10,
        help="Fraction of recorded rows used for converged prices (default: 0.10).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (_REPO_ROOT / run_dir).resolve()

    result = store_price_heterogeneity(
        run_dir,
        radius_m=args.radius_m,
        tail_frac=args.tail_frac,
    )

    _print_summary(result)

    written_paths: list[Path] = []

    report_path = run_dir / "heterogeneity_report.parquet"
    result["per_store"].to_parquet(report_path, index=False)
    written_paths.append(report_path.resolve())

    summary_path = run_dir / "heterogeneity_summary.json"
    summary_path.write_text(json.dumps(_json_ready(result), indent=2))
    written_paths.append(summary_path.resolve())

    plot_path = _maybe_plot(result, run_dir)
    if plot_path is not None:
        written_paths.append(plot_path.resolve())

    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
