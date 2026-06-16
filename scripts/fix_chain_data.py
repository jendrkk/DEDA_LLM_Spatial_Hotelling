#!/usr/bin/env python
"""Targeted one-off data patch for the chain table — NOT a pipeline stage.

Fixes two data issues in data/processed/ without re-running the GEO pipeline.
Validated against the current parquet contents (494 active stores, 16 170 cells).

(1) Netto merge  [ALWAYS RUN]
    Relabel chain "Netto Marken-Discount" -> "Netto". Both tiers are already
    chain_type="discount", so this is a pure label change: geometry, row count,
    and positional store index are all untouched. travel_times.parquet and any
    converged Q-table stay aligned; `run_strategic.py --from-run` keeps working
    (it simply builds one fewer CEO).
      supermarkets.parquet      : 47 rows relabel -> Netto = 62
      supermarkets_full.parquet : 54 rows relabel -> Netto = 71

(2) Nah & Frisch removal  [ONLY WITH --drop-nah-frisch ; DESTRUCTIVE]
    Removes the single "Nah & Frisch" store (positional index 82 in
    supermarkets.parquet) and remaps travel_times.parquet.to_id so the store
    index space stays contiguous 0..N-2 (drops the 16 170 rows with to_id==82,
    decrements every to_id>82 by one).
      => N changes 494 -> 493, which changes the benchmark-derived price grid
         and INVALIDATES every converged Q-table. You MUST re-run
         scripts/run_baseline.py afterwards (burn-in only; NO LLM/API calls).
         benchmarks_cache.npz is moved aside so it is recomputed.
      => The existing results/runs/20260615_192416_* baseline and
         results/strategic_runs/runs/20260615_225956_* strategic run become
         incompatible with the new 493-store data.

Safety:
  * dry-run by default; nothing is written without --apply
  * every file is copied to <name>.<timestamp>.bak before being overwritten
  * each step asserts that ONLY the intended rows/columns changed; the script
    aborts (writing nothing) if any invariant fails

Usage (from the repository root):
    conda run -n py314 python scripts/fix_chain_data.py                          # dry-run, Netto merge only
    conda run -n py314 python scripts/fix_chain_data.py --apply                  # apply Netto merge
    conda run -n py314 python scripts/fix_chain_data.py --apply --drop-nah-frisch # + remove Nah & Frisch
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
SM = PROCESSED / "supermarkets.parquet"
SM_FULL = PROCESSED / "supermarkets_full.parquet"
TT = PROCESSED / "travel_times.parquet"
BENCH = PROCESSED / "benchmarks_cache.npz"

NETTO_FROM = "Netto Marken-Discount"
NETTO_TO = "Netto"
DROP_CHAIN = "Nah & Frisch"

_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + f".{_STAMP}.bak")
    shutil.copy2(path, bak)
    return bak


def _merge_netto(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    n_from = int((gdf["chain"] == NETTO_FROM).sum())
    n_to0 = int((gdf["chain"] == NETTO_TO).sum())
    ct0 = gdf["chain_type"].value_counts().to_dict()
    geom0 = gdf.geometry.to_wkb().tolist()
    n_rows = len(gdf)

    out = gdf.copy()
    out.loc[out["chain"] == NETTO_FROM, "chain"] = NETTO_TO

    assert len(out) == n_rows, "row count changed during Netto merge"
    assert int((out["chain"] == NETTO_FROM).sum()) == 0, "residual Netto-MD rows"
    assert int((out["chain"] == NETTO_TO).sum()) == n_to0 + n_from, "Netto count mismatch"
    assert out["chain_type"].value_counts().to_dict() == ct0, "chain_type distribution changed"
    assert out.geometry.to_wkb().tolist() == geom0, "geometry changed"

    print(f"  [{label}] relabel {n_from} '{NETTO_FROM}' -> '{NETTO_TO}' "
          f"(Netto: {n_to0} -> {n_to0 + n_from}); rows={n_rows} unchanged; geometry & chain_type intact.")
    return out


def _drop_chain(gdf: gpd.GeoDataFrame, label: str) -> tuple[gpd.GeoDataFrame, int]:
    g = gdf.reset_index(drop=True)
    rows = g.index[g["chain"] == DROP_CHAIN].tolist()
    assert len(rows) == 1, f"expected exactly 1 '{DROP_CHAIN}' row, found {len(rows)}"
    pos = int(rows[0])
    n0 = len(g)
    out = g.drop(index=pos).reset_index(drop=True)
    assert len(out) == n0 - 1 and int((out["chain"] == DROP_CHAIN).sum()) == 0
    print(f"  [{label}] dropped '{DROP_CHAIN}' at positional index {pos}; rows {n0} -> {len(out)}.")
    return out, pos


def _remap_travel_times(tt: pd.DataFrame, drop_idx: int, n_before: int) -> pd.DataFrame:
    to_int = tt["to_id"].astype(int)
    assert sorted(to_int.unique()) == list(range(n_before)), \
        "travel_times.to_id is not contiguous 0..N-1; aborting remap"
    n_drop = int((to_int == drop_idx).sum())
    keep = tt[to_int != drop_idx].copy()
    ki = keep["to_id"].astype(int)
    ki = ki.where(ki < drop_idx, ki - 1)          # ids > drop_idx shift down by one
    keep["to_id"] = ki.astype(str)
    assert sorted(keep["to_id"].astype(int).unique()) == list(range(n_before - 1)), \
        "remapped to_id not contiguous 0..N-2"
    print(f"  [travel_times] dropped {n_drop} rows (to_id=={drop_idx}); decremented to_id>{drop_idx}; "
          f"to_id now 0..{n_before - 2}; rows {len(tt)} -> {len(keep)}.")
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--drop-nah-frisch", action="store_true",
                    help="ALSO remove the single 'Nah & Frisch' store (destructive; requires re-burn-in)")
    args = ap.parse_args()
    print(f"=== fix_chain_data.py [{'APPLY' if args.apply else 'DRY-RUN'}] ===")

    for p in (SM, SM_FULL, TT):
        if not p.exists():
            sys.exit(f"missing required file: {p}")

    sm = gpd.read_parquet(SM)
    smf = gpd.read_parquet(SM_FULL)
    n_before = len(sm)

    print("\n(1) Netto merge")
    sm = _merge_netto(sm, "supermarkets")
    smf = _merge_netto(smf, "supermarkets_full")

    tt_out: pd.DataFrame | None = None
    if args.drop_nah_frisch:
        print("\n(2) Nah & Frisch removal (DESTRUCTIVE)")
        sm, drop_pos = _drop_chain(sm, "supermarkets")     # this index drives the tt remap
        smf, _ = _drop_chain(smf, "supermarkets_full")
        tt_out = _remap_travel_times(pd.read_parquet(TT), drop_pos, n_before)
    else:
        print("\n(2) Nah & Frisch removal: SKIPPED (pass --drop-nah-frisch to enable).")

    if not args.apply:
        print("\nDRY-RUN complete — no files written. Re-run with --apply to commit.")
        return

    print("\nWriting (backup -> overwrite):")
    print("  backup:", _backup(SM)); sm.to_parquet(SM); print("  wrote :", SM)
    print("  backup:", _backup(SM_FULL)); smf.to_parquet(SM_FULL); print("  wrote :", SM_FULL)
    if tt_out is not None:
        print("  backup:", _backup(TT)); tt_out.to_parquet(TT, index=False); print("  wrote :", TT)
        if BENCH.exists():
            moved = _backup(BENCH); BENCH.unlink()
            print(f"  benchmarks_cache.npz moved to {moved} (stale; will recompute on next baseline).")
        print(f"\n*** REQUIRED NEXT STEP (store set changed: N {n_before} -> {n_before - 1}) ***")
        print("    Re-run the SAME run_baseline.py invocation you used to produce")
        print("    results/runs/20260615_192416_* (it regenerates qtable.npz + price grid).")
        print("    Then point run_strategic.py --from-run at the NEW baseline directory.")
        print("    The 20260615 baseline/strategic runs are now incompatible with this data.")
    print("\nDone.")


if __name__ == "__main__":
    main()
