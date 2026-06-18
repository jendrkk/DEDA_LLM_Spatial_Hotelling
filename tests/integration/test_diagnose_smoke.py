"""Smoke test: diagnose_calibration_methods.py runs end-to-end."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_diagnose_script_smokes() -> None:
    """Both methods complete and the report prints without error.

    Marked `slow` because it triggers two Bertrand-Nash solves on the
    inner-ring city (~tens of seconds). Skip via `-m "not slow"`.
    """
    if not (_REPO_ROOT / "data" / "processed" / "demand_grid.parquet").exists():
        pytest.skip("Berlin parquet data not present; skipping smoke test.")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "diagnose_calibration_methods.py"),
            "--max-nfev", "4",
        ],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, (
        f"Diagnostic script exited {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "CALIBRATION-METHOD COMPARISON" in result.stdout
    assert "FOC-inversion diagnostics" in result.stdout
    assert "Mean Bertrand-Nash price per chain type" in result.stdout
