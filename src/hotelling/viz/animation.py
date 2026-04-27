"""GIF animation of training trajectory.

Responsibility: assemble a sequence of matplotlib figure snapshots into an
animated GIF file using imageio.

Public API: animate_training

Key dependencies: imageio, matplotlib, pathlib

References:
    imageio https://imageio.readthedocs.io/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np


def animate_training(
    frame_snapshots: List[Any],
    output_path: Path,
    fps: int = 10,
) -> Path:
    """Save a list of matplotlib figures as an animated GIF.

    Each figure is rasterized to a NumPy array and written as one frame.
    The figures are NOT closed by this function; the caller is responsible
    for closing them if memory is a concern.

    Parameters
    ----------
    frame_snapshots : list of matplotlib.figure.Figure objects, one per frame.
        Figures should already be fully rendered (all artists drawn).
    output_path : destination file path (should end in .gif)
    fps : frames per second for the animation

    Returns
    -------
    Path to the written GIF file
    """
    raise NotImplementedError
