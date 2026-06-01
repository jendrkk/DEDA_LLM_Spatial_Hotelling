"""GIF animation of training trajectory.

Responsibility: assemble a sequence of matplotlib figure snapshots into an
animated GIF file using imageio.  Also re-exports
:func:`~hotelling.viz.spatial_map.animate_market` for spatial run animations.

Public API: animate_training, animate_market

Key dependencies: imageio, matplotlib, pathlib

References:
    imageio https://imageio.readthedocs.io/.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, List

import numpy as np


def animate_training(
    frame_snapshots: List[Any],
    output_path: Path,
    fps: int = 10,
) -> Path:
    """Save a list of matplotlib figures as an animated GIF.

    Each figure is rasterized to a NumPy array via an in-memory PNG buffer
    and written as one frame using imageio.  The figures are **not** closed
    by this function; the caller is responsible for closing them to free
    memory.

    Parameters
    ----------
    frame_snapshots : list of matplotlib.figure.Figure
        Figures should already be fully rendered (all artists drawn).
    output_path : Path
        Destination file path.  Conventionally ends in ``.gif``.
    fps : int, optional
        Frames per second for the animation.  Default 10.

    Returns
    -------
    Path
        The path to the written GIF file (same as *output_path*).

    Raises
    ------
    ImportError
        If ``imageio`` is not installed.  Install with
        ``pip install 'hotelling[viz]'``.
    ValueError
        If *frame_snapshots* is empty.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> figs = [plt.figure() for _ in range(5)]
    >>> animate_training(figs, Path("training.gif"), fps=5)
    PosixPath('training.gif')
    """
    if not frame_snapshots:
        raise ValueError("frame_snapshots must contain at least one figure.")

    try:
        import imageio.v3 as iio
        _v3 = True
    except ImportError:
        try:
            import imageio as iio  # type: ignore[no-redef]
            _v3 = False
        except ImportError as exc:
            raise ImportError(
                "imageio is required for GIF animation. "
                "Install with: pip install 'hotelling[viz]'"
            ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: List[np.ndarray] = []
    for fig in frame_snapshots:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
        buf.seek(0)
        if _v3:
            frame = iio.imread(buf, extension=".png")
        else:
            frame = iio.imread(buf, format="png")
        frames.append(frame)

    if _v3:
        iio.imwrite(
            str(output_path),
            frames,
            extension=".gif",
            loop=0,
            fps=fps,
        )
    else:
        iio.mimsave(str(output_path), frames, fps=fps)

    return output_path


# ---------------------------------------------------------------------------
# Re-export spatial run animation under its canonical name
# ---------------------------------------------------------------------------

def animate_market(*args: Any, **kwargs: Any) -> Path:
    """Delegate to :func:`hotelling.viz.spatial_map.animate_market`.

    See that function for the full signature and documentation.  This
    re-export exists so callers can import either from
    ``hotelling.viz.animation`` or ``hotelling.viz.spatial_map``.
    """
    from hotelling.viz.spatial_map import animate_market as _animate_market

    return _animate_market(*args, **kwargs)
