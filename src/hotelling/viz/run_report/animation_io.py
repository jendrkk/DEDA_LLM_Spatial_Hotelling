"""Animation encoding + basemap helpers for the run-report pipeline.

Transparency engine choice
--------------------------
Animated **GIF** carries only 1-bit transparency (a single fully-transparent
palette index → hard fringing), and **MP4** has none.  For genuine per-pixel
alpha the pipeline assembles frames as **WebP** (preferred: wide support,
lossless alpha, animation) or **APNG** (PNG-native alpha).  ``save_animation``
therefore renders each matplotlib figure to an RGBA buffer
(``savefig(transparent=True)``) and muxes with Pillow.  GIF/MP4 remain available
for the opaque case.

Frame-size stability
--------------------
Frames are written WITHOUT ``bbox_inches="tight"`` so every frame shares the
exact canvas size (a hard requirement for muxing).  Callers must therefore add
all colorbars/legends ONCE before the loop and only update artist data per
frame, and should reserve right-margin space via ``subplots_adjust``.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── figure -> RGBA ───────────────────────────────────────────────────────────

def fig_to_rgba(fig, dpi: int, transparent: bool) -> np.ndarray:
    """Rasterise a fully-drawn figure to an ``(H, W, 4)`` uint8 RGBA array."""
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=transparent,
                facecolor=("none" if transparent else "white"))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGBA"))


def _pad_to_common(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Pad frames with transparent pixels to a common (max H, max W)."""
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    out = []
    for f in frames:
        if f.shape[0] == h and f.shape[1] == w:
            out.append(f)
            continue
        pad = np.zeros((h, w, 4), dtype=np.uint8)
        pad[: f.shape[0], : f.shape[1]] = f
        out.append(pad)
    return out


# ── animation muxing ─────────────────────────────────────────────────────────

def save_animation(
    frames: List[np.ndarray], path: Path, fps: int, fmt: str,
    transparent: bool, loop: int = 0, lossless_webp: bool = True,
) -> Path:
    """Mux RGBA frames into an animation, honouring transparency.

    Returns the actual written path (the suffix is forced to match *fmt*).
    """
    from PIL import Image

    path = Path(path)
    fmt = fmt.lower()
    if transparent and fmt in ("gif", "mp4"):
        logger.info("transparent=True with fmt=%s cannot carry alpha; using webp.", fmt)
        fmt = "webp"
    path = path.with_suffix("." + fmt)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames = _pad_to_common(frames)
    duration = max(1, int(round(1000.0 / max(fps, 1))))

    if fmt == "mp4":
        return _save_mp4(frames, path, fps)

    imgs = [Image.fromarray(f, mode="RGBA") for f in frames]

    if fmt == "webp":
        imgs[0].save(path, format="WEBP", save_all=True, append_images=imgs[1:],
                     duration=duration, loop=loop, lossless=lossless_webp,
                     allow_mixed=not lossless_webp)
        return path

    if fmt == "apng":
        imgs[0].save(path, format="PNG", save_all=True, append_images=imgs[1:],
                     duration=duration, loop=loop, disposal=1, blend=0)
        return path

    # GIF
    if transparent:
        pal = [_rgba_to_gif_frame(f) for f in frames]
        pal[0].save(path, format="GIF", save_all=True, append_images=pal[1:],
                    duration=duration, loop=loop, transparency=255, disposal=2)
    else:
        rgb = [Image.fromarray(_composite_white(f)) for f in frames]
        rgb[0].save(path, format="GIF", save_all=True, append_images=rgb[1:],
                    duration=duration, loop=loop)
    return path


def _composite_white(rgba: np.ndarray) -> np.ndarray:
    a = rgba[:, :, 3:4].astype(np.float64) / 255.0
    rgb = rgba[:, :, :3].astype(np.float64)
    white = 255.0 * (1.0 - a)
    return (rgb * a + white).astype(np.uint8)


def _rgba_to_gif_frame(rgba: np.ndarray):
    """RGBA -> P-mode GIF frame with palette index 255 reserved as transparent."""
    from PIL import Image
    alpha = rgba[:, :, 3]
    rgb = Image.fromarray(rgba[:, :, :3], mode="RGB")
    p = rgb.convert("P", palette=Image.ADAPTIVE, colors=255)
    mask = alpha < 128
    arr = np.asarray(p).copy()
    arr[mask] = 255
    out = Image.fromarray(arr, mode="P")
    out.putpalette(p.getpalette())
    return out


def _save_mp4(frames: List[np.ndarray], path: Path, fps: int) -> Path:
    try:
        import imageio.v3 as iio
        rgb = [_composite_white(f) for f in frames]
        iio.imwrite(str(path), np.stack(rgb), fps=fps, codec="libx264")
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("MP4 export failed (%s); falling back to webp.", exc)
        from PIL import Image
        path = path.with_suffix(".webp")
        imgs = [Image.fromarray(f, mode="RGBA") for f in frames]
        imgs[0].save(path, format="WEBP", save_all=True, append_images=imgs[1:],
                     duration=max(1, int(1000 / max(fps, 1))), loop=0, lossless=True)
        return path


# ── OSM basemap ──────────────────────────────────────────────────────────────

def resolve_provider(provider_path: str):
    """Resolve a dotted contextily provider path, e.g. 'OpenStreetMap.Mapnik'."""
    import contextily as ctx
    node: Any = ctx.providers
    for part in provider_path.split("."):
        node = getattr(node, part)
    return node


def add_osm_basemap(ax, extent_3857, provider_path: str, alpha: float = 1.0,
                    zoom="auto") -> None:
    """Add an OSM basemap to *ax*, locking the axis extent before & after.

    ``extent_3857`` = (minx, miny, maxx, maxy) in EPSG:3857.
    """
    import contextily as ctx
    minx, miny, maxx, maxy = extent_3857
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ctx.add_basemap(ax, source=resolve_provider(provider_path), zoom=zoom,
                    reset_extent=False, alpha=alpha)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()
