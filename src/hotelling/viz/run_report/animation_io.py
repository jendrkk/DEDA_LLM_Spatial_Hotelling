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
    mov_codec: str = "prores_ks",
    mov_bits_per_mb: int = 8000,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Mux RGBA frames into an animation, honouring transparency.

    Returns the actual written path (the suffix is forced to match *fmt*).

    For ``fmt='mov'``: delegates to ``_save_mov`` which encodes via ffmpeg
    rawvideo pipe (ProRes 4444 by default).  FPS is exact and constant —
    stored in the container's rational time base, not per-frame Pillow
    duration fields.  Requires system ffmpeg (``brew install ffmpeg``).

    For ``fmt='webp'`` / ``'apng'`` / ``'gif'``: uses Pillow as before.
    Note that Pillow WebP animation timing is decoder-dependent and may
    exhibit irregular playback on macOS; use ``mov`` for Keynote output.
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

    # ── .mov via ffmpeg (ProRes 4444 or fallback) ────────────────────────────
    if fmt == "mov":
        return _save_mov(frames, path, fps, transparent,
                         codec=mov_codec, bits_per_mb=mov_bits_per_mb,
                         ffmpeg_path=ffmpeg_path)

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


def _save_mov(
    frames: List[np.ndarray],
    path: Path,
    fps: int,
    transparent: bool,
    codec: str = "prores_ks",
    bits_per_mb: int = 8000,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Encode RGBA frames to .mov via ffmpeg rawvideo pipe.

    Primary codec: ProRes 4444 (``prores_ks -profile:v 4444``), which carries
    true per-pixel alpha in ``yuva444p10le`` and is natively supported by
    Keynote. Frame timing is stored in the container's rational time base —
    FPS is exactly constant with zero per-frame rounding drift.

    Fallback chain (automatic, logged as WARNING):
        prores_ks (any failure) → png codec in .mov → APNG via Pillow

    Requires ffmpeg ≥ 4.0 with prores_ks (available in homebrew ffmpeg):
        brew install ffmpeg

    Parameters
    ----------
    frames       : list of (H, W, 4) uint8 RGBA arrays, all same spatial size
                   (caller must have run _pad_to_common first).
    path         : desired output path; suffix will be forced to .mov.
    fps          : constant frame rate embedded in the container time base.
    transparent  : if True → yuva444p10le (ProRes 4444 with alpha channel);
                   if False → yuv444p10le (no alpha; slightly smaller file).
    codec        : 'prores_ks' | 'png' | 'qtrle'.
    bits_per_mb  : prores_ks quality hint (bits per 16×16 macroblock).
                   8000 = standard ProRes 4444. Reduce to 2000–4000 for
                   smaller files with imperceptible quality loss on screen
                   content (choropleths, scatter plots).
    ffmpeg_path  : path to ffmpeg binary, e.g. '/opt/homebrew/bin/ffmpeg'.
    """
    import shutil
    import subprocess

    path = Path(path).with_suffix(".mov")
    path.parent.mkdir(parents=True, exist_ok=True)

    # ── ffmpeg availability check ────────────────────────────────────────────
    if shutil.which(ffmpeg_path) is None:
        logger.warning(
            "_save_mov: ffmpeg not found at '%s'. "
            "Install via 'brew install ffmpeg' for .mov / ProRes output. "
            "Falling back to APNG.",
            ffmpeg_path,
        )
        return _mov_apng_fallback(frames, path, fps)

    h, w = frames[0].shape[:2]

    # ── build codec-specific encoder args ───────────────────────────────────
    pix_fmt_out = "yuva444p10le" if transparent else "yuv444p10le"

    if codec == "prores_ks":
        enc_args: List[str] = [
            "-c:v", "prores_ks",
            "-profile:v", "4444",
            "-pix_fmt", pix_fmt_out,
            "-vendor", "apl0",          # signals Apple-compatible origin to QuickTime
            "-bits_per_mb", str(bits_per_mb),
        ]
    elif codec == "png":
        enc_args = ["-c:v", "png", "-pix_fmt", "rgba"]
    elif codec == "qtrle":
        enc_args = ["-c:v", "qtrle", "-pix_fmt", "argb"]
    else:
        logger.warning("_save_mov: unknown codec '%s'; using prores_ks.", codec)
        enc_args = [
            "-c:v", "prores_ks", "-profile:v", "4444",
            "-pix_fmt", pix_fmt_out, "-vendor", "apl0",
            "-bits_per_mb", str(bits_per_mb),
        ]

    cmd: List[str] = [
        ffmpeg_path, "-y",
        # rawvideo input from stdin
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgba",
        "-r", str(fps),
        "-i", "pipe:0",
        # output
        *enc_args,
        "-movflags", "+faststart",      # moov atom at front (enables streaming)
        str(path),
    ]
    logger.debug("_save_mov ffmpeg cmd: %s", " ".join(cmd))

    # ── pipe frames ──────────────────────────────────────────────────────────
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    _, stderr_bytes = proc.communicate()

    # ── error handling + fallback ────────────────────────────────────────────
    if proc.returncode != 0:
        err_tail = stderr_bytes.decode(errors="replace")[-600:]
        if codec == "prores_ks":
            logger.warning(
                "_save_mov: prores_ks failed (exit %d). "
                "Retrying with png codec (lossless, larger file).\n%s",
                proc.returncode, err_tail,
            )
            return _save_mov(frames, path, fps, transparent,
                             codec="png", bits_per_mb=bits_per_mb,
                             ffmpeg_path=ffmpeg_path)
        else:
            logger.warning(
                "_save_mov: codec '%s' failed (exit %d). Falling back to APNG.\n%s",
                codec, proc.returncode, err_tail,
            )
            return _mov_apng_fallback(frames, path, fps)

    # ── post-encode size check ───────────────────────────────────────────────
    size_mb = path.stat().st_size / 1_048_576
    logger.info(
        "Saved %s (%.1f MB, %d frames, %d fps, codec=%s).",
        path.name, size_mb, len(frames), fps, codec,
    )
    if size_mb > 50.0:
        logger.warning(
            "_save_mov: output %.1f MB exceeds 50 MB target. "
            "Reduce animation.anim_dpi (current uses global_.dpi if unset), "
            "frames.n_frames, or animation.mov_bits_per_mb in run_report.yaml.",
            size_mb,
        )
    return path


def _mov_apng_fallback(frames: List[np.ndarray], path: Path, fps: int) -> Path:
    """Last-resort APNG fallback when ffmpeg is unavailable or all codecs fail."""
    from PIL import Image
    apng_path = path.with_suffix(".apng")
    duration = max(1, int(round(1000.0 / max(fps, 1))))
    imgs = [Image.fromarray(f, mode="RGBA") for f in frames]
    imgs[0].save(apng_path, format="PNG", save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0, disposal=1, blend=0)
    logger.info("Wrote APNG fallback: %s", apng_path)
    return apng_path


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
