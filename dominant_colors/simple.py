# dominant_colors/simple.py
# This module provides functions to extract dominant colors from images
# and render color palettes. It uses KMeans clustering for color quantization.

from __future__ import annotations
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

# ---------------------------
# Utilities
# ---------------------------

def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def _exclude_mask(arr: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of pixels to KEEP (True) after excluding near-black/near-white.
    Criteria: all channels <= 25 (near-black) OR all >= 230 (near-white) are excluded.
    """
    near_black = np.all(arr <= 25, axis=1)
    near_white = np.all(arr >= 230, axis=1)
    keep = ~(near_black | near_white)
    return keep

def _load_image_pixels(path: str | Path, sample: int) -> np.ndarray:
    """
    Load image and downscale such that the longest side == sample (if needed).
    Returns pixels as Nx3 RGB uint8.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if sample and max(w, h) > sample:
        scale = sample / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
    return arr

def _kmeans_colors(pixels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run KMeans on pixels (Nx3). Returns (centers uint8, proportions float32).
    """
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=np.float32)

    # If too many pixels, random subsample for speed
    if pixels.shape[0] > 100_000:
        idx = np.random.choice(pixels.shape[0], 100_000, replace=False)
        pixels_fit = pixels[idx]
    else:
        pixels_fit = pixels

    # Edge case: less unique colors than k
    uniq = np.unique(pixels_fit, axis=0)
    k_eff = min(k, max(1, uniq.shape[0]))

    km = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
    labels = km.fit_predict(pixels_fit.astype(np.float32))
    centers = km.cluster_centers_.astype(np.uint8)

    # Compute proportions on the FULL set (assign nearest center)
    # This avoids bias from subsampling.
    # If huge, project on a subset for speed but keep accuracy reasonable.
    proj = pixels
    if pixels.shape[0] > 300_000:
        j = np.random.choice(pixels.shape[0], 300_000, replace=False)
        proj = pixels[j]

    # Nearest center via L2
    diffs = proj[:, None, :].astype(np.int32) - centers[None, :, :].astype(np.int32)
    d2 = np.sum(diffs * diffs, axis=2)
    proj_labels = np.argmin(d2, axis=1)

    counts = np.bincount(proj_labels, minlength=centers.shape[0]).astype(np.float32)
    props = counts / counts.sum() if counts.sum() > 0 else counts
    return centers, props

# ---------------------------
# Public API (used by app.py)
# ---------------------------

def dominant_colors(path: str | Path, k: int = 5, sample: int = 400, exclude_extremes: bool = True) -> List[Dict]:
    """
    Compute top-k dominant colors from an image.
    Returns: list of dicts [{"hex": str, "rgb": (r,g,b), "percent": float}, ...] sorted by percent desc.
    """
    pixels = _load_image_pixels(path, sample)
    if exclude_extremes and pixels.size:
        keep = _exclude_mask(pixels)
        pixels = pixels[keep]

    centers, props = _kmeans_colors(pixels, k)
    # Sort by proportion descending
    order = np.argsort(props)[::-1]
    centers = centers[order]
    props = props[order]

    out = []
    for c, p in zip(centers, props):
        rgb = (int(c[0]), int(c[1]), int(c[2]))
        out.append({
            "hex": _rgb_to_hex(rgb),
            "rgb": rgb,
            "percent": round(float(p) * 100.0, 2),
        })
    return out

def render_palette(
    colors: List[Dict],
    width: int = 900,
    height: int = 140,
    show_labels: bool = True,
    outfile: str | Path | None = None,
) -> Image.Image:
    """
    Render a horizontal palette image. If outfile is given, saves PNG there.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Bars
    x = 0
    total = sum(max(0.0, c.get("percent", 0.0)) for c in colors) or 100.0
    for c in colors:
        pct = max(0.0, float(c.get("percent", 0.0)))
        w = int(round((pct / total) * width))
        rgb = tuple(int(v) for v in c["rgb"])
        draw.rectangle([x, 0, x + w, height], fill=rgb)
        x += w

    # Labels
    if show_labels and colors:
        try:
            # Using a default PIL font keeps things portable on HF Spaces
            font = ImageFont.load_default()
        except Exception:
            font = None
        pad = 8
        x = 0
        for c in colors:
            pct = max(0.0, float(c.get("percent", 0.0)))
            w = int(round((pct / total) * width))
            label = f'{c["hex"]} • {pct:.1f}%'
            # Choose contrasting text color
            r, g, b = c["rgb"]
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
            # Draw within each bar if wide enough
            if w >= 90:
                draw.text((x + pad, height // 2 - 6), label, fill=text_color, font=font)
            x += w

    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        img.save(outfile, format="PNG")
    return img
