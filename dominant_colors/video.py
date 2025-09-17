# dominant_colors/video.py
# This module provides functions to extract dominant colors from videos
# using KMeans clustering for color quantization.

from __future__ import annotations
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

# Reuse helpers (duplicated locally to avoid cross-file imports for simplicity)

def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def _exclude_mask(arr: np.ndarray) -> np.ndarray:
    near_black = np.all(arr <= 25, axis=1)
    near_white = np.all(arr >= 230, axis=1)
    return ~(near_black | near_white)

def _kmeans_colors(pixels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=np.float32)

    if pixels.shape[0] > 150_000:
        idx = np.random.choice(pixels.shape[0], 150_000, replace=False)
        pixels_fit = pixels[idx]
    else:
        pixels_fit = pixels

    uniq = np.unique(pixels_fit, axis=0)
    k_eff = min(k, max(1, uniq.shape[0]))

    km = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
    labels = km.fit_predict(pixels_fit.astype(np.float32))
    centers = km.cluster_centers_.astype(np.uint8)

    proj = pixels
    if pixels.shape[0] > 400_000:
        j = np.random.choice(pixels.shape[0], 400_000, replace=False)
        proj = pixels[j]

    diffs = proj[:, None, :].astype(np.int32) - centers[None, :, :].astype(np.int32)
    d2 = np.sum(diffs * diffs, axis=2)
    proj_labels = np.argmin(d2, axis=1)

    counts = np.bincount(proj_labels, minlength=centers.shape[0]).astype(np.float32)
    props = counts / counts.sum() if counts.sum() > 0 else counts
    return centers, props

# --------------- Public API (used by app.py) ----------------

def sample_video_pixels(
    path: str | Path,
    frame_step: int = 15,
    resize: int = 256,
    limit_frames: int = 300,
    exclude_extremes: bool = True,
) -> np.ndarray:
    """
    Sample RGB pixels from a video.
    - Reads every `frame_step` frame.
    - Resizes such that the longest side == `resize`.
    - Stops after `limit_frames` sampled frames.
    Returns Nx3 uint8 array (RGB).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.empty((0, 3), dtype=np.uint8)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sampled = 0
    frame_idx = 0
    pixels_list: List[np.ndarray] = []

    while True:
        ret = cap.grab()  # faster than read() every time
        if not ret:
            break

        if frame_idx % frame_step == 0:
            ret2, frame = cap.retrieve()
            if not ret2:
                break

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w = frame.shape[:2]
            if max(w, h) > resize:
                scale = resize / float(max(w, h))
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            arr = frame.reshape(-1, 3)
            if exclude_extremes and arr.size:
                keep = _exclude_mask(arr)
                arr = arr[keep]

            if arr.size:
                pixels_list.append(arr)

            sampled += 1
            if sampled >= limit_frames:
                break

        frame_idx += 1

    cap.release()
    if not pixels_list:
        return np.empty((0, 3), dtype=np.uint8)
    return np.vstack(pixels_list).astype(np.uint8)

def dominant_colors_from_pixels(pixels: np.ndarray, k: int = 5) -> List[Dict]:
    """
    KMeans over pre-sampled video pixels.
    Returns list of dicts [{"hex","rgb","percent"}, ...] sorted by percent desc.
    """
    centers, props = _kmeans_colors(pixels, k)
    order = np.argsort(props)[::-1]
    centers, props = centers[order], props[order]

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
):
    """
    Same renderer as in simple.py; duplicated here with no external dependency.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    x = 0
    total = sum(max(0.0, c.get("percent", 0.0)) for c in colors) or 100.0
    for c in colors:
        pct = max(0.0, float(c.get("percent", 0.0)))
        w = int(round((pct / total) * width))
        rgb = tuple(int(v) for v in c["rgb"])
        draw.rectangle([x, 0, x + w, height], fill=rgb)
        x += w

    if show_labels and colors:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        pad = 8
        x = 0
        for c in colors:
            pct = max(0.0, float(c.get("percent", 0.0)))
            w = int(round((pct / total) * width))
            label = f'{c["hex"]} • {pct:.1f}%'
            r, g, b = c["rgb"]
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
            if w >= 90:
                draw.text((x + pad, height // 2 - 6), label, fill=text_color, font=font)
            x += w

    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        img.save(outfile, format="PNG")
    return img
