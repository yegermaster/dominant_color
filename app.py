# app.py
# Gradio app for extracting dominant colors from images and videos.

import os
import tempfile
from pathlib import Path
import pandas as pd

# --- load env (HF_TOKEN, PORT, etc.) ---
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

HF_TOKEN = os.getenv("HF_TOKEN")  # optional for any HF calls

import gradio as gr

# Local imports
from ui import build_demo
from dominant_colors.simple import dominant_colors as img_dominant_colors, render_palette as render_palette_img
from dominant_colors.video import (
    sample_video_pixels,
    dominant_colors_from_pixels,
    render_palette as render_palette_vid,
)

PALETTE_WIDTH = 900
PALETTE_HEIGHT = 140


def to_dataframe(colors):
    if not colors:
        return pd.DataFrame(columns=["hex", "rgb", "percent"])
    rows = [{"hex": c["hex"], "rgb": c["rgb"], "percent": c["percent"]} for c in colors]
    return pd.DataFrame(rows)


def process_image(image_path, k, sample, exclude_extremes, show_labels):
    if not image_path:
        return None, to_dataframe([])

    colors = img_dominant_colors(
        path=image_path,
        k=int(k),
        sample=int(sample),
        exclude_extremes=bool(exclude_extremes),
    )
    if not colors:
        return None, to_dataframe([])

    out_path = os.path.join(tempfile.gettempdir(), f"palette_image_{Path(image_path).stem}.png")
    render_palette_img(
        colors,
        width=PALETTE_WIDTH,
        height=PALETTE_HEIGHT,
        show_labels=bool(show_labels),
        outfile=out_path,
    )
    return out_path, to_dataframe(colors)


def process_video(video_path, k, frame_step, resize, limit_frames, exclude_extremes, show_labels):
    if not video_path:
        return None, to_dataframe([])

    frame_step = max(1, int(frame_step))
    resize = max(64, int(resize))
    limit_frames = max(1, int(limit_frames))

    pixels = sample_video_pixels(
        path=video_path,
        frame_step=frame_step,
        resize=resize,
        limit_frames=limit_frames,
        exclude_extremes=bool(exclude_extremes),
    )
    colors = dominant_colors_from_pixels(pixels, k=int(k))
    if not colors:
        return None, to_dataframe([])

    out_path = os.path.join(tempfile.gettempdir(), f"palette_video_{Path(video_path).stem}.png")
    render_palette_vid(
        colors,
        width=PALETTE_WIDTH,
        height=PALETTE_HEIGHT,
        show_labels=bool(show_labels),
        outfile=out_path,
    )
    return out_path, to_dataframe(colors)


if __name__ == "__main__":
    demo = build_demo(process_image, process_video)  # your ui.py should wire the widgets ↔ functions
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
