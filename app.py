# app.py
# This is a Gradio web app for extracting dominant colors from images and videos.
# It uses the dominant_colors package for analysis.

import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

# Local imports: functions for image and video dominant color extraction
from ui import build_demo
from dominant_colors.simple import dominant_colors as img_dominant_colors, render_palette as render_palette_img
from dominant_colors.video import (
    sample_video_pixels,
    dominant_colors_from_pixels,
    render_palette as render_palette_vid,
)


# Global constants for palette rendering dimensions
PALETTE_WIDTH = 900
PALETTE_HEIGHT = 140


def to_dataframe(colors):
    """
    Convert list of color dictionaries to a pandas DataFrame.

    Args:
        colors (list[dict]): Each dict contains keys 'hex', 'rgb', 'percent'.

    Returns:
        pd.DataFrame: Structured table for Gradio output.
    """
    if not colors:
        return pd.DataFrame(columns=["hex", "rgb", "percent"])
    rows = [{"hex": c["hex"], "rgb": c["rgb"], "percent": c["percent"]} for c in colors]
    return pd.DataFrame(rows)


def process_image(image_path, k, sample, exclude_extremes, show_labels):
    """
    Analyze an uploaded image and extract top dominant colors.

    Args:
        image_path (str): Path to uploaded image.
        k (int): Number of clusters (dominant colors).
        sample (int): Max side length for downscaling.
        exclude_extremes (bool): Whether to remove near-white/near-black.
        show_labels (bool): Display HEX and percentage on palette.

    Returns:
        (str, DataFrame): Path to rendered palette PNG and color table.
    """
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
    """
    Analyze an uploaded video and extract global dominant colors.

    Args:
        video_path (str): Path to uploaded video.
        k (int): Number of clusters (dominant colors).
        frame_step (int): Process every Nth frame.
        resize (int): Downscale longest side for performance.
        limit_frames (int): Safety cap on number of frames.
        exclude_extremes (bool): Remove near-white/near-black.
        show_labels (bool): Display HEX and percentage on palette.

    Returns:
        (str, DataFrame): Path to rendered palette PNG and color table.
    """
    if not video_path:
        return None, to_dataframe([])

    # Ensure minimum safe parameter values
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


# Build Gradio interface

demo_title = "Dominant Colors – Image & Video"
demo_description = (
    "# Dominant Colors – Image & Video\n"
    "Upload an image **or** a video to extract the top dominant colors. "
    "You will receive both a palette visualization and a table of HEX/RGB values with percentages."
)

with gr.Blocks(title=demo_title, theme=gr.themes.Soft()) as demo:
    gr.Markdown(demo_description)

    with gr.Tabs():
        # Image analysis tab
        with gr.Tab("Image"):
            with gr.Row():
                image_in = gr.Image(type="filepath", label="Upload image", sources=["upload"], height=300)
            with gr.Accordion("Parameters", open=False):
                k_img = gr.Slider(1, 12, value=5, step=1, label="Top K colors")
                sample_img = gr.Slider(100, 1024, value=400, step=10, label="Downsample max side (px)")
                excl_img = gr.Checkbox(value=True, label="Exclude near-black/near-white")
                labels_img = gr.Checkbox(value=True, label="Show labels on palette")
            run_img = gr.Button("Analyze Image")
            palette_img = gr.Image(type="filepath", label="Palette", interactive=False)
            table_img = gr.Dataframe(headers=["hex", "rgb", "percent"], label="Colors", interactive=False)

            run_img.click(
                fn=process_image,
                inputs=[image_in, k_img, sample_img, excl_img, labels_img],
                outputs=[palette_img, table_img],
                concurrency_limit=2,
            )

        # Video analysis tab
        with gr.Tab("Video"):
            with gr.Row():
                video_in = gr.Video(label="Upload video", sources=["upload"], height=300)
            with gr.Accordion("Parameters", open=False):
                k_vid = gr.Slider(1, 12, value=5, step=1, label="Top K colors")
                step_vid = gr.Slider(1, 60, value=15, step=1, label="Frame step (every Nth frame)")
                resize_vid = gr.Slider(64, 720, value=256, step=16, label="Downsample max side (px)")
                limit_vid = gr.Slider(10, 1000, value=300, step=10, label="Max sampled frames")
                excl_vid = gr.Checkbox(value=True, label="Exclude near-black/near-white")
                labels_vid = gr.Checkbox(value=True, label="Show labels on palette")
            run_vid = gr.Button("Analyze Video")
            palette_vid = gr.Image(type="filepath", label="Palette", interactive=False)
            table_vid = gr.Dataframe(headers=["hex", "rgb", "percent"], label="Colors", interactive=False)

            run_vid.click(
                fn=process_video,
                inputs=[video_in, k_vid, step_vid, resize_vid, limit_vid, excl_vid, labels_vid],
                outputs=[palette_vid, table_vid],
                concurrency_limit=2,
            )

    gr.Markdown(
        """
**Notes**
- Results depend on K, sampling, and whether extremes are excluded.
- Video analysis uses frame sampling for a global palette; no per-frame breakdown.
- Processing runs on CPU and is suitable for free hosting tiers.
"""
    )

if __name__ == "__main__":
    demo = build_demo(process_image, process_video)
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))