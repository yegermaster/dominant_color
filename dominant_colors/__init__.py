# __init__.py
# This file makes it easier to import functions from the dominant_colors package.

from .simple import dominant_colors, render_palette
from .video import sample_video_pixels, dominant_colors_from_pixels
from .video import render_palette as render_palette_video

__all__ = [
    "dominant_colors",
    "render_palette",
    "sample_video_pixels",
    "dominant_colors_from_pixels",
    "render_palette_video",
]
