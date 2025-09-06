# dominant_colors_simple.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

def rgb_to_hex(rgb):
    """Convert an (R, G, B) tuple to HEX string."""
    r, g, b = map(int, rgb)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def dominant_colors(path, k=3, sample=400, exclude_extremes=False):
    """
    Compute the top-k dominant colors in an image using K-Means clustering.

    Args:
        path (str): Path to the image file.
        k (int): Number of clusters (top colors) to extract.
        sample (int): Max side length used for downsampling to speed up clustering.
        exclude_extremes (bool): If True, filter out near-black and near-white pixels.

    Returns:
        list[dict]: Sorted list of dominant colors with:
                    - 'rgb': (R, G, B)
                    - 'hex': "#RRGGBB"
                    - 'percent': float percentage of pixels
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((sample, sample))
    data = np.asarray(img, dtype=np.uint8).reshape(-1, 3)

    if exclude_extremes:
        # Filter near-white and near-black to avoid bias
        mask = ~((data > 245).all(axis=1) | (data < 10).all(axis=1))
        if mask.any():
            data = data[mask]

    unique_colors = np.unique(data, axis=0)
    eff_k = min(k, len(unique_colors))
    if eff_k == 0:
        return []

    km = KMeans(n_clusters=eff_k, n_init="auto", random_state=0)
    labels = km.fit_predict(data)

    labels_u, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    percents = counts / total * 100.0

    results = []
    for lbl, cnt, pct in zip(labels_u, counts, percents):
        center = km.cluster_centers_[lbl].clip(0, 255).astype(int)
        results.append({
            "rgb": tuple(center.tolist()),
            "hex": rgb_to_hex(center),
            "percent": float(round(pct, 2)),
        })
    results.sort(key=lambda x: x["percent"], reverse=True)
    return results

def render_palette(colors, width=800, height=120, show_labels=True, outfile="palette.png"):
    """
    Render a simple horizontal palette where each swatch width is proportional to its percent.

    Args:
        colors (list[dict]): Output of dominant_colors().
        width (int): Output image width in pixels.
        height (int): Output image height in pixels.
        show_labels (bool): Draw HEX and percent labels on swatches when space allows.
        outfile (str): Path to save the rendered palette image.
    """
    if not colors:
        raise ValueError("No colors to render.")

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load a default font; fall back to PIL's basic bitmap font
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, height // 6))
    except Exception:
        font = ImageFont.load_default()

    # Compute swatch rectangles
    x = 0
    for c in colors:
        w = int(round((c["percent"] / 100.0) * width))
        # Ensure at least 1px width to keep visibility
        w = max(w, 1)
        draw.rectangle([x, 0, x + w, height], fill=c["rgb"])

        if show_labels:
            label = f'{c["hex"]}  {c["percent"]:.1f}%'
            # Choose text color by luminance for readability
            r, g, b = c["rgb"]
            luminance = 0.2126*r + 0.7152*g + 0.0722*b
            text_fill = (0, 0, 0) if luminance > 140 else (255, 255, 255)

            # Only draw label if swatch is wide enough
            text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
            if w >= text_w + 10:  # margin
                draw.text((x + (w - text_w) // 2, (height - text_h) // 2),
                          label, fill=text_fill, font=font)

        x += w

    img.save(outfile)
    return outfile

# Example usage without sys.argv
if __name__ == "__main__":
    image_path = "img\sample3.png" 
    top_colors = dominant_colors(image_path, k=5, exclude_extremes=True)

    print("Top colors:")
    for i, c in enumerate(top_colors, 1):
        print(f"{i}. RGB={c['rgb']}, HEX={c['hex']}, {c['percent']}%")

    out = render_palette(top_colors, width=900, height=140, show_labels=True, outfile=f"{image_path}_palette.png")
    print(f"Saved palette visualization to: {Path(out).resolve()}")
