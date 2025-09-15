# dominant_colors_video.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
import cv2
from pathlib import Path

def rgb_to_hex(rgb):
    r, g, b = map(int, rgb)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def sample_video_pixels(path, frame_step=15, resize=256, limit_frames=300, exclude_extremes=False):
    """
    Grab pixels from every `frame_step` frame, downscale to `resize` (max side),
    optionally filter near-black/near-white, and return a (N,3) uint8 array.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    pixels = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    grabbed = 0
    idx = 0
    while True:
        ret = cap.grab()  # cheap grab
        if not ret:
            break
        if idx % frame_step == 0:
            ret2, frame = cap.retrieve()
            if not ret2:
                break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            scale = resize / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            data = frame.reshape(-1, 3)

            if exclude_extremes:
                mask = ~((data > 245).all(axis=1) | (data < 10).all(axis=1))
                if mask.any():
                    data = data[mask]

            # To cap memory, randomly subsample up to ~50k pixels per sampled frame
            if data.shape[0] > 50_000:
                sel = np.random.choice(data.shape[0], 50_000, replace=False)
                data = data[sel]

            pixels.append(data.astype(np.uint8))
            grabbed += 1
            if grabbed >= limit_frames:
                break
        idx += 1

    cap.release()
    if not pixels:
        return np.empty((0,3), dtype=np.uint8)
    return np.concatenate(pixels, axis=0)

def dominant_colors_from_pixels(data, k=5):
    """
    K-Means on provided pixel array -> list of dicts (rgb, hex, percent).
    """
    if data.size == 0:
        return []
    unique_colors = np.unique(data, axis=0)
    eff_k = min(k, len(unique_colors))
    if eff_k == 0:
        return []
    km = KMeans(n_clusters=eff_k, n_init="auto", random_state=0)
    labels = km.fit_predict(data)
    labels_u, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    results = []
    for lbl, cnt in zip(labels_u, counts):
        center = km.cluster_centers_[lbl].clip(0, 255).astype(int)
        pct = float(round((cnt / total) * 100.0, 2))
        results.append({"rgb": tuple(center.tolist()), "hex": rgb_to_hex(center), "percent": pct})
    results.sort(key=lambda x: x["percent"], reverse=True)
    return results

def render_palette(colors, width=900, height=140, show_labels=True, outfile="palette_video.png"):
    if not colors:
        raise ValueError("No colors to render.")
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, height // 6))
    except Exception:
        font = ImageFont.load_default()

    x = 0
    for c in colors:
        w = int(round((c["percent"] / 100.0) * width))
        w = max(w, 1)
        draw.rectangle([x, 0, x + w, height], fill=c["rgb"])

        if show_labels:
            label = f'{c["hex"]}  {c["percent"]:.1f}%'
            r, g, b = c["rgb"]
            luminance = 0.2126*r + 0.7152*g + 0.0722*b
            text_fill = (0, 0, 0) if luminance > 140 else (255, 255, 255)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w >= text_w + 10:
                draw.text((x + (w - text_w) // 2, (height - text_h) // 2),
                          label, fill=text_fill, font=font)
        x += w

    img.save(outfile)
    return outfile

# Example usage without sys.argv
if __name__ == "__main__":
    video_path = "video\matrix3.mp4"
    pixels = sample_video_pixels(
        video_path,
        frame_step=15,       # analyze ~every 15th frame
        resize=256,          # downscale longest side to speed up
        limit_frames=300,    # safety cap
        exclude_extremes=True
    )
    top_colors = dominant_colors_from_pixels(pixels, k=5)

    print("Video – top colors (global):")
    for i, c in enumerate(top_colors, 1):
        print(f"{i}. RGB={c['rgb']}, HEX={c['hex']}, {c['percent']}%")

    out = render_palette(top_colors, width=900, height=140, show_labels=True, outfile=f"{video_path}palette_video.png")
    print(f"Saved palette visualization to: {Path(out).resolve()}")
