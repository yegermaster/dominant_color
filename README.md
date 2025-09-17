Here’s a clean, ready-to-paste README for your Gradio app.

# Dominant Colors – Image & Video (Gradio)

Extract the top dominant colors from images **or** videos, with a palette preview and a table of HEX/RGB percentages. Built for local use and easy deployment to **Hugging Face Spaces**.

---

## Features

* Image tab: upload an image, get a palette + table.
* Video tab: sample frames to compute a **global** palette (not per-frame).
* Tunable parameters: `K` clusters, downsampling, frame step, limit frames, exclude near-black/near-white, show labels.
* CPU-friendly; suitable for free tiers.

---

## Quickstart (Local)

Requirements: Python 3.9–3.11

```bash
# 1) Clone your repo (or copy files into a folder)
# app.py
# dominant_colors_simple.py
# dominant_colors_video.py
# requirements.txt

# 2) Create & activate a virtualenv (Windows PowerShell shown)
python -m venv .venv
.venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the app
python app.py
```

Open [http://localhost:7860](http://localhost:7860)

> The app sets `server_name="0.0.0.0"` and respects `PORT` if provided (useful for cloud hosts).

**Example `requirements.txt`**

```
gradio>=4.0
pandas
numpy
pillow
scikit-learn
opencv-python
```

---

## Interface Overview

### Image tab

* **Inputs**

  * Image file
  * **Top K colors** (1–12)
  * **Downsample max side (px)**: resize longest side for speed
  * **Exclude near-black/near-white**: filter out extremes
  * **Show labels**: annotate palette with HEX and %
* **Outputs**

  * Palette PNG
  * Table: `hex`, `rgb`, `percent`

### Video tab

* **Inputs**

  * Video file
  * **Top K colors** (1–12)
  * **Frame step**: sample every Nth frame
  * **Downsample max side (px)**: resize frames for speed
  * **Max sampled frames**: safety cap
  * **Exclude near-black/near-white**, **Show labels**
* **Outputs**

  * Palette PNG
  * Table: `hex`, `rgb`, `percent`

> Video analysis computes a **global** palette across sampled frames (no timeline breakdown).

---

## How It Works (High Level)

* **Images**: downsample → flatten pixels → K-Means → sort by frequency → render palette + table.
* **Videos**: sample frames (`frame_step`, `limit_frames`) → optional downsample → collect pixels → K-Means on all sampled pixels → render palette + table.
* Palette images are written to a temp directory; Gradio displays the generated file.

---

## Project Structure

```
.
├─ app.py                      # Gradio Blocks UI
├─ dominant_colors_simple.py   # Image pipeline + palette render
├─ dominant_colors_video.py    # Video sampling + K-Means + palette render
└─ requirements.txt
```

---

## Deploy to Hugging Face Spaces

1. Create a new Space → **Gradio** template.
2. Upload `app.py`, `dominant_colors_simple.py`, `dominant_colors_video.py`, `requirements.txt`.
3. Hardware: CPU Basic is fine.
4. Space will install deps and run `python app.py` (uses `PORT` env var automatically).

---

## Troubleshooting

* **OpenCV import errors** → reinstall `opencv-python` in your venv.
* **NumPy/SciPy ABI mismatch** → upgrade/downgrade to compatible versions (`pip install --upgrade numpy scikit-learn`).
* **No palette shown** → check parameter ranges and file type; verify logs in terminal.
* **Large videos slow** → increase `Frame step`, decrease `Downsample max side`, or lower `Max sampled frames`.

---

## License

Choose a license for your repo (e.g., MIT). Add a `LICENSE` file if needed.

---

## Acknowledgements

Built with **Gradio**, **NumPy**, **scikit-learn**, **OpenCV**, and **Pillow**.
