# Dominant Colors – Image & Video

A **Gradio web app** that extracts the **top dominant colors** from images and videos.
It provides both a **palette visualization** and a **color table** (HEX, RGB, and percentage).

---

## 🌟 Features

* **Image analysis** – upload any picture and get its dominant colors.
* **Video analysis** – sample frames from a video to compute global color palettes.
* **Customizable parameters** – adjust K (number of colors), downsampling, frame step, and whether to exclude near-white/near-black.
* **Palette visualization** – clean horizontal color bar with optional HEX/percentage labels.
* **Color table** – full breakdown of HEX codes, RGB values, and proportions.

---

## 📸 Screenshots

![alt text](barney_img.png)


![alt text]matrix_img.png.png)
---

## 🚀 Getting Started

### Local Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yegermaster/dominant-colors.git
   cd dominant-colors
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   python app.py
   ```

5. Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## ⚙️ Tech Stack

* [Python](https://www.python.org/)
* [Gradio](https://gradio.app/) – Web UI
* [OpenCV](https://opencv.org/) – Video frame sampling
* [scikit-learn](https://scikit-learn.org/) – KMeans clustering
* [Pandas](https://pandas.pydata.org/) – Data display
* [Pillow](https://python-pillow.org/) – Image handling
