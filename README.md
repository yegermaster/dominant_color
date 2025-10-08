---
title: Dominant Colors
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Dominant Colors – Image & Video

A **Gradio web app** that extracts the **top dominant colors** from images and videos.  
It provides both a **palette visualization** and a **color table** (HEX, RGB, and percentage).  

👉 **Check out the live version on Hugging Face:**  
[https://huggingface.co/spaces/yegermaster/Dominant_Colors](https://huggingface.co/spaces/yegermaster/Dominant_Colors)

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
![alt text](matrix_img.png)

---

## 🚀 Getting Started

### Local Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yegermaster/dominant-colors.git
   cd dominant-colors
