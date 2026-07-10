# CloudVision AI — Cloud Speed & Classification Project

A Python-based sky analysis system that classifies cloud types, estimates cloud height and movement speed from images or video, and visualizes motion using optical flow. The main interface is a **Streamlit** web app (`app.py`).

**Repository:** [github.com/AliHasanJilani/Cloud_speed_project](https://github.com/AliHasanJilani/Cloud_speed_project)

---

## Features

| Area | What it does |
|------|----------------|
| **Cloud classification** | 7 cloud types via MobileNetV2 (`cloud_model.keras`) |
| **Cloud detection** | Contour + mask pipeline with optional EfficientNet-B0/B3 validation |
| **Speed estimation** | Farneback optical flow + camera geometry (FOV, focal length, cloud height) |
| **Height estimation** | Type-based default heights; triangulation with two cameras |
| **Live analysis** | Webcam / uploaded video / YouTube URL support |
| **Wind blending** | Live wind from Open-Meteo to cross-check cloud drift direction |
| **Solar tracking** | Sun azimuth/elevation via `pvlib` from latitude/longitude |
| **Mobile export** | H.264 re-encoding for WhatsApp-compatible MP4 sharing |
| **Motion visualization** | Annotated output video with boxes, speed arrows, and formulas |

### Supported cloud types

| Type | Typical height (m) |
|------|-------------------|
| Cumulus | 1,500 |
| Altocumulus | 4,500 |
| Cirrus | 9,000 |
| Stratocumulus | 1,200 |
| Cumulonimbus | 6,000 |
| Mixed | 3,500 |
| ClearSky | 0 |

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/AliHasanJilani/Cloud_speed_project.git
cd Cloud_speed_project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
python -m streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## App tabs

| Tab | Purpose |
|-----|---------|
| **Video Analysis** | Upload or paste a YouTube link; get speed, direction, and annotated output |
| **Multi Image Analysis** | Compare motion across multiple sky images |
| **Two-Camera Triangulation** | Estimate distance/height using two camera views |
| **Live Dual-Camera** | Real-time dual webcam analysis |
| **Quick Identify** | Single image — cloud type and estimated height |
| **Live Single Camera** | Real-time single webcam full-view tracking |

### Sidebar options

- **Theme** — dark / light UI
- **Solar location** — latitude & longitude for sun position
- **Wind blend** — compare optical-flow speed with live wind data
- **EfficientNet filter** — B0 (faster) or B3 (more accurate) cloud box validation
- **Timestamp override** — manual date/time when EXIF metadata is missing

---

## Speed calculation (overview)

Cloud speed is derived from pixel displacement between frames:

1. **Optical flow** — OpenCV Farneback estimates per-pixel motion in the sky region.
2. **Focal length** — computed from horizontal FOV and frame width.
3. **Real-world distance** — `distance = (pixel_disp × cloud_height) / focal_length_px`
4. **Speed** — `speed_mps = distance / delta_t` → converted to km/h
5. **Clamp** — detected motion below ~11 km/h is floored to a realistic minimum when clouds are clearly moving.

The app also shows a step-by-step formula breakdown in the UI.

---

## Project structure

```
Cloud_speed_project/
├── app.py                        # Main Streamlit application
├── cloud_box_detector.py         # Contour/mask cloud boxes + optical-flow tracking
├── efficientnet_cloud_detector.py# EfficientNet patch scoring for box validation
├── motion_visualizer.py          # Draws motion vectors and annotations on frames
├── predict.py                    # CLI: classify a single image
├── train.py                      # Train / fine-tune MobileNetV2 classifier
├── cloud_speed.py                # Basic optical-flow speed demo script
├── extract_frames.py             # Extract frames from video
├── forecast.py                   # Forecast utilities
├── cloud_model.keras             # Trained classification model
├── requirements.txt
├── dataset/                      # Training images (7 class folders)
├── frames/                       # Sample extracted frames
├── videos/                       # Sample input videos
└── cloud_motion_frames/          # Motion visualization outputs
```

---

## Training the classifier

Place images under `dataset/` in class-named subfolders, then run:

```bash
python train.py
```

This trains MobileNetV2 with augmentation and saves `cloud_model.keras`.

**Classes:** `Cumulus`, `Altocumulus`, `Cirrus`, `ClearSky`, `Stratocumulus`, `Cumulonimbus`, `Mixed`

---

## Standalone prediction

```bash
python predict.py path/to/sky_image.jpg
```

---

## Utility scripts

| Script | Description |
|--------|-------------|
| `extract_frames.py` | Pull frames from a video file |
| `fps_check.py` | Check video frame rate |
| `image_size.py` | Inspect image dimensions |
| `corrupt_check.py` | Detect corrupt images in dataset |
| `dataset_info.py` | Print dataset class counts |
| `test_motion_viz.py` | Test motion visualizer output |

---

## Requirements

| Package | Role |
|---------|------|
| `streamlit` | Web UI |
| `tensorflow` | Classification + EfficientNet |
| `opencv-python` | Optical flow, video I/O |
| `numpy`, `pandas` | Numerics |
| `Pillow` | Image loading |
| `pvlib` | Solar position |
| `yt-dlp` | YouTube video download |
| `imageio-ffmpeg` | FFmpeg fallback for H.264 export |

**Note:** `ffmpeg` on PATH improves video encoding. If missing, `imageio-ffmpeg` is used automatically.

---

## Files not included in Git

The following are excluded via `.gitignore` or kept local due to size:

- `venv311/` — virtual environment (never commit)
- `yolo*.pt` — YOLO weight files (download separately if needed)
- `weights/` — additional model weights
- Large media folders may exist locally but are optional for running the app

---

## Author

**Ali Hasan Jilani** — [GitHub](https://github.com/AliHasanJilani)

---

## License

This project is provided as-is for research and educational use. Add a license file if you plan to distribute commercially.
