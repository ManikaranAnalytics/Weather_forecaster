# CloudVision AI — Complete Technical Documentation

**Project name:** Cloud Speed Project / CloudVision AI  
**Main application:** `app.py` (Streamlit web app)  
**Author:** Ali Hasan Jilani  
**Repository:** [github.com/AliHasanJilani/Cloud_speed_project](https://github.com/AliHasanJilani/Cloud_speed_project)

---

## 1. Purpose — Why This Project Exists

**CloudVision AI** is a computer-vision system for analyzing the sky from images, video, webcam feeds, or YouTube streams. It was built to answer practical questions that matter for solar power plants, meteorology research, and sky monitoring:

| Question | What the system provides |
|----------|--------------------------|
| What kind of cloud is this? | Classification into 7 cloud types |
| How high is the cloud? | Type-based height lookup, or two-camera triangulation |
| How fast is it moving, and in which direction? | Optical-flow speed in km/h and compass direction |
| Will its shadow hit a solar plant? | Solar shadow forecast (time + expected power drop) |
| Does the video motion match real wind? | Live Open-Meteo wind blend and comparison |
| Where is the sun right now? | Solar position via `pvlib` (or image-based estimate) |

### Why it was built

1. **Solar plant operations** — Passing clouds cause sudden drops in photovoltaic (PV) output (“ramp events”). Knowing cloud type, speed, and direction helps operators anticipate when power will fall and for how long.
2. **Low-cost instrumentation** — Traditional ceilometers and wind profilers are expensive. This project uses ordinary cameras plus physics formulas.
3. **End-to-end demo** — Classification, detection, tracking, height, speed, wind cross-check, and shadow forecasting in one Streamlit UI.
4. **Research / education** — Documents camera geometry, optical flow, triangulation, and wind profiles with formulas you can verify in the UI.

---

## 2. What the System Does (Feature Summary)

| Feature | Description | Main technology |
|---------|-------------|-----------------|
| Cloud classification | 7 classes from sky images/frames | MobileNetV2 (`models/cloud_model.keras`) |
| Cloud detection | Bounding boxes on sky region | Contour + HSV mask; optional EfficientNet-B0/B3 |
| Motion estimation | Pixel displacement between frames | OpenCV Farneback optical flow |
| Speed estimation | Pixel motion → real-world km/h | Camera FOV + cloud height geometry |
| Height estimation | Default by type, or measured | Lookup table / two-camera triangulation |
| GNN refinement (optional) | Smooths multi-cloud speeds / heights | PyTorch Geometric GCN (NumPy fallback) |
| Live wind | Surface wind → wind at cloud height | Open-Meteo API + power-law profile |
| Solar tracking | Sun azimuth & elevation | `pvlib` + lat/lon + timestamp |
| Shadow forecast | Time until shadow hits Plant 1 / Plant 2 | Trigonometry + cloud lifetime model |
| Video I/O | Upload, webcam, YouTube (incl. live clip) | OpenCV, `yt-dlp`, FFmpeg / imageio-ffmpeg |
| Mobile export | WhatsApp-compatible MP4 | H.264 re-encode via FFmpeg |

---

## 3. Supported Cloud Types

| Class | Typical base height used in app (m) | Notes |
|-------|--------------------------------------|--------|
| Cumulus | 1,500 | Low convective clouds |
| Altocumulus | 4,500 | Mid-level |
| Cirrus | 9,000 | High ice clouds |
| Stratocumulus | 1,200 | Low layered |
| Cumulonimbus | 6,000 | Storm / tall convective |
| Mixed | 3,500 | Multiple types |
| ClearSky | 0 | No cloud |

These heights are **default meteorological averages**, not measured altitudes. For a measured height, use the **Two-Camera Triangulation** tab.

---

## 4. Application Tabs

| Tab | Purpose |
|-----|---------|
| **Video Analysis** | Upload a video or paste a YouTube link; classify clouds; estimate speed, direction, height; annotated output video |
| **Multi Image Analysis** | Compare motion across a sequence of still images with a known time interval |
| **Two-Camera Triangulation** | Estimate real cloud height from two views and then compute speed |
| **Live Dual-Camera** | Real-time dual webcam analysis |
| **Quick Identify** | Single image — cloud type and estimated height only |
| **Live Single Camera** | Real-time single webcam full-view tracking |

### Sidebar options

- Theme (dark / light)
- Solar location (latitude & longitude)
- Wind blend (compare optical-flow speed with live wind)
- EfficientNet filter (B0 faster / B3 more accurate)
- Timestamp override (when EXIF / video metadata is missing)

---

## 5. System Architecture

```
Input (image / video / webcam / YouTube)
        │
        ▼
┌───────────────────┐
│ Sky scene check   │  Reject non-sky images when possible
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Classification    │  MobileNetV2 → cloud type + confidence
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Detection         │  Cloud mask + boxes (+ optional EfficientNet)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Optical flow      │  Farneback → median pixel displacement + angle
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Optional GNN      │  Refine per-cloud speed / angle
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Geometry → speed  │  FOV + height → m/s → km/h
└─────────┬─────────┘
          ▼
┌───────────────────────────────────────┐
│ Wind blend │ Solar shadow │ Metrics UI │
└───────────────────────────────────────┘
```

---

## 6. Complete Formula Reference

All core formulas used by the application are listed below.

### 6.1 Optical flow (Farneback)

Motion between two grayscale frames is estimated with OpenCV:

```text
flow = calcOpticalFlowFarneback(gray1, gray2,
         pyr_scale=0.5, levels=3, winsize=15,
         iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
```

Then flow vectors \((u, v)\) are converted to magnitude and angle:

\[
\text{magnitude} = \sqrt{u^2 + v^2},\quad
\text{angle} = \operatorname{atan2}(v, u)
\]

The app uses the **median** magnitude and **median** angle over the frame (robust to outliers).

- **Pixel displacement** \(\Delta p\) = median magnitude (pixels over \(\Delta t\))
- **Pixel speed** = \(\lvert\Delta p\rvert / \Delta t\) (px/s)
- **Direction** = angle mapped to: North, NE, East, SE, South, SW, West, NW

---

### 6.2 Focal length from field of view (FOV)

Camera horizontal FOV and frame width \(W\) (pixels) give focal length in pixels:

\[
f_{px} = \frac{W / 2}{\tan(\mathrm{FOV}/2)}
\]

**Example:** \(W = 1920\), \(\mathrm{FOV} = 90^\circ\)

\[
f_{px} = \frac{960}{\tan(45^\circ)} = 960
\]

---

### 6.3 Pixel motion → real-world distance

With cloud height \(h\) (metres) and pixel displacement \(\Delta p\):

\[
d = \frac{\lvert\Delta p\rvert \times h}{f_{px}}
\]

**Interpretation:** At height \(h\), one pixel on the image corresponds to \(h / f_{px}\) metres on the cloud plane. Multiplying by \(\lvert\Delta p\rvert\) gives ground-track distance travelled by the cloud between frames.

---

### 6.4 Speed

\[
v_{m/s} = \frac{d}{\Delta t}
\]

\[
v_{km/h} = v_{m/s} \times 3.6
\]

**Minimum speed floor:** If motion is detected but \(v_{km/h} < 11\), the app floors the result to **11 km/h** (`MIN_CLOUD_SPEED_KMH = 11.0`) so tiny FOV/geometry errors do not report unrealistically slow “moving” clouds.

---

### 6.5 Distance projections (forecast horizons)

\[
\text{Distance in } t \text{ minutes (km)} = v_{km/h} \times \frac{t}{60}
\]

| Horizon | Formula |
|---------|---------|
| +5 minutes | \(v_{km/h} \times (5/60)\) |
| +15 minutes | \(v_{km/h} \times (15/60)\) |

---

### 6.6 Two-camera triangulation (cloud height)

Two cameras are separated by baseline distance \(d\) (metres). Each measures elevation angle above the horizon to the same cloud feature:

- \(\alpha\) = elevation from Camera 1 (degrees)
- \(\beta\) = elevation from Camera 2 (degrees)

\[
h = \frac{d}{\dfrac{1}{\tan\alpha} + \dfrac{1}{\tan\beta}}
\]

**Notes in code:**

- Angles below \(0.5^\circ\) are clamped to avoid division by zero.
- Elevations below \(5^\circ\) are treated as unreliable near the horizon.
- If triangulation fails, height falls back to **1500 m**.

#### Elevation angle from pixel Y

Horizon is assumed at fraction \(f_h\) of frame height (default \(0.75\)):

\[
\text{horizon}_y = H \times f_h
\]

\[
\text{deg/px} = \frac{\mathrm{FOV}_v}{H}
\]

\[
\alpha = (\text{horizon}_y - y_{cloud}) \times \text{deg/px}
\]

(Positive when the cloud is above the horizon.)

#### Speed with triangulated height

After height \(h\) is known, angular motion \(\theta\) from optical flow is converted:

\[
d = h \times \tan(\theta),\quad
v_{m/s} = \frac{d}{\Delta t},\quad
v_{km/h} = v_{m/s} \times 3.6
\]

---

### 6.7 Wind profile (power law)

Open-Meteo returns wind at **10 m** above ground. Wind usually increases with height. The app uses a power-law profile:

\[
v(z) = v_{10} \times \left(\frac{z}{z_{ref}}\right)^{\alpha}
\]

Defaults: \(z_{ref} = 10\,\mathrm{m}\), \(\alpha = 0.15\) (open terrain exponent).

---

### 6.8 Wind–cloud speed blending

| Condition | Blend used |
|-----------|------------|
| Wind and video **aligned** (diff ≤ 40%) | \(0.6\,v_{\text{video}} + 0.4\,v_{\text{wind}}\) |
| Video **slower** than wind | \(0.45\,v_{\text{video}} + 0.55\,v_{\text{wind}}\) |
| Otherwise | Use video speed only |

Cloud drift direction is opposite meteorological **wind-from** bearing:

\[
\text{drift bearing} = (\text{wind-from} + 180^\circ) \bmod 360^\circ
\]

---

### 6.9 Solar position

With latitude, longitude, and timestamp, `pvlib` computes:

- **Azimuth** — compass bearing of the sun
- **Elevation** — angle above the horizon

If `pvlib` is unavailable, the app can estimate elevation from sky brightness (horizon glow, blue ratio, saturation cues).

Sun azimuth from a pixel (when the sun is visible):

\[
\text{offset}^\circ = (x_{sun} - W/2) \times \frac{\mathrm{FOV}}{W}
\]

\[
\text{azimuth} = (\text{device heading} + \text{offset}) \bmod 360
\]

---

### 6.10 Solar shadow forecast (Plant 1 — camera site)

Approximate ground offset of the cloud shadow from the plant:

\[
\text{ground\_offset} = h \times \tan(\theta)
\]

where \(\theta\) is a conservative angle based on half-FOV.

\[
t_{\text{shadow}} = \frac{\text{ground\_offset}}{v_{m/s}}
\]

Expected PV power drop depends on cloud type and coverage. Base factors:

| Cloud type | Power attenuation factor (approx.) |
|------------|--------------------------------------|
| Cirrus | 0.18 |
| Altocumulus | 0.45 |
| Mixed | 0.50 |
| Cumulus | 0.55 |
| Stratocumulus | 0.72 |
| Cumulonimbus | 0.85 |
| ClearSky | 0.0 |

Drop is scaled by coverage and capped near 95%.

---

### 6.11 Second-plant shadow forecast

Plant 2 is at distance \(D\) km and bearing \(\phi\) from Plant 1. Cloud travels at bearing \(\psi\). Angle difference:

\[
\delta = \min\big(\lvert\psi - \phi\rvert \bmod 360,\; 360 - \lvert\psi - \phi\rvert \bmod 360\big)
\]

If \(\delta \ge 85^\circ\), the cloud is not heading toward Plant 2 (safe).

Otherwise effective travel distance:

\[
D_{\text{eff}} = \frac{D}{\cos\delta}
\]

\[
t_{\text{arrival (min)}} = \frac{D_{\text{eff}}}{v_{km/h}} \times 60
\]

If arrival time exceeds typical cloud lifetime, the shadow is judged unlikely to reach Plant 2.

---

### 6.12 Cloud lifetime / visibility estimate

\[
t_{\text{life}} = \left(\frac{t_{lo}+t_{hi}}{2} + \text{coverage\_bonus}\right) \times \text{speed\_factor}
\]

\[
\text{coverage\_bonus} = \frac{\text{coverage\%}}{100} \times (t_{hi}-t_{lo}) \times 0.3
\]

\[
\text{speed\_factor} = \max\!\left(0.7,\; 1 - \frac{v_{km/h}}{200}\right)
\]

Typical lifetime ranges (minutes):

| Type | Range |
|------|-------|
| Cumulus | 10–60 |
| Altocumulus | 30–120 |
| Cirrus | 60–360 |
| Stratocumulus | 60–480 |
| Cumulonimbus | 30–90 |
| Mixed | 20–90 |

---

### 6.13 Ramp risk score (solar)

Integer score from speed, coverage, and cloud type:

| Signal | Points |
|--------|--------|
| Speed ≥ 30 km/h | +3 |
| Speed ≥ 15 km/h | +2 |
| Speed ≥ 5 km/h | +1 |
| Coverage ≥ 70% | +3 |
| Coverage ≥ 40% | +2 |
| Coverage ≥ 15% | +1 |
| Type in Cumulonimbus / Cumulus / Stratocumulus | +1 |

| Score | Risk |
|-------|------|
| ≥ 6 | High |
| ≥ 3 | Medium |
| else | Low |

---

### 6.14 Pseudo-depth → horizontal distance (display)

For visualization of box distances:

\[
d_m = h \times \big(1 + 2(1 - \text{depth})\big)
\]

where `depth` ∈ [0, 1] (1 ≈ overhead, 0 ≈ near horizon).

---

### 6.15 GNN graph (optional enhancement)

Each detected cloud is a **node** with features:

\[
\mathbf{x} = \big[\hat{c}_x,\; \hat{c}_y,\; \hat{w},\; \hat{h},\; \widehat{\text{speed}},\; \sin\theta,\; \cos\theta,\; \text{density}\big]
\]

Edges connect each cloud to its \(k\) nearest neighbors (\(k=3\)), with weight \(1/\text{pixel distance}\).

- **CloudMotionGNN** — refines speed/angle using neighbor consensus (GCN if PyTorch Geometric is installed; otherwise Laplacian-style smoothing).
- **TriangulationGNN** — consensus height from multiple \((\alpha,\beta)\) pairs.

Classical triangulation remains the base; GNN is an optional correction layer.

---

## 7. Machine Learning Models Used

### 7.1 MobileNetV2 cloud classifier

| Item | Detail |
|------|--------|
| Architecture | MobileNetV2 (ImageNet pretrained) + custom head |
| Input size | 224 × 224 × 3 |
| Classes | 7 (listed in Section 3) |
| Loss | Categorical cross-entropy |
| Training script | `scripts/train.py` |
| Weights file | `models/cloud_model.keras` |

**Training phases:**

1. Freeze base; train top layers (Adam lr = \(1\times10^{-3}\), ~10 epochs)
2. Unfreeze last 30 base layers; fine-tune (Adam lr = \(1\times10^{-5}\), ~5 epochs)

**Augmentation:** rotation ±20°, shifts, horizontal flip, brightness [0.8, 1.2], zoom 0.1, rescale 1/255.

**Prediction:** Softmax; majority vote across frames; confidence = mean of max-class probabilities. Diagnostics include top-2 probability gap and normalized entropy.

### 7.2 EfficientNet (optional box validation)

EfficientNet-B0 or B3 embeddings compare candidate patches to sky/cloud prototypes to reduce false-positive boxes. Not required for basic classification.

### 7.3 Models parked / unused by the main app

YOLO (`.pt`) and CLIP weights under `models/` are local optional assets and are **not** required by the current Streamlit pipeline.

---

## 8. Computer Vision Techniques Used

| Technique | Role | Library / params |
|-----------|------|------------------|
| Farneback optical flow | Motion between frames | OpenCV `calcOpticalFlowFarneback` |
| HSV + CLAHE masking | Cloud vs blue sky segmentation | OpenCV |
| Contours / NMS | Bounding boxes | OpenCV + DNN NMS |
| Distance transform | Split merged cloud blobs | OpenCV |
| Adaptive threshold | Horizon-band cloud detection | OpenCV |
| Brightest-pixel sun find | Sun in frame if max brightness > 240 | OpenCV `minMaxLoc` |
| H.264 encode | Mobile/WhatsApp video | FFmpeg (`libx264`, yuv420p, baseline) |

---

## 9. External APIs and Libraries

### Dependencies (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `tensorflow` | MobileNetV2 / EfficientNet |
| `opencv-python` | Video, optical flow, drawing |
| `numpy`, `pandas` | Numerics / timestamps |
| `Pillow` | Image load / EXIF |
| `pvlib` | Solar azimuth & elevation |
| `yt-dlp` | YouTube download / stream resolve |
| `imageio-ffmpeg` | FFmpeg binary fallback on Windows |

### APIs

| Service | Use |
|---------|-----|
| [Open-Meteo](https://api.open-meteo.com) | Current wind speed/direction/gusts at 10 m, temperature, cloud cover (no API key) |

### Optional

| Package | Use |
|---------|-----|
| `torch` + `torch-geometric` | Full GNN path in `src/gnn/cloud_graph.py` |
| System `ffmpeg` / `ffprobe` | Encoding and video metadata |

---

## 10. Project Structure

```text
Cloud_speed_project/
├── app.py                         # Main Streamlit application
├── requirements.txt
├── README.md                      # Quick start
├── DOCUMENTATION.md               # This file
├── src/
│   ├── detection/
│   │   ├── cloud_box_detector.py      # Mask + contour boxes
│   │   └── efficientnet_cloud_detector.py
│   ├── tracking/
│   │   └── motion_visualizer.py       # Simple predicted-motion MP4
│   ├── forecasting/
│   │   └── forecast.py                # Pixel displacement demo script
│   ├── gnn/
│   │   └── cloud_graph.py             # Motion + triangulation GNN
│   └── utils/
│       ├── fps_check.py
│       ├── image_size.py
│       ├── corrupt_check.py
│       └── dataset_info.py
├── scripts/
│   ├── train.py                   # Train MobileNetV2 classifier
│   ├── predict.py                 # Single-image CLI predict
│   ├── extract_frames.py
│   ├── make_dummy_model.py
│   └── cloud_speed.py             # Minimal optical-flow demo
├── models/
│   └── cloud_model.keras          # Required for classification
├── data/
│   ├── dataset/                   # Training images by class folder
│   ├── frames/
│   ├── videos/
│   └── outputs/
├── assets/                        # UI background
├── tests/
│   └── test_motion_viz.py
└── archive/                       # Older experiments / notes
```

---

## 11. End-to-End Pipelines

### 11.1 Single-camera video speed

1. Load video (upload or YouTube via `yt-dlp`).
2. Sample frame pairs separated by \(\Delta t\).
3. Classify cloud type with MobileNetV2 → look up default height \(h\).
4. Detect cloud regions (mask / boxes).
5. Farneback optical flow → \(\Delta p\), angle → compass direction.
6. Optional GNN refinement of speeds.
7. Convert \(\Delta p\) → \(v_{km/h}\) using FOV and \(h\).
8. Optional Open-Meteo wind blend.
9. Solar shadow / second-plant forecasts when location is set.
10. Draw boxes, arrows, overlays; export H.264 MP4 if requested.

### 11.2 Two-camera height + speed

1. Two synchronized (or near-synchronized) frame pairs.
2. Optical flow on each camera.
3. Convert feature Y positions → \(\alpha\), \(\beta\).
4. Triangulate \(h\) (optionally GNN consensus).
5. Convert angular motion → ground speed with measured \(h\).

---

## 12. How to Run

```bash
git clone https://github.com/AliHasanJilani/Cloud_speed_project.git
cd Cloud_speed_project

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

### Train the classifier

Place images in `data/dataset/<ClassName>/`, then:

```bash
python scripts/train.py
```

### Single-image prediction (CLI)

```bash
python scripts/predict.py path/to/sky_image.jpg
```

---

## 13. Assumptions and Limitations

1. **Default heights are averages** — Real cloud bases vary; triangulation is more accurate when available.
2. **Pinhole FOV model** — Lens distortion is not fully calibrated.
3. **Median flow** — Assumes dominant motion is cloud drift (camera should be fixed).
4. **Minimum speed floor (11 km/h)** — Can overestimate very slow clouds.
5. **Open-Meteo is model data** — Not a ground weather station (updates ~every 15 minutes).
6. **Shadow forecasts are heuristics** — Useful for operational awareness, not certified power-market forecasts.
7. **GNN needs torch-geometric for full model** — Without it, a NumPy physics fallback is used.

---

## 14. Accuracy Improves When You Provide

- Correct **camera FOV** (horizontal and vertical)
- Accurate **location** (lat/lon) and **timestamp**
- Measured **baseline** between dual cameras
- Fixed camera mount (no shake)
- Higher resolution and stable FPS
- Longer clips with clear cloud motion

---

## 15. Intended Users

| Audience | Use case |
|----------|----------|
| Solar plant operators | Anticipating cloud-driven ramp events |
| Researchers / students | CV + meteorology / solar irradiance projects |
| Developers | Extending detection, GNN, or forecasting modules |
| Hobbyists | Analyzing sky timelapses / webcam feeds |

---

## 16. License and Disclaimer

This project is provided **as-is for research and educational use**. Speed, height, and shadow forecasts are **estimates** based on computer vision and meteorological assumptions. They must not replace certified meteorological instruments or operational forecasting systems without further validation.

---

## 17. Quick Formula Cheat Sheet

| Quantity | Formula |
|----------|---------|
| Focal length (px) | \(f_{px} = (W/2)/\tan(\mathrm{FOV}/2)\) |
| Real distance (m) | \(d = (\lvert\Delta p\rvert \times h) / f_{px}\) |
| Speed (m/s) | \(v = d / \Delta t\) |
| Speed (km/h) | \(v_{km/h} = v \times 3.6\) |
| +5 min distance (km) | \(v_{km/h} \times 5/60\) |
| Triangulated height | \(h = d / (1/\tan\alpha + 1/\tan\beta)\) |
| Wind at height \(z\) | \(v(z) = v_{10} (z/10)^{0.15}\) |
| Shadow time | \(t = (h\tan\theta) / v_{m/s}\) |
| Plant-2 ETA (min) | \(60 \times D_{\text{eff}} / v_{km/h}\) |

---

*End of documentation — CloudVision AI / Cloud Speed Project*
