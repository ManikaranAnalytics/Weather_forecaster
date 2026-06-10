import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import math
from collections import Counter
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from motion_visualizer import CloudMotionVisualizer

# ─────────────────────────── CONFIG ────────────────────────────
st.set_page_config(page_title="CloudVision AI", page_icon="☁️", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #4fa3e0, #a0c4f1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a2535; border-radius: 8px 8px 0 0;
        color: #a0b8d0; padding: 8px 20px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── MODEL ─────────────────────────────
@st.cache_resource
def load_cloud_model():
    return load_model("cloud_model.keras")

model = load_cloud_model()

class_names  = ["Cumulus","Altocumulus","Cirrus","ClearSky","Stratocumulus","Cumulonimbus","Mixed"]
cloud_height = {"Cumulus":1500,"Altocumulus":4500,"Cirrus":9000,
                "ClearSky":0,"Stratocumulus":1200,"Cumulonimbus":6000,"Mixed":3500}
cloud_emoji  = {"Cumulus":"⛅","Altocumulus":"🌤️","Cirrus":"🌬️",
                "ClearSky":"☀️","Stratocumulus":"🌥️","Cumulonimbus":"⛈️","Mixed":"🌦️"}

# ─────────────────────────── HELPERS ───────────────────────────
def angle_to_direction(angle_deg):
    a = angle_deg % 360
    if   45  <= a < 135: return "North"
    elif 135 <= a < 225: return "West"
    elif 225 <= a < 315: return "South"
    else:                return "East"

def compute_optical_flow(gray1, gray2):
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.median(mag), np.mean(ang)

def pixels_to_kmh(pixel_displacement, delta_t_sec, cloud_type, frame_width, fov):
    height_m      = cloud_height.get(cloud_type, 2000)
    # Pixel → Angle
    degree_per_px = fov / frame_width
    theta_deg     = pixel_displacement * degree_per_px
    theta_rad     = math.radians(theta_deg)
    # Tan formula: d = h * tan(θ)  — right triangle, cloud horizontally moves
    distance_m    = height_m * math.tan(theta_rad)
    speed_mps     = distance_m / delta_t_sec
    speed_kmh     = speed_mps * 3.6
    return speed_mps, speed_kmh, degree_per_px, theta_deg, distance_m, height_m

def compute_cloud_density(frame, sky_h):
    """
    Cloud coverage % calculate karta hai sky region mein.
    Returns: coverage_percent (0-100), density_label, density_color
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sky_gray = gray[:sky_h, :]
    sky_hsv  = hsv[:sky_h, :]

    _, bright_mask = cv2.threshold(sky_gray, 140, 255, cv2.THRESH_BINARY)
    sat = sky_hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)
    hue = sky_hsv[:, :, 0]
    blue_sky = cv2.inRange(hue, 95, 135)
    not_blue = cv2.bitwise_not(blue_sky)
    combined = cv2.bitwise_or(bright_mask, cv2.bitwise_and(sat_mask, not_blue))

    total_pixels = sky_gray.shape[0] * sky_gray.shape[1]
    cloud_pixels = int(cv2.countNonZero(combined))
    coverage = min(100.0, (cloud_pixels / total_pixels) * 100)

    if coverage < 20:
        label, color = "Low ☀️", "#22c55e"
    elif coverage < 55:
        label, color = "Medium 🌤️", "#f59e0b"
    else:
        label, color = "High ⛅", "#ef4444"

    return round(coverage, 1), label, color


def predict_visibility(cloud_type, speed_kmh, direction, coverage_pct, fov_deg):
    """
    Predict karta hai ki cloud aage jaane ke baad dikhega ya miss ho jayega.
    Logic: cloud ki speed, direction, coverage aur FOV ke basis par.
    Returns: verdict (str), reason (str), color (str)
    """
    # Frame edge tak pahunchne ka time estimate (assume 640px wide frame, cloud center ~middle)
    # distance to edge in km at current speed
    if speed_kmh < 0.5:
        return (
            "🟡 Stationary — Still Visible",
            "Cloud bahut slow chal raha hai, kaafi der tak dikh\u00e9ga.",
            "#f59e0b"
        )

    # Kitne minutes mein cloud FOV ke bahar jayega
    # FOV coverage: zyada FOV = zyada area cover = cloud zyada der dikhega
    fov_factor   = fov_deg / 75.0          # normalize to default 75°
    cover_factor = coverage_pct / 100.0    # dense cloud = zyada area = zyada time

    # Rough estimate: at speed_kmh, cloud travels ~1 km in (60/speed_kmh) minutes
    # Frame captures approx 0.5–2 km depending on FOV & height
    frame_km = 0.8 * fov_factor            # km visible in frame (approx)
    time_to_exit_min = (frame_km / speed_kmh) * 60.0 * (0.5 + cover_factor)

    if cloud_type == "ClearSky":
        verdict = "☀️ No Cloud — Clear Sky"
        reason  = "Abhi koi cloud nahi hai."
        color   = "#22c55e"
        time_to_exit_min = 999
    elif time_to_exit_min > 15:
        verdict = "🟢 Will Stay Visible (>15 min)"
        reason  = (f"Cloud ki speed {speed_kmh:.1f} km/h hai aur coverage {coverage_pct}% — "
                   f"yeh frame mein ~{time_to_exit_min:.0f} min tak dikhta rahega.")
        color   = "#22c55e"
    elif time_to_exit_min > 5:
        verdict = f"🟡 Visible for ~{time_to_exit_min:.0f} min, then Exits"
        reason  = (f"Cloud {direction} direction mein {speed_kmh:.1f} km/h se chal raha hai. "
                   f"~{time_to_exit_min:.0f} min baad frame se bahar chala jayega.")
        color   = "#f59e0b"
    else:
        verdict = f"🔴 Will Disappear Soon (~{time_to_exit_min:.1f} min)"
        reason  = (f"Cloud tezi se ({speed_kmh:.1f} km/h) {direction} mein ja raha hai. "
                   f"~{time_to_exit_min:.1f} min mein frame se miss ho jayega.")
        color   = "#ef4444"

    return verdict, reason, color, time_to_exit_min


def predict_cloud_type(frames_or_images):
    preds, confs = [], []
    for img in frames_or_images:
        img_pil = Image.fromarray(img).resize((224,224)) if isinstance(img, np.ndarray) else img.resize((224,224))
        arr = np.expand_dims(image.img_to_array(img_pil), 0) / 255.0
        pred = model.predict(arr, verbose=0)
        preds.append(class_names[np.argmax(pred)])
        confs.append(float(np.max(pred)) * 100)
    return Counter(preds).most_common(1)[0][0], float(np.mean(confs))

def show_metrics(cloud_type, confidence, direction, height_m, fov,
                 frame_width, pixel_disp, delta_t, deg_per_px,
                 theta_deg, distance_m, speed_mps, speed_kmh, dist_5, dist_15,
                 coverage_pct=None, density_label=None, density_color=None,
                 vis_verdict=None, vis_reason=None, vis_color=None,
                 time_to_exit_min=999):
    emoji = cloud_emoji.get(cloud_type, "☁️")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"{emoji} Cloud Type", cloud_type)
    c2.metric("🎯 Confidence",       f"{confidence:.1f}%")
    c3.metric("🧭 Direction",        direction)
    c4.metric("📍 Height",           f"{height_m:,} m")

    # Smart +5 / +15 min labels
    label_5  = "❌ Out of Frame" if time_to_exit_min <= 5  else f"~{dist_5:.2f} km"
    label_15 = "❌ Out of Frame" if time_to_exit_min <= 15 else f"~{dist_15:.2f} km"
    delta_5  = f"exits ~{time_to_exit_min:.1f} min" if time_to_exit_min <= 5  else None
    delta_15 = f"exits ~{time_to_exit_min:.1f} min" if time_to_exit_min <= 15 else None

    c5,c6,c7,c8 = st.columns(4)
    c5.metric("⚡ Speed",       f"{speed_kmh:.1f} km/h")
    c6.metric("⚡ Speed (m/s)", f"{speed_mps:.2f} m/s")
    c7.metric("⏳ +5 min",     label_5,  delta=delta_5,  delta_color="off")
    c8.metric("⏳ +15 min",    label_15, delta=delta_15, delta_color="off")

    # ── Cloud Density Card ──
    if coverage_pct is not None:
        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"""
<div style="background:#1a2535;border:1px solid {density_color};border-radius:10px;padding:16px;">
  <div style="font-size:0.85rem;color:#a0b8d0;margin-bottom:4px;">☁️ Cloud Density</div>
  <div style="font-size:1.5rem;font-weight:700;color:{density_color};">{density_label}</div>
  <div style="margin-top:6px;">
    <div style="background:#0f172a;border-radius:6px;height:14px;width:100%;">
      <div style="background:{density_color};width:{coverage_pct}%;height:14px;border-radius:6px;"></div>
    </div>
    <div style="color:#cbd5e1;font-size:0.8rem;margin-top:4px;">Sky Coverage: <b>{coverage_pct}%</b></div>
  </div>
  <div style="color:#94a3b8;font-size:0.75rem;margin-top:6px;">
    Low &lt;20% &nbsp;|&nbsp; Medium 20–55% &nbsp;|&nbsp; High &gt;55%
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Visibility Prediction Card ──
        with d2:
            st.markdown(f"""
<div style="background:#1a2535;border:1px solid {vis_color};border-radius:10px;padding:16px;">
  <div style="font-size:0.85rem;color:#a0b8d0;margin-bottom:4px;">👁️ Visibility Prediction</div>
  <div style="font-size:1.1rem;font-weight:700;color:{vis_color};">{vis_verdict}</div>
  <div style="color:#cbd5e1;font-size:0.82rem;margin-top:8px;">{vis_reason}</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

    with st.expander("🔬 Calculation Details"):
        st.markdown(f"""
| Parameter | Value |
|---|---|
| FOV | {fov}° | Frame Width | {frame_width} px |
| Degree/Pixel | {deg_per_px:.4f} °/px | Pixel Disp | {pixel_disp:.2f} px/{delta_t:.2f}s |
| Theta | {theta_deg:.4f}° | Tan Distance | {distance_m:.2f} m |
| Speed | {speed_mps:.2f} m/s → {speed_kmh:.1f} km/h |
""")

# ─────────────────────────── CLOUD DETECTION ───────────────────
def detect_clouds(frame, sky_h):
    """
    Multi-method cloud detection:
    1. Brightness threshold (white clouds)
    2. HSV saturation (low saturation = cloud/white)
    3. Combine both masks
    Alag-alag clouds ko alag boxes milein isliye
    watershed-style distance based separation use karta hai.
    """
    OUT_W = frame.shape[1]

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sky_gray = gray[:sky_h, :]
    sky_hsv  = hsv[:sky_h, :]

    # Method 1: brightness — lower threshold to catch grey clouds too
    _, bright_mask = cv2.threshold(sky_gray, 140, 255, cv2.THRESH_BINARY)

    # Method 2: low saturation = white/grey cloud (not blue sky)
    sat = sky_hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)

    # Method 3: not-blue sky — blue sky has high hue (100-130)
    hue = sky_hsv[:, :, 0]
    blue_sky = cv2.inRange(hue, 95, 135)
    not_blue = cv2.bitwise_not(blue_sky)

    # Combine: bright OR (low-sat AND not-blue-sky)
    combined = cv2.bitwise_or(bright_mask,
                cv2.bitwise_and(sat_mask, not_blue))

    # Morphology — smaller kernels to keep clouds separate
    k_close = np.ones((12, 12), np.uint8)
    k_open  = np.ones((6,  6),  np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)

    # --- Watershed separation to split merged clouds ---
    dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg    = np.uint8(sure_fg)

    sure_bg    = cv2.dilate(combined, np.ones((3,3), np.uint8), iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0

    # Watershed needs 3-channel BGR image
    sky_bgr = frame[:sky_h, :].copy()
    markers = cv2.watershed(sky_bgr, markers)

    # Extract bounding boxes from each watershed region
    boxes = []
    unique_labels = np.unique(markers)
    for lbl in unique_labels:
        if lbl <= 1:   # background or border
            continue
        mask_lbl = np.zeros_like(combined)
        mask_lbl[markers == lbl] = 255
        cnts, _ = cv2.findContours(mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 600:   # ignore tiny noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h, mask_lbl))

    # Fallback: if watershed gave nothing, use simple contours
    if not boxes:
        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < 600:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h, None))

    return boxes, gray

# ─────────────────────── STEREO DEPTH VISION ───────────────────
def compute_pseudo_depth_map(frame_bgr, sky_h):
    """
    Single-image pseudo stereo depth map for cloud regions.

    Physics cues used (all monocular):
      1. Brightness  — brighter cloud core = optically thicker = visually 'closer'
      2. Texture     — high-freq detail = nearer; smooth/hazy = farther
      3. Saturation  — desaturated (grey/white) regions = cloud mass present
      4. Vertical pos— lower in sky frame ≈ closer horizon clouds

    Output: depth_map (H x W float32, 0=far / blue … 1=near / red)
            depth_color (H x W x 3 uint8, COLORMAP_JET applied)
    """
    sky = frame_bgr[:sky_h, :].copy()
    H, W = sky.shape[:2]

    gray = cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv  = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat  = hsv[:, :, 1] / 255.0
    val  = hsv[:, :, 2] / 255.0

    # Cue 1: brightness (brighter = more cloud mass = closer)
    bright_cue = val

    # Cue 2: texture energy via Laplacian (sharp edges = nearer)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex_cue = np.abs(lap)
    tex_cue = cv2.GaussianBlur(tex_cue, (15, 15), 0)
    tex_max = tex_cue.max()
    if tex_max > 0:
        tex_cue /= tex_max

    # Cue 3: low saturation = cloud (not blue sky) → weight up
    cloud_presence = 1.0 - np.clip(sat, 0, 1)   # white/grey = high weight

    # Cue 4: vertical position — lower row = closer (nearer horizon)
    row_idx  = np.linspace(1.0, 0.0, H, dtype=np.float32)   # top=far, bottom=near
    vert_cue = np.tile(row_idx[:, None], (1, W))

    # Weighted fusion
    depth = (0.40 * bright_cue +
             0.25 * tex_cue    +
             0.20 * cloud_presence +
             0.15 * vert_cue)

    # Smooth for clean visualization
    depth = cv2.GaussianBlur(depth, (21, 21), 0)
    cv2.normalize(depth, depth, 0, 1, cv2.NORM_MINMAX)

    # Colorize: COLORMAP_JET  blue=far → green=mid → red=near
    depth_u8    = (depth * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    return depth, depth_color


def depth_to_distance_km(depth_val, cloud_height_m, fov_deg):
    """
    Depth value (0–1) → estimated slant distance in km.
    Uses trigonometry: closer clouds (higher depth) are nearer to cloud_height_m;
    farther (lower depth) are assumed to be 1.5–3x that height away (oblique angle).
    """
    # depth=1 → distance = cloud_height_m (directly overhead)
    # depth=0 → distance = 3 * cloud_height_m (far horizon, shallow angle)
    distance_m = cloud_height_m * (1.0 + 2.0 * (1.0 - float(depth_val)))
    return round(distance_m / 1000.0, 2)


# ─────────────────────────── BOUNDING BOX FUNCTION ─────────────
def draw_boxes_on_frame(frame, speed_kmh, direction, cloud_type, height_m,
                         dist_5, dist_15, elapsed_sec, prev_gray=None, delta_t=None,
                         fov=75, time_to_exit_min=999):
    OUT_W = frame.shape[1]
    OUT_H = frame.shape[0]
    sky_h = int(OUT_H * 0.78)   # slightly more sky area

    # Pre-compute full dense optical flow if prev frame available
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sky_gray = gray[:sky_h, :]
    full_flow = None
    if prev_gray is not None and delta_t is not None and delta_t > 0:
        prev_sky = prev_gray[:sky_h, :]
        full_flow = cv2.calcOpticalFlowFarneback(
            prev_sky, sky_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

    # Detect clouds with improved method
    boxes, _ = detect_clouds(frame, sky_h)

    # ── Compute pseudo stereo depth map for full sky region ──
    depth_map, depth_color = compute_pseudo_depth_map(frame, sky_h)

    for (x, y, w, h, _) in boxes:
        pad = 8
        x1 = max(0,       x - pad);    y1 = max(0,       y - pad)
        x2 = min(OUT_W-1, x+w + pad);  y2 = min(sky_h,   y+h + pad)

        # ── Per-cloud speed from optical flow ROI ──
        if full_flow is not None:
            roi_flow = full_flow[y1:y2, x1:x2]
            if roi_flow.size > 0:
                mag, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
                roi_pixel_disp = float(np.median(mag))
                if roi_pixel_disp > 0.1:
                    _, cloud_speed_kmh, _, _, _, _ = pixels_to_kmh(
                        roi_pixel_disp, delta_t, cloud_type, OUT_W, fov
                    )
                else:
                    cloud_speed_kmh = 0.0
            else:
                cloud_speed_kmh = speed_kmh
        else:
            cloud_speed_kmh = speed_kmh

        # ── Stereo Depth Overlay inside box ──
        roi_depth_color = depth_color[y1:y2, x1:x2]
        roi_frame       = frame[y1:y2, x1:x2]
        if roi_depth_color.shape == roi_frame.shape and roi_frame.size > 0:
            # Blend depth colormap (40%) with original frame (60%)
            cv2.addWeighted(roi_depth_color, 0.40, roi_frame, 0.60, 0,
                            frame[y1:y2, x1:x2])

        # Estimated distance from depth at box center
        cy_box = min((y1 + y2) // 2, depth_map.shape[0] - 1)
        cx_box = min((x1 + x2) // 2, depth_map.shape[1] - 1)
        center_depth = float(depth_map[cy_box, cx_box])
        est_dist_km  = depth_to_distance_km(center_depth, height_m, fov)

        # Glow effect
        glow = frame.copy()
        cv2.rectangle(glow, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 100), 4)
        cv2.addWeighted(glow, 0.3, frame, 0.7, 0, frame)

        # Main box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

        # Corner ticks
        t = 14
        for (px_, py_, sdx, sdy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px_, py_), (px_+sdx*t, py_),    (0, 255, 60), 2)
            cv2.line(frame, (px_, py_), (px_, py_+sdy*t),    (0, 255, 60), 2)

        # Per-cloud speed + depth distance label
        label = f"{cloud_speed_kmh:.1f} km/h  |  D:{est_dist_km:.1f}km"
        fs    = 0.46
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        lx = x1
        ly = y1 - 5 if y1 - 5 - th > 2 else y1 + th + 6
        cv2.rectangle(frame, (lx-2, ly-th-3), (lx+tw+6, ly+bl+1), (0, 150, 55), -1)
        cv2.rectangle(frame, (lx-2, ly-th-3), (lx+tw+6, ly+bl+1), (0, 255, 100), 1)
        cv2.putText(frame, label, (lx+3, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

    # ── HUD top-left ──
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (360,125), (0,0,0), -1)
    cv2.addWeighted(ov, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (0,0), (360,125), (0,200,80), 1)

    def txt(t, y, sc=0.52, c=(255,255,255), b=1):
        cv2.putText(frame, t, (12,y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, b, cv2.LINE_AA)

    txt(f"Cloud  : {cloud_type}",   22, c=(140,230,255), b=2)
    txt(f"Height : {height_m:,} m", 43)
    txt(f"Speed  : {speed_kmh:.1f} km/h  ({speed_kmh/3.6:.2f} m/s)", 64, c=(80,255,160))

    # Smart +5 / +15 min — show "OUT OF FRAME" if cloud will have exited by then
    lbl_5  = f"~{dist_5:.2f} km"  if time_to_exit_min > 5  else "OUT OF FRAME"
    lbl_15 = f"~{dist_15:.2f} km" if time_to_exit_min > 15 else "OUT OF FRAME"
    hud_color_4 = (80, 80, 255) if (time_to_exit_min <= 5) else (200, 200, 200)
    txt(f"Dir:{direction}  +5m:{lbl_5}  +15m:{lbl_15}", 86, sc=0.40, c=hud_color_4)

    mins = int(elapsed_sec)//60;  secs = int(elapsed_sec)%60
    cv2.putText(frame, f"T+ {mins:02d}:{secs:02d}",
                (OUT_W-155, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,180), 2, cv2.LINE_AA)

    # Direction arrow
    dir_vec = {"East":(1,0),"West":(-1,0),"North":(0,-1),"South":(0,1)}.get(direction,(1,0))
    cx, cy  = OUT_W//2, OUT_H - 35
    cv2.arrowedLine(frame, (cx,cy),
                    (int(cx+dir_vec[0]*55), int(cy+dir_vec[1]*55)),
                    (255,255,255), 3, tipLength=0.35)
    cv2.putText(frame, direction, (cx-30, cy+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # ── Depth Legend (colorbar) — bottom right ──
    bar_x, bar_y, bar_w, bar_h = OUT_W - 120, OUT_H - 90, 18, 70
    for i in range(bar_h):
        val   = int(255 * (1.0 - i / bar_h))
        color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
        cv2.line(frame, (bar_x, bar_y + i), (bar_x + bar_w, bar_y + i), color, 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    cv2.putText(frame, "Near", (bar_x + bar_w + 4, bar_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, "Far",  (bar_x + bar_w + 4, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Depth", (bar_x - 2, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def generate_boxed_video(input_path, output_path, speed_kmh, speed_mps,
                          direction, cloud_type, height_m, dist_5, dist_15, fov=75,
                          time_to_exit_min=999):
    cap     = cv2.VideoCapture(input_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    OUT_W, OUT_H = 960, 540
    delta_t = 1.0 / fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, OUT_H))

    prev_gray = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame   = cv2.resize(frame, (OUT_W, OUT_H))
        elapsed = frame_idx / fps
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame   = draw_boxes_on_frame(
            frame, speed_kmh, direction, cloud_type,
            height_m, dist_5, dist_15, elapsed,
            prev_gray=prev_gray, delta_t=delta_t, fov=fov,
            time_to_exit_min=time_to_exit_min
        )
        out.write(frame)
        prev_gray = gray
        frame_idx += 1

    cap.release()
    out.release()


# ─────────────────────────── UI ────────────────────────────────
st.markdown('<p class="main-header">☁️ CloudVision AI</p>', unsafe_allow_html=True)
st.markdown("**Cloud Classification & Motion Prediction System**")
st.divider()

tab1, tab2 = st.tabs(["🎬 Video Analysis", "🖼️ Multi Image Analysis"])

# ══════════════════════════ VIDEO TAB ══════════════════════════
with tab1:
    st.subheader("Upload Cloud Video")

    uploaded_video = st.file_uploader("Choose a video file",
                                       type=["mp4","avi","mov"], key="video_upload")
    fov_video = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
                          help="Phone: 70-80° | Wide angle: 90-120° | Telephoto: 30-50°")

    if uploaded_video is not None:

        st.subheader("📹 Uploaded Video")
        st.video(uploaded_video)

        uploaded_video.seek(0)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()

        with st.spinner("🔍 Analysing video..."):
            cap          = cv2.VideoCapture(tfile.name)
            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            sample_frames = []
            for pt in [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pt))
                ret, frm = cap.read()
                if ret:
                    sample_frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            cloud_type, avg_conf = predict_cloud_type(sample_frames)

            frame_gap   = max(1, int(fps))
            delta_t_sec = frame_gap / fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0);         ret1, f1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_gap); ret2, f2 = cap.read()
            cap.release()

        if ret1 and ret2:
            fw = f1.shape[1]
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            pixel_disp, avg_angle = compute_optical_flow(g1, g2)
            direction = angle_to_direction(np.degrees(avg_angle))

            speed_mps, speed_kmh, deg_per_px, theta_deg, distance_m, height_m = \
                pixels_to_kmh(pixel_disp, delta_t_sec, cloud_type, fw, fov_video)
            pixel_speed = pixel_disp / delta_t_sec
            dist_5  = speed_kmh * (5  / 60)
            dist_15 = speed_kmh * (15 / 60)

            st.divider()
            st.subheader("📊 Analysis Results")

            # Density & Visibility compute
            sample_bgr = cv2.cvtColor(sample_frames[len(sample_frames)//2], cv2.COLOR_RGB2BGR)
            sky_h_sample = int(sample_bgr.shape[0] * 0.78)
            cov_pct, den_label, den_color = compute_cloud_density(sample_bgr, sky_h_sample)
            vis_verdict, vis_reason, vis_color, time_to_exit_min = predict_visibility(
                cloud_type, speed_kmh, direction, cov_pct, fov_video)

            show_metrics(cloud_type, avg_conf, direction, height_m, fov_video,
                         fw, pixel_disp, delta_t_sec, deg_per_px, theta_deg,
                         distance_m, speed_mps, speed_kmh, dist_5, dist_15,
                         coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
                         vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
                         time_to_exit_min=time_to_exit_min)

            st.divider()
            st.subheader("🎬 Output Videos")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📦 Real Video with Cloud Boxes**")
                st.caption("Tumhara actual video — har cloud ke around box aur speed")
                if st.button("🟩 Generate Boxed Video", key="vid_box"):
                    with st.spinner("⏳ Processing video and drawing boxes..."):
                        tmp_box = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmp_box.close()
                        generate_boxed_video(
                            tfile.name, tmp_box.name,
                            speed_kmh, speed_mps, direction,
                            cloud_type, height_m, dist_5, dist_15, fov=fov_video,
                            time_to_exit_min=time_to_exit_min
                        )
                        with open(tmp_box.name, "rb") as f:
                            vdata = f.read()
                        st.success("✅ Boxed video ready!")
                        st.video(vdata)
                        st.download_button("📥 Download Boxed Video", data=vdata,
                                           file_name=f"cloud_{cloud_type}_boxes.mp4",
                                           mime="video/mp4")
                        try: os.unlink(tmp_box.name)
                        except: pass

            with col2:
                st.markdown("**🔮 Prediction Video (+5 min / +15 min)**")
                st.caption("Simulated animation — agle 15 min mein cloud kahan jayega")
                if st.button("🎬 Generate Prediction Video", key="vid_pred"):
                    with st.spinner("⏳ Prediction video generate ho rahi hai..."):
                        viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
                                                    direction=direction, pixel_speed=pixel_speed)
                        tmp_pred = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmp_pred.close()
                        viz.save_video_with_prediction(tmp_pred.name, prediction_minutes=15)
                        with open(tmp_pred.name, "rb") as f:
                            vdata2 = f.read()
                        st.success("✅ Prediction video ready!")
                        st.video(vdata2)
                        st.download_button("📥 Download Prediction Video", data=vdata2,
                                           file_name=f"cloud_{cloud_type}_prediction.mp4",
                                           mime="video/mp4")
                        try: os.unlink(tmp_pred.name)
                        except: pass

        try: os.unlink(tfile.name)
        except: pass

# ══════════════════════ MULTI IMAGE TAB ════════════════════════
with tab2:
    st.subheader("Upload Multiple Cloud Images")

    uploaded_images = st.file_uploader("Choose cloud images (minimum 2)",
                                        type=["jpg","jpeg","png"],
                                        accept_multiple_files=True, key="img_upload")
    interval   = st.number_input("⏱️ Time Gap Between Images (seconds)", min_value=1, value=60)
    fov_images = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
                           help="Phone: 70-80° | Wide angle: 90-120°", key="fov_images")

    if uploaded_images:
        st.success(f"✅ {len(uploaded_images)} image(s) uploaded")

        with st.expander("🖼️ View Uploaded Images"):
            img_cols = st.columns(min(len(uploaded_images), 4))
            for i, img_file in enumerate(uploaded_images):
                img_file.seek(0)
                with img_cols[i % 4]:
                    st.image(img_file, caption=img_file.name, use_container_width=True)

        if len(uploaded_images) >= 2:
            with st.spinner("🔍 Analysing images..."):
                dirs_deg, px_disps = [], []
                for i in range(len(uploaded_images) - 1):
                    uploaded_images[i].seek(0); uploaded_images[i+1].seek(0)
                    img1 = np.array(Image.open(uploaded_images[i]).convert("RGB").resize((640,480)))
                    img2 = np.array(Image.open(uploaded_images[i+1]).convert("RGB").resize((640,480)))
                    med, ang = compute_optical_flow(
                        cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
                        cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                    )
                    px_disps.append(med); dirs_deg.append(np.degrees(ang))

                avg_disp    = float(np.mean(px_disps))
                direction   = angle_to_direction(float(np.mean(dirs_deg)))
                pil_imgs    = []
                for f in uploaded_images:
                    f.seek(0); pil_imgs.append(Image.open(f))
                cloud_type, avg_conf = predict_cloud_type(pil_imgs)

            fw = 640
            speed_mps, speed_kmh, deg_per_px, theta_deg, distance_m, height_m = \
                pixels_to_kmh(avg_disp, interval, cloud_type, fw, fov_images)
            pixel_speed = avg_disp / interval
            dist_5  = speed_kmh * (5  / 60)
            dist_15 = speed_kmh * (15 / 60)

            st.divider()
            st.subheader("📊 Analysis Results")

            # Density & Visibility compute from first image
            uploaded_images[0].seek(0)
            first_bgr = cv2.resize(
                cv2.cvtColor(np.array(Image.open(uploaded_images[0]).convert("RGB")), cv2.COLOR_RGB2BGR),
                (640, 480)
            )
            sky_h_img = int(first_bgr.shape[0] * 0.78)
            cov_pct, den_label, den_color = compute_cloud_density(first_bgr, sky_h_img)
            vis_verdict, vis_reason, vis_color, time_to_exit_min = predict_visibility(
                cloud_type, speed_kmh, direction, cov_pct, fov_images)

            show_metrics(cloud_type, avg_conf, direction, height_m, fov_images,
                         fw, avg_disp, interval, deg_per_px, theta_deg,
                         distance_m, speed_mps, speed_kmh, dist_5, dist_15,
                         coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
                         vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
                         time_to_exit_min=time_to_exit_min)

            st.divider()
            st.subheader("🎬 Output")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📦 Images with Cloud Boxes**")
                if st.button("🟩 Show Boxes on Images", key="img_box"):
                    cols3 = st.columns(min(len(uploaded_images), 3))
                    for i, img_file in enumerate(uploaded_images):
                        img_file.seek(0)
                        arr = cv2.resize(
                            cv2.cvtColor(np.array(Image.open(img_file).convert("RGB")),
                                         cv2.COLOR_RGB2BGR), (640, 480)
                        )
                        prev_arr = None
                        if i > 0:
                            uploaded_images[i-1].seek(0)
                            prev_arr_bgr = cv2.resize(
                                cv2.cvtColor(np.array(Image.open(uploaded_images[i-1]).convert("RGB")),
                                             cv2.COLOR_RGB2BGR), (640, 480)
                            )
                            prev_arr = cv2.cvtColor(prev_arr_bgr, cv2.COLOR_BGR2GRAY)
                        arr = draw_boxes_on_frame(arr, speed_kmh, direction, cloud_type,
                                                   height_m, dist_5, dist_15, i * interval,
                                                   prev_gray=prev_arr, delta_t=float(interval),
                                                   fov=fov_images,
                                                   time_to_exit_min=time_to_exit_min)
                        with cols3[i % 3]:
                            st.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),
                                     caption=f"Image {i+1}", use_container_width=True)

            with col2:
                st.markdown("**🔮 Prediction Video**")
                if st.button("🎬 Generate Prediction Video", key="img_pred"):
                    with st.spinner("Generating..."):
                        viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
                                                    direction=direction, pixel_speed=pixel_speed)
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmp.close()
                        viz.save_video_with_prediction(tmp.name, prediction_minutes=15)
                        with open(tmp.name, "rb") as f:
                            vdata = f.read()
                        st.success("✅ Ready!")
                        st.video(vdata)
                        st.download_button("📥 Download", data=vdata,
                                           file_name=f"cloud_{cloud_type}_prediction.mp4",
                                           mime="video/mp4")
                        try: os.unlink(tmp.name)
                        except: pass
        else:
            st.warning("⚠️ Please upload at least 2 images.")