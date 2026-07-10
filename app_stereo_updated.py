import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import math
import subprocess
import shutil
import json
import datetime
from collections import Counter
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from motion_visualizer import CloudMotionVisualizer

# ===== SOLAR SHADOW & SUN TRACKING =====
# pip install pvlib pandas
try:
    import pvlib
    import pandas as pd
except ImportError:
    pvlib = None
    pd = None

def get_solar_position(lat, lon, timestamp):
    if pvlib is None:
        return None, None
    times = pd.DatetimeIndex([timestamp])
    sol = pvlib.solarposition.get_solarposition(times, lat, lon)
    return float(sol["azimuth"].iloc[0]), float(sol["elevation"].iloc[0])


def extract_exif_datetime(pil_image):
    """Extract capture datetime from EXIF. Returns datetime or None."""
    try:
        exif_data = pil_image._getexif()
        if exif_data is None:
            return None
        for tag_id in (36867, 36868, 306):
            if tag_id in exif_data:
                return datetime.datetime.strptime(exif_data[tag_id], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def extract_video_datetime(video_path):
    """Extract creation_time from MP4/MOV via ffprobe. Returns datetime or None."""
    if not shutil.which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, timeout=10
        )
        meta = json.loads(result.stdout)
        tags = meta.get("format", {}).get("tags", {})
        for key in ("creation_time", "com.apple.quicktime.creationdate"):
            val = tags.get(key)
            if val:
                val = val.rstrip("Z").split(".")[0]
                return datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
    except Exception:
        pass
    return None


def estimate_sun_elevation_from_image(frame_bgr):
    """
    Estimate sun elevation from sky brightness when sun is NOT visible in frame.

    Uses 4 monocular cues:
      1. Overall sky brightness  → proxy for solar irradiance
      2. Horizon glow ratio      → low ratio = high sun, high ratio = low sun
      3. Blue channel dominance  → clear vs overcast sky
      4. Saturation              → confidence modifier

    Returns: (elevation_deg, confidence, method_note)
    """
    sky_h = int(frame_bgr.shape[0] * 0.78)
    sky   = frame_bgr[:sky_h, :]
    H, W  = sky.shape[:2]

    sky_f  = sky.astype(np.float32)
    hsv    = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV).astype(np.float32)
    bright = hsv[:, :, 2] / 255.0

    mean_bright  = float(np.mean(bright))
    horizon_mean = float(np.mean(bright[int(H * 0.80):, :]))
    top_mean     = float(np.mean(bright[:int(H * 0.20), :]))
    horizon_ratio = horizon_mean / (top_mean + 1e-6)

    b_ch = sky_f[:, :, 0] / 255.0
    r_ch = sky_f[:, :, 2] / 255.0
    blue_ratio = float(np.mean(b_ch)) / (float(np.mean(r_ch)) + 1e-6)
    mean_sat   = float(np.mean(hsv[:, :, 1] / 255.0))

    if mean_bright < 0.15:
        el_base = 2.0
        note = "Very dark sky — sun likely below horizon or nighttime"
        conf = "low"
    elif mean_bright < 0.30:
        if horizon_ratio > 1.3:
            el_base = 8.0 + (horizon_ratio - 1.3) * 10
            note = "Horizon glow detected — estimated sunrise/sunset angle"
            conf = "medium"
        else:
            el_base = 15.0 + mean_bright * 40
            note = "Dim sky — low sun elevation estimated from brightness"
            conf = "low"
    elif mean_bright < 0.55:
        el_base = 25.0 + (mean_bright - 0.30) / 0.25 * 30
        if horizon_ratio > 1.15:
            el_base -= 10
        note = "Moderate brightness — mid-range elevation estimated"
        conf = "medium"
    else:
        el_base = 55.0 + (mean_bright - 0.55) / 0.45 * 25
        note = "Bright sky — high elevation estimated (near noon)"
        conf = "medium" if blue_ratio > 1.1 else "low"

    if mean_sat < 0.10 and blue_ratio < 1.05:
        conf = "low"
        note += " (overcast — estimate less reliable)"

    return round(float(np.clip(el_base, 0.0, 85.0)), 1), conf, note


def detect_sun_in_frame(frame_bgr):
    """
    Detect the sun by finding the brightest spot in the sky region.

    Looks at the top 78% of the frame (sky), finds the pixel with maximum
    brightness. If that brightness exceeds 240 → sun is visible.

    Args:
        frame_bgr: BGR image (numpy array from cv2).

    Returns:
        (sun_x, sun_y, True)    — pixel position of sun if detected
        (None,  None,  False)   — if sun is not visible
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sky_h = int(frame_bgr.shape[0] * 0.78)
    sky = gray[:sky_h, :]

    # Sabse bright pixel dhundho
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sky)

    # Agar brightness bahut high hai = sun visible
    if max_val > 240:
        sun_x, sun_y = max_loc
        return sun_x, sun_y, True   # pixel position
    return None, None, False


def sun_pixel_to_azimuth(sun_x, frame_width, fov_deg, device_heading=0):
    """
    Convert sun's pixel X position to an estimated azimuth (compass bearing).

    Calculates the angular offset of the sun from the frame center using
    the camera's field of view, then adds the device heading to get true azimuth.

    Args:
        sun_x:          Horizontal pixel position of the sun in the frame.
        frame_width:    Total width of the frame in pixels.
        fov_deg:        Camera horizontal field of view in degrees.
        device_heading: Compass heading the camera is pointing (0=N, 90=E, etc.).

    Returns:
        Estimated sun azimuth in degrees [0, 360).
    """
    # Center se kitna door hai sun
    offset_px = sun_x - frame_width / 2
    offset_deg = offset_px * (fov_deg / frame_width)
    azimuth = (device_heading + offset_deg) % 360
    return azimuth


def sun_azimuth_to_direction(azimuth_deg):
    """Convert sun azimuth (0=N, 90=E, 180=S, 270=W) to compass label."""
    if azimuth_deg is None:
        return "Unknown"
    a = azimuth_deg % 360
    if   a < 22.5 or a >= 337.5: return "North"
    elif a < 67.5:  return "NE"
    elif a < 112.5: return "East"
    elif a < 157.5: return "SE"
    elif a < 202.5: return "South"
    elif a < 247.5: return "SW"
    elif a < 292.5: return "West"
    else:           return "NW"


def get_cloud_sun_alignment(cloud_direction, sun_azimuth_deg):
    """
    Check if cloud is moving TOWARD or AWAY from the sun.
    Returns: alignment_status, angle_diff, description
    """
    if sun_azimuth_deg is None:
        return "unknown", None, "Sun position unavailable (enable location)"

    # Map cloud direction to azimuth
    dir_to_az = {"North": 0, "South": 180, "East": 90, "West": 270}
    cloud_az = dir_to_az.get(cloud_direction, 0)

    # Angular difference between cloud movement and sun direction
    diff = abs(cloud_az - sun_azimuth_deg) % 360
    if diff > 180:
        diff = 360 - diff  # shortest arc

    if diff < 30:
        return "toward_sun", diff, f"Cloud is moving directly toward the sun ({diff:.0f}° offset). Shadow will likely cross your solar panel soon."
    elif diff < 90:
        return "glancing", diff, f"Cloud path is at {diff:.0f}° from sun direction — glancing alignment. Shadow may partially affect the panel."
    elif diff < 150:
        return "crossing", diff, f"Cloud is moving roughly perpendicular to the sun ({diff:.0f}° offset). Shadow will cross and clear quickly."
    else:
        return "away_from_sun", diff, f"Cloud is moving away from the sun ({diff:.0f}° offset). Shadow risk is low."

# ═══════════════════════════════════════════════════════════════



# ─────────────────────────── CONFIG ────────────────────────────
st.set_page_config(page_title="CloudVision AI", page_icon="☁️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #070d14;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1280px; }

/* ── Header ── */
.cv-header {
    display: flex; align-items: center; gap: 16px;
    padding: 28px 0 8px 0; border-bottom: 1px solid #1a2d44;
    margin-bottom: 24px;
}
.cv-logo {
    width: 48px; height: 48px; border-radius: 12px;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
    box-shadow: 0 0 24px rgba(14,165,233,0.35);
}
.cv-title { font-size: 1.75rem; font-weight: 700; color: #f0f6ff; letter-spacing: -0.02em; }
.cv-sub   { font-size: 0.82rem; color: #4a6580; font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1a2d44;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px 8px 0 0;
    color: #4a6580;
    padding: 10px 22px;
    font-weight: 500;
    font-size: 0.88rem;
    transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: #94b8d4; background: #0d1a27; }
.stTabs [aria-selected="true"] {
    background: #0d1a27 !important;
    color: #38bdf8 !important;
    border-color: #1a2d44 #1a2d44 transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0d1a27;
    border: 1px solid #1a2d44;
    border-radius: 12px;
    padding: 18px 20px !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #2a4a64; }
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #4a6580 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}
[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: #e2ecf6 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 0.875rem;
    letter-spacing: 0.01em;
    transition: opacity 0.15s, transform 0.1s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #0d1a27;
    border: 1.5px dashed #1e3650;
    border-radius: 12px;
    padding: 12px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #0ea5e9; }

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0d1a27;
    border: 1px solid #1a2d44;
    border-radius: 10px;
}
[data-testid="stExpander"] summary {
    color: #94b8d4 !important;
    font-size: 0.85rem;
    font-weight: 500;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #38bdf8 !important; }

/* ── Divider ── */
hr { border-color: #1a2d44 !important; margin: 20px 0 !important; }

/* ── Subheader ── */
h2, h3 { color: #c8dff0 !important; font-weight: 600 !important; }

/* ── Number input / selectbox ── */
[data-baseweb="input"], [data-baseweb="select"] {
    background: #0d1a27 !important;
    border-color: #1a2d44 !important;
    border-radius: 8px !important;
    color: #e2ecf6 !important;
}

/* ── Markdown tables ── */
table { width: 100%; border-collapse: collapse; }
th { background: #0d1a27; color: #4a6580; font-family: 'JetBrains Mono', monospace;
     font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
     padding: 10px 14px; border-bottom: 1px solid #1a2d44; }
td { color: #c8dff0; padding: 9px 14px; font-family: 'JetBrains Mono', monospace;
     font-size: 0.82rem; border-bottom: 1px solid #0f1e2d; }
tr:last-child td { border-bottom: none; }

/* ── Success / Warning / Info ── */
[data-testid="stAlert"] { border-radius: 10px !important; border-width: 1px !important; }

/* ── Video ── */
video { border-radius: 10px; border: 1px solid #1a2d44; }

/* ── Caption ── */
.stCaption { color: #4a6580 !important; font-size: 0.78rem !important; }

/* ── Stat pill used in custom cards ── */
.cv-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.04em;
    background: #0a1929;
    border: 1px solid #1a2d44;
    color: #94b8d4;
    margin-right: 4px;
}

/* ── Section label eyebrow ── */
.cv-eyebrow {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a6580;
    margin-bottom: 10px;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #0d1a27 !important;
    border: 1px solid #1a2d44 !important;
    color: #38bdf8 !important;
    font-weight: 500 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #38bdf8 !important;
    background: #0a2035 !important;
}
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
st.sidebar.subheader("Stereo Camera Setup")

baseline_m = st.sidebar.number_input(
    "Camera 1 ↔ Camera 2 Distance (meters)",
    min_value=0.1,
    value=50.0,
    step=0.1
)

focal_length_mm = st.sidebar.number_input(
    "Camera Focal Length (mm)",
    value=4.0
)

sensor_width_mm = st.sidebar.number_input(
    "Sensor Width (mm)",
    value=6.4
)
cloud_emoji  = {"Cumulus":"⛅","Altocumulus":"🌤️","Cirrus":"🌬️",
                "ClearSky":"☀️","Stratocumulus":"🌥️","Cumulonimbus":"⛈️","Mixed":"🌦️"}

# ─────────────────────────── HELPERS ───────────────────────────
def stereo_cloud_height(
    cloud_x_cam1,
    cloud_x_cam2,
    baseline_m,
    focal_length_px
):
    disparity = abs(cloud_x_cam1 - cloud_x_cam2)

    if disparity < 1:
        return None

    return (focal_length_px * baseline_m) / disparity
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
    Calculates cloud coverage percentage within the sky region.
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
    Estimates how long the cloud will remain visible in the sky (until dissipation).
    Estimate is based on cloud type's atmospheric lifetime, speed, and coverage.
    Returns: verdict, reason, color, lifetime_min
    """

    # ── Typical atmospheric lifetimes by cloud type (minutes) ──
    # Based on meteorological averages:
    # Cumulus: 10–60 min (convective, quickly form/dissipate)
    # Altocumulus: 30–120 min (mid-level, moderate lifetime)
    # Cirrus: 60–360 min (high-altitude ice, very long lasting)
    # ClearSky: no cloud
    # Stratocumulus: 60–480 min (layer cloud, very persistent)
    # Cumulonimbus: 30–90 min (active storm cell, intense but burns out)
    # Mixed: 20–90 min (varied)
    cloud_lifetime_range = {
        "Cumulus":       (10,  60),
        "Altocumulus":   (30,  120),
        "Cirrus":        (60,  360),
        "ClearSky":      (0,   0),
        "Stratocumulus": (60,  480),
        "Cumulonimbus":  (30,  90),
        "Mixed":         (20,  90),
    }

    if cloud_type == "ClearSky":
        return (
            "☀️ Clear Sky — No Cloud",
            "Sky is currently clear. No clouds are visible.",
            "#22c55e",
            999,
        )

    if speed_kmh < 0.5:
        lo, hi = cloud_lifetime_range.get(cloud_type, (20, 90))
        mid = (lo + hi) // 2
        return (
            f"🟡 Stationary — ~{mid} min remaining",
            (f"{cloud_type} clouds typically last {lo}–{hi} min. "
             f"This cloud is currently stationary and will remain visible for approximately {mid} more minutes."),
            "#f59e0b",
            float(mid),
        )

    lo, hi = cloud_lifetime_range.get(cloud_type, (20, 90))

    # High coverage = more moisture/mass = longer lifetime
    coverage_bonus = (coverage_pct / 100.0) * (hi - lo) * 0.3

    # Fast-moving clouds dissipate faster (turbulence, mixing)
    # Normalize: >60 km/h = fast, cuts lifetime by up to 30%
    speed_factor = max(0.7, 1.0 - (speed_kmh / 200.0))

    lifetime_min = ((lo + hi) / 2 + coverage_bonus) * speed_factor
    lifetime_min = round(max(lo, min(hi, lifetime_min)))

    # ── Verdict tiers ──
    if lifetime_min > 60:
        hrs = lifetime_min // 60
        mins = lifetime_min % 60
        time_str = f"~{hrs}h {mins}min" if mins else f"~{hrs}h"
        verdict = f"🟢 Will Stay — {time_str} remaining"
        reason  = (f"{cloud_type} clouds are long-lasting (typically {lo}–{hi} min). "
                   f"With {coverage_pct}% coverage and a speed of {speed_kmh:.1f} km/h, "
                   f"this cloud is expected to remain visible for approximately {time_str}.")
        color   = "#22c55e"
    elif lifetime_min > 20:
        verdict = f"🟡 Moderate — ~{lifetime_min} min remaining"
        reason  = (f"{cloud_type} clouds typically last {lo}–{hi} min. "
                   f"Based on the current speed ({speed_kmh:.1f} km/h) and coverage ({coverage_pct}%), "
                   f"this cloud is expected to remain visible for approximately {lifetime_min} more minutes.")
        color   = "#f59e0b"
    else:
        verdict = f"🔴 Dissipating — ~{lifetime_min} min remaining"
        reason  = (f"{cloud_type} clouds dissipate quickly (lifetime {lo}–{hi} min). "
                   f"The high speed ({speed_kmh:.1f} km/h) and low coverage ({coverage_pct}%) suggest "
                   f"this cloud will disappear from the sky in approximately {lifetime_min} minutes.")
        color   = "#ef4444"

    return verdict, reason, color, float(lifetime_min)


def predict_cloud_type(frames_or_images):
    """Batch-predict cloud type from a list of frames or PIL images."""
    batch = []
    for img in frames_or_images:
        img_pil = Image.fromarray(img).resize((224, 224)) if isinstance(img, np.ndarray) else img.resize((224, 224))
        batch.append(image.img_to_array(img_pil) / 255.0)

    batch_arr = np.stack(batch, axis=0)          # shape: (N, 224, 224, 3)
    preds_all = model.predict(batch_arr, verbose=0)  # single forward pass

    preds = [class_names[np.argmax(p)] for p in preds_all]
    confs = [float(np.max(p)) * 100 for p in preds_all]
    return Counter(preds).most_common(1)[0][0], float(np.mean(confs))

def compute_solar_shadow_forecast(cloud_type, height_m, speed_mps, speed_kmh,
                                   direction, pixel_disp, frame_width, fov_deg,
                                   coverage_pct):
    """
    Camera is mounted on or near the solar plant.
    When a cloud enters the field of view, its shadow falls somewhere on the ground.

    Physics:
      - Cloud's current angular position from frame center → ground_offset
        ground_offset = height_m * tan(angle_from_zenith)
      - Shadow ground offset = cloud's horizontal distance from directly overhead
      - If cloud is moving toward the camera center → shadow WILL hit the panel
      - Time to shadow = ground_offset / speed_mps
      - If cloud is moving away from center → shadow WON'T hit the panel

    Returns dict with all forecast info.
    """
    cloud_power_factor = {
        "Cumulus": 0.55, "Altocumulus": 0.45, "Cirrus": 0.18,
        "ClearSky": 0.0, "Stratocumulus": 0.72, "Cumulonimbus": 0.85, "Mixed": 0.50
    }

    if cloud_type == "ClearSky":
        return {
            "will_hit": False,
            "reason": "☀️ Sky is completely clear — no clouds, no shadow.",
            "status": "clear",
            "shadow_time_min": None,
            "power_drop_pct": 0.0,
            "ground_offset_m": 0.0,
        }

    # Angular position of cloud from frame center (pixels → degrees → radians)
    deg_per_px     = fov_deg / frame_width
    # pixel_disp is displacement magnitude; we use half-frame as rough center-offset
    # Center offset: assume cloud centroid is ~frame center (worst case / mean case)
    # More accurate: use half FOV = cloud is within FOV, so it's within ±fov/2 of zenith
    # We use the actual angle the cloud has already traveled (theta) as proxy for offset
    half_fov_deg   = fov_deg / 2.0
    # Angle from zenith to cloud edge (conservative: use half-FOV)
    angle_rad      = math.radians(half_fov_deg * 0.5)   # ~centre of visible sky arc
    ground_offset_m = height_m * math.tan(angle_rad)    # metres from panel

    # Direction logic: is cloud moving TOWARD overhead (will shadow hit) or AWAY?
    # "toward overhead" = cloud is off to one side and moving toward center
    # Simplified: if cloud is IN the FOV it is either overhead now or will cross overhead
    # based on direction + frame position.
    # We check: does the cloud's travel path cross the zenith column?
    # Heuristic: if cloud is moving and it's within FOV → it will pass overhead → shadow hits
    # UNLESS cloud is already past center and moving further away.

    # We estimate current cloud X position from optical flow direction
    # North/South movement means cloud crosses overhead (shadow hits)
    # East/West also crosses — it just depends on whether it already passed
    # Use pixel_disp relative to frame to estimate if approaching or receding
    
    # Simple model: cloud in FOV = overhead within half-FOV → shadow WILL hit
    # Time for shadow to reach panel = ground_offset / speed_mps
    
    if speed_mps < 0.05:
        # Nearly stationary
        return {
            "will_hit": True,
            "reason": f"Cloud is still in the sky and is practically stationary. Shadow may already be present on the solar panel or has stalled overhead.",
            "status": "stationary",
            "shadow_time_min": 0.0,
            "power_drop_pct": _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor),
            "ground_offset_m": ground_offset_m,
        }

    time_to_shadow_sec = ground_offset_m / speed_mps
    time_to_shadow_min = time_to_shadow_sec / 60.0

    # If time is very small (< 0.5 min) → shadow is basically overhead now
    # If cloud is moving away (already past zenith), offset increases over time → no hit
    # Proxy: use pixel displacement direction vs frame center
    # If the optical flow vector points toward frame center → cloud approaching
    # Rough approximation: if time_to_shadow_min < 0 conceptually, cloud already passed
    # We use: if time_to_shadow_min > time_to_exit_min-equivalent → won't cross
    
    # Practical cutoff: if it takes longer than cloud lifetime to arrive → won't hit
    cloud_lifetime = {
        "Cumulus": 35, "Altocumulus": 75, "Cirrus": 200,
        "ClearSky": 0, "Stratocumulus": 270, "Cumulonimbus": 60, "Mixed": 55
    }
    lifetime_min = cloud_lifetime.get(cloud_type, 60)

    power_drop = _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor)

    if time_to_shadow_min <= 0.3:
        return {
            "will_hit": True,
            "reason": f"Cloud is almost directly overhead — shadow is currently falling on the solar panel or is about to.",
            "status": "now",
            "shadow_time_min": 0.0,
            "power_drop_pct": power_drop,
            "ground_offset_m": ground_offset_m,
        }
    elif time_to_shadow_min <= lifetime_min:
        return {
            "will_hit": True,
            "reason": f"{cloud_type} cloud is offset by {ground_offset_m/1000:.2f} km. At {speed_kmh:.1f} km/h, the shadow will reach the solar panel in {time_to_shadow_min:.1f} minutes.",
            "status": "incoming",
            "shadow_time_min": time_to_shadow_min,
            "power_drop_pct": power_drop,
            "ground_offset_m": ground_offset_m,
        }
    else:
        return {
            "will_hit": False,
            "reason": (f"The estimated shadow arrival time for this {cloud_type} cloud is ~{time_to_shadow_min:.0f} min, "
                       f"but the cloud's expected lifetime is only ~{lifetime_min} min. "
                       f"The cloud will dissipate or exit the field of view before the shadow reaches the panel."),
            "status": "miss",
            "shadow_time_min": time_to_shadow_min,
            "power_drop_pct": 0.0,
            "ground_offset_m": ground_offset_m,
        }


def _calc_power_drop(cloud_type, coverage_pct, factor_map):
    base = factor_map.get(cloud_type, 0.50)
    cov  = (coverage_pct / 100.0) if coverage_pct is not None else 0.5
    drop = base * 0.55 * 100 + cov * base * 0.45 * 100
    return round(min(95.0, drop), 1)


def show_metrics(cloud_type, confidence, direction, height_m, fov,
                 frame_width, pixel_disp, delta_t, deg_per_px,
                 theta_deg, distance_m, speed_mps, speed_kmh, dist_5, dist_15,
                 coverage_pct=None, density_label=None, density_color=None,
                 vis_verdict=None, vis_reason=None, vis_color=None,
                 time_to_exit_min=999, solar_dist_km=1.0,
                 media_timestamp=None, timestamp_source="now",
                 image_elevation_est=None, image_elevation_conf=None,
                 image_elevation_note=None):
    emoji = cloud_emoji.get(cloud_type, "☁️")

    # ── Low confidence warning ──
    if confidence < 60.0:
        st.warning(
            f"⚠️ **Low Model Confidence ({confidence:.1f}%)** — The model is uncertain about this cloud type. "
            f"Results may be less accurate. Try uploading a clearer image or more frames."
        )

    # ── Section divider helper ──
    def section_header(icon, title):
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:28px 0 14px 0;
            padding-bottom:10px;border-bottom:1px solid #1a2d44;">
  <span style="font-size:1.1rem;">{icon}</span>
  <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
               letter-spacing:0.12em;color:#4a6580;font-weight:600;">{title}</span>
</div>
""", unsafe_allow_html=True)

    # ── Section 1: Cloud Classification ──
    section_header("☁", "Cloud Classification")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{emoji} Cloud Type",    cloud_type)
    c2.metric("🎯 Model Confidence",    f"{confidence:.1f}%")
    c3.metric("🧭 Wind Direction",      direction)
    c4.metric("📍 Estimated Altitude",  f"{height_m:,} m")

    # ── Section 2: Motion & Displacement ──
    section_header("⚡", "Motion & Displacement")
    label_5  = f"~{dist_5:.2f} km"  if dist_5  > 0 else "—"
    label_15 = f"~{dist_15:.2f} km" if dist_15 > 0 else "—"

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Wind Speed",            f"{speed_kmh:.1f} km/h")
    c6.metric("Wind Speed (m/s)",      f"{speed_mps:.2f} m/s")
    c7.metric("Projected Dist. +5 min",  label_5,  delta_color="off")
    c8.metric("Projected Dist. +15 min", label_15, delta_color="off")

    # ── Section 3: Atmospheric Analysis ──
    if coverage_pct is not None:
        section_header("📡", "Atmospheric Analysis")
        d1, d2 = st.columns(2)

        with d1:
            bar_filled = int(coverage_pct)
            st.markdown(f"""
<div style="background:#0d1a27;border:1px solid {density_color}44;border-radius:12px;padding:22px;">
  <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
              letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">Sky Coverage</div>
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:16px;">
    <span style="font-size:2.2rem;font-weight:700;color:{density_color};font-family:'Inter',sans-serif;line-height:1;">{coverage_pct}%</span>
    <span style="font-size:0.82rem;color:{density_color};font-weight:600;padding:3px 10px;
                 background:{density_color}18;border-radius:999px;border:1px solid {density_color}33;">{density_label}</span>
  </div>
  <div style="background:#050c14;border-radius:8px;height:10px;width:100%;overflow:hidden;margin-bottom:8px;">
    <div style="background:linear-gradient(90deg,{density_color}77,{density_color});
                width:{bar_filled}%;height:10px;border-radius:8px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:6px;">
    <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">Low — &lt;20%</span>
    <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">Medium — 20–55%</span>
    <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">High — &gt;55%</span>
  </div>
</div>
""", unsafe_allow_html=True)

        with d2:
            st.markdown(f"""
<div style="background:#0d1a27;border:1px solid {vis_color}44;border-radius:12px;padding:22px;height:100%;">
  <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
              letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">Cloud Lifetime Forecast</div>
  <div style="font-size:1rem;font-weight:700;color:{vis_color};margin-bottom:12px;
              padding:8px 12px;background:{vis_color}14;border-radius:8px;border-left:3px solid {vis_color};">{vis_verdict}</div>
  <div style="font-size:0.82rem;color:#7a9ab4;line-height:1.65;">{vis_reason}</div>
</div>
""", unsafe_allow_html=True)

    # ── Section 4: Solar Plant Impact Forecast ──
    section_header("☀️", "Solar Plant Impact Forecast")

    solar = compute_solar_shadow_forecast(
        cloud_type, height_m, speed_mps, speed_kmh,
        direction, pixel_disp, frame_width, fov,
        coverage_pct if coverage_pct is not None else 50.0
    )

    status = solar["status"]

    if status == "clear":
        st.markdown("""
<div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:12px;padding:24px;display:flex;align-items:center;gap:20px;">
  <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
              display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">☀️</div>
  <div>
    <div style="font-size:1.05rem;font-weight:700;color:#22c55e;margin-bottom:6px;">Clear Sky — Solar Plant Fully Safe</div>
    <div style="font-size:0.82rem;color:#4a8060;line-height:1.5;">No clouds detected. No shadow risk. Solar plant is operating at full capacity.</div>
  </div>
  <div style="margin-left:auto;text-align:right;flex-shrink:0;">
    <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Power Status</div>
    <div style="font-size:1.1rem;font-weight:700;color:#22c55e;">🟢 100%</div>
  </div>
</div>
""", unsafe_allow_html=True)

    elif status == "now":
        pdrop = solar["power_drop_pct"]
        st.markdown(f"""
<div style="background:#0d1a27;border:1.5px solid #ef444455;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:#ef444418;border:1px solid #ef444433;
                display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">⚠️</div>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:#ef4444;margin-bottom:5px;">Shadow Currently Falling on Solar Panel</div>
      <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
                  background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
        {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; {speed_kmh:.1f} km/h
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
    <div style="background:#100505;border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
      <div style="font-size:2.1rem;font-weight:700;color:#ef4444;line-height:1;">{pdrop}%</div>
    </div>
    <div style="background:#100505;border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
      <div style="font-size:1rem;font-weight:700;color:#ff6b6b;margin-top:4px;">🔴 Active Now</div>
    </div>
    <div style="background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 2;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
      <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    elif status == "stationary":
        pdrop = solar["power_drop_pct"]
        st.markdown(f"""
<div style="background:#0d1a27;border:1.5px solid #f59e0b55;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:#f59e0b18;border:1px solid #f59e0b33;
                display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🟡</div>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:#f59e0b;margin-bottom:5px;">Cloud Stationary — Shadow Present on Panel</div>
      <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
                  background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
        {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; Nearly stationary
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
    <div style="background:#100a00;border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
      <div style="font-size:2.1rem;font-weight:700;color:#f59e0b;line-height:1;">{pdrop}%</div>
    </div>
    <div style="background:#100a00;border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Movement</div>
      <div style="font-size:1rem;font-weight:700;color:#f59e0b;margin-top:4px;">⏸ Stationary</div>
    </div>
    <div style="background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 2;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
      <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    elif status == "incoming":
        pdrop    = solar["power_drop_pct"]
        arr_min  = solar["shadow_time_min"]
        off_km   = solar["ground_offset_m"] / 1000.0
        if arr_min < 10:
            urg_color = "#ef4444"; urg_icon = "🔴"
        elif arr_min < 30:
            urg_color = "#f59e0b"; urg_icon = "🟡"
        else:
            urg_color = "#22c55e"; urg_icon = "🟢"
        arr_str = f"{arr_min:.1f} min" if arr_min < 60 else f"{int(arr_min//60)}h {int(arr_min%60)}m"
        st.markdown(f"""
<div style="background:#0d1a27;border:1.5px solid {urg_color}55;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:{urg_color}18;border:1px solid {urg_color}33;
                display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">☁️</div>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:{urg_color};margin-bottom:5px;">
        {urg_icon} Shadow Will Reach Solar Plant in {arr_str}
      </div>
      <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
                  background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
        {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;">
    <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Shadow Arrives In</div>
      <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{arr_str}</div>
    </div>
    <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
      <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{pdrop}%</div>
    </div>
    <div style="background:#050c14;border:1px solid #1e3a50;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Ground Offset</div>
      <div style="font-size:2.1rem;font-weight:700;color:#38bdf8;line-height:1;">{off_km:.2f} km</div>
    </div>
    <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 3;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Forecast Analysis</div>
      <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    else:  # status == "miss"
        arr_min = solar["shadow_time_min"]
        arr_str = f"{arr_min:.0f} min" if arr_min is not None else "N/A"
        st.markdown(f"""
<div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
                display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">✅</div>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:#22c55e;margin-bottom:5px;">Shadow Will Not Reach the Solar Plant</div>
      <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
                  background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
        {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
      </div>
    </div>
  </div>
  <div style="background:#050c14;border:1px solid #22c55e22;border-radius:10px;padding:16px 18px;margin-bottom:12px;">
    <div style="font-size:0.82rem;color:#6aaa84;line-height:1.7;">{solar["reason"]}</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
    <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Power Drop</div>
      <div style="font-size:2.1rem;font-weight:700;color:#22c55e;line-height:1;">0%</div>
    </div>
    <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
      <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-top:4px;">🟢 Safe</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    # ── Section 5: Sun Position & Cloud Alignment ──
    _lat = st.session_state.get("user_lat", 28.6)
    _lon = st.session_state.get("user_lon", 77.2)

    # Use media timestamp if available, else current time
    _ts = media_timestamp if media_timestamp is not None else datetime.datetime.utcnow()
    sun_az, sun_el = get_solar_position(_lat, _lon, _ts)

    # Timestamp source badge
    _ts_badge_map = {
        "exif":    ("📷 From EXIF",        "#22c55e"),
        "ffprobe": ("🎬 From Video Meta",  "#22c55e"),
        "manual":  ("🕐 Manual Input",      "#f59e0b"),
        "now":     ("⏱️ Current Time",      "#4a6580"),
    }
    _ts_label, _ts_color = _ts_badge_map.get(timestamp_source, ("⏱️ Current Time", "#4a6580"))

    section_header("🌞", "Sun Position & Cloud Alignment")

    if sun_az is None:
        st.info("📍 Install pvlib + pandas and set your location in the sidebar for live sun tracking.")
    else:
        sun_dir  = sun_azimuth_to_direction(sun_az)
        align_status, angle_diff, align_desc = get_cloud_sun_alignment(direction, sun_az)

        align_color_map = {
            "toward_sun":    "#ef4444",
            "glancing":      "#f59e0b",
            "crossing":      "#38bdf8",
            "away_from_sun": "#22c55e",
            "unknown":       "#4a6580",
        }
        align_icon_map = {
            "toward_sun":    "🔴 Heading Toward Sun",
            "glancing":      "🟡 Glancing Sun",
            "crossing":      "🔵 Crossing Sun Path",
            "away_from_sun": "🟢 Moving Away from Sun",
            "unknown":       "❓ Unknown",
        }
        a_color = align_color_map.get(align_status, "#4a6580")
        a_label = align_icon_map.get(align_status, "")

        if sun_el < 0:
            sun_status_label = "🌙 Below Horizon"
            sun_el_color = "#4a6580"
        elif sun_el < 15:
            sun_status_label = "🌅 Near Horizon"
            sun_el_color = "#f59e0b"
        else:
            sun_status_label = "☀️ Above Horizon"
            sun_el_color = "#fbbf24"

        # ── Determine which elevation to show ──
        # pvlib = authoritative; image estimate = fallback shown alongside
        show_img_est = (image_elevation_est is not None)
        img_conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(
            image_elevation_conf, "#4a6580")

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("☀️ Sun Azimuth",   f"{sun_az:.1f}°")
        sc2.metric("📐 Sun Elevation (pvlib)", f"{sun_el:.1f}°")
        sc3.metric("🧭 Sun Direction", sun_dir)
        sc4.metric("☁️ Cloud Moving",  direction)

        # Timestamp badge + image estimate row
        badge_html = f"""
<div style="display:flex;align-items:center;gap:10px;margin:10px 0 14px 0;flex-wrap:wrap;">
  <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;padding:3px 10px;
               border-radius:999px;background:{_ts_color}18;border:1px solid {_ts_color}44;
               color:{_ts_color};">{_ts_label}: {_ts.strftime('%Y-%m-%d %H:%M UTC')}</span>"""

        if show_img_est:
            badge_html += f"""
  <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;padding:3px 10px;
               border-radius:999px;background:{img_conf_color}18;border:1px solid {img_conf_color}44;
               color:{img_conf_color};">
    📸 Image Estimate: {image_elevation_est}° ({image_elevation_conf} confidence)
  </span>"""

        badge_html += "</div>"
        st.markdown(badge_html, unsafe_allow_html=True)

        # Image elevation detail card (only when sun not visible in frame)
        if show_img_est:
            diff_el = abs(sun_el - image_elevation_est)
            st.markdown(f"""
<div style="background:#0d1a27;border:1px solid {img_conf_color}44;border-radius:12px;
            padding:16px 20px;margin-bottom:14px;">
  <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
              letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">
    📸 Image-Based Sun Elevation Estimate (Sun not visible in frame)
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;">
    <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
      <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:6px;">Image Estimate</div>
      <div style="font-size:1.6rem;font-weight:700;color:{img_conf_color};line-height:1;">
        {image_elevation_est}°</div>
    </div>
    <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
      <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:6px;">pvlib (Location+Time)</div>
      <div style="font-size:1.6rem;font-weight:700;color:#fbbf24;line-height:1;">{sun_el:.1f}°</div>
    </div>
    <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
      <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:6px;">Difference</div>
      <div style="font-size:1.6rem;font-weight:700;color:{'#22c55e' if diff_el < 10 else '#f59e0b' if diff_el < 25 else '#ef4444'};line-height:1;">
        ±{diff_el:.1f}°</div>
    </div>
  </div>
  <div style="font-size:0.8rem;color:#7a9ab4;line-height:1.6;font-style:italic;">{image_elevation_note}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:#0d1a27;border:1.5px solid {a_color}55;border-radius:14px;padding:22px;margin-top:4px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
    <div style="width:48px;height:48px;border-radius:12px;background:{a_color}18;border:1px solid {a_color}33;
                display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;">🌞</div>
    <div>
      <div style="font-size:1rem;font-weight:700;color:{a_color};margin-bottom:4px;">
        Cloud–Sun Alignment: {a_label}
      </div>
      <div style="font-size:0.78rem;font-family:'JetBrains Mono',monospace;color:#4a6580;
                  background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:3px 10px;display:inline-block;">
        {sun_status_label} &nbsp;·&nbsp; Azimuth {sun_az:.1f}° &nbsp;·&nbsp; Elevation {sun_el:.1f}°
        {'&nbsp;·&nbsp; ' + str(round(angle_diff)) + '° offset' if angle_diff is not None else ''}
      </div>
    </div>
  </div>
  <div style="font-size:0.84rem;color:#94b8d4;line-height:1.7;background:#050c14;
              border-radius:10px;padding:14px 18px;border:1px solid #1a2d44;">
    {align_desc}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # ── Export Analysis Results ──
    with st.expander("📤 Export Analysis Report"):
        import csv, io
        report_data = {
            "cloud_type": cloud_type,
            "confidence_pct": round(confidence, 1),
            "direction": direction,
            "altitude_m": height_m,
            "speed_kmh": round(speed_kmh, 1),
            "speed_mps": round(speed_mps, 2),
            "projected_dist_5min_km": round(dist_5, 2),
            "projected_dist_15min_km": round(dist_15, 2),
            "sky_coverage_pct": coverage_pct,
            "density_label": density_label,
            "visibility_forecast": vis_verdict,
            "timestamp_utc": _ts.strftime("%Y-%m-%d %H:%M UTC") if media_timestamp is not None else datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "timestamp_source": timestamp_source,
        }
        # JSON download
        import json as _json
        json_str = _json.dumps(report_data, indent=2)
        st.download_button(
            "📥 Download JSON Report",
            data=json_str,
            file_name=f"cloudvision_report_{cloud_type}.json",
            mime="application/json",
            key="dl_json_report"
        )
        # CSV download
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=report_data.keys())
        writer.writeheader()
        writer.writerow(report_data)
        st.download_button(
            "📥 Download CSV Report",
            data=csv_buf.getvalue(),
            file_name=f"cloudvision_report_{cloud_type}.csv",
            mime="text/csv",
            key="dl_csv_report"
        )
        st.code(json_str, language="json")

    with st.expander("🔬 Optical Flow — Calculation Details"):
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Camera FOV | {fov}° |
| Frame Width | {frame_width} px |
| Degrees per Pixel | {deg_per_px:.4f} °/px |
| Pixel Displacement | {pixel_disp:.2f} px over {delta_t:.2f} s |
| Angular Displacement (θ) | {theta_deg:.4f}° |
| Horizontal Distance (tan formula) | {distance_m:.2f} m |
| Derived Speed | {speed_mps:.2f} m/s → {speed_kmh:.1f} km/h |
""")

# ─────────────────────────── CLOUD DETECTION ───────────────────
def detect_clouds(frame, sky_h):
    """
    Multi-method cloud detection:
    1. Brightness threshold (white clouds)
    2. HSV saturation (low saturation = cloud/white)
    3. Combine both masks
    Uses watershed-style distance-based separation to assign distinct
    bounding boxes to individual cloud regions.
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
def get_cloud_centroid(box):
    x, y, w, h = box[:4]
    return x + w // 2

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
    lbl_5  = f"~{dist_5:.2f} km"
    lbl_15 = f"~{dist_15:.2f} km"
    txt(f"Dir:{direction}  +5m:{lbl_5}  +15m:{lbl_15}", 86, sc=0.40, c=(200, 200, 200))

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

    # ── Sun Position Arrow ──
    try:
        _lat2 = st.session_state.get("user_lat", 28.6)
        _lon2 = st.session_state.get("user_lon", 77.2)
        _sun_az2, _sun_el2 = get_solar_position(_lat2, _lon2, datetime.datetime.utcnow())
    except Exception:
        _sun_az2, _sun_el2 = None, None

    if _sun_az2 is not None and _sun_el2 > 0:
        # Draw sun arrow at bottom-center right of cloud arrow
        sun_az_rad = math.radians(_sun_az2)   # 0=N, 90=E
        # Convert azimuth to screen vector (x right=East, y down=South)
        sun_dx = math.sin(sun_az_rad)   # East component
        sun_dy = -math.cos(sun_az_rad)  # North component (inverted for screen)
        scx, scy = OUT_W//2 + 120, OUT_H - 35
        cv2.arrowedLine(frame, (scx, scy),
                        (int(scx + sun_dx * 50), int(scy + sun_dy * 50)),
                        (30, 220, 255), 2, tipLength=0.4)
        cv2.putText(frame, f"Sun {_sun_az2:.0f}", (scx - 28, scy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 220, 255), 1, cv2.LINE_AA)

        # Cloud-Sun alignment indicator
        align_status, angle_diff, _ = get_cloud_sun_alignment(direction, _sun_az2)
        align_color_cv = {
            "toward_sun":    (60, 80, 255),
            "glancing":      (60, 180, 255),
            "crossing":      (255, 200, 60),
            "away_from_sun": (60, 220, 100),
            "unknown":       (150, 150, 150),
        }.get(align_status, (150, 150, 150))
        align_text = {
            "toward_sun":    "TO SUN",
            "glancing":      "GLANCING",
            "crossing":      "CROSSING",
            "away_from_sun": "FROM SUN",
            "unknown":       "?",
        }.get(align_status, "?")
        if angle_diff is not None:
            align_label_full = f"{align_text} {angle_diff:.0f}deg"
        else:
            align_label_full = align_text
        (aw, _ah), _ = cv2.getTextSize(align_label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        ax = scx - aw // 2
        ay = scy - 14
        cv2.rectangle(frame, (ax - 3, ay - 14), (ax + aw + 4, ay + 4), (10, 10, 10), -1)
        cv2.putText(frame, align_label_full, (ax, ay),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, align_color_cv, 1, cv2.LINE_AA)

    # ── Solar Plant Shadow HUD (camera IS on solar plant) ──
    if cloud_type != "ClearSky":
        solar = compute_solar_shadow_forecast(
            cloud_type, height_m, speed_mps, speed_kmh,
            direction, OUT_W * 0.05,   # small pixel_disp proxy for HUD
            OUT_W, fov, coverage_pct=50.0
        )
        status = solar["status"]
        if status == "now" or status == "stationary":
            s_line1 = f"SHADOW ON SOLAR PLANT NOW!"
            s_line2 = f"Power drop: {solar['power_drop_pct']}%"
            box_col  = (0, 60, 220)   # red-orange
            txt_col1 = (60, 80, 255)
            txt_col2 = (60, 255, 160)
        elif status == "incoming":
            arr = solar["shadow_time_min"]
            arr_str = f"{arr:.1f}min" if arr < 60 else f"{int(arr//60)}h{int(arr%60)}m"
            s_line1 = f"Shadow arrives: {arr_str}"
            s_line2 = f"Power drop: {solar['power_drop_pct']}%"
            box_col  = (0, 160, 240)
            txt_col1 = (80, 220, 255)
            txt_col2 = (80, 255, 160)
        elif status == "miss":
            s_line1 = "Shadow will NOT hit solar plant"
            s_line2 = "Power drop: 0%  [SAFE]"
            box_col  = (0, 130, 40)
            txt_col1 = (80, 255, 120)
            txt_col2 = (80, 255, 120)
        else:
            s_line1 = "Clear sky — solar plant safe"
            s_line2 = "No shadow expected"
            box_col  = (0, 130, 40)
            txt_col1 = (80, 255, 120)
            txt_col2 = (80, 255, 120)

        (sw1, _), _ = cv2.getTextSize(s_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        (sw2, _), _ = cv2.getTextSize(s_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        box_w = max(sw1, sw2) + 22
        box_h = 58
        bx, by = OUT_W - box_w - 8, 8

        sol_ov = frame.copy()
        cv2.rectangle(sol_ov, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
        cv2.addWeighted(sol_ov, 0.62, frame, 0.38, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), box_col, 1)
        cv2.putText(frame, s_line1, (bx + 8, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col1, 1, cv2.LINE_AA)
        cv2.putText(frame, s_line2, (bx + 8, by + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col2, 1, cv2.LINE_AA)

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

    # Re-encode to H.264 so browser can play it in st.video()
    if shutil.which("ffmpeg"):
        tmp_h264 = output_path.replace(".mp4", "_h264.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", output_path,
            "-vcodec", "libx264", "-crf", "23",
            "-preset", "fast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            tmp_h264
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
            os.replace(tmp_h264, output_path)


# ─────────────────────────── UI HEADER ────────────────────────
st.markdown("""
<div class="cv-header">
  <div class="cv-logo">☁️</div>
  <div>
    <div class="cv-title">CloudVision AI</div>
    <div class="cv-sub">Cloud Classification &amp; Motion Prediction System</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎬 Video Analysis", "🖼️ Multi Image Analysis"])

# ── Solar location inputs (shared across tabs) ──
with st.sidebar:
    st.markdown("### ☀️ Solar Location")
    st.caption("Enter your location for real-time sun position tracking")
    user_lat = st.number_input("Latitude",  value=28.6, min_value=-90.0,  max_value=90.0,  step=0.1, format="%.4f", key="user_lat")
    user_lon = st.number_input("Longitude", value=77.2, min_value=-180.0, max_value=180.0, step=0.1, format="%.4f", key="user_lon")
    st.caption("🇮🇳 Default: New Delhi")

    st.markdown("---")
    st.markdown("### 🕐 Media Timestamp")
    st.caption("App auto-reads EXIF/video metadata. Override manually if needed.")
    use_manual_time = st.checkbox("✏️ Override timestamp manually", value=False, key="use_manual_time")
    if use_manual_time:
        _today = datetime.date.today()
        manual_date = st.date_input("Date", value=_today, key="manual_date")
        manual_time = st.time_input("Time (local)", value=datetime.time(12, 0), key="manual_time")
        tz_offset   = st.number_input("Timezone offset (hrs from UTC)", value=5.5,
                                       min_value=-12.0, max_value=14.0, step=0.5, key="tz_offset")
        # Convert to UTC datetime
        local_dt = datetime.datetime.combine(manual_date, manual_time)
        manual_utc = local_dt - datetime.timedelta(hours=tz_offset)
        st.session_state["manual_utc"] = manual_utc
        st.caption(f"UTC: {manual_utc.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.session_state["manual_utc"] = None

    if pvlib is not None:
        _ts_sb = st.session_state.get("manual_utc") or datetime.datetime.utcnow()
        _az, _el = get_solar_position(user_lat, user_lon, _ts_sb)
        if _az is not None:
            _sun_dir = sun_azimuth_to_direction(_az)
            if _el < 0:
                _sun_status = "🌙 Sun below horizon"
                _sun_color  = "#4a6580"
            elif _el < 15:
                _sun_status = "🌅 Sun near horizon"
                _sun_color  = "#f59e0b"
            else:
                _sun_status = "☀️ Sun above horizon"
                _sun_color  = "#fbbf24"
            st.markdown(f"""
<div style='background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:14px;margin-top:8px;'>
  <div style='font-size:0.68rem;font-family:monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Live Sun Position</div>
  <div style='font-size:0.92rem;font-weight:600;color:{_sun_color};margin-bottom:6px;'>{_sun_status}</div>
  <div style='font-size:0.78rem;color:#94b8d4;'>Azimuth: <b style='color:#e2ecf6;'>{_az:.1f}°</b> ({_sun_dir})</div>
  <div style='font-size:0.78rem;color:#94b8d4;'>Elevation: <b style='color:#e2ecf6;'>{_el:.1f}°</b></div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Install pvlib + pandas for live sun tracking:\n`pip install pvlib pandas`")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "**CloudVision AI** — Cloud classification, motion analysis & "
        "solar shadow forecasting.\n\n"
        "Model: Keras CNN · Classes: Cumulus, Altocumulus, Cirrus, "
        "ClearSky, Stratocumulus, Cumulonimbus, Mixed"
    )

# ══════════════════════════ VIDEO TAB ══════════════════════════
with tab1:
    st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🎬 Video Analysis</div>', unsafe_allow_html=True)
    st.subheader("Upload a Sky Video")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_video_cam1 = st.file_uploader(
            "Camera 1 Video",
            type=["mp4","avi","mov"],
            key="video_upload_cam1"
        )

    with col2:
        uploaded_video_cam2 = st.file_uploader(
            "Camera 2 Video",
            type=["mp4","avi","mov"],
            key="video_upload_cam2"
        )

    uploaded_video = uploaded_video_cam1
    fov_video = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
                          help="Phone: 70-80° | Wide angle: 90-120° | Telephoto: 30-50°")

    # ── Preview uploaded videos immediately ──
    if uploaded_video_cam1 is not None or uploaded_video_cam2 is not None:
        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            if uploaded_video_cam1 is not None:
                st.markdown('<div class="cv-eyebrow" style="margin-top:16px;margin-bottom:8px;">📹 Camera 1 — Uploaded Video</div>', unsafe_allow_html=True)
                uploaded_video_cam1.seek(0)
                st.video(uploaded_video_cam1)
        with prev_col2:
            if uploaded_video_cam2 is not None:
                st.markdown('<div class="cv-eyebrow" style="margin-top:16px;margin-bottom:8px;">📹 Camera 2 — Uploaded Video</div>', unsafe_allow_html=True)
                uploaded_video_cam2.seek(0)
                st.video(uploaded_video_cam2)
        st.divider()

    if uploaded_video is not None:

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

            # Density & Visibility compute
            sample_bgr = cv2.cvtColor(sample_frames[len(sample_frames)//2], cv2.COLOR_RGB2BGR)
            sky_h_sample = int(sample_bgr.shape[0] * 0.78)
            cov_pct, den_label, den_color = compute_cloud_density(sample_bgr, sky_h_sample)
            vis_verdict, vis_reason, vis_color, time_to_exit_min = predict_visibility(
                cloud_type, speed_kmh, direction, cov_pct, fov_video)

            # ── Timestamp resolution (priority: manual > ffprobe > now) ──
            _manual_utc = st.session_state.get("manual_utc")
            if _manual_utc is not None:
                media_ts      = _manual_utc
                ts_source     = "manual"
            else:
                media_ts = extract_video_datetime(tfile.name)
                ts_source = "ffprobe" if media_ts is not None else "now"
                if media_ts is None:
                    media_ts = datetime.datetime.utcnow()

            # ── Image-based sun elevation (for frames where sun not visible) ──
            img_el_est, img_el_conf, img_el_note = estimate_sun_elevation_from_image(sample_bgr)

            # ── Sun detection from frame (if sun is visible in sky) ──
            sun_x, sun_y, sun_visible = detect_sun_in_frame(sample_bgr)
            sun_az_from_frame = None
            if sun_visible:
                sun_az_from_frame = sun_pixel_to_azimuth(sun_x, sample_bgr.shape[1], fov_video)
                img_el_note = (f"☀️ Sun detected in frame at pixel ({sun_x}, {sun_y}). "
                               f"Estimated azimuth from camera: {sun_az_from_frame:.1f}°")

            show_metrics(cloud_type, avg_conf, direction, height_m, fov_video,
                         fw, pixel_disp, delta_t_sec, deg_per_px, theta_deg,
                         distance_m, speed_mps, speed_kmh, dist_5, dist_15,
                         coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
                         vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
                         time_to_exit_min=time_to_exit_min,
                         media_timestamp=media_ts, timestamp_source=ts_source,
                         image_elevation_est=img_el_est, image_elevation_conf=img_el_conf,
                         image_elevation_note=img_el_note)
            
                        # ── Detection video — shown RIGHT HERE under uploader ──
            st.markdown('<div class="cv-eyebrow">📦 Cloud Detection Video</div>', unsafe_allow_html=True)
            with st.spinner("Generating detection video…"):
                tmp_box = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_box.close()
                generate_boxed_video(
                    tfile.name, tmp_box.name,
                    speed_kmh, speed_mps, direction,
                    cloud_type, height_m, dist_5, dist_15, fov=fov_video,
                    time_to_exit_min=999
                )
                with open(tmp_box.name, "rb") as f:
                    vdata = f.read()
            st.video(vdata)
            st.download_button("📥 Download Detection Video", data=vdata,
                               file_name=f"cloud_{cloud_type}_boxes.mp4",
                               mime="video/mp4", key="dl_box")
            try: os.unlink(tmp_box.name)
            except: pass

            st.divider()
            st.markdown('<div class="cv-eyebrow">📊 Analysis Results</div>', unsafe_allow_html=True)


            # ── Prediction video ──
            st.divider()
            st.markdown('<div class="cv-eyebrow">🔮 Motion Prediction Video</div>', unsafe_allow_html=True)
            with st.spinner("Generating prediction video…"):
                viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
                                            direction=direction, pixel_speed=pixel_speed)
                tmp_pred = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_pred.close()
                viz.save_video_with_prediction(tmp_pred.name, prediction_minutes=15)
                # Re-encode to H.264 for browser playback
                if shutil.which("ffmpeg"):
                    tmp_h264 = tmp_pred.name.replace(".mp4", "_h264.mp4")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", tmp_pred.name,
                        "-vcodec", "libx264", "-crf", "23",
                        "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", tmp_h264
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
                        os.replace(tmp_h264, tmp_pred.name)
                with open(tmp_pred.name, "rb") as f:
                    vdata2 = f.read()
            st.video(vdata2)
            st.download_button("📥 Download Prediction Video", data=vdata2,
                               file_name=f"cloud_{cloud_type}_prediction.mp4",
                               mime="video/mp4", key="dl_pred")
            try: os.unlink(tmp_pred.name)
            except: pass

        try: os.unlink(tfile.name)
        except: pass

# ══════════════════════ MULTI IMAGE TAB ════════════════════════
with tab2:
    st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🖼️ Image Analysis</div>', unsafe_allow_html=True)
    st.subheader("Upload Sky Images")

    uploaded_images = st.file_uploader("Upload 2 or more images taken at a fixed time interval",
                                        type=["jpg","jpeg","png"],
                                        accept_multiple_files=True, key="img_upload")
    interval   = st.number_input("⏱️ Time Between Images (seconds)", min_value=1, value=60)
    fov_images = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
                           help="Phone: 70-80° | Wide angle: 90-120°", key="fov_images")

    if uploaded_images:
        st.success(f"{len(uploaded_images)} image(s) uploaded.")

        with st.expander("🖼️ Preview Uploaded Images"):
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
            st.markdown('<div class="cv-eyebrow">📊 Analysis Results</div>', unsafe_allow_html=True)

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

            # ── Timestamp resolution for images (manual > EXIF > now) ──
            _manual_utc_i = st.session_state.get("manual_utc")
            if _manual_utc_i is not None:
                media_ts_i  = _manual_utc_i
                ts_source_i = "manual"
            else:
                uploaded_images[0].seek(0)
                _pil_first = Image.open(uploaded_images[0])
                media_ts_i = extract_exif_datetime(_pil_first)
                ts_source_i = "exif" if media_ts_i is not None else "now"
                if media_ts_i is None:
                    media_ts_i = datetime.datetime.utcnow()

            # ── Image-based elevation estimate from first uploaded image ──
            img_el_est_i, img_el_conf_i, img_el_note_i = estimate_sun_elevation_from_image(first_bgr)

            # ── Sun detection from image (if sun is visible in sky) ──
            sun_x_i, sun_y_i, sun_visible_i = detect_sun_in_frame(first_bgr)
            sun_az_from_img = None
            if sun_visible_i:
                sun_az_from_img = sun_pixel_to_azimuth(sun_x_i, first_bgr.shape[1], fov_images)
                img_el_note_i = (f"☀️ Sun detected in image at pixel ({sun_x_i}, {sun_y_i}). "
                                 f"Estimated azimuth from camera: {sun_az_from_img:.1f}°")

            show_metrics(cloud_type, avg_conf, direction, height_m, fov_images,
                         fw, avg_disp, interval, deg_per_px, theta_deg,
                         distance_m, speed_mps, speed_kmh, dist_5, dist_15,
                         coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
                         vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
                         time_to_exit_min=time_to_exit_min,
                         media_timestamp=media_ts_i, timestamp_source=ts_source_i,
                         image_elevation_est=img_el_est_i, image_elevation_conf=img_el_conf_i,
                         image_elevation_note=img_el_note_i)

            st.divider()
            st.markdown('<div class="cv-eyebrow">🎬 Export</div>', unsafe_allow_html=True)
            st.subheader("Generate Output")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📦 Cloud Detection on Images**")
                st.caption("Bounding boxes with speed and depth overlay on each uploaded image")
                if st.button("Show Detection Boxes", key="img_box"):
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
                st.markdown("**🔮 Motion Prediction Video**")
                st.caption("Simulated animation — +5 min and +15 min forecast")
                if st.button("Generate Prediction Video", key="img_pred"):
                    with st.spinner("Simulating cloud motion…"):
                        viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
                                                    direction=direction, pixel_speed=pixel_speed)
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmp.close()
                        viz.save_video_with_prediction(tmp.name, prediction_minutes=15)
                        if shutil.which("ffmpeg"):
                            tmp_h264 = tmp.name.replace(".mp4", "_h264.mp4")
                            subprocess.run([
                                "ffmpeg", "-y", "-i", tmp.name,
                                "-vcodec", "libx264", "-crf", "23",
                                "-preset", "fast", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", tmp_h264
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
                                os.replace(tmp_h264, tmp.name)
                        with open(tmp.name, "rb") as f:
                            vdata = f.read()
                        st.success("Prediction video ready.")
                        st.video(vdata)
                        st.download_button("📥 Download Prediction Video", data=vdata,
                                           file_name=f"cloud_{cloud_type}_prediction.mp4",
                                           mime="video/mp4")
                        try: os.unlink(tmp.name)
                        except: pass
        else:
            st.warning("Upload at least 2 images to run analysis.")

# import streamlit as st
# import numpy as np
# import cv2
# import tempfile
# import os
# import math
# import subprocess
# import shutil
# import json
# import datetime
# from collections import Counter
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from motion_visualizer import CloudMotionVisualizer

# # ===== SOLAR SHADOW & SUN TRACKING =====
# # pip install pvlib pandas
# try:
#     import pvlib
#     import pandas as pd
# except ImportError:
#     pvlib = None
#     pd = None

# def get_solar_position(lat, lon, timestamp):
#     if pvlib is None:
#         return None, None
#     times = pd.DatetimeIndex([timestamp])
#     sol = pvlib.solarposition.get_solarposition(times, lat, lon)
#     return float(sol["azimuth"].iloc[0]), float(sol["elevation"].iloc[0])


# def extract_exif_datetime(pil_image):
#     """Extract capture datetime from EXIF. Returns datetime or None."""
#     try:
#         exif_data = pil_image._getexif()
#         if exif_data is None:
#             return None
#         for tag_id in (36867, 36868, 306):
#             if tag_id in exif_data:
#                 return datetime.datetime.strptime(exif_data[tag_id], "%Y:%m:%d %H:%M:%S")
#     except Exception:
#         pass
#     return None


# def extract_video_datetime(video_path):
#     """Extract creation_time from MP4/MOV via ffprobe. Returns datetime or None."""
#     if not shutil.which("ffprobe"):
#         return None
#     try:
#         result = subprocess.run(
#             ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
#             capture_output=True, text=True, timeout=10
#         )
#         meta = json.loads(result.stdout)
#         tags = meta.get("format", {}).get("tags", {})
#         for key in ("creation_time", "com.apple.quicktime.creationdate"):
#             val = tags.get(key)
#             if val:
#                 val = val.rstrip("Z").split(".")[0]
#                 return datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
#     except Exception:
#         pass
#     return None


# def estimate_sun_elevation_from_image(frame_bgr):
#     """
#     Estimate sun elevation from sky brightness when sun is NOT visible in frame.

#     Uses 4 monocular cues:
#       1. Overall sky brightness  → proxy for solar irradiance
#       2. Horizon glow ratio      → low ratio = high sun, high ratio = low sun
#       3. Blue channel dominance  → clear vs overcast sky
#       4. Saturation              → confidence modifier

#     Returns: (elevation_deg, confidence, method_note)
#     """
#     sky_h = int(frame_bgr.shape[0] * 0.78)
#     sky   = frame_bgr[:sky_h, :]
#     H, W  = sky.shape[:2]

#     sky_f  = sky.astype(np.float32)
#     hsv    = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV).astype(np.float32)
#     bright = hsv[:, :, 2] / 255.0

#     mean_bright  = float(np.mean(bright))
#     horizon_mean = float(np.mean(bright[int(H * 0.80):, :]))
#     top_mean     = float(np.mean(bright[:int(H * 0.20), :]))
#     horizon_ratio = horizon_mean / (top_mean + 1e-6)

#     b_ch = sky_f[:, :, 0] / 255.0
#     r_ch = sky_f[:, :, 2] / 255.0
#     blue_ratio = float(np.mean(b_ch)) / (float(np.mean(r_ch)) + 1e-6)
#     mean_sat   = float(np.mean(hsv[:, :, 1] / 255.0))

#     if mean_bright < 0.15:
#         el_base = 2.0
#         note = "Very dark sky — sun likely below horizon or nighttime"
#         conf = "low"
#     elif mean_bright < 0.30:
#         if horizon_ratio > 1.3:
#             el_base = 8.0 + (horizon_ratio - 1.3) * 10
#             note = "Horizon glow detected — estimated sunrise/sunset angle"
#             conf = "medium"
#         else:
#             el_base = 15.0 + mean_bright * 40
#             note = "Dim sky — low sun elevation estimated from brightness"
#             conf = "low"
#     elif mean_bright < 0.55:
#         el_base = 25.0 + (mean_bright - 0.30) / 0.25 * 30
#         if horizon_ratio > 1.15:
#             el_base -= 10
#         note = "Moderate brightness — mid-range elevation estimated"
#         conf = "medium"
#     else:
#         el_base = 55.0 + (mean_bright - 0.55) / 0.45 * 25
#         note = "Bright sky — high elevation estimated (near noon)"
#         conf = "medium" if blue_ratio > 1.1 else "low"

#     if mean_sat < 0.10 and blue_ratio < 1.05:
#         conf = "low"
#         note += " (overcast — estimate less reliable)"

#     return round(float(np.clip(el_base, 0.0, 85.0)), 1), conf, note


# def detect_sun_in_frame(frame_bgr):
#     """
#     Detect the sun by finding the brightest spot in the sky region.

#     Looks at the top 78% of the frame (sky), finds the pixel with maximum
#     brightness. If that brightness exceeds 240 → sun is visible.

#     Args:
#         frame_bgr: BGR image (numpy array from cv2).

#     Returns:
#         (sun_x, sun_y, True)    — pixel position of sun if detected
#         (None,  None,  False)   — if sun is not visible
#     """
#     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     sky_h = int(frame_bgr.shape[0] * 0.78)
#     sky = gray[:sky_h, :]

#     # Sabse bright pixel dhundho
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sky)

#     # Agar brightness bahut high hai = sun visible
#     if max_val > 240:
#         sun_x, sun_y = max_loc
#         return sun_x, sun_y, True   # pixel position
#     return None, None, False


# def sun_pixel_to_azimuth(sun_x, frame_width, fov_deg, device_heading=0):
#     """
#     Convert sun's pixel X position to an estimated azimuth (compass bearing).

#     Calculates the angular offset of the sun from the frame center using
#     the camera's field of view, then adds the device heading to get true azimuth.

#     Args:
#         sun_x:          Horizontal pixel position of the sun in the frame.
#         frame_width:    Total width of the frame in pixels.
#         fov_deg:        Camera horizontal field of view in degrees.
#         device_heading: Compass heading the camera is pointing (0=N, 90=E, etc.).

#     Returns:
#         Estimated sun azimuth in degrees [0, 360).
#     """
#     # Center se kitna door hai sun
#     offset_px = sun_x - frame_width / 2
#     offset_deg = offset_px * (fov_deg / frame_width)
#     azimuth = (device_heading + offset_deg) % 360
#     return azimuth


# def sun_azimuth_to_direction(azimuth_deg):
#     """Convert sun azimuth (0=N, 90=E, 180=S, 270=W) to compass label."""
#     if azimuth_deg is None:
#         return "Unknown"
#     a = azimuth_deg % 360
#     if   a < 22.5 or a >= 337.5: return "North"
#     elif a < 67.5:  return "NE"
#     elif a < 112.5: return "East"
#     elif a < 157.5: return "SE"
#     elif a < 202.5: return "South"
#     elif a < 247.5: return "SW"
#     elif a < 292.5: return "West"
#     else:           return "NW"



# def get_cloud_sun_alignment(cloud_direction, sun_azimuth_deg, flow_angle_deg=None):
#     if sun_azimuth_deg is None:
#         return "unknown", None, "Sun position unavailable (set location or timestamp)."

#     if flow_angle_deg is not None:
#         cloud_az = flow_angle_deg % 360
#         diff = abs((cloud_az - sun_azimuth_deg + 180) % 360 - 180)
#     else:
#         dir_to_az = {"North": 0, "NE": 45, "East": 90, "SE": 135, "South": 180, "SW": 225, "West": 270, "NW": 315}
#         cloud_az = dir_to_az.get(cloud_direction, 0)
#         diff = abs((cloud_az - sun_azimuth_deg + 180) % 360 - 180)

#     if diff < 20:
#         return "toward_sun", diff, f"Cloud motion is nearly aligned with the sun ({diff:.0f}° offset). Shadow risk is high."
#     elif diff < 60:
#         return "glancing", diff, f"Cloud motion is partly aligned with sun ({diff:.0f}° offset). Shadow may partially affect the panel."
#     elif diff < 120:
#         return "crossing", diff, f"Cloud path is crossing the sun direction ({diff:.0f}° offset). Shadow may be brief."
#     else:
#         return "away_from_sun", diff, f"Cloud is moving away from sun direction ({diff:.0f}° offset). Shadow risk is lower."

# # ═══════════════════════════════════════════════════════════════



# # ─────────────────────────── CONFIG ────────────────────────────
# st.set_page_config(page_title="CloudVision AI", page_icon="☁️", layout="wide")

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

# /* ── Base ── */
# html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

# .stApp {
#     background: #070d14;
# }

# /* ── Hide default streamlit chrome ── */
# #MainMenu, footer { visibility: hidden; }
# .block-container { padding-top: 1.5rem !important; max-width: 1280px; }

# /* ── Header ── */
# .cv-header {
#     display: flex; align-items: center; gap: 16px;
#     padding: 28px 0 8px 0; border-bottom: 1px solid #1a2d44;
#     margin-bottom: 24px;
# }
# .cv-logo {
#     width: 48px; height: 48px; border-radius: 12px;
#     background: linear-gradient(135deg, #0ea5e9, #6366f1);
#     display: flex; align-items: center; justify-content: center;
#     font-size: 24px; flex-shrink: 0;
#     box-shadow: 0 0 24px rgba(14,165,233,0.35);
# }
# .cv-title { font-size: 1.75rem; font-weight: 700; color: #f0f6ff; letter-spacing: -0.02em; }
# .cv-sub   { font-size: 0.82rem; color: #4a6580; font-family: 'JetBrains Mono', monospace;
#             text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

# /* ── Tabs ── */
# .stTabs [data-baseweb="tab-list"] {
#     background: transparent;
#     border-bottom: 1px solid #1a2d44;
#     gap: 4px;
# }
# .stTabs [data-baseweb="tab"] {
#     background: transparent;
#     border: 1px solid transparent;
#     border-radius: 8px 8px 0 0;
#     color: #4a6580;
#     padding: 10px 22px;
#     font-weight: 500;
#     font-size: 0.88rem;
#     transition: all 0.15s;
# }
# .stTabs [data-baseweb="tab"]:hover { color: #94b8d4; background: #0d1a27; }
# .stTabs [aria-selected="true"] {
#     background: #0d1a27 !important;
#     color: #38bdf8 !important;
#     border-color: #1a2d44 #1a2d44 transparent !important;
# }
# .stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }

# /* ── Metric cards ── */
# [data-testid="metric-container"] {
#     background: #0d1a27;
#     border: 1px solid #1a2d44;
#     border-radius: 12px;
#     padding: 18px 20px !important;
#     transition: border-color 0.2s;
# }
# [data-testid="metric-container"]:hover { border-color: #2a4a64; }
# [data-testid="stMetricLabel"] {
#     font-size: 0.75rem !important;
#     color: #4a6580 !important;
#     text-transform: uppercase;
#     letter-spacing: 0.08em;
#     font-family: 'JetBrains Mono', monospace;
# }
# [data-testid="stMetricValue"] {
#     font-size: 1.35rem !important;
#     font-weight: 600 !important;
#     color: #e2ecf6 !important;
# }
# [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

# /* ── Buttons ── */
# .stButton > button {
#     background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
#     color: #fff;
#     border: none;
#     border-radius: 8px;
#     padding: 10px 22px;
#     font-weight: 600;
#     font-size: 0.875rem;
#     letter-spacing: 0.01em;
#     transition: opacity 0.15s, transform 0.1s;
#     width: 100%;
# }
# .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
# .stButton > button:active { transform: translateY(0); }

# /* ── Upload area ── */
# [data-testid="stFileUploader"] {
#     background: #0d1a27;
#     border: 1.5px dashed #1e3650;
#     border-radius: 12px;
#     padding: 12px;
#     transition: border-color 0.2s;
# }
# [data-testid="stFileUploader"]:hover { border-color: #0ea5e9; }

# /* ── Sliders ── */
# [data-testid="stSlider"] > div > div > div > div {
#     background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
# }

# /* ── Expander ── */
# [data-testid="stExpander"] {
#     background: #0d1a27;
#     border: 1px solid #1a2d44;
#     border-radius: 10px;
# }
# [data-testid="stExpander"] summary {
#     color: #94b8d4 !important;
#     font-size: 0.85rem;
#     font-weight: 500;
# }

# /* ── Spinner ── */
# [data-testid="stSpinner"] { color: #38bdf8 !important; }

# /* ── Divider ── */
# hr { border-color: #1a2d44 !important; margin: 20px 0 !important; }

# /* ── Subheader ── */
# h2, h3 { color: #c8dff0 !important; font-weight: 600 !important; }

# /* ── Number input / selectbox ── */
# [data-baseweb="input"], [data-baseweb="select"] {
#     background: #0d1a27 !important;
#     border-color: #1a2d44 !important;
#     border-radius: 8px !important;
#     color: #e2ecf6 !important;
# }

# /* ── Markdown tables ── */
# table { width: 100%; border-collapse: collapse; }
# th { background: #0d1a27; color: #4a6580; font-family: 'JetBrains Mono', monospace;
#      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
#      padding: 10px 14px; border-bottom: 1px solid #1a2d44; }
# td { color: #c8dff0; padding: 9px 14px; font-family: 'JetBrains Mono', monospace;
#      font-size: 0.82rem; border-bottom: 1px solid #0f1e2d; }
# tr:last-child td { border-bottom: none; }

# /* ── Success / Warning / Info ── */
# [data-testid="stAlert"] { border-radius: 10px !important; border-width: 1px !important; }

# /* ── Video ── */
# video { border-radius: 10px; border: 1px solid #1a2d44; }

# /* ── Caption ── */
# .stCaption { color: #4a6580 !important; font-size: 0.78rem !important; }

# /* ── Stat pill used in custom cards ── */
# .cv-pill {
#     display: inline-block;
#     padding: 3px 10px;
#     border-radius: 999px;
#     font-size: 0.72rem;
#     font-family: 'JetBrains Mono', monospace;
#     font-weight: 500;
#     letter-spacing: 0.04em;
#     background: #0a1929;
#     border: 1px solid #1a2d44;
#     color: #94b8d4;
#     margin-right: 4px;
# }

# /* ── Section label eyebrow ── */
# .cv-eyebrow {
#     font-size: 0.7rem;
#     font-family: 'JetBrains Mono', monospace;
#     text-transform: uppercase;
#     letter-spacing: 0.12em;
#     color: #4a6580;
#     margin-bottom: 10px;
# }

# /* ── Download button ── */
# [data-testid="stDownloadButton"] > button {
#     background: #0d1a27 !important;
#     border: 1px solid #1a2d44 !important;
#     color: #38bdf8 !important;
#     font-weight: 500 !important;
# }
# [data-testid="stDownloadButton"] > button:hover {
#     border-color: #38bdf8 !important;
#     background: #0a2035 !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # ─────────────────────────── MODEL ─────────────────────────────
# @st.cache_resource
# def load_cloud_model():
#     return load_model("cloud_model.keras")

# model = load_cloud_model()

# class_names  = ["Cumulus","Altocumulus","Cirrus","ClearSky","Stratocumulus","Cumulonimbus","Mixed"]
# cloud_height = {"Cumulus":1500,"Altocumulus":4500,"Cirrus":9000,
#                 "ClearSky":0,"Stratocumulus":1200,"Cumulonimbus":6000,"Mixed":3500}
# cloud_emoji  = {"Cumulus":"⛅","Altocumulus":"🌤️","Cirrus":"🌬️",
#                 "ClearSky":"☀️","Stratocumulus":"🌥️","Cumulonimbus":"⛈️","Mixed":"🌦️"}

# # ─────────────────────────── HELPERS ───────────────────────────
# def angle_to_direction(angle_deg):
#     a = angle_deg % 360
#     if 22.5 <= a < 67.5:
#         return "NE"
#     elif 67.5 <= a < 112.5:
#         return "East"
#     elif 112.5 <= a < 157.5:
#         return "SE"
#     elif 157.5 <= a < 202.5:
#         return "South"
#     elif 202.5 <= a < 247.5:
#         return "SW"
#     elif 247.5 <= a < 292.5:
#         return "West"
#     elif 292.5 <= a < 337.5:
#         return "NW"
#     return "North"

# def compute_optical_flow(gray1, gray2):
#     flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
#     return float(np.median(mag)), float(np.median(ang))

# def pixels_to_kmh(pixel_displacement, delta_t_sec, cloud_type, frame_width, fov):
#     height_m = float(cloud_height.get(cloud_type, 2000))
#     degree_per_px = fov / max(frame_width, 1)
#     theta_deg = abs(pixel_displacement) * degree_per_px
#     theta_rad = math.radians(theta_deg)
#     distance_m = height_m * math.tan(theta_rad)
#     speed_mps = distance_m / max(delta_t_sec, 1e-6)
#     speed_kmh = speed_mps * 3.6
#     return speed_mps, speed_kmh, degree_per_px, theta_deg, distance_m, height_m

# def compute_cloud_density(frame, sky_h):
#     """
#     Calculates cloud coverage percentage within the sky region.
#     Returns: coverage_percent (0-100), density_label, density_color
#     """
#     gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     sky_gray = gray[:sky_h, :]
#     sky_hsv  = hsv[:sky_h, :]

#     _, bright_mask = cv2.threshold(sky_gray, 140, 255, cv2.THRESH_BINARY)
#     sat = sky_hsv[:, :, 1]
#     _, sat_mask = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)
#     hue = sky_hsv[:, :, 0]
#     blue_sky = cv2.inRange(hue, 95, 135)
#     not_blue = cv2.bitwise_not(blue_sky)
#     combined = cv2.bitwise_or(bright_mask, cv2.bitwise_and(sat_mask, not_blue))

#     total_pixels = sky_gray.shape[0] * sky_gray.shape[1]
#     cloud_pixels = int(cv2.countNonZero(combined))
#     coverage = min(100.0, (cloud_pixels / total_pixels) * 100)

#     if coverage < 20:
#         label, color = "Low ☀️", "#22c55e"
#     elif coverage < 55:
#         label, color = "Medium 🌤️", "#f59e0b"
#     else:
#         label, color = "High ⛅", "#ef4444"

#     return round(coverage, 1), label, color


# def predict_visibility(cloud_type, speed_kmh, direction, coverage_pct, fov_deg):
#     """
#     Estimates how long the cloud will remain visible in the sky (until dissipation).
#     Estimate is based on cloud type's atmospheric lifetime, speed, and coverage.
#     Returns: verdict, reason, color, lifetime_min
#     """

#     # ── Typical atmospheric lifetimes by cloud type (minutes) ──
#     # Based on meteorological averages:
#     # Cumulus: 10–60 min (convective, quickly form/dissipate)
#     # Altocumulus: 30–120 min (mid-level, moderate lifetime)
#     # Cirrus: 60–360 min (high-altitude ice, very long lasting)
#     # ClearSky: no cloud
#     # Stratocumulus: 60–480 min (layer cloud, very persistent)
#     # Cumulonimbus: 30–90 min (active storm cell, intense but burns out)
#     # Mixed: 20–90 min (varied)
#     cloud_lifetime_range = {
#         "Cumulus": (10, 60),
#         "Altocumulus": (30, 120),
#         "Cirrus": (60, 360),
#         "ClearSky": (0, 0),
#         "Stratocumulus": (60, 480),
#         "Cumulonimbus": (30, 90),
#         "Mixed": (20, 90),
#     }

#     if cloud_type == "ClearSky":
#         return (
#             "☀️ Clear Sky — No Cloud",
#             "Sky is currently clear. No clouds are visible.",
#             "#22c55e",
#             999,
#         )

#     if speed_kmh < 0.5:
#         lo, hi = cloud_lifetime_range.get(cloud_type, (20, 90))
#         mid = (lo + hi) // 2
#         return (
#             f"🟡 Stationary — ~{mid} min remaining",
#             (f"{cloud_type} clouds typically last {lo}–{hi} min. "
#              f"This cloud is currently stationary and will remain visible for approximately {mid} more minutes."),
#             "#f59e0b",
#             float(mid),
#         )

#     lo, hi = cloud_lifetime_range.get(cloud_type, (20, 90))

#     # High coverage = more moisture/mass = longer lifetime
#     coverage_bonus = (coverage_pct / 100.0) * (hi - lo) * 0.3

#     # Fast-moving clouds dissipate faster (turbulence, mixing)
#     # Normalize: >60 km/h = fast, cuts lifetime by up to 30%
#     speed_factor = max(0.7, 1.0 - (speed_kmh / 200.0))

#     lifetime_min = ((lo + hi) / 2 + coverage_bonus) * speed_factor
#     lifetime_min = round(max(lo, min(hi, lifetime_min)))

#     # ── Verdict tiers ──
#     if lifetime_min > 60:
#         hrs = lifetime_min // 60
#         mins = lifetime_min % 60
#         time_str = f"~{hrs}h {mins}min" if mins else f"~{hrs}h"
#         verdict = f"🟢 Will Stay — {time_str} remaining"
#         reason  = (f"{cloud_type} clouds are long-lasting (typically {lo}–{hi} min). "
#                    f"With {coverage_pct}% coverage and a speed of {speed_kmh:.1f} km/h, "
#                    f"this cloud is expected to remain visible for approximately {time_str}.")
#         color   = "#22c55e"
#     elif lifetime_min > 20:
#         verdict = f"🟡 Moderate — ~{lifetime_min} min remaining"
#         reason  = (f"{cloud_type} clouds typically last {lo}–{hi} min. "
#                    f"Based on the current speed ({speed_kmh:.1f} km/h) and coverage ({coverage_pct}%), "
#                    f"this cloud is expected to remain visible for approximately {lifetime_min} more minutes.")
#         color   = "#f59e0b"
#     else:
#         verdict = f"🔴 Dissipating — ~{lifetime_min} min remaining"
#         reason  = (f"{cloud_type} clouds dissipate quickly (lifetime {lo}–{hi} min). "
#                    f"The high speed ({speed_kmh:.1f} km/h) and low coverage ({coverage_pct}%) suggest "
#                    f"this cloud will disappear from the sky in approximately {lifetime_min} minutes.")
#         color   = "#ef4444"

#     return verdict, reason, color, float(lifetime_min)


# def predict_cloud_type(frames_or_images):
#     """Batch-predict cloud type from a list of frames or PIL images."""
#     batch = []
#     for img in frames_or_images:
#         img_pil = Image.fromarray(img).resize((224, 224)) if isinstance(img, np.ndarray) else img.resize((224, 224))
#         batch.append(image.img_to_array(img_pil) / 255.0)

#     batch_arr = np.stack(batch, axis=0)          # shape: (N, 224, 224, 3)
#     preds_all = model.predict(batch_arr, verbose=0)  # single forward pass

#     preds = [class_names[np.argmax(p)] for p in preds_all]
#     confs = [float(np.max(p)) * 100 for p in preds_all]
#     return Counter(preds).most_common(1)[0][0], float(np.mean(confs))

# def compute_solar_shadow_forecast(cloud_type, height_m, speed_mps, speed_kmh,
#                                    direction, pixel_disp, frame_width, fov_deg,
#                                    coverage_pct):
#     """
#     Camera is mounted on or near the solar plant.
#     When a cloud enters the field of view, its shadow falls somewhere on the ground.

#     Physics:
#       - Cloud's current angular position from frame center → ground_offset
#         ground_offset = height_m * tan(angle_from_zenith)
#       - Shadow ground offset = cloud's horizontal distance from directly overhead
#       - If cloud is moving toward the camera center → shadow WILL hit the panel
#       - Time to shadow = ground_offset / speed_mps
#       - If cloud is moving away from center → shadow WON'T hit the panel

#     Returns dict with all forecast info.
#     """
#     cloud_power_factor = {
#         "Cumulus": 0.55, "Altocumulus": 0.45, "Cirrus": 0.18,
#         "ClearSky": 0.0, "Stratocumulus": 0.72, "Cumulonimbus": 0.85, "Mixed": 0.50
#     }

#     if cloud_type == "ClearSky":
#         return {
#             "will_hit": False,
#             "reason": "☀️ Sky is completely clear — no clouds, no shadow.",
#             "status": "clear",
#             "shadow_time_min": None,
#             "power_drop_pct": 0.0,
#             "ground_offset_m": 0.0,
#         }

#     # Angular position of cloud from frame center (pixels → degrees → radians)
#     deg_per_px     = fov_deg / frame_width
#     # pixel_disp is displacement magnitude; we use half-frame as rough center-offset
#     # Center offset: assume cloud centroid is ~frame center (worst case / mean case)
#     # More accurate: use half FOV = cloud is within FOV, so it's within ±fov/2 of zenith
#     # We use the actual angle the cloud has already traveled (theta) as proxy for offset
#     half_fov_deg   = fov_deg / 2.0
#     # Angle from zenith to cloud edge (conservative: use half-FOV)
#     angle_rad      = math.radians(half_fov_deg * 0.5)   # ~centre of visible sky arc
#     ground_offset_m = height_m * math.tan(angle_rad)    # metres from panel

#     # Direction logic: is cloud moving TOWARD overhead (will shadow hit) or AWAY?
#     # "toward overhead" = cloud is off to one side and moving toward center
#     # Simplified: if cloud is IN the FOV it is either overhead now or will cross overhead
#     # based on direction + frame position.
#     # We check: does the cloud's travel path cross the zenith column?
#     # Heuristic: if cloud is moving and it's within FOV → it will pass overhead → shadow hits
#     # UNLESS cloud is already past center and moving further away.

#     # We estimate current cloud X position from optical flow direction
#     # North/South movement means cloud crosses overhead (shadow hits)
#     # East/West also crosses — it just depends on whether it already passed
#     # Use pixel_disp relative to frame to estimate if approaching or receding
    
#     # Simple model: cloud in FOV = overhead within half-FOV → shadow WILL hit
#     # Time for shadow to reach panel = ground_offset / speed_mps
    
#     if speed_mps < 0.05:
#         # Nearly stationary
#         return {
#             "will_hit": True,
#             "reason": f"Cloud is still in the sky and is practically stationary. Shadow may already be present on the solar panel or has stalled overhead.",
#             "status": "stationary",
#             "shadow_time_min": 0.0,
#             "power_drop_pct": _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor),
#             "ground_offset_m": ground_offset_m,
#         }

#     time_to_shadow_sec = ground_offset_m / speed_mps
#     time_to_shadow_min = time_to_shadow_sec / 60.0

#     # If time is very small (< 0.5 min) → shadow is basically overhead now
#     # If cloud is moving away (already past zenith), offset increases over time → no hit
#     # Proxy: use pixel displacement direction vs frame center
#     # If the optical flow vector points toward frame center → cloud approaching
#     # Rough approximation: if time_to_shadow_min < 0 conceptually, cloud already passed
#     # We use: if time_to_shadow_min > time_to_exit_min-equivalent → won't cross
    
#     # Practical cutoff: if it takes longer than cloud lifetime to arrive → won't hit
#     cloud_lifetime = {
#         "Cumulus": 35, "Altocumulus": 75, "Cirrus": 200,
#         "ClearSky": 0, "Stratocumulus": 270, "Cumulonimbus": 60, "Mixed": 55
#     }
#     lifetime_min = cloud_lifetime.get(cloud_type, 60)

#     power_drop = _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor)

#     if time_to_shadow_min <= 0.3:
#         return {
#             "will_hit": True,
#             "reason": f"Cloud is almost directly overhead — shadow is currently falling on the solar panel or is about to.",
#             "status": "now",
#             "shadow_time_min": 0.0,
#             "power_drop_pct": power_drop,
#             "ground_offset_m": ground_offset_m,
#         }
#     elif time_to_shadow_min <= lifetime_min:
#         return {
#             "will_hit": True,
#             "reason": f"{cloud_type} cloud is offset by {ground_offset_m/1000:.2f} km. At {speed_kmh:.1f} km/h, the shadow will reach the solar panel in {time_to_shadow_min:.1f} minutes.",
#             "status": "incoming",
#             "shadow_time_min": time_to_shadow_min,
#             "power_drop_pct": power_drop,
#             "ground_offset_m": ground_offset_m,
#         }
#     else:
#         return {
#             "will_hit": False,
#             "reason": (f"The estimated shadow arrival time for this {cloud_type} cloud is ~{time_to_shadow_min:.0f} min, "
#                        f"but the cloud's expected lifetime is only ~{lifetime_min} min. "
#                        f"The cloud will dissipate or exit the field of view before the shadow reaches the panel."),
#             "status": "miss",
#             "shadow_time_min": time_to_shadow_min,
#             "power_drop_pct": 0.0,
#             "ground_offset_m": ground_offset_m,
#         }


# def _calc_power_drop(cloud_type, coverage_pct, factor_map):
#     # Updated power-drop heuristic with optional confidence weighting
#     base = float(factor_map.get(cloud_type, 0.50))
#     cov = max(0.0, min(1.0, (coverage_pct or 0.0) / 100.0))
#     conf = 1.0
#     drop = (base * (0.50 + 0.50 * cov) * 100.0) * conf
#     return round(min(95.0, drop), 1)
# def _calc_power_drop(cloud_type, coverage_pct, factor_map, confidence_pct=100.0):
#     base = float(factor_map.get(cloud_type, 0.50))
#     cov = max(0.0, min(1.0, (coverage_pct or 0.0) / 100.0))
#     conf = max(0.35, min(1.0, confidence_pct / 100.0))
#     drop = (base * (0.50 + 0.50 * cov) * 100.0) * conf
#     return round(min(95.0, drop), 1)


# def show_metrics(cloud_type, confidence, direction, height_m, fov,
#                  frame_width, pixel_disp, delta_t, deg_per_px,
#                  theta_deg, distance_m, speed_mps, speed_kmh, dist_5, dist_15,
#                  coverage_pct=None, density_label=None, density_color=None,
#                  vis_verdict=None, vis_reason=None, vis_color=None,
#                  time_to_exit_min=999, solar_dist_km=1.0,
#                  media_timestamp=None, timestamp_source="now",
#                  image_elevation_est=None, image_elevation_conf=None,
#                  image_elevation_note=None):
#     emoji = cloud_emoji.get(cloud_type, "☁️")

#     # ── Low confidence warning ──
#     if confidence < 60:
#         st.markdown(
#             f"⚠️ **Low Model Confidence ({confidence:.1f}%)** — The model is uncertain about this cloud type. Results may be less accurate.",
#             unsafe_allow_html=True,
#         )

#     # ── Section divider helper ──
#     def section_header(icon, title):
#         st.markdown(
#             f"<div style='font-size:0.9rem;font-weight:700;color:#94b8d4;margin:8px 0 6px 0;'>{icon} {title}</div>",
#             unsafe_allow_html=True,
#         )

#     # ── Section 1: Cloud Classification ──
#     section_header("☁", "Cloud Classification")
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric(f"{emoji} Cloud Type",    cloud_type)
#     c2.metric("🎯 Model Confidence",    f"{confidence:.1f}%")
#     c3.metric("🧭 Cloud Direction",      direction)
#     c4.metric("📍 Estimated Altitude",  f"{height_m:,} m")

#     # ── Section 2: Motion & Displacement ──
#     section_header("⚡", "Motion & Displacement")
#     label_5  = f"~{dist_5:.2f} km"  if dist_5  > 0 else "—"
#     label_15 = f"~{dist_15:.2f} km" if dist_15 > 0 else "—"

#     c5, c6, c7, c8 = st.columns(4)
#     c5.metric("Cloud Speed",            f"{speed_kmh:.1f} km/h")
#     c6.metric("Cloud Speed (m/s)",      f"{speed_mps:.2f} m/s")
#     c7.metric("Projected Dist. +5 min",  label_5,  delta_color="off")
#     c8.metric("Projected Dist. +15 min", label_15, delta_color="off")

#     # ── Section 3: Atmospheric Analysis ──
#     if coverage_pct is not None:
#         section_header("📡", "Atmospheric Analysis")
#         d1, d2 = st.columns(2)

#         with d1:
#             bar_filled = int(coverage_pct)
#             st.markdown(f"""
# <div style="background:#0d1a27;border:1px solid {density_color}44;border-radius:12px;padding:22px;">
#   <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#               letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">Sky Coverage</div>
#   <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:16px;">
#     <span style="font-size:2.2rem;font-weight:700;color:{density_color};font-family:'Inter',sans-serif;line-height:1;">{coverage_pct}%</span>
#     <span style="font-size:0.82rem;color:{density_color};font-weight:600;padding:3px 10px;
#                  background:{density_color}18;border-radius:999px;border:1px solid {density_color}33;">{density_label}</span>
#   </div>
#   <div style="background:#050c14;border-radius:8px;height:10px;width:100%;overflow:hidden;margin-bottom:8px;">
#     <div style="background:linear-gradient(90deg,{density_color}77,{density_color});
#                 width:{bar_filled}%;height:10px;border-radius:8px;"></div>
#   </div>
#   <div style="display:flex;justify-content:space-between;margin-top:6px;">
#     <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">Low — &lt;20%</span>
#     <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">Medium — 20–55%</span>
#     <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">High — &gt;55%</span>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#         with d2:
#             st.markdown(f"""
# <div style="background:#0d1a27;border:1px solid {vis_color}44;border-radius:12px;padding:22px;height:100%;">
#   <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#               letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">Cloud Lifetime Forecast</div>
#   <div style="font-size:1rem;font-weight:700;color:{vis_color};margin-bottom:12px;
#               padding:8px 12px;background:{vis_color}14;border-radius:8px;border-left:3px solid {vis_color};">{vis_verdict}</div>
#   <div style="font-size:0.82rem;color:#7a9ab4;line-height:1.65;">{vis_reason}</div>
# </div>
# """, unsafe_allow_html=True)

#     # ── Section 4: Solar Plant Impact Forecast ──
#     section_header("☀️", "Solar Plant Impact Forecast")

#     solar = compute_solar_shadow_forecast(
#         cloud_type, height_m, speed_mps, speed_kmh,
#         direction, pixel_disp, frame_width, fov,
#         coverage_pct if coverage_pct is not None else 50.0
#     )

#     status = solar["status"]

#     if status == "clear":
#         st.markdown("""
# <div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:12px;padding:24px;display:flex;align-items:center;gap:20px;">
#   <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
#               display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">☀️</div>
#   <div>
#     <div style="font-size:1.05rem;font-weight:700;color:#22c55e;margin-bottom:6px;">Clear Sky — Solar Plant Fully Safe</div>
#     <div style="font-size:0.82rem;color:#4a8060;line-height:1.5;">No clouds detected. No shadow risk. Solar plant is operating at full capacity.</div>
#   </div>
#   <div style="margin-left:auto;text-align:right;flex-shrink:0;">
#     <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Power Status</div>
#     <div style="font-size:1.1rem;font-weight:700;color:#22c55e;">🟢 100%</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif status == "now":
#         pdrop = solar["power_drop_pct"]
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #ef444455;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:#ef444418;border:1px solid #ef444433;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">⚠️</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:#ef4444;margin-bottom:5px;">Shadow Currently Falling on Solar Panel</div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; {speed_kmh:.1f} km/h
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
#     <div style="background:#100505;border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#ef4444;line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#100505;border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
#       <div style="font-size:1rem;font-weight:700;color:#ff6b6b;margin-top:4px;">🔴 Active Now</div>
#     </div>
#     <div style="background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 2;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif status == "stationary":
#         pdrop = solar["power_drop_pct"]
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #f59e0b55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:#f59e0b18;border:1px solid #f59e0b33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🟡</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:#f59e0b;margin-bottom:5px;">Cloud Stationary — Shadow Present on Panel</div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; Nearly stationary
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
#     <div style="background:#100a00;border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#f59e0b;line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#100a00;border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Movement</div>
#       <div style="font-size:1rem;font-weight:700;color:#f59e0b;margin-top:4px;">⏸ Stationary</div>
#     </div>
#     <div style="background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 2;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif status == "incoming":
#         pdrop    = solar["power_drop_pct"]
#         arr_min  = solar["shadow_time_min"]
#         off_km   = solar["ground_offset_m"] / 1000.0
#         if arr_min < 10:
#             urg_color = "#ef4444"; urg_icon = "🔴"
#         elif arr_min < 30:
#             urg_color = "#f59e0b"; urg_icon = "🟡"
#         else:
#             urg_color = "#22c55e"; urg_icon = "🟢"
#         arr_str = f"{arr_min:.1f} min" if arr_min < 60 else f"{int(arr_min//60)}h {int(arr_min%60)}m"
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid {urg_color}55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:{urg_color}18;border:1px solid {urg_color}33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">☁️</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:{urg_color};margin-bottom:5px;">
#         {urg_icon} Shadow Will Reach Solar Plant in {arr_str}
#       </div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;">
#     <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Shadow Arrives In</div>
#       <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{arr_str}</div>
#     </div>
#     <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1e3a50;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Ground Offset</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#38bdf8;line-height:1;">{off_km:.2f} km</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 3;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Forecast Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     else:  # status == "miss"
#         arr_min = solar["shadow_time_min"]
#         arr_str = f"{arr_min:.0f} min" if arr_min is not None else "N/A"
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">✅</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:#22c55e;margin-bottom:5px;">Shadow Will Not Reach the Solar Plant</div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
#       </div>
#     </div>
#   </div>
#   <div style="background:#050c14;border:1px solid #22c55e22;border-radius:10px;padding:16px 18px;margin-bottom:12px;">
#     <div style="font-size:0.82rem;color:#6aaa84;line-height:1.7;">{solar["reason"]}</div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#22c55e;line-height:1;">0%</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
#       <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-top:4px;">🟢 Safe</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
#     # ── Section 5: Sun Position & Cloud Alignment ──
#     _lat = st.session_state.get("user_lat", 28.6)
#     _lon = st.session_state.get("user_lon", 77.2)

#     # Use media timestamp if available, else current time
#     _ts = media_timestamp if media_timestamp is not None else datetime.datetime.utcnow()
#     sun_az, sun_el = get_solar_position(_lat, _lon, _ts)

#     # Timestamp source badge
#     _ts_badge_map = {
#         "exif":    ("📷 From EXIF",        "#22c55e"),
#         "ffprobe": ("🎬 From Video Meta",  "#22c55e"),
#         "manual":  ("🕐 Manual Input",      "#f59e0b"),
#         "now":     ("⏱️ Current Time",      "#4a6580"),
#     }
#     _ts_label, _ts_color = _ts_badge_map.get(timestamp_source, ("⏱️ Current Time", "#4a6580"))

#     section_header("🌞", "Sun Position & Cloud Alignment")

#     if sun_az is None:
#         st.info("📍 Install pvlib + pandas and set your location in the sidebar for live sun tracking.")
#     else:
#         sun_dir  = sun_azimuth_to_direction(sun_az)
#         align_status, angle_diff, align_desc = get_cloud_sun_alignment(direction, sun_az)

#         align_color_map = {
#             "toward_sun":    "#ef4444",
#             "glancing":      "#f59e0b",
#             "crossing":      "#38bdf8",
#             "away_from_sun": "#22c55e",
#             "unknown":       "#4a6580",
#         }
#         align_icon_map = {
#             "toward_sun":    "🔴 Heading Toward Sun",
#             "glancing":      "🟡 Glancing Sun",
#             "crossing":      "🔵 Crossing Sun Path",
#             "away_from_sun": "🟢 Moving Away from Sun",
#             "unknown":       "❓ Unknown",
#         }
#         a_color = align_color_map.get(align_status, "#4a6580")
#         a_label = align_icon_map.get(align_status, "")

#         if sun_el < 0:
#             sun_status_label = "🌙 Below Horizon"
#             sun_el_color = "#4a6580"
#         elif sun_el < 15:
#             sun_status_label = "🌅 Near Horizon"
#             sun_el_color = "#f59e0b"
#         else:
#             sun_status_label = "☀️ Above Horizon"
#             sun_el_color = "#fbbf24"

#         # ── Determine which elevation to show ──
#         # pvlib = authoritative; image estimate = fallback shown alongside
#         show_img_est = (image_elevation_est is not None)
#         img_conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(
#             image_elevation_conf, "#4a6580")

#         sc1, sc2, sc3, sc4 = st.columns(4)
#         sc1.metric("☀️ Sun Azimuth",   f"{sun_az:.1f}°")
#         sc2.metric("📐 Sun Elevation (pvlib)", f"{sun_el:.1f}°")
#         sc3.metric("🧭 Sun Direction", sun_dir)
#         sc4.metric("☁️ Cloud Moving",  direction)

#         # Timestamp badge + image estimate row
#         badge_html = f"""
# <div style="display:flex;align-items:center;gap:10px;margin:10px 0 14px 0;flex-wrap:wrap;">
#   <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;padding:3px 10px;
#                border-radius:999px;background:{_ts_color}18;border:1px solid {_ts_color}44;
#                color:{_ts_color};">{_ts_label}: {_ts.strftime('%Y-%m-%d %H:%M UTC')}</span>"""

#         if show_img_est:
#             badge_html += f"""
#   <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;padding:3px 10px;
#                border-radius:999px;background:{img_conf_color}18;border:1px solid {img_conf_color}44;
#                color:{img_conf_color};">
#     📸 Image Estimate: {image_elevation_est}° ({image_elevation_conf} confidence)
#   </span>"""

#         badge_html += "</div>"
#         st.markdown(badge_html, unsafe_allow_html=True)

#         # Image elevation detail card (only when sun not visible in frame)
#         if show_img_est:
#             diff_el = abs(sun_el - image_elevation_est)
#             st.markdown(f"""
# <div style="background:#0d1a27;border:1px solid {img_conf_color}44;border-radius:12px;
#             padding:16px 20px;margin-bottom:14px;">
#   <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#               letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">
#     📸 Image-Based Sun Elevation Estimate (Sun not visible in frame)
#   </div>
#   <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;">
#     <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
#       <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:6px;">Image Estimate</div>
#       <div style="font-size:1.6rem;font-weight:700;color:{img_conf_color};line-height:1;">
#         {image_elevation_est}°</div>
#     </div>
#     <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
#       <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:6px;">pvlib (Location+Time)</div>
#       <div style="font-size:1.6rem;font-weight:700;color:#fbbf24;line-height:1;">{sun_el:.1f}°</div>
#     </div>
#     <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
#       <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:6px;">Difference</div>
#       <div style="font-size:1.6rem;font-weight:700;color:{'#22c55e' if diff_el < 10 else '#f59e0b' if diff_el < 25 else '#ef4444'};line-height:1;">
#         ±{diff_el:.1f}°</div>
#     </div>
#   </div>
#   <div style="font-size:0.8rem;color:#7a9ab4;line-height:1.6;font-style:italic;">{image_elevation_note}</div>
# </div>
# """, unsafe_allow_html=True)

#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid {a_color}55;border-radius:14px;padding:22px;margin-top:4px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
#     <div style="width:48px;height:48px;border-radius:12px;background:{a_color}18;border:1px solid {a_color}33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;">🌞</div>
#     <div>
#       <div style="font-size:1rem;font-weight:700;color:{a_color};margin-bottom:4px;">
#         Cloud–Sun Alignment: {a_label}
#       </div>
#       <div style="font-size:0.78rem;font-family:'JetBrains Mono',monospace;color:#4a6580;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:3px 10px;display:inline-block;">
#         {sun_status_label} &nbsp;·&nbsp; Azimuth {sun_az:.1f}° &nbsp;·&nbsp; Elevation {sun_el:.1f}°
#         {'&nbsp;·&nbsp; ' + str(round(angle_diff)) + '° offset' if angle_diff is not None else ''}
#       </div>
#     </div>
#   </div>
#   <div style="font-size:0.84rem;color:#94b8d4;line-height:1.7;background:#050c14;
#               border-radius:10px;padding:14px 18px;border:1px solid #1a2d44;">
#     {align_desc}
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

#     # ── Export Analysis Results ──
#     with st.expander("📤 Export Analysis Report"):
#         import csv, io
#         report_data = {
#             "cloud_type": cloud_type,
#             "confidence_pct": round(confidence, 1),
#             "direction": direction,
#             "altitude_m": height_m,
#             "speed_kmh": round(speed_kmh, 1),
#             "speed_mps": round(speed_mps, 2),
#             "projected_dist_5min_km": round(dist_5, 2),
#             "projected_dist_15min_km": round(dist_15, 2),
#             "sky_coverage_pct": coverage_pct,
#             "density_label": density_label,
#             "visibility_forecast": vis_verdict,
#             "timestamp_utc": _ts.strftime("%Y-%m-%d %H:%M UTC") if media_timestamp is not None else datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
#             "timestamp_source": timestamp_source,
#         }
#         # JSON download
#         import json as _json
#         json_str = _json.dumps(report_data, indent=2)
#         st.download_button(
#             "📥 Download JSON Report",
#             data=json_str,
#             file_name=f"cloudvision_report_{cloud_type}.json",
#             mime="application/json",
#             key="dl_json_report"
#         )
#         # CSV download
#         csv_buf = io.StringIO()
#         writer = csv.DictWriter(csv_buf, fieldnames=report_data.keys())
#         writer.writeheader()
#         writer.writerow(report_data)
#         st.download_button(
#             "📥 Download CSV Report",
#             data=csv_buf.getvalue(),
#             file_name=f"cloudvision_report_{cloud_type}.csv",
#             mime="text/csv",
#             key="dl_csv_report"
#         )
#         st.code(json_str, language="json")

#     with st.expander("🔬 Optical Flow — Calculation Details"):
#         st.markdown(f"""
# | Parameter | Value |
# |---|---|
# | Camera FOV | {fov}° |
# | Frame Width | {frame_width} px |
# | Degrees per Pixel | {deg_per_px:.4f} °/px |
# | Pixel Displacement | {pixel_disp:.2f} px over {delta_t:.2f} s |
# | Angular Displacement (θ) | {theta_deg:.4f}° |
# | Horizontal Distance (tan formula) | {distance_m:.2f} m |
# | Derived Speed | {speed_mps:.2f} m/s → {speed_kmh:.1f} km/h |
# """)

# # ─────────────────────────── CLOUD DETECTION ───────────────────
# def detect_clouds(frame, sky_h):
#     """
#     Multi-method cloud detection:
#     1. Brightness threshold (white clouds)
#     2. HSV saturation (low saturation = cloud/white)
#     3. Combine both masks
#     Uses watershed-style distance-based separation to assign distinct
#     bounding boxes to individual cloud regions.
#     """
#     OUT_W = frame.shape[1]

#     gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     sky_gray = gray[:sky_h, :]
#     sky_hsv  = hsv[:sky_h, :]

#     # Method 1: brightness — lower threshold to catch grey clouds too
#     _, bright_mask = cv2.threshold(sky_gray, 140, 255, cv2.THRESH_BINARY)

#     # Method 2: low saturation = white/grey cloud (not blue sky)
#     sat = sky_hsv[:, :, 1]
#     _, sat_mask = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)

#     # Method 3: not-blue sky — blue sky has high hue (100-130)
#     hue = sky_hsv[:, :, 0]
#     blue_sky = cv2.inRange(hue, 95, 135)
#     not_blue = cv2.bitwise_not(blue_sky)

#     # Combine: bright OR (low-sat AND not-blue-sky)
#     combined = cv2.bitwise_or(bright_mask,
#                 cv2.bitwise_and(sat_mask, not_blue))

#     # Morphology — smaller kernels to keep clouds separate
#     k_close = np.ones((12, 12), np.uint8)
#     k_open  = np.ones((6,  6),  np.uint8)
#     combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
#     combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)

#     # --- Watershed separation to split merged clouds ---
#     dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
#     cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
#     _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
#     sure_fg    = np.uint8(sure_fg)

#     sure_bg    = cv2.dilate(combined, np.ones((3,3), np.uint8), iterations=3)
#     unknown    = cv2.subtract(sure_bg, sure_fg)

#     _, markers = cv2.connectedComponents(sure_fg)
#     markers    = markers + 1
#     markers[unknown == 255] = 0

#     # Watershed needs 3-channel BGR image
#     sky_bgr = frame[:sky_h, :].copy()
#     markers = cv2.watershed(sky_bgr, markers)

#     # Extract bounding boxes from each watershed region
#     boxes = []
#     unique_labels = np.unique(markers)
#     for lbl in unique_labels:
#         if lbl <= 1:   # background or border
#             continue
#         mask_lbl = np.zeros_like(combined)
#         mask_lbl[markers == lbl] = 255
#         cnts, _ = cv2.findContours(mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in cnts:
#             area = cv2.contourArea(cnt)
#             if area < 600:   # ignore tiny noise
#                 continue
#             x, y, w, h = cv2.boundingRect(cnt)
#             boxes.append((x, y, w, h, mask_lbl))

#     # Fallback: if watershed gave nothing, use simple contours
#     if not boxes:
#         cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in cnts:
#             if cv2.contourArea(cnt) < 600:
#                 continue
#             x, y, w, h = cv2.boundingRect(cnt)
#             boxes.append((x, y, w, h, None))

#     return boxes, gray

# # ─────────────────────── STEREO DEPTH VISION ───────────────────
# def compute_pseudo_depth_map(frame_bgr, sky_h):
#     """
#     Single-image pseudo stereo depth map for cloud regions.

#     Physics cues used (all monocular):
#       1. Brightness  — brighter cloud core = optically thicker = visually 'closer'
#       2. Texture     — high-freq detail = nearer; smooth/hazy = farther
#       3. Saturation  — desaturated (grey/white) regions = cloud mass present
#       4. Vertical pos— lower in sky frame ≈ closer horizon clouds

#     Output: depth_map (H x W float32, 0=far / blue … 1=near / red)
#             depth_color (H x W x 3 uint8, COLORMAP_JET applied)
#     """
#     sky = frame_bgr[:sky_h, :].copy()
#     H, W = sky.shape[:2]

#     gray = cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
#     hsv  = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV).astype(np.float32)
#     sat  = hsv[:, :, 1] / 255.0
#     val  = hsv[:, :, 2] / 255.0

#     # Cue 1: brightness (brighter = more cloud mass = closer)
#     bright_cue = val

#     # Cue 2: texture energy via Laplacian (sharp edges = nearer)
#     lap = cv2.Laplacian(gray, cv2.CV_32F)
#     tex_cue = np.abs(lap)
#     tex_cue = cv2.GaussianBlur(tex_cue, (15, 15), 0)
#     tex_max = tex_cue.max()
#     if tex_max > 0:
#         tex_cue /= tex_max

#     # Cue 3: low saturation = cloud (not blue sky) → weight up
#     cloud_presence = 1.0 - np.clip(sat, 0, 1)   # white/grey = high weight

#     # Cue 4: vertical position — lower row = closer (nearer horizon)
#     row_idx  = np.linspace(1.0, 0.0, H, dtype=np.float32)   # top=far, bottom=near
#     vert_cue = np.tile(row_idx[:, None], (1, W))

#     # Weighted fusion
#     depth = (0.40 * bright_cue +
#              0.25 * tex_cue    +
#              0.20 * cloud_presence +
#              0.15 * vert_cue)

#     # Smooth for clean visualization
#     depth = cv2.GaussianBlur(depth, (21, 21), 0)
#     cv2.normalize(depth, depth, 0, 1, cv2.NORM_MINMAX)

#     # Colorize: COLORMAP_JET  blue=far → green=mid → red=near
#     depth_u8    = (depth * 255).astype(np.uint8)
#     depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

#     return depth, depth_color


# def depth_to_distance_km(depth_val, cloud_height_m, fov_deg):
#     """
#     Depth value (0–1) → estimated slant distance in km.
#     Uses trigonometry: closer clouds (higher depth) are nearer to cloud_height_m;
#     farther (lower depth) are assumed to be 1.5–3x that height away (oblique angle).
#     """
#     # depth=1 → distance = cloud_height_m (directly overhead)
#     # depth=0 → distance = 3 * cloud_height_m (far horizon, shallow angle)
#     distance_m = cloud_height_m * (1.0 + 2.0 * (1.0 - float(depth_val)))
#     return round(distance_m / 1000.0, 2)


# # ─────────────────────────── BOUNDING BOX FUNCTION ─────────────
# def draw_boxes_on_frame(frame, speed_kmh, direction, cloud_type, height_m,
#                          dist_5, dist_15, elapsed_sec, prev_gray=None, delta_t=None,
#                          fov=75, time_to_exit_min=999):
#     OUT_W = frame.shape[1]
#     OUT_H = frame.shape[0]
#     sky_h = int(OUT_H * 0.78)   # slightly more sky area

#     # Pre-compute full dense optical flow if prev frame available
#     gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     sky_gray = gray[:sky_h, :]
#     full_flow = None
#     if prev_gray is not None and delta_t is not None and delta_t > 0:
#         prev_sky = prev_gray[:sky_h, :]
#         full_flow = cv2.calcOpticalFlowFarneback(
#             prev_sky, sky_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
#         )

#     # Detect clouds with improved method
#     boxes, _ = detect_clouds(frame, sky_h)

#     # ── Compute pseudo stereo depth map for full sky region ──
#     depth_map, depth_color = compute_pseudo_depth_map(frame, sky_h)

#     for (x, y, w, h, _) in boxes:
#         pad = 8
#         x1 = max(0,       x - pad);    y1 = max(0,       y - pad)
#         x2 = min(OUT_W-1, x+w + pad);  y2 = min(sky_h,   y+h + pad)

#         # ── Per-cloud speed from optical flow ROI ──
#         if full_flow is not None:
#             roi_flow = full_flow[y1:y2, x1:x2]
#             if roi_flow.size > 0:
#                 mag, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
#                 roi_pixel_disp = float(np.median(mag))
#                 if roi_pixel_disp > 0.1:
#                     _, cloud_speed_kmh, _, _, _, _ = pixels_to_kmh(
#                         roi_pixel_disp, delta_t, cloud_type, OUT_W, fov
#                     )
#                 else:
#                     cloud_speed_kmh = 0.0
#             else:
#                 cloud_speed_kmh = speed_kmh
#         else:
#             cloud_speed_kmh = speed_kmh

#         # ── Stereo Depth Overlay inside box ──
#         roi_depth_color = depth_color[y1:y2, x1:x2]
#         roi_frame       = frame[y1:y2, x1:x2]
#         if roi_depth_color.shape == roi_frame.shape and roi_frame.size > 0:
#             # Blend depth colormap (40%) with original frame (60%)
#             cv2.addWeighted(roi_depth_color, 0.40, roi_frame, 0.60, 0,
#                             frame[y1:y2, x1:x2])

#         # Estimated distance from depth at box center
#         cy_box = min((y1 + y2) // 2, depth_map.shape[0] - 1)
#         cx_box = min((x1 + x2) // 2, depth_map.shape[1] - 1)
#         center_depth = float(depth_map[cy_box, cx_box])
#         est_dist_km  = depth_to_distance_km(center_depth, height_m, fov)

#         # Glow effect
#         glow = frame.copy()
#         cv2.rectangle(glow, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 100), 4)
#         cv2.addWeighted(glow, 0.3, frame, 0.7, 0, frame)

#         # Main box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

#         # Corner ticks
#         t = 14
#         for (px_, py_, sdx, sdy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
#             cv2.line(frame, (px_, py_), (px_+sdx*t, py_),    (0, 255, 60), 2)
#             cv2.line(frame, (px_, py_), (px_, py_+sdy*t),    (0, 255, 60), 2)

#         # Per-cloud speed + depth distance label
#         label = f"{cloud_speed_kmh:.1f} km/h  |  D:{est_dist_km:.1f}km"
#         fs    = 0.46
#         (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
#         lx = x1
#         ly = y1 - 5 if y1 - 5 - th > 2 else y1 + th + 6
#         cv2.rectangle(frame, (lx-2, ly-th-3), (lx+tw+6, ly+bl+1), (0, 150, 55), -1)
#         cv2.rectangle(frame, (lx-2, ly-th-3), (lx+tw+6, ly+bl+1), (0, 255, 100), 1)
#         cv2.putText(frame, label, (lx+3, ly),
#                     cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

#     # ── HUD top-left ──
#     ov = frame.copy()
#     cv2.rectangle(ov, (0,0), (360,125), (0,0,0), -1)
#     cv2.addWeighted(ov, 0.58, frame, 0.42, 0, frame)
#     cv2.rectangle(frame, (0,0), (360,125), (0,200,80), 1)

#     def txt(t, y, sc=0.52, c=(255,255,255), b=1):
#         cv2.putText(frame, t, (12,y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, b, cv2.LINE_AA)

#     txt(f"Cloud  : {cloud_type}",   22, c=(140,230,255), b=2)
#     txt(f"Height : {height_m:,} m", 43)
#     txt(f"Speed  : {speed_kmh:.1f} km/h  ({speed_kmh/3.6:.2f} m/s)", 64, c=(80,255,160))

#     # Smart +5 / +15 min — show "OUT OF FRAME" if cloud will have exited by then
#     lbl_5  = f"~{dist_5:.2f} km"
#     lbl_15 = f"~{dist_15:.2f} km"
#     txt(f"Dir:{direction}  +5m:{lbl_5}  +15m:{lbl_15}", 86, sc=0.40, c=(200, 200, 200))

#     mins = int(elapsed_sec)//60;  secs = int(elapsed_sec)%60
#     cv2.putText(frame, f"T+ {mins:02d}:{secs:02d}",
#                 (OUT_W-155, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,180), 2, cv2.LINE_AA)

#     # Direction arrow
#     dir_vec = {"East":(1,0),"West":(-1,0),"North":(0,-1),"South":(0,1)}.get(direction,(1,0))
#     cx, cy  = OUT_W//2, OUT_H - 35
#     cv2.arrowedLine(frame, (cx,cy),
#                     (int(cx+dir_vec[0]*55), int(cy+dir_vec[1]*55)),
#                     (255,255,255), 3, tipLength=0.35)
#     cv2.putText(frame, direction, (cx-30, cy+20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

#     # ── Sun Position Arrow ──
#     try:
#         _lat2 = st.session_state.get("user_lat", 28.6)
#         _lon2 = st.session_state.get("user_lon", 77.2)
#         _sun_az2, _sun_el2 = get_solar_position(_lat2, _lon2, datetime.datetime.utcnow())
#     except Exception:
#         _sun_az2, _sun_el2 = None, None

#     if _sun_az2 is not None and _sun_el2 > 0:
#         # Draw sun arrow at bottom-center right of cloud arrow
#         sun_az_rad = math.radians(_sun_az2)   # 0=N, 90=E
#         # Convert azimuth to screen vector (x right=East, y down=South)
#         sun_dx = math.sin(sun_az_rad)   # East component
#         sun_dy = -math.cos(sun_az_rad)  # North component (inverted for screen)
#         scx, scy = OUT_W//2 + 120, OUT_H - 35
#         cv2.arrowedLine(frame, (scx, scy),
#                         (int(scx + sun_dx * 50), int(scy + sun_dy * 50)),
#                         (30, 220, 255), 2, tipLength=0.4)
#         cv2.putText(frame, f"Sun {_sun_az2:.0f}", (scx - 28, scy + 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 220, 255), 1, cv2.LINE_AA)

#         # Cloud-Sun alignment indicator
#         align_status, angle_diff, _ = get_cloud_sun_alignment(direction, _sun_az2)
#         align_color_cv = {
#             "toward_sun":    (60, 80, 255),
#             "glancing":      (60, 180, 255),
#             "crossing":      (255, 200, 60),
#             "away_from_sun": (60, 220, 100),
#             "unknown":       (150, 150, 150),
#         }.get(align_status, (150, 150, 150))
#         align_text = {
#             "toward_sun":    "TO SUN",
#             "glancing":      "GLANCING",
#             "crossing":      "CROSSING",
#             "away_from_sun": "FROM SUN",
#             "unknown":       "?",
#         }.get(align_status, "?")
#         if angle_diff is not None:
#             align_label_full = f"{align_text} {angle_diff:.0f}deg"
#         else:
#             align_label_full = align_text
#         (aw, _ah), _ = cv2.getTextSize(align_label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
#         ax = scx - aw // 2
#         ay = scy - 14
#         cv2.rectangle(frame, (ax - 3, ay - 14), (ax + aw + 4, ay + 4), (10, 10, 10), -1)
#         cv2.putText(frame, align_label_full, (ax, ay),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.44, align_color_cv, 1, cv2.LINE_AA)

#     # ── Solar Plant Shadow HUD (camera IS on solar plant) ──
#     if cloud_type != "ClearSky":
#         solar = compute_solar_shadow_forecast(
#             cloud_type, height_m, speed_mps, speed_kmh,
#             direction, OUT_W * 0.05,   # small pixel_disp proxy for HUD
#             OUT_W, fov, coverage_pct=50.0
#         )
#         status = solar["status"]
#         if status == "now" or status == "stationary":
#             s_line1 = f"SHADOW ON SOLAR PLANT NOW!"
#             s_line2 = f"Power drop: {solar['power_drop_pct']}%"
#             box_col  = (0, 60, 220)   # red-orange
#             txt_col1 = (60, 80, 255)
#             txt_col2 = (60, 255, 160)
#         elif status == "incoming":
#             arr = solar["shadow_time_min"]
#             arr_str = f"{arr:.1f}min" if arr < 60 else f"{int(arr//60)}h{int(arr%60)}m"
#             s_line1 = f"Shadow arrives: {arr_str}"
#             s_line2 = f"Power drop: {solar['power_drop_pct']}%"
#             box_col  = (0, 160, 240)
#             txt_col1 = (80, 220, 255)
#             txt_col2 = (80, 255, 160)
#         elif status == "miss":
#             s_line1 = "Shadow will NOT hit solar plant"
#             s_line2 = "Power drop: 0%  [SAFE]"
#             box_col  = (0, 130, 40)
#             txt_col1 = (80, 255, 120)
#             txt_col2 = (80, 255, 120)
#         else:
#             s_line1 = "Clear sky — solar plant safe"
#             s_line2 = "No shadow expected"
#             box_col  = (0, 130, 40)
#             txt_col1 = (80, 255, 120)
#             txt_col2 = (80, 255, 120)

#         (sw1, _), _ = cv2.getTextSize(s_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
#         (sw2, _), _ = cv2.getTextSize(s_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
#         box_w = max(sw1, sw2) + 22
#         box_h = 58
#         bx, by = OUT_W - box_w - 8, 8

#         sol_ov = frame.copy()
#         cv2.rectangle(sol_ov, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
#         cv2.addWeighted(sol_ov, 0.62, frame, 0.38, 0, frame)
#         cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), box_col, 1)
#         cv2.putText(frame, s_line1, (bx + 8, by + 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col1, 1, cv2.LINE_AA)
#         cv2.putText(frame, s_line2, (bx + 8, by + 44),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col2, 1, cv2.LINE_AA)

#     # ── Depth Legend (colorbar) — bottom right ──
#     bar_x, bar_y, bar_w, bar_h = OUT_W - 120, OUT_H - 90, 18, 70
#     for i in range(bar_h):
#         val   = int(255 * (1.0 - i / bar_h))
#         color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
#         cv2.line(frame, (bar_x, bar_y + i), (bar_x + bar_w, bar_y + i), color, 1)
#     cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
#     cv2.putText(frame, "Near", (bar_x + bar_w + 4, bar_y + 8),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 80, 80), 1, cv2.LINE_AA)
#     cv2.putText(frame, "Far",  (bar_x + bar_w + 4, bar_y + bar_h),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 200), 1, cv2.LINE_AA)
#     cv2.putText(frame, "Depth", (bar_x - 2, bar_y - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

#     return frame


# def generate_boxed_video(input_path, output_path, speed_kmh, speed_mps,
#                           direction, cloud_type, height_m, dist_5, dist_15, fov=75,
#                           time_to_exit_min=999):
#     cap     = cv2.VideoCapture(input_path)
#     fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     OUT_W, OUT_H = 960, 540
#     delta_t = 1.0 / fps

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out    = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, OUT_H))

#     prev_gray = None
#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame   = cv2.resize(frame, (OUT_W, OUT_H))
#         elapsed = frame_idx / fps
#         gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame   = draw_boxes_on_frame(
#             frame, speed_kmh, direction, cloud_type,
#             height_m, dist_5, dist_15, elapsed,
#             prev_gray=prev_gray, delta_t=delta_t, fov=fov,
#             time_to_exit_min=time_to_exit_min
#         )
#         out.write(frame)
#         prev_gray = gray
#         frame_idx += 1

#     cap.release()
#     out.release()

#     # Re-encode to H.264 so browser can play it in st.video()
#     if shutil.which("ffmpeg"):
#         tmp_h264 = output_path.replace(".mp4", "_h264.mp4")
#         subprocess.run([
#             "ffmpeg", "-y", "-i", output_path,
#             "-vcodec", "libx264", "-crf", "23",
#             "-preset", "fast", "-pix_fmt", "yuv420p",
#             "-movflags", "+faststart",
#             tmp_h264
#         ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
#             os.replace(tmp_h264, output_path)


# # ─────────────────────────── UI HEADER ────────────────────────
# st.markdown("""
# <div class="cv-header">
#   <div class="cv-logo">☁️</div>
#   <div>
#     <div class="cv-title">CloudVision AI</div>
#     <div class="cv-sub">Cloud Classification &amp; Motion Prediction System</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

# tab1, tab2 = st.tabs(["🎬 Video Analysis", "🖼️ Multi Image Analysis"])

# # ── Solar location inputs (shared across tabs) ──
# with st.sidebar:
#     st.markdown("### ☀️ Solar Location")
#     st.caption("Enter your location for real-time sun position tracking")
#     user_lat = st.number_input("Latitude",  value=28.6, min_value=-90.0,  max_value=90.0,  step=0.1, format="%.4f", key="user_lat")
#     user_lon = st.number_input("Longitude", value=77.2, min_value=-180.0, max_value=180.0, step=0.1, format="%.4f", key="user_lon")
#     st.caption("🇮🇳 Default: New Delhi")

#     st.markdown("---")
#     st.markdown("### 🕐 Media Timestamp")
#     st.caption("App auto-reads EXIF/video metadata. Override manually if needed.")
#     use_manual_time = st.checkbox("✏️ Override timestamp manually", value=False, key="use_manual_time")
#     if use_manual_time:
#         _today = datetime.date.today()
#         manual_date = st.date_input("Date", value=_today, key="manual_date")
#         manual_time = st.time_input("Time (local)", value=datetime.time(12, 0), key="manual_time")
#         tz_offset   = st.number_input("Timezone offset (hrs from UTC)", value=5.5,
#                                        min_value=-12.0, max_value=14.0, step=0.5, key="tz_offset")
#         # Convert to UTC datetime
#         local_dt = datetime.datetime.combine(manual_date, manual_time)
#         manual_utc = local_dt - datetime.timedelta(hours=tz_offset)
#         st.session_state["manual_utc"] = manual_utc
#         st.caption(f"UTC: {manual_utc.strftime('%Y-%m-%d %H:%M')}")
#     else:
#         st.session_state["manual_utc"] = None

#     if pvlib is not None:
#         _ts_sb = st.session_state.get("manual_utc") or datetime.datetime.utcnow()
#         _az, _el = get_solar_position(user_lat, user_lon, _ts_sb)
#         if _az is not None:
#             _sun_dir = sun_azimuth_to_direction(_az)
#             if _el < 0:
#                 _sun_status = "🌙 Sun below horizon"
#                 _sun_color  = "#4a6580"
#             elif _el < 15:
#                 _sun_status = "🌅 Sun near horizon"
#                 _sun_color  = "#f59e0b"
#             else:
#                 _sun_status = "☀️ Sun above horizon"
#                 _sun_color  = "#fbbf24"
#             st.markdown(f"""
# <div style='background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:14px;margin-top:8px;'>
#   <div style='font-size:0.68rem;font-family:monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Live Sun Position</div>
#   <div style='font-size:0.92rem;font-weight:600;color:{_sun_color};margin-bottom:6px;'>{_sun_status}</div>
#   <div style='font-size:0.78rem;color:#94b8d4;'>Azimuth: <b style='color:#e2ecf6;'>{_az:.1f}°</b> ({_sun_dir})</div>
#   <div style='font-size:0.78rem;color:#94b8d4;'>Elevation: <b style='color:#e2ecf6;'>{_el:.1f}°</b></div>
# </div>
# """, unsafe_allow_html=True)
#     else:
#         st.info("Install pvlib + pandas for live sun tracking:\n`pip install pvlib pandas`")

#     st.markdown("---")
#     st.markdown("### ℹ️ About")
#     st.caption(
#         "**CloudVision AI** — Cloud classification, motion analysis & "
#         "solar shadow forecasting.\n\n"
#         "Model: Keras CNN · Classes: Cumulus, Altocumulus, Cirrus, "
#         "ClearSky, Stratocumulus, Cumulonimbus, Mixed"
#     )

# # ══════════════════════════ VIDEO TAB ══════════════════════════
# with tab1:
#     st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🎬 Video Analysis</div>', unsafe_allow_html=True)
#     st.subheader("Upload a Sky Video")

#     uploaded_video = st.file_uploader("Drop an MP4 / AVI / MOV file here",
#                                        type=["mp4","avi","mov"], key="video_upload")
#     fov_video = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
#                           help="Phone: 70-80° | Wide angle: 90-120° | Telephoto: 30-50°")

#     if uploaded_video is not None:

#         uploaded_video.seek(0)
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         tfile.write(uploaded_video.read())
#         tfile.flush()
#         tfile.close()

#         with st.spinner("🔍 Analysing video..."):
#             cap          = cv2.VideoCapture(tfile.name)
#             fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#             sample_frames = []
#             for pt in [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pt))
#                 ret, frm = cap.read()
#                 if ret:
#                     sample_frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

#             cloud_type, avg_conf = predict_cloud_type(sample_frames)

#             frame_gap   = max(1, int(fps))
#             delta_t_sec = frame_gap / fps
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0);         ret1, f1 = cap.read()
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_gap); ret2, f2 = cap.read()
#             cap.release()

#         if ret1 and ret2:
#             fw = f1.shape[1]
#             g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
#             g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
#             pixel_disp, avg_angle = compute_optical_flow(g1, g2)
#             direction = angle_to_direction(np.degrees(avg_angle))

#             speed_mps, speed_kmh, deg_per_px, theta_deg, distance_m, height_m = \
#                 pixels_to_kmh(pixel_disp, delta_t_sec, cloud_type, fw, fov_video)
#             pixel_speed = pixel_disp / delta_t_sec
#             dist_5  = speed_kmh * (5  / 60)
#             dist_15 = speed_kmh * (15 / 60)

#             # Density & Visibility compute
#             sample_bgr = cv2.cvtColor(sample_frames[len(sample_frames)//2], cv2.COLOR_RGB2BGR)
#             sky_h_sample = int(sample_bgr.shape[0] * 0.78)
#             cov_pct, den_label, den_color = compute_cloud_density(sample_bgr, sky_h_sample)
#             vis_verdict, vis_reason, vis_color, time_to_exit_min = predict_visibility(
#                 cloud_type, speed_kmh, direction, cov_pct, fov_video)

#             # ── Timestamp resolution (priority: manual > ffprobe > now) ──
#             _manual_utc = st.session_state.get("manual_utc")
#             if _manual_utc is not None:
#                 media_ts      = _manual_utc
#                 ts_source     = "manual"
#             else:
#                 media_ts = extract_video_datetime(tfile.name)
#                 ts_source = "ffprobe" if media_ts is not None else "now"
#                 if media_ts is None:
#                     media_ts = datetime.datetime.utcnow()

#             # ── Image-based sun elevation (for frames where sun not visible) ──
#             img_el_est, img_el_conf, img_el_note = estimate_sun_elevation_from_image(sample_bgr)

#             # ── Sun detection from frame (if sun is visible in sky) ──
#             sun_x, sun_y, sun_visible = detect_sun_in_frame(sample_bgr)
#             sun_az_from_frame = None
#             if sun_visible:
#                 sun_az_from_frame = sun_pixel_to_azimuth(sun_x, sample_bgr.shape[1], fov_video)
#                 img_el_note = (f"☀️ Sun detected in frame at pixel ({sun_x}, {sun_y}). "
#                                f"Estimated azimuth from camera: {sun_az_from_frame:.1f}°")

#             show_metrics(cloud_type, avg_conf, direction, height_m, fov_video,
#                          fw, pixel_disp, delta_t_sec, deg_per_px, theta_deg,
#                          distance_m, speed_mps, speed_kmh, dist_5, dist_15,
#                          coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
#                          vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
#                          time_to_exit_min=time_to_exit_min,
#                          media_timestamp=media_ts, timestamp_source=ts_source,
#                          image_elevation_est=img_el_est, image_elevation_conf=img_el_conf,
#                          image_elevation_note=img_el_note)
            
#                         # ── Detection video — shown RIGHT HERE under uploader ──
#             st.markdown('<div class="cv-eyebrow">📦 Cloud Detection Video</div>', unsafe_allow_html=True)
#             with st.spinner("Generating detection video…"):
#                 tmp_box = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 tmp_box.close()
#                 generate_boxed_video(
#                     tfile.name, tmp_box.name,
#                     speed_kmh, speed_mps, direction,
#                     cloud_type, height_m, dist_5, dist_15, fov=fov_video,
#                     time_to_exit_min=999
#                 )
#                 with open(tmp_box.name, "rb") as f:
#                     vdata = f.read()
#             st.video(vdata)
#             st.download_button("📥 Download Detection Video", data=vdata,
#                                file_name=f"cloud_{cloud_type}_boxes.mp4",
#                                mime="video/mp4", key="dl_box")
#             try: os.unlink(tmp_box.name)
#             except: pass

#             st.divider()
#             st.markdown('<div class="cv-eyebrow">📊 Analysis Results</div>', unsafe_allow_html=True)


#             # ── Prediction video ──
#             st.divider()
#             st.markdown('<div class="cv-eyebrow">🔮 Motion Prediction Video</div>', unsafe_allow_html=True)
#             with st.spinner("Generating prediction video…"):
#                 viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
#                                             direction=direction, pixel_speed=pixel_speed)
#                 tmp_pred = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 tmp_pred.close()
#                 viz.save_video_with_prediction(tmp_pred.name, prediction_minutes=15)
#                 # Re-encode to H.264 for browser playback
#                 if shutil.which("ffmpeg"):
#                     tmp_h264 = tmp_pred.name.replace(".mp4", "_h264.mp4")
#                     subprocess.run([
#                         "ffmpeg", "-y", "-i", tmp_pred.name,
#                         "-vcodec", "libx264", "-crf", "23",
#                         "-preset", "fast", "-pix_fmt", "yuv420p",
#                         "-movflags", "+faststart", tmp_h264
#                     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                     if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
#                         os.replace(tmp_h264, tmp_pred.name)
#                 with open(tmp_pred.name, "rb") as f:
#                     vdata2 = f.read()
#             st.video(vdata2)
#             st.download_button("📥 Download Prediction Video", data=vdata2,
#                                file_name=f"cloud_{cloud_type}_prediction.mp4",
#                                mime="video/mp4", key="dl_pred")
#             try: os.unlink(tmp_pred.name)
#             except: pass

#         try: os.unlink(tfile.name)
#         except: pass

# # ══════════════════════ MULTI IMAGE TAB ════════════════════════
# with tab2:
#     st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🖼️ Image Analysis</div>', unsafe_allow_html=True)
#     st.subheader("Upload Sky Images")

#     uploaded_images = st.file_uploader("Upload 2 or more images taken at a fixed time interval",
#                                         type=["jpg","jpeg","png"],
#                                         accept_multiple_files=True, key="img_upload")
#     interval   = st.number_input("⏱️ Time Between Images (seconds)", min_value=1, value=60)
#     fov_images = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
#                            help="Phone: 70-80° | Wide angle: 90-120°", key="fov_images")

#     if uploaded_images:
#         st.success(f"{len(uploaded_images)} image(s) uploaded.")

#         with st.expander("🖼️ Preview Uploaded Images"):
#             img_cols = st.columns(min(len(uploaded_images), 4))
#             for i, img_file in enumerate(uploaded_images):
#                 img_file.seek(0)
#                 with img_cols[i % 4]:
#                     st.image(img_file, caption=img_file.name, use_container_width=True)

#         if len(uploaded_images) >= 2:
#             with st.spinner("🔍 Analysing images..."):
#                 dirs_deg, px_disps = [], []
#                 for i in range(len(uploaded_images) - 1):
#                     uploaded_images[i].seek(0); uploaded_images[i+1].seek(0)
#                     img1 = np.array(Image.open(uploaded_images[i]).convert("RGB").resize((640,480)))
#                     img2 = np.array(Image.open(uploaded_images[i+1]).convert("RGB").resize((640,480)))
#                     med, ang = compute_optical_flow(
#                         cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
#                         cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
#                     )
#                     px_disps.append(med); dirs_deg.append(np.degrees(ang))

#                 avg_disp    = float(np.mean(px_disps))
#                 direction   = angle_to_direction(float(np.mean(dirs_deg)))
#                 pil_imgs    = []
#                 for f in uploaded_images:
#                     f.seek(0); pil_imgs.append(Image.open(f))
#                 cloud_type, avg_conf = predict_cloud_type(pil_imgs)

#             fw = 640
#             speed_mps, speed_kmh, deg_per_px, theta_deg, distance_m, height_m = \
#                 pixels_to_kmh(avg_disp, interval, cloud_type, fw, fov_images)
#             pixel_speed = avg_disp / interval
#             dist_5  = speed_kmh * (5  / 60)
#             dist_15 = speed_kmh * (15 / 60)

#             st.divider()
#             st.markdown('<div class="cv-eyebrow">📊 Analysis Results</div>', unsafe_allow_html=True)

#             # Density & Visibility compute from first image
#             uploaded_images[0].seek(0)
#             first_bgr = cv2.resize(
#                 cv2.cvtColor(np.array(Image.open(uploaded_images[0]).convert("RGB")), cv2.COLOR_RGB2BGR),
#                 (640, 480)
#             )
#             sky_h_img = int(first_bgr.shape[0] * 0.78)
#             cov_pct, den_label, den_color = compute_cloud_density(first_bgr, sky_h_img)
#             vis_verdict, vis_reason, vis_color, time_to_exit_min = predict_visibility(
#                 cloud_type, speed_kmh, direction, cov_pct, fov_images)

#             # ── Timestamp resolution for images (manual > EXIF > now) ──
#             _manual_utc_i = st.session_state.get("manual_utc")
#             if _manual_utc_i is not None:
#                 media_ts_i  = _manual_utc_i
#                 ts_source_i = "manual"
#             else:
#                 uploaded_images[0].seek(0)
#                 _pil_first = Image.open(uploaded_images[0])
#                 media_ts_i = extract_exif_datetime(_pil_first)
#                 ts_source_i = "exif" if media_ts_i is not None else "now"
#                 if media_ts_i is None:
#                     media_ts_i = datetime.datetime.utcnow()

#             # ── Image-based elevation estimate from first uploaded image ──
#             img_el_est_i, img_el_conf_i, img_el_note_i = estimate_sun_elevation_from_image(first_bgr)

#             # ── Sun detection from image (if sun is visible in sky) ──
#             sun_x_i, sun_y_i, sun_visible_i = detect_sun_in_frame(first_bgr)
#             sun_az_from_img = None
#             if sun_visible_i:
#                 sun_az_from_img = sun_pixel_to_azimuth(sun_x_i, first_bgr.shape[1], fov_images)
#                 img_el_note_i = (f"☀️ Sun detected in image at pixel ({sun_x_i}, {sun_y_i}). "
#                                  f"Estimated azimuth from camera: {sun_az_from_img:.1f}°")

#             show_metrics(cloud_type, avg_conf, direction, height_m, fov_images,
#                          fw, avg_disp, interval, deg_per_px, theta_deg,
#                          distance_m, speed_mps, speed_kmh, dist_5, dist_15,
#                          coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
#                          vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
#                          time_to_exit_min=time_to_exit_min,
#                          media_timestamp=media_ts_i, timestamp_source=ts_source_i,
#                          image_elevation_est=img_el_est_i, image_elevation_conf=img_el_conf_i,
#                          image_elevation_note=img_el_note_i)

#             st.divider()
#             st.markdown('<div class="cv-eyebrow">🎬 Export</div>', unsafe_allow_html=True)
#             st.subheader("Generate Output")

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.markdown("**📦 Cloud Detection on Images**")
#                 st.caption("Bounding boxes with speed and depth overlay on each uploaded image")
#                 if st.button("Show Detection Boxes", key="img_box"):
#                     cols3 = st.columns(min(len(uploaded_images), 3))
#                     for i, img_file in enumerate(uploaded_images):
#                         img_file.seek(0)
#                         arr = cv2.resize(
#                             cv2.cvtColor(np.array(Image.open(img_file).convert("RGB")),
#                                          cv2.COLOR_RGB2BGR), (640, 480)
#                         )
#                         prev_arr = None
#                         if i > 0:
#                             uploaded_images[i-1].seek(0)
#                             prev_arr_bgr = cv2.resize(
#                                 cv2.cvtColor(np.array(Image.open(uploaded_images[i-1]).convert("RGB")),
#                                              cv2.COLOR_RGB2BGR), (640, 480)
#                             )
#                             prev_arr = cv2.cvtColor(prev_arr_bgr, cv2.COLOR_BGR2GRAY)
#                         arr = draw_boxes_on_frame(arr, speed_kmh, direction, cloud_type,
#                                                    height_m, dist_5, dist_15, i * interval,
#                                                    prev_gray=prev_arr, delta_t=float(interval),
#                                                    fov=fov_images,
#                                                    time_to_exit_min=time_to_exit_min)
#                         with cols3[i % 3]:
#                             st.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),
#                                      caption=f"Image {i+1}", use_container_width=True)

#             with col2:
#                 st.markdown("**🔮 Motion Prediction Video**")
#                 st.caption("Simulated animation — +5 min and +15 min forecast")
#                 if st.button("Generate Prediction Video", key="img_pred"):
#                     with st.spinner("Simulating cloud motion…"):
#                         viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
#                                                     direction=direction, pixel_speed=pixel_speed)
#                         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                         tmp.close()
#                         viz.save_video_with_prediction(tmp.name, prediction_minutes=15)
#                         if shutil.which("ffmpeg"):
#                             tmp_h264 = tmp.name.replace(".mp4", "_h264.mp4")
#                             subprocess.run([
#                                 "ffmpeg", "-y", "-i", tmp.name,
#                                 "-vcodec", "libx264", "-crf", "23",
#                                 "-preset", "fast", "-pix_fmt", "yuv420p",
#                                 "-movflags", "+faststart", tmp_h264
#                             ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                             if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
#                                 os.replace(tmp_h264, tmp.name)
#                         with open(tmp.name, "rb") as f:
#                             vdata = f.read()
#                         st.success("Prediction video ready.")
#                         st.video(vdata)
#                         st.download_button("📥 Download Prediction Video", data=vdata,
#                                            file_name=f"cloud_{cloud_type}_prediction.mp4",
#                                            mime="video/mp4")
#                         try: os.unlink(tmp.name)
#                         except: pass
#         else:
#             st.warning("Upload at least 2 images to run analysis.")

# # import streamlit as st
# # import numpy as np
# # import cv2
# # import tempfile
# # import os
# # import math
# # import subprocess
# # import shutil
# # import json
# # import datetime
# # from collections import Counter
# # from PIL import Image
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# # from motion_visualizer import CloudMotionVisualizer

# # # ===== SOLAR SHADOW & SUN TRACKING =====
# # # pip install pvlib pandas
# # try:
# #     import pvlib
# #     import pandas as pd
# # except ImportError:
# #     pvlib = None
# #     pd = None

# # def get_solar_position(lat, lon, timestamp):
# #     if pvlib is None:
# #         return None, None
# #     times = pd.DatetimeIndex([timestamp])
# #     sol = pvlib.solarposition.get_solarposition(times, lat, lon)
# #     return float(sol["azimuth"].iloc[0]), float(sol["elevation"].iloc[0])


# # def extract_exif_datetime(pil_image):
# #     """Extract capture datetime from EXIF. Returns datetime or None."""
# #     try:
# #         exif_data = pil_image._getexif()
# #         if exif_data is None:
# #             return None
# #         for tag_id in (36867, 36868, 306):
# #             if tag_id in exif_data:
# #                 return datetime.datetime.strptime(exif_data[tag_id], "%Y:%m:%d %H:%M:%S")
# #     except Exception:
# #         pass
# #     return None


# # def extract_video_datetime(video_path):
# #     """Extract creation_time from MP4/MOV via ffprobe. Returns datetime or None."""
# #     if not shutil.which("ffprobe"):
# #         return None
# #     try:
# #         result = subprocess.run(
# #             ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
# #             capture_output=True, text=True, timeout=10
# #         )
# #         meta = json.loads(result.stdout)
# #         tags = meta.get("format", {}).get("tags", {})
# #         for key in ("creation_time", "com.apple.quicktime.creationdate"):
# #             val = tags.get(key)
# #             if val:
# #                 val = val.rstrip("Z").split(".")[0]
# #                 return datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
# #     except Exception:
# #         pass
# #     return None


# # def estimate_sun_elevation_from_image(frame_bgr):
# #     """
# #     Estimate sun elevation from sky brightness when sun is NOT visible in frame.

# #     Uses 4 monocular cues:
# #       1. Overall sky brightness  → proxy for solar irradiance
# #       2. Horizon glow ratio      → low ratio = high sun, high ratio = low sun
# #       3. Blue channel dominance  → clear vs overcast sky
# #       4. Saturation              → confidence modifier

# #     Returns: (elevation_deg, confidence, method_note)
# #     """
# #     sky_h = int(frame_bgr.shape[0] * 0.78)
# #     sky   = frame_bgr[:sky_h, :]
# #     H, W  = sky.shape[:2]

# #     sky_f  = sky.astype(np.float32)
# #     hsv    = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV).astype(np.float32)
# #     bright = hsv[:, :, 2] / 255.0

# #     mean_bright  = float(np.mean(bright))
# #     horizon_mean = float(np.mean(bright[int(H * 0.80):, :]))
# #     top_mean     = float(np.mean(bright[:int(H * 0.20), :]))
# #     horizon_ratio = horizon_mean / (top_mean + 1e-6)

# #     b_ch = sky_f[:, :, 0] / 255.0
# #     r_ch = sky_f[:, :, 2] / 255.0
# #     blue_ratio = float(np.mean(b_ch)) / (float(np.mean(r_ch)) + 1e-6)
# #     mean_sat   = float(np.mean(hsv[:, :, 1] / 255.0))

# #     if mean_bright < 0.15:
# #         el_base = 2.0
# #         note = "Very dark sky — sun likely below horizon or nighttime"
# #         conf = "low"
# #     elif mean_bright < 0.30:
# #         if horizon_ratio > 1.3:
# #             el_base = 8.0 + (horizon_ratio - 1.3) * 10
# #             note = "Horizon glow detected — estimated sunrise/sunset angle"
# #             conf = "medium"
# #         else:
# #             el_base = 15.0 + mean_bright * 40
# #             note = "Dim sky — low sun elevation estimated from brightness"
# #             conf = "low"
# #     elif mean_bright < 0.55:
# #         el_base = 25.0 + (mean_bright - 0.30) / 0.25 * 30
# #         if horizon_ratio > 1.15:
# #             el_base -= 10
# #         note = "Moderate brightness — mid-range elevation estimated"
# #         conf = "medium"
# #     else:
# #         el_base = 55.0 + (mean_bright - 0.55) / 0.45 * 25
# #         note = "Bright sky — high elevation estimated (near noon)"
# #         conf = "medium" if blue_ratio > 1.1 else "low"

# #     if mean_sat < 0.10 and blue_ratio < 1.05:
# #         conf = "low"
# #         note += " (overcast — estimate less reliable)"

# #     return round(float(np.clip(el_base, 0.0, 85.0)), 1), conf, note


# # def detect_sun_in_frame(frame_bgr):
# #     """
# #     Detect the sun by finding the brightest spot in the sky region.

# #     Looks at the top 78% of the frame (sky), finds the pixel with maximum
# #     brightness. If that brightness exceeds 240 → sun is visible.

# #     Args:
# #         frame_bgr: BGR image (numpy array from cv2).

# #     Returns:
# #         (sun_x, sun_y, True)    — pixel position of sun if detected
# #         (None,  None,  False)   — if sun is not visible
# #     """
# #     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
# #     sky_h = int(frame_bgr.shape[0] * 0.78)
# #     sky = gray[:sky_h, :]

# #     # Sabse bright pixel dhundho
# #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sky)

# #     # Agar brightness bahut high hai = sun visible
# #     if max_val > 240:
# #         sun_x, sun_y = max_loc
# #         return sun_x, sun_y, True   # pixel position
# #     return None, None, False


# # def sun_pixel_to_azimuth(sun_x, frame_width, fov_deg, device_heading=0):
# #     """
# #     Convert sun's pixel X position to an estimated azimuth (compass bearing).

# #     Calculates the angular offset of the sun from the frame center using
# #     the camera's field of view, then adds the device heading to get true azimuth.

# #     Args:
# #         sun_x:          Horizontal pixel position of the sun in the frame.
# #         frame_width:    Total width of the frame in pixels.
# #         fov_deg:        Camera horizontal field of view in degrees.
# #         device_heading: Compass heading the camera is pointing (0=N, 90=E, etc.).

# #     Returns:
# #         Estimated sun azimuth in degrees [0, 360).
# #     """
# #     # Center se kitna door hai sun
# #     offset_px = sun_x - frame_width / 2
# #     offset_deg = offset_px * (fov_deg / frame_width)
# #     azimuth = (device_heading + offset_deg) % 360
# #     return azimuth


# # def sun_azimuth_to_direction(azimuth_deg):
# #     """Convert sun azimuth (0=N, 90=E, 180=S, 270=W) to compass label."""
# #     if azimuth_deg is None:
# #         return "Unknown"
# #     a = azimuth_deg % 360
# #     if   a < 22.5 or a >= 337.5: return "North"
# #     elif a < 67.5:  return "NE"
# #     elif a < 112.5: return "East"
# #     elif a < 157.5: return "SE"
# #     elif a < 202.5: return "South"
# #     elif a < 247.5: return "SW"
# #     elif a < 292.5: return "West"
# #     else:           return "NW"


# # def get_cloud_sun_alignment(cloud_direction, sun_azimuth_deg):
# #     """
# #     Check if cloud is moving TOWARD or AWAY from the sun.
# #     Returns: alignment_status, angle_diff, description
# #     """
# #     if sun_azimuth_deg is None:
# #         return "unknown", None, "Sun position unavailable (enable location)"

# #     # Map cloud direction to azimuth
# #     dir_to_az = {"North": 0, "South": 180, "East": 90, "West": 270}
# #     cloud_az = dir_to_az.get(cloud_direction, 0)

# #     # Angular difference between cloud movement and sun direction
# #     diff = abs(cloud_az - sun_azimuth_deg) % 360
# #     if diff > 180:
# #         diff = 360 - diff  # shortest arc

# #     if diff < 30:
# #         return "toward_sun", diff, f"Cloud is moving directly toward the sun ({diff:.0f}° offset). Shadow will likely cross your solar panel soon."
# #     elif diff < 90:
# #         return "glancing", diff, f"Cloud path is at {diff:.0f}° from sun direction — glancing alignment. Shadow may partially affect the panel."
# #     elif diff < 150:
# #         return "crossing", diff, f"Cloud is moving roughly perpendicular to the sun ({diff:.0f}° offset). Shadow will cross and clear quickly."
# #     else:
# #         return "away_from_sun", diff, f"Cloud is moving away from the sun ({diff:.0f}° offset). Shadow risk is low."

# # # ═══════════════════════════════════════════════════════════════



# # # ─────────────────────────── CONFIG ────────────────────────────
# # st.set_page_config(page_title="CloudVision AI", page_icon="☁️", layout="wide", initial_sidebar_state="expanded")

# # st.markdown("""
# # <style>
# # @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

# # /* ── Base ── */
# # html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

# # .stApp {
# #     background: #070d14;
# # }

# # /* ── Hide default streamlit chrome ── */
# # #MainMenu, footer{ visibility: hidden; }
# # .block-container { padding-top: 1.5rem !important; max-width: 1280px; }

# # /* ── Header ── */
# # .cv-header {
# #     display: flex; align-items: center; gap: 16px;
# #     padding: 28px 0 8px 0; border-bottom: 1px solid #1a2d44;
# #     margin-bottom: 24px;
# # }
# # .cv-logo {
# #     width: 48px; height: 48px; border-radius: 12px;
# #     background: linear-gradient(135deg, #0ea5e9, #6366f1);
# #     display: flex; align-items: center; justify-content: center;
# #     font-size: 24px; flex-shrink: 0;
# #     box-shadow: 0 0 24px rgba(14,165,233,0.35);
# # }
# # .cv-title { font-size: 1.75rem; font-weight: 700; color: #f0f6ff; letter-spacing: -0.02em; }
# # .cv-sub   { font-size: 0.82rem; color: #4a6580; font-family: 'JetBrains Mono', monospace;
# #             text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

# # /* ── Tabs ── */
# # .stTabs [data-baseweb="tab-list"] {
# #     background: transparent;
# #     border-bottom: 1px solid #1a2d44;
# #     gap: 4px;
# # }
# # .stTabs [data-baseweb="tab"] {
# #     background: transparent;
# #     border: 1px solid transparent;
# #     border-radius: 8px 8px 0 0;
# #     color: #4a6580;
# #     padding: 10px 22px;
# #     font-weight: 500;
# #     font-size: 0.88rem;
# #     transition: all 0.15s;
# # }
# # .stTabs [data-baseweb="tab"]:hover { color: #94b8d4; background: #0d1a27; }
# # .stTabs [aria-selected="true"] {
# #     background: #0d1a27 !important;
# #     color: #38bdf8 !important;
# #     border-color: #1a2d44 #1a2d44 transparent !important;
# # }
# # .stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }

# # /* ── Metric cards ── */
# # [data-testid="metric-container"] {
# #     background: #0d1a27;
# #     border: 1px solid #1a2d44;
# #     border-radius: 12px;
# #     padding: 18px 20px !important;
# #     transition: border-color 0.2s;
# # }
# # [data-testid="metric-container"]:hover { border-color: #2a4a64; }
# # [data-testid="stMetricLabel"] {
# #     font-size: 0.75rem !important;
# #     color: #4a6580 !important;
# #     text-transform: uppercase;
# #     letter-spacing: 0.08em;
# #     font-family: 'JetBrains Mono', monospace;
# # }
# # [data-testid="stMetricValue"] {
# #     font-size: 1.35rem !important;
# #     font-weight: 600 !important;
# #     color: #e2ecf6 !important;
# # }
# # [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

# # /* ── Buttons ── */
# # .stButton > button {
# #     background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
# #     color: #fff;
# #     border: none;
# #     border-radius: 8px;
# #     padding: 10px 22px;
# #     font-weight: 600;
# #     font-size: 0.875rem;
# #     letter-spacing: 0.01em;
# #     transition: opacity 0.15s, transform 0.1s;
# #     width: 100%;
# # }
# # .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
# # .stButton > button:active { transform: translateY(0); }

# # /* ── Upload area ── */
# # [data-testid="stFileUploader"] {
# #     background: #0d1a27;
# #     border: 1.5px dashed #1e3650;
# #     border-radius: 12px;
# #     padding: 12px;
# #     transition: border-color 0.2s;
# # }
# # [data-testid="stFileUploader"]:hover { border-color: #0ea5e9; }

# # /* ── Sliders ── */
# # [data-testid="stSlider"] > div > div > div > div {
# #     background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
# }

# /* ── Expander ── */
# [data-testid="stExpander"] {
#     background: #0d1a27;
#     border: 1px solid #1a2d44;
#     border-radius: 10px;
# }
# [data-testid="stExpander"] summary {
#     color: #94b8d4 !important;
#     font-size: 0.85rem;
#     font-weight: 500;
# }

# /* ── Spinner ── */
# [data-testid="stSpinner"] { color: #38bdf8 !important; }

# /* ── Divider ── */
# hr { border-color: #1a2d44 !important; margin: 20px 0 !important; }

# /* ── Subheader ── */
# h2, h3 { color: #c8dff0 !important; font-weight: 600 !important; }

# /* ── Number input / selectbox ── */
# [data-baseweb="input"], [data-baseweb="select"] {
#     background: #0d1a27 !important;
#     border-color: #1a2d44 !important;
#     border-radius: 8px !important;
#     color: #e2ecf6 !important;
# }

# /* ── Markdown tables ── */
# table { width: 100%; border-collapse: collapse; }
# th { background: #0d1a27; color: #4a6580; font-family: 'JetBrains Mono', monospace;
#      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
#      padding: 10px 14px; border-bottom: 1px solid #1a2d44; }
# td { color: #c8dff0; padding: 9px 14px; font-family: 'JetBrains Mono', monospace;
#      font-size: 0.82rem; border-bottom: 1px solid #0f1e2d; }
# tr:last-child td { border-bottom: none; }

# /* ── Success / Warning / Info ── */
# [data-testid="stAlert"] { border-radius: 10px !important; border-width: 1px !important; }

# /* ── Video ── */
# video { border-radius: 10px; border: 1px solid #1a2d44; }

# /* ── Caption ── */
# .stCaption { color: #4a6580 !important; font-size: 0.78rem !important; }

# /* ── Stat pill used in custom cards ── */
# .cv-pill {
#     display: inline-block;
#     padding: 3px 10px;
#     border-radius: 999px;
#     font-size: 0.72rem;
#     font-family: 'JetBrains Mono', monospace;
#     font-weight: 500;
#     letter-spacing: 0.04em;
#     background: #0a1929;
#     border: 1px solid #1a2d44;
#     color: #94b8d4;
#     margin-right: 4px;
# }

# /* ── Section label eyebrow ── */
# .cv-eyebrow {
#     font-size: 0.7rem;
#     font-family: 'JetBrains Mono', monospace;
#     text-transform: uppercase;
#     letter-spacing: 0.12em;
#     color: #4a6580;
#     margin-bottom: 10px;
# }

# /* ── Download button ── */
# [data-testid="stDownloadButton"] > button {
#     background: #0d1a27 !important;
#     border: 1px solid #1a2d44 !important;
#     color: #38bdf8 !important;
#     font-weight: 500 !important;
# }
# [data-testid="stDownloadButton"] > button:hover {
#     border-color: #38bdf8 !important;
#     background: #0a2035 !important;
# }

# /* FIX SIDEBAR */
# section[data-testid="stSidebar"]{
# min-width:320px !important;
# max-width:320px !important;
# }
# [data-testid="collapsedControl"]{display:none !important;}
# button[kind="header"]{display:none !important;}

# </style>
# """, unsafe_allow_html=True)

# # ─────────────────────────── MODEL ─────────────────────────────
# @st.cache_resource
# def load_cloud_model():
#     return load_model("cloud_model.keras")

# model = load_cloud_model()

# class_names  = ["Cumulus","Altocumulus","Cirrus","ClearSky","Stratocumulus","Cumulonimbus","Mixed"]
# cloud_height = {"Cumulus":1500,"Altocumulus":4500,"Cirrus":9000,
#                 "ClearSky":0,"Stratocumulus":1200,"Cumulonimbus":6000,"Mixed":3500}
# st.sidebar.subheader("Stereo Camera Setup")

# baseline_m = st.sidebar.number_input(
#     "Camera 1 ↔ Camera 2 Distance (meters)",
#     min_value=0.1,
#     value=50.0,
#     step=0.1
# )

# focal_length_mm = st.sidebar.number_input(
#     "Camera Focal Length (mm)",
#     value=4.0
# )

# sensor_width_mm = st.sidebar.number_input(
#     "Sensor Width (mm)",
#     value=6.4
# )
# cloud_emoji  = {"Cumulus":"⛅","Altocumulus":"🌤️","Cirrus":"🌬️",
#                 "ClearSky":"☀️","Stratocumulus":"🌥️","Cumulonimbus":"⛈️","Mixed":"🌦️"}

# # ─────────────────────────── HELPERS ───────────────────────────
# def stereo_cloud_height(
#     cloud_x_cam1,
#     cloud_x_cam2,
#     baseline_m,
#     focal_length_px
# ):
#     disparity = abs(cloud_x_cam1 - cloud_x_cam2)

#     if disparity < 1:
#         return None

#     return (focal_length_px * baseline_m) / disparity
# def angle_to_direction(angle_deg):
#     a = angle_deg % 360
#     if   45  <= a < 135: return "North"
#     elif 135 <= a < 225: return "West"
#     elif 225 <= a < 315: return "South"
#     else:                return "East"

# def compute_optical_flow(gray1, gray2):
#     flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     return np.median(mag), np.mean(ang)

# def pixels_to_kmh(pixel_displacement, delta_t_sec, cloud_type, frame_width, fov):
#     height_m      = cloud_height.get(cloud_type, 2000)
#     # Pixel → Angle
#     degree_per_px = fov / frame_width
#     theta_deg     = pixel_displacement * degree_per_px
#     theta_rad     = math.radians(theta_deg)
#     # Tan formula: d = h * tan(θ)  — right triangle, cloud horizontally moves
#     distance_m    = height_m * math.tan(theta_rad)
#     speed_mps     = distance_m / delta_t_sec
#     speed_kmh     = speed_mps * 3.6
#     return speed_mps, speed_kmh, degree_per_px, theta_deg, distance_m, height_m

# def compute_cloud_density(frame, sky_h):
#     """
#     Calculates cloud coverage percentage within the sky region.
#     Returns: coverage_percent (0-100), density_label, density_color
#     """
#     gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     sky_gray = gray[:sky_h, :]
#     sky_hsv  = hsv[:sky_h, :]

#     _, bright_mask = cv2.threshold(sky_gray, 140, 255, cv2.THRESH_BINARY)
#     sat = sky_hsv[:, :, 1]
#     _, sat_mask = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)
#     hue = sky_hsv[:, :, 0]
#     blue_sky = cv2.inRange(hue, 95, 135)
#     not_blue = cv2.bitwise_not(blue_sky)
#     combined = cv2.bitwise_or(bright_mask, cv2.bitwise_and(sat_mask, not_blue))

#     total_pixels = sky_gray.shape[0] * sky_gray.shape[1]
#     cloud_pixels = int(cv2.countNonZero(combined))
#     coverage = min(100.0, (cloud_pixels / total_pixels) * 100)

#     if coverage < 20:
#         label, color = "Low ☀️", "#22c55e"
#     elif coverage < 55:
#         label, color = "Medium 🌤️", "#f59e0b"
#     else:
#         label, color = "High ⛅", "#ef4444"

#     return round(coverage, 1), label, color


# def predict_visibility(cloud_type, speed_kmh, direction, coverage_pct, fov_deg):
#     """
#     Estimates how long the cloud will remain visible in the sky (until dissipation).
#     Estimate is based on cloud type's atmospheric lifetime, speed, and coverage.
#     Returns: verdict, reason, color, lifetime_min
#     """

#     # ── Typical atmospheric lifetimes by cloud type (minutes) ──
#     # Based on meteorological averages:
#     # Cumulus: 10–60 min (convective, quickly form/dissipate)
#     # Altocumulus: 30–120 min (mid-level, moderate lifetime)
#     # Cirrus: 60–360 min (high-altitude ice, very long lasting)
#     # ClearSky: no cloud
#     # Stratocumulus: 60–480 min (layer cloud, very persistent)
#     # Cumulonimbus: 30–90 min (active storm cell, intense but burns out)
#     # Mixed: 20–90 min (varied)
#     cloud_lifetime_range = {
#         "Cumulus":       (10,  60),
#         "Altocumulus":   (30,  120),
#         "Cirrus":        (60,  360),
#         "ClearSky":      (0,   0),
#         "Stratocumulus": (60,  480),
#         "Cumulonimbus":  (30,  90),
#         "Mixed":         (20,  90),
#     }

#     if cloud_type == "ClearSky":
#         return (
#             "☀️ Clear Sky — No Cloud",
#             "Sky is currently clear. No clouds are visible.",
#             "#22c55e",
#             999,
#         )

#     if speed_kmh < 0.5:
#         lo, hi = cloud_lifetime_range.get(cloud_type, (20, 90))
#         mid = (lo + hi) // 2
#         return (
#             f"🟡 Stationary — ~{mid} min remaining",
#             (f"{cloud_type} clouds typically last {lo}–{hi} min. "
#              f"This cloud is currently stationary and will remain visible for approximately {mid} more minutes."),
#             "#f59e0b",
#             float(mid),
#         )

#     lo, hi = cloud_lifetime_range.get(cloud_type, (20, 90))

#     # High coverage = more moisture/mass = longer lifetime
#     coverage_bonus = (coverage_pct / 100.0) * (hi - lo) * 0.3

#     # Fast-moving clouds dissipate faster (turbulence, mixing)
#     # Normalize: >60 km/h = fast, cuts lifetime by up to 30%
#     speed_factor = max(0.7, 1.0 - (speed_kmh / 200.0))

#     lifetime_min = ((lo + hi) / 2 + coverage_bonus) * speed_factor
#     lifetime_min = round(max(lo, min(hi, lifetime_min)))

#     # ── Verdict tiers ──
#     if lifetime_min > 60:
#         hrs = lifetime_min // 60
#         mins = lifetime_min % 60
#         time_str = f"~{hrs}h {mins}min" if mins else f"~{hrs}h"
#         verdict = f"🟢 Will Stay — {time_str} remaining"
#         reason  = (f"{cloud_type} clouds are long-lasting (typically {lo}–{hi} min). "
#                    f"With {coverage_pct}% coverage and a speed of {speed_kmh:.1f} km/h, "
#                    f"this cloud is expected to remain visible for approximately {time_str}.")
#         color   = "#22c55e"
#     elif lifetime_min > 20:
#         verdict = f"🟡 Moderate — ~{lifetime_min} min remaining"
#         reason  = (f"{cloud_type} clouds typically last {lo}–{hi} min. "
#                    f"Based on the current speed ({speed_kmh:.1f} km/h) and coverage ({coverage_pct}%), "
#                    f"this cloud is expected to remain visible for approximately {lifetime_min} more minutes.")
#         color   = "#f59e0b"
#     else:
#         verdict = f"🔴 Dissipating — ~{lifetime_min} min remaining"
#         reason  = (f"{cloud_type} clouds dissipate quickly (lifetime {lo}–{hi} min). "
#                    f"The high speed ({speed_kmh:.1f} km/h) and low coverage ({coverage_pct}%) suggest "
#                    f"this cloud will disappear from the sky in approximately {lifetime_min} minutes.")
#         color   = "#ef4444"

#     return verdict, reason, color, float(lifetime_min)


# def predict_cloud_type(frames_or_images):
#     """Batch-predict cloud type from a list of frames or PIL images."""
#     batch = []
#     for img in frames_or_images:
#         img_pil = Image.fromarray(img).resize((224, 224)) if isinstance(img, np.ndarray) else img.resize((224, 224))
#         batch.append(image.img_to_array(img_pil) / 255.0)

#     batch_arr = np.stack(batch, axis=0)          # shape: (N, 224, 224, 3)
#     preds_all = model.predict(batch_arr, verbose=0)  # single forward pass

#     preds = [class_names[np.argmax(p)] for p in preds_all]
#     confs = [float(np.max(p)) * 100 for p in preds_all]
#     return Counter(preds).most_common(1)[0][0], float(np.mean(confs))

# def compute_solar_shadow_forecast(cloud_type, height_m, speed_mps, speed_kmh,
#                                    direction, pixel_disp, frame_width, fov_deg,
#                                    coverage_pct):
#     """
#     Camera is mounted on or near the solar plant.
#     When a cloud enters the field of view, its shadow falls somewhere on the ground.

#     Physics:
#       - Cloud's current angular position from frame center → ground_offset
#         ground_offset = height_m * tan(angle_from_zenith)
#       - Shadow ground offset = cloud's horizontal distance from directly overhead
#       - If cloud is moving toward the camera center → shadow WILL hit the panel
#       - Time to shadow = ground_offset / speed_mps
#       - If cloud is moving away from center → shadow WON'T hit the panel

#     Returns dict with all forecast info.
#     """
#     cloud_power_factor = {
#         "Cumulus": 0.55, "Altocumulus": 0.45, "Cirrus": 0.18,
#         "ClearSky": 0.0, "Stratocumulus": 0.72, "Cumulonimbus": 0.85, "Mixed": 0.50
#     }

#     if cloud_type == "ClearSky":
#         return {
#             "will_hit": False,
#             "reason": "☀️ Sky is completely clear — no clouds, no shadow.",
#             "status": "clear",
#             "shadow_time_min": None,
#             "power_drop_pct": 0.0,
#             "ground_offset_m": 0.0,
#         }

#     # Angular position of cloud from frame center (pixels → degrees → radians)
#     deg_per_px     = fov_deg / frame_width
#     # pixel_disp is displacement magnitude; we use half-frame as rough center-offset
#     # Center offset: assume cloud centroid is ~frame center (worst case / mean case)
#     # More accurate: use half FOV = cloud is within FOV, so it's within ±fov/2 of zenith
#     # We use the actual angle the cloud has already traveled (theta) as proxy for offset
#     half_fov_deg   = fov_deg / 2.0
#     # Angle from zenith to cloud edge (conservative: use half-FOV)
#     angle_rad      = math.radians(half_fov_deg * 0.5)   # ~centre of visible sky arc
#     ground_offset_m = height_m * math.tan(angle_rad)    # metres from panel

#     # Direction logic: is cloud moving TOWARD overhead (will shadow hit) or AWAY?
#     # "toward overhead" = cloud is off to one side and moving toward center
#     # Simplified: if cloud is IN the FOV it is either overhead now or will cross overhead
#     # based on direction + frame position.
#     # We check: does the cloud's travel path cross the zenith column?
#     # Heuristic: if cloud is moving and it's within FOV → it will pass overhead → shadow hits
#     # UNLESS cloud is already past center and moving further away.

#     # We estimate current cloud X position from optical flow direction
#     # North/South movement means cloud crosses overhead (shadow hits)
#     # East/West also crosses — it just depends on whether it already passed
#     # Use pixel_disp relative to frame to estimate if approaching or receding
    
#     # Simple model: cloud in FOV = overhead within half-FOV → shadow WILL hit
#     # Time for shadow to reach panel = ground_offset / speed_mps
    
#     if speed_mps < 0.05:
#         # Nearly stationary
#         return {
#             "will_hit": True,
#             "reason": f"Cloud is still in the sky and is practically stationary. Shadow may already be present on the solar panel or has stalled overhead.",
#             "status": "stationary",
#             "shadow_time_min": 0.0,
#             "power_drop_pct": _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor),
#             "ground_offset_m": ground_offset_m,
#         }

#     time_to_shadow_sec = ground_offset_m / speed_mps
#     time_to_shadow_min = time_to_shadow_sec / 60.0

#     # If time is very small (< 0.5 min) → shadow is basically overhead now
#     # If cloud is moving away (already past zenith), offset increases over time → no hit
#     # Proxy: use pixel displacement direction vs frame center
#     # If the optical flow vector points toward frame center → cloud approaching
#     # Rough approximation: if time_to_shadow_min < 0 conceptually, cloud already passed
#     # We use: if time_to_shadow_min > time_to_exit_min-equivalent → won't cross
    
#     # Practical cutoff: if it takes longer than cloud lifetime to arrive → won't hit
#     cloud_lifetime = {
#         "Cumulus": 35, "Altocumulus": 75, "Cirrus": 200,
#         "ClearSky": 0, "Stratocumulus": 270, "Cumulonimbus": 60, "Mixed": 55
#     }
#     lifetime_min = cloud_lifetime.get(cloud_type, 60)

#     power_drop = _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor)

#     if time_to_shadow_min <= 0.3:
#         return {
#             "will_hit": True,
#             "reason": f"Cloud is almost directly overhead — shadow is currently falling on the solar panel or is about to.",
#             "status": "now",
#             "shadow_time_min": 0.0,
#             "power_drop_pct": power_drop,
#             "ground_offset_m": ground_offset_m,
#         }
#     elif time_to_shadow_min <= lifetime_min:
#         return {
#             "will_hit": True,
#             "reason": f"{cloud_type} cloud is offset by {ground_offset_m/1000:.2f} km. At {speed_kmh:.1f} km/h, the shadow will reach the solar panel in {time_to_shadow_min:.1f} minutes.",
#             "status": "incoming",
#             "shadow_time_min": time_to_shadow_min,
#             "power_drop_pct": power_drop,
#             "ground_offset_m": ground_offset_m,
#         }
#     else:
#         return {
#             "will_hit": False,
#             "reason": (f"The estimated shadow arrival time for this {cloud_type} cloud is ~{time_to_shadow_min:.0f} min, "
#                        f"but the cloud's expected lifetime is only ~{lifetime_min} min. "
#                        f"The cloud will dissipate or exit the field of view before the shadow reaches the panel."),
#             "status": "miss",
#             "shadow_time_min": time_to_shadow_min,
#             "power_drop_pct": 0.0,
#             "ground_offset_m": ground_offset_m,
#         }


# def _calc_power_drop(cloud_type, coverage_pct, factor_map):
#     base = factor_map.get(cloud_type, 0.50)
#     cov  = (coverage_pct / 100.0) if coverage_pct is not None else 0.5
#     drop = base * 0.55 * 100 + cov * base * 0.45 * 100
#     return round(min(95.0, drop), 1)


# def show_metrics(cloud_type, confidence, direction, height_m, fov,
#                  frame_width, pixel_disp, delta_t, deg_per_px,
#                  theta_deg, distance_m, speed_mps, speed_kmh, dist_5, dist_15,
#                  coverage_pct=None, density_label=None, density_color=None,
#                  vis_verdict=None, vis_reason=None, vis_color=None,
#                  time_to_exit_min=999, solar_dist_km=1.0,
#                  media_timestamp=None, timestamp_source="now",
#                  image_elevation_est=None, image_elevation_conf=None,
#                  image_elevation_note=None):
#     emoji = cloud_emoji.get(cloud_type, "☁️")

#     # ── Low confidence warning ──
#     if confidence < 60.0:
#         st.warning(
#             f"⚠️ **Low Model Confidence ({confidence:.1f}%)** — The model is uncertain about this cloud type. "
#             f"Results may be less accurate. Try uploading a clearer image or more frames."
#         )

#     # ── Section divider helper ──
#     def section_header(icon, title):
#         st.markdown(f"""
# <div style="display:flex;align-items:center;gap:10px;margin:28px 0 14px 0;
#             padding-bottom:10px;border-bottom:1px solid #1a2d44;">
#   <span style="font-size:1.1rem;">{icon}</span>
#   <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#                letter-spacing:0.12em;color:#4a6580;font-weight:600;">{title}</span>
# </div>
# """, unsafe_allow_html=True)

#     # ── Section 1: Cloud Classification ──
#     section_header("☁", "Cloud Classification")
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric(f"{emoji} Cloud Type",    cloud_type)
#     c2.metric("🎯 Model Confidence",    f"{confidence:.1f}%")
#     c3.metric("🧭 Wind Direction",      direction)
#     c4.metric("📍 Estimated Altitude",  f"{height_m:,} m")

#     # ── Section 2: Motion & Displacement ──
#     section_header("⚡", "Motion & Displacement")
#     label_5  = f"~{dist_5:.2f} km"  if dist_5  > 0 else "—"
#     label_15 = f"~{dist_15:.2f} km" if dist_15 > 0 else "—"

#     c5, c6, c7, c8 = st.columns(4)
#     c5.metric("Wind Speed",            f"{speed_kmh:.1f} km/h")
#     c6.metric("Wind Speed (m/s)",      f"{speed_mps:.2f} m/s")
#     c7.metric("Projected Dist. +5 min",  label_5,  delta_color="off")
#     c8.metric("Projected Dist. +15 min", label_15, delta_color="off")

#     # ── Section 3: Atmospheric Analysis ──
#     if coverage_pct is not None:
#         section_header("📡", "Atmospheric Analysis")
#         d1, d2 = st.columns(2)

#         with d1:
#             bar_filled = int(coverage_pct)
#             st.markdown(f"""
# <div style="background:#0d1a27;border:1px solid {density_color}44;border-radius:12px;padding:22px;">
#   <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#               letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">Sky Coverage</div>
#   <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:16px;">
#     <span style="font-size:2.2rem;font-weight:700;color:{density_color};font-family:'Inter',sans-serif;line-height:1;">{coverage_pct}%</span>
#     <span style="font-size:0.82rem;color:{density_color};font-weight:600;padding:3px 10px;
#                  background:{density_color}18;border-radius:999px;border:1px solid {density_color}33;">{density_label}</span>
#   </div>
#   <div style="background:#050c14;border-radius:8px;height:10px;width:100%;overflow:hidden;margin-bottom:8px;">
#     <div style="background:linear-gradient(90deg,{density_color}77,{density_color});
#                 width:{bar_filled}%;height:10px;border-radius:8px;"></div>
#   </div>
#   <div style="display:flex;justify-content:space-between;margin-top:6px;">
#     <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">Low — &lt;20%</span>
#     <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">Medium — 20–55%</span>
#     <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">High — &gt;55%</span>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#         with d2:
#             st.markdown(f"""
# <div style="background:#0d1a27;border:1px solid {vis_color}44;border-radius:12px;padding:22px;height:100%;">
#   <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#               letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">Cloud Lifetime Forecast</div>
#   <div style="font-size:1rem;font-weight:700;color:{vis_color};margin-bottom:12px;
#               padding:8px 12px;background:{vis_color}14;border-radius:8px;border-left:3px solid {vis_color};">{vis_verdict}</div>
#   <div style="font-size:0.82rem;color:#7a9ab4;line-height:1.65;">{vis_reason}</div>
# </div>
# """, unsafe_allow_html=True)

#     # ── Section 4: Solar Plant Impact Forecast ──
#     section_header("☀️", "Solar Plant Impact Forecast")

#     solar = compute_solar_shadow_forecast(
#         cloud_type, height_m, speed_mps, speed_kmh,
#         direction, pixel_disp, frame_width, fov,
#         coverage_pct if coverage_pct is not None else 50.0
#     )

#     status = solar["status"]

#     if status == "clear":
#         st.markdown("""
# <div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:12px;padding:24px;display:flex;align-items:center;gap:20px;">
#   <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
#               display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">☀️</div>
#   <div>
#     <div style="font-size:1.05rem;font-weight:700;color:#22c55e;margin-bottom:6px;">Clear Sky — Solar Plant Fully Safe</div>
#     <div style="font-size:0.82rem;color:#4a8060;line-height:1.5;">No clouds detected. No shadow risk. Solar plant is operating at full capacity.</div>
#   </div>
#   <div style="margin-left:auto;text-align:right;flex-shrink:0;">
#     <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Power Status</div>
#     <div style="font-size:1.1rem;font-weight:700;color:#22c55e;">🟢 100%</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif status == "now":
#         pdrop = solar["power_drop_pct"]
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #ef444455;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:#ef444418;border:1px solid #ef444433;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">⚠️</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:#ef4444;margin-bottom:5px;">Shadow Currently Falling on Solar Panel</div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; {speed_kmh:.1f} km/h
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
#     <div style="background:#100505;border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#ef4444;line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#100505;border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
#       <div style="font-size:1rem;font-weight:700;color:#ff6b6b;margin-top:4px;">🔴 Active Now</div>
#     </div>
#     <div style="background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 2;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif status == "stationary":
#         pdrop = solar["power_drop_pct"]
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #f59e0b55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:#f59e0b18;border:1px solid #f59e0b33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🟡</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:#f59e0b;margin-bottom:5px;">Cloud Stationary — Shadow Present on Panel</div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; Nearly stationary
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
#     <div style="background:#100a00;border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#f59e0b;line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#100a00;border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Movement</div>
#       <div style="font-size:1rem;font-weight:700;color:#f59e0b;margin-top:4px;">⏸ Stationary</div>
#     </div>
#     <div style="background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 2;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif status == "incoming":
#         pdrop    = solar["power_drop_pct"]
#         arr_min  = solar["shadow_time_min"]
#         off_km   = solar["ground_offset_m"] / 1000.0
#         if arr_min < 10:
#             urg_color = "#ef4444"; urg_icon = "🔴"
#         elif arr_min < 30:
#             urg_color = "#f59e0b"; urg_icon = "🟡"
#         else:
#             urg_color = "#22c55e"; urg_icon = "🟢"
#         arr_str = f"{arr_min:.1f} min" if arr_min < 60 else f"{int(arr_min//60)}h {int(arr_min%60)}m"
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid {urg_color}55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:{urg_color}18;border:1px solid {urg_color}33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">☁️</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:{urg_color};margin-bottom:5px;">
#         {urg_icon} Shadow Will Reach Solar Plant in {arr_str}
#       </div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;">
#     <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Shadow Arrives In</div>
#       <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{arr_str}</div>
#     </div>
#     <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1e3a50;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Ground Offset</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#38bdf8;line-height:1;">{off_km:.2f} km</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 3;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Forecast Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     else:  # status == "miss"
#         arr_min = solar["shadow_time_min"]
#         arr_str = f"{arr_min:.0f} min" if arr_min is not None else "N/A"
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">✅</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:#22c55e;margin-bottom:5px;">Shadow Will Not Reach the Solar Plant</div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
#       </div>
#     </div>
#   </div>
#   <div style="background:#050c14;border:1px solid #22c55e22;border-radius:10px;padding:16px 18px;margin-bottom:12px;">
#     <div style="font-size:0.82rem;color:#6aaa84;line-height:1.7;">{solar["reason"]}</div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#22c55e;line-height:1;">0%</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
#       <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-top:4px;">🟢 Safe</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
#     # ── Section 5: Sun Position & Cloud Alignment ──
#     _lat = st.session_state.get("user_lat", 28.6)
#     _lon = st.session_state.get("user_lon", 77.2)

#     # Use media timestamp if available, else current time
#     _ts = media_timestamp if media_timestamp is not None else datetime.datetime.utcnow()
#     sun_az, sun_el = get_solar_position(_lat, _lon, _ts)

#     # Timestamp source badge
#     _ts_badge_map = {
#         "exif":    ("📷 From EXIF",        "#22c55e"),
#         "ffprobe": ("🎬 From Video Meta",  "#22c55e"),
#         "manual":  ("🕐 Manual Input",      "#f59e0b"),
#         "now":     ("⏱️ Current Time",      "#4a6580"),
#     }
#     _ts_label, _ts_color = _ts_badge_map.get(timestamp_source, ("⏱️ Current Time", "#4a6580"))

#     section_header("🌞", "Sun Position & Cloud Alignment")

#     if sun_az is None:
#         st.info("📍 Install pvlib + pandas and set your location in the sidebar for live sun tracking.")
#     else:
#         sun_dir  = sun_azimuth_to_direction(sun_az)
#         align_status, angle_diff, align_desc = get_cloud_sun_alignment(direction, sun_az)

#         align_color_map = {
#             "toward_sun":    "#ef4444",
#             "glancing":      "#f59e0b",
#             "crossing":      "#38bdf8",
#             "away_from_sun": "#22c55e",
#             "unknown":       "#4a6580",
#         }
#         align_icon_map = {
#             "toward_sun":    "🔴 Heading Toward Sun",
#             "glancing":      "🟡 Glancing Sun",
#             "crossing":      "🔵 Crossing Sun Path",
#             "away_from_sun": "🟢 Moving Away from Sun",
#             "unknown":       "❓ Unknown",
#         }
#         a_color = align_color_map.get(align_status, "#4a6580")
#         a_label = align_icon_map.get(align_status, "")

#         if sun_el < 0:
#             sun_status_label = "🌙 Below Horizon"
#             sun_el_color = "#4a6580"
#         elif sun_el < 15:
#             sun_status_label = "🌅 Near Horizon"
#             sun_el_color = "#f59e0b"
#         else:
#             sun_status_label = "☀️ Above Horizon"
#             sun_el_color = "#fbbf24"

#         # ── Determine which elevation to show ──
#         # pvlib = authoritative; image estimate = fallback shown alongside
#         show_img_est = (image_elevation_est is not None)
#         img_conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(
#             image_elevation_conf, "#4a6580")

#         sc1, sc2, sc3, sc4 = st.columns(4)
#         sc1.metric("☀️ Sun Azimuth",   f"{sun_az:.1f}°")
#         sc2.metric("📐 Sun Elevation (pvlib)", f"{sun_el:.1f}°")
#         sc3.metric("🧭 Sun Direction", sun_dir)
#         sc4.metric("☁️ Cloud Moving",  direction)

#         # Timestamp badge + image estimate row
#         badge_html = f"""
# <div style="display:flex;align-items:center;gap:10px;margin:10px 0 14px 0;flex-wrap:wrap;">
#   <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;padding:3px 10px;
#                border-radius:999px;background:{_ts_color}18;border:1px solid {_ts_color}44;
#                color:{_ts_color};">{_ts_label}: {_ts.strftime('%Y-%m-%d %H:%M UTC')}</span>"""

#         if show_img_est:
#             badge_html += f"""
#   <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;padding:3px 10px;
#                border-radius:999px;background:{img_conf_color}18;border:1px solid {img_conf_color}44;
#                color:{img_conf_color};">
#     📸 Image Estimate: {image_elevation_est}° ({image_elevation_conf} confidence)
#   </span>"""

#         badge_html += "</div>"
#         st.markdown(badge_html, unsafe_allow_html=True)

#         # Image elevation detail card (only when sun not visible in frame)
#         if show_img_est:
#             diff_el = abs(sun_el - image_elevation_est)
#             st.markdown(f"""
# <div style="background:#0d1a27;border:1px solid {img_conf_color}44;border-radius:12px;
#             padding:16px 20px;margin-bottom:14px;">
#   <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#               letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">
#     📸 Image-Based Sun Elevation Estimate (Sun not visible in frame)
#   </div>
#   <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;">
#     <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
#       <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:6px;">Image Estimate</div>
#       <div style="font-size:1.6rem;font-weight:700;color:{img_conf_color};line-height:1;">
#         {image_elevation_est}°</div>
#     </div>
#     <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
#       <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:6px;">pvlib (Location+Time)</div>
#       <div style="font-size:1.6rem;font-weight:700;color:#fbbf24;line-height:1;">{sun_el:.1f}°</div>
#     </div>
#     <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
#       <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:6px;">Difference</div>
#       <div style="font-size:1.6rem;font-weight:700;color:{'#22c55e' if diff_el < 10 else '#f59e0b' if diff_el < 25 else '#ef4444'};line-height:1;">
#         ±{diff_el:.1f}°</div>
#     </div>
#   </div>
#   <div style="font-size:0.8rem;color:#7a9ab4;line-height:1.6;font-style:italic;">{image_elevation_note}</div>
# </div>
# """, unsafe_allow_html=True)

#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid {a_color}55;border-radius:14px;padding:22px;margin-top:4px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
#     <div style="width:48px;height:48px;border-radius:12px;background:{a_color}18;border:1px solid {a_color}33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;">🌞</div>
#     <div>
#       <div style="font-size:1rem;font-weight:700;color:{a_color};margin-bottom:4px;">
#         Cloud–Sun Alignment: {a_label}
#       </div>
#       <div style="font-size:0.78rem;font-family:'JetBrains Mono',monospace;color:#4a6580;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:3px 10px;display:inline-block;">
#         {sun_status_label} &nbsp;·&nbsp; Azimuth {sun_az:.1f}° &nbsp;·&nbsp; Elevation {sun_el:.1f}°
#         {'&nbsp;·&nbsp; ' + str(round(angle_diff)) + '° offset' if angle_diff is not None else ''}
#       </div>
#     </div>
#   </div>
#   <div style="font-size:0.84rem;color:#94b8d4;line-height:1.7;background:#050c14;
#               border-radius:10px;padding:14px 18px;border:1px solid #1a2d44;">
#     {align_desc}
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

#     # ── Export Analysis Results ──
#     with st.expander("📤 Export Analysis Report"):
#         import csv, io
#         report_data = {
#             "cloud_type": cloud_type,
#             "confidence_pct": round(confidence, 1),
#             "direction": direction,
#             "altitude_m": height_m,
#             "speed_kmh": round(speed_kmh, 1),
#             "speed_mps": round(speed_mps, 2),
#             "projected_dist_5min_km": round(dist_5, 2),
#             "projected_dist_15min_km": round(dist_15, 2),
#             "sky_coverage_pct": coverage_pct,
#             "density_label": density_label,
#             "visibility_forecast": vis_verdict,
#             "timestamp_utc": _ts.strftime("%Y-%m-%d %H:%M UTC") if media_timestamp is not None else datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
#             "timestamp_source": timestamp_source,
#         }
#         # JSON download
#         import json as _json
#         json_str = _json.dumps(report_data, indent=2)
#         st.download_button(
#             "📥 Download JSON Report",
#             data=json_str,
#             file_name=f"cloudvision_report_{cloud_type}.json",
#             mime="application/json",
#             key="dl_json_report"
#         )
#         # CSV download
#         csv_buf = io.StringIO()
#         writer = csv.DictWriter(csv_buf, fieldnames=report_data.keys())
#         writer.writeheader()
#         writer.writerow(report_data)
#         st.download_button(
#             "📥 Download CSV Report",
#             data=csv_buf.getvalue(),
#             file_name=f"cloudvision_report_{cloud_type}.csv",
#             mime="text/csv",
#             key="dl_csv_report"
#         )
#         st.code(json_str, language="json")

#     with st.expander("🔬 Optical Flow — Calculation Details"):
#         st.markdown(f"""
# | Parameter | Value |
# |---|---|
# | Camera FOV | {fov}° |
# | Frame Width | {frame_width} px |
# | Degrees per Pixel | {deg_per_px:.4f} °/px |
# | Pixel Displacement | {pixel_disp:.2f} px over {delta_t:.2f} s |
# | Angular Displacement (θ) | {theta_deg:.4f}° |
# | Horizontal Distance (tan formula) | {distance_m:.2f} m |
# | Derived Speed | {speed_mps:.2f} m/s → {speed_kmh:.1f} km/h |
# """)

# # ─────────────────────────── CLOUD DETECTION ───────────────────
# def detect_clouds(frame, sky_h):
#     """
#     Multi-method cloud detection:
#     1. Brightness threshold (white clouds)
#     2. HSV saturation (low saturation = cloud/white)
#     3. Combine both masks
#     Uses watershed-style distance-based separation to assign distinct
#     bounding boxes to individual cloud regions.
#     """
#     OUT_W = frame.shape[1]

#     gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     sky_gray = gray[:sky_h, :]
#     sky_hsv  = hsv[:sky_h, :]

#     # Method 1: brightness — lower threshold to catch grey clouds too
#     _, bright_mask = cv2.threshold(sky_gray, 140, 255, cv2.THRESH_BINARY)

#     # Method 2: low saturation = white/grey cloud (not blue sky)
#     sat = sky_hsv[:, :, 1]
#     _, sat_mask = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)

#     # Method 3: not-blue sky — blue sky has high hue (100-130)
#     hue = sky_hsv[:, :, 0]
#     blue_sky = cv2.inRange(hue, 95, 135)
#     not_blue = cv2.bitwise_not(blue_sky)

#     # Combine: bright OR (low-sat AND not-blue-sky)
#     combined = cv2.bitwise_or(bright_mask,
#                 cv2.bitwise_and(sat_mask, not_blue))

#     # Morphology — smaller kernels to keep clouds separate
#     k_close = np.ones((12, 12), np.uint8)
#     k_open  = np.ones((6,  6),  np.uint8)
#     combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
#     combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)

#     # --- Watershed separation to split merged clouds ---
#     dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
#     cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
#     _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
#     sure_fg    = np.uint8(sure_fg)

#     sure_bg    = cv2.dilate(combined, np.ones((3,3), np.uint8), iterations=3)
#     unknown    = cv2.subtract(sure_bg, sure_fg)

#     _, markers = cv2.connectedComponents(sure_fg)
#     markers    = markers + 1
#     markers[unknown == 255] = 0

#     # Watershed needs 3-channel BGR image
#     sky_bgr = frame[:sky_h, :].copy()
#     markers = cv2.watershed(sky_bgr, markers)

#     # Extract bounding boxes from each watershed region
#     boxes = []
#     unique_labels = np.unique(markers)
#     for lbl in unique_labels:
#         if lbl <= 1:   # background or border
#             continue
#         mask_lbl = np.zeros_like(combined)
#         mask_lbl[markers == lbl] = 255
#         cnts, _ = cv2.findContours(mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in cnts:
#             area = cv2.contourArea(cnt)
#             if area < 600:   # ignore tiny noise
#                 continue
#             x, y, w, h = cv2.boundingRect(cnt)
#             boxes.append((x, y, w, h, mask_lbl))

#     # Fallback: if watershed gave nothing, use simple contours
#     if not boxes:
#         cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in cnts:
#             if cv2.contourArea(cnt) < 600:
#                 continue
#             x, y, w, h = cv2.boundingRect(cnt)
#             boxes.append((x, y, w, h, None))

#     return boxes, gray
# def get_cloud_centroid(box):
#     x, y, w, h = box[:4]
#     return x + w // 2

# # ─────────────────────── STEREO DEPTH VISION ───────────────────
# def compute_pseudo_depth_map(frame_bgr, sky_h):
#     """
#     Single-image pseudo stereo depth map for cloud regions.

#     Physics cues used (all monocular):
#       1. Brightness  — brighter cloud core = optically thicker = visually 'closer'
#       2. Texture     — high-freq detail = nearer; smooth/hazy = farther
#       3. Saturation  — desaturated (grey/white) regions = cloud mass present
#       4. Vertical pos— lower in sky frame ≈ closer horizon clouds

#     Output: depth_map (H x W float32, 0=far / blue … 1=near / red)
#             depth_color (H x W x 3 uint8, COLORMAP_JET applied)
#     """
#     sky = frame_bgr[:sky_h, :].copy()
#     H, W = sky.shape[:2]

#     gray = cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
#     hsv  = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV).astype(np.float32)
#     sat  = hsv[:, :, 1] / 255.0
#     val  = hsv[:, :, 2] / 255.0

#     # Cue 1: brightness (brighter = more cloud mass = closer)
#     bright_cue = val

#     # Cue 2: texture energy via Laplacian (sharp edges = nearer)
#     lap = cv2.Laplacian(gray, cv2.CV_32F)
#     tex_cue = np.abs(lap)
#     tex_cue = cv2.GaussianBlur(tex_cue, (15, 15), 0)
#     tex_max = tex_cue.max()
#     if tex_max > 0:
#         tex_cue /= tex_max

#     # Cue 3: low saturation = cloud (not blue sky) → weight up
#     cloud_presence = 1.0 - np.clip(sat, 0, 1)   # white/grey = high weight

#     # Cue 4: vertical position — lower row = closer (nearer horizon)
#     row_idx  = np.linspace(1.0, 0.0, H, dtype=np.float32)   # top=far, bottom=near
#     vert_cue = np.tile(row_idx[:, None], (1, W))

#     # Weighted fusion
#     depth = (0.40 * bright_cue +
#              0.25 * tex_cue    +
#              0.20 * cloud_presence +
#              0.15 * vert_cue)

#     # Smooth for clean visualization
#     depth = cv2.GaussianBlur(depth, (21, 21), 0)
#     cv2.normalize(depth, depth, 0, 1, cv2.NORM_MINMAX)

#     # Colorize: COLORMAP_JET  blue=far → green=mid → red=near
#     depth_u8    = (depth * 255).astype(np.uint8)
#     depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

#     return depth, depth_color


# def depth_to_distance_km(depth_val, cloud_height_m, fov_deg):
#     """
#     Depth value (0–1) → estimated slant distance in km.
#     Uses trigonometry: closer clouds (higher depth) are nearer to cloud_height_m;
#     farther (lower depth) are assumed to be 1.5–3x that height away (oblique angle).
#     """
#     # depth=1 → distance = cloud_height_m (directly overhead)
#     # depth=0 → distance = 3 * cloud_height_m (far horizon, shallow angle)
#     distance_m = cloud_height_m * (1.0 + 2.0 * (1.0 - float(depth_val)))
#     return round(distance_m / 1000.0, 2)


# # ─────────────────────────── BOUNDING BOX FUNCTION ─────────────
# def draw_boxes_on_frame(frame, speed_kmh, direction, cloud_type, height_m,
#                          dist_5, dist_15, elapsed_sec, prev_gray=None, delta_t=None,
#                          fov=75, time_to_exit_min=999):
#     OUT_W = frame.shape[1]
#     OUT_H = frame.shape[0]
#     sky_h = int(OUT_H * 0.78)   # slightly more sky area

#     # Pre-compute full dense optical flow if prev frame available
#     gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     sky_gray = gray[:sky_h, :]
#     full_flow = None
#     if prev_gray is not None and delta_t is not None and delta_t > 0:
#         prev_sky = prev_gray[:sky_h, :]
#         full_flow = cv2.calcOpticalFlowFarneback(
#             prev_sky, sky_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
#         )

#     # Detect clouds with improved method
#     boxes, _ = detect_clouds(frame, sky_h)

#     # ── Compute pseudo stereo depth map for full sky region ──
#     depth_map, depth_color = compute_pseudo_depth_map(frame, sky_h)

#     for (x, y, w, h, _) in boxes:
#         pad = 8
#         x1 = max(0,       x - pad);    y1 = max(0,       y - pad)
#         x2 = min(OUT_W-1, x+w + pad);  y2 = min(sky_h,   y+h + pad)

#         # ── Per-cloud speed from optical flow ROI ──
#         if full_flow is not None:
#             roi_flow = full_flow[y1:y2, x1:x2]
#             if roi_flow.size > 0:
#                 mag, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
#                 roi_pixel_disp = float(np.median(mag))
#                 if roi_pixel_disp > 0.1:
#                     _, cloud_speed_kmh, _, _, _, _ = pixels_to_kmh(
#                         roi_pixel_disp, delta_t, cloud_type, OUT_W, fov
#                     )
#                 else:
#                     cloud_speed_kmh = 0.0
#             else:
#                 cloud_speed_kmh = speed_kmh
#         else:
#             cloud_speed_kmh = speed_kmh

#         # ── Stereo Depth Overlay inside box ──
#         roi_depth_color = depth_color[y1:y2, x1:x2]
#         roi_frame       = frame[y1:y2, x1:x2]
#         if roi_depth_color.shape == roi_frame.shape and roi_frame.size > 0:
#             # Blend depth colormap (40%) with original frame (60%)
#             cv2.addWeighted(roi_depth_color, 0.40, roi_frame, 0.60, 0,
#                             frame[y1:y2, x1:x2])

#         # Estimated distance from depth at box center
#         cy_box = min((y1 + y2) // 2, depth_map.shape[0] - 1)
#         cx_box = min((x1 + x2) // 2, depth_map.shape[1] - 1)
#         center_depth = float(depth_map[cy_box, cx_box])
#         est_dist_km  = depth_to_distance_km(center_depth, height_m, fov)

#         # Glow effect
#         glow = frame.copy()
#         cv2.rectangle(glow, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 100), 4)
#         cv2.addWeighted(glow, 0.3, frame, 0.7, 0, frame)

#         # Main box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

#         # Corner ticks
#         t = 14
#         for (px_, py_, sdx, sdy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
#             cv2.line(frame, (px_, py_), (px_+sdx*t, py_),    (0, 255, 60), 2)
#             cv2.line(frame, (px_, py_), (px_, py_+sdy*t),    (0, 255, 60), 2)

#         # Per-cloud speed + depth distance label
#         label = f"{cloud_speed_kmh:.1f} km/h  |  D:{est_dist_km:.1f}km"
#         fs    = 0.46
#         (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
#         lx = x1
#         ly = y1 - 5 if y1 - 5 - th > 2 else y1 + th + 6
#         cv2.rectangle(frame, (lx-2, ly-th-3), (lx+tw+6, ly+bl+1), (0, 150, 55), -1)
#         cv2.rectangle(frame, (lx-2, ly-th-3), (lx+tw+6, ly+bl+1), (0, 255, 100), 1)
#         cv2.putText(frame, label, (lx+3, ly),
#                     cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

#     # ── HUD top-left ──
#     ov = frame.copy()
#     cv2.rectangle(ov, (0,0), (360,125), (0,0,0), -1)
#     cv2.addWeighted(ov, 0.58, frame, 0.42, 0, frame)
#     cv2.rectangle(frame, (0,0), (360,125), (0,200,80), 1)

#     def txt(t, y, sc=0.52, c=(255,255,255), b=1):
#         cv2.putText(frame, t, (12,y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, b, cv2.LINE_AA)

#     txt(f"Cloud  : {cloud_type}",   22, c=(140,230,255), b=2)
#     txt(f"Height : {height_m:,} m", 43)
#     txt(f"Speed  : {speed_kmh:.1f} km/h  ({speed_kmh/3.6:.2f} m/s)", 64, c=(80,255,160))

#     # Smart +5 / +15 min — show "OUT OF FRAME" if cloud will have exited by then
#     lbl_5  = f"~{dist_5:.2f} km"
#     lbl_15 = f"~{dist_15:.2f} km"
#     txt(f"Dir:{direction}  +5m:{lbl_5}  +15m:{lbl_15}", 86, sc=0.40, c=(200, 200, 200))

#     mins = int(elapsed_sec)//60;  secs = int(elapsed_sec)%60
#     cv2.putText(frame, f"T+ {mins:02d}:{secs:02d}",
#                 (OUT_W-155, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,180), 2, cv2.LINE_AA)

#     # Direction arrow
#     dir_vec = {"East":(1,0),"West":(-1,0),"North":(0,-1),"South":(0,1)}.get(direction,(1,0))
#     cx, cy  = OUT_W//2, OUT_H - 35
#     cv2.arrowedLine(frame, (cx,cy),
#                     (int(cx+dir_vec[0]*55), int(cy+dir_vec[1]*55)),
#                     (255,255,255), 3, tipLength=0.35)
#     cv2.putText(frame, direction, (cx-30, cy+20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

#     # ── Sun Position Arrow ──
#     try:
#         _lat2 = st.session_state.get("user_lat", 28.6)
#         _lon2 = st.session_state.get("user_lon", 77.2)
#         _sun_az2, _sun_el2 = get_solar_position(_lat2, _lon2, datetime.datetime.utcnow())
#     except Exception:
#         _sun_az2, _sun_el2 = None, None

#     if _sun_az2 is not None and _sun_el2 > 0:
#         # Draw sun arrow at bottom-center right of cloud arrow
#         sun_az_rad = math.radians(_sun_az2)   # 0=N, 90=E
#         # Convert azimuth to screen vector (x right=East, y down=South)
#         sun_dx = math.sin(sun_az_rad)   # East component
#         sun_dy = -math.cos(sun_az_rad)  # North component (inverted for screen)
#         scx, scy = OUT_W//2 + 120, OUT_H - 35
#         cv2.arrowedLine(frame, (scx, scy),
#                         (int(scx + sun_dx * 50), int(scy + sun_dy * 50)),
#                         (30, 220, 255), 2, tipLength=0.4)
#         cv2.putText(frame, f"Sun {_sun_az2:.0f}", (scx - 28, scy + 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 220, 255), 1, cv2.LINE_AA)

#         # Cloud-Sun alignment indicator
#         align_status, angle_diff, _ = get_cloud_sun_alignment(direction, _sun_az2)
#         align_color_cv = {
#             "toward_sun":    (60, 80, 255),
#             "glancing":      (60, 180, 255),
#             "crossing":      (255, 200, 60),
#             "away_from_sun": (60, 220, 100),
#             "unknown":       (150, 150, 150),
#         }.get(align_status, (150, 150, 150))
#         align_text = {
#             "toward_sun":    "TO SUN",
#             "glancing":      "GLANCING",
#             "crossing":      "CROSSING",
#             "away_from_sun": "FROM SUN",
#             "unknown":       "?",
#         }.get(align_status, "?")
#         if angle_diff is not None:
#             align_label_full = f"{align_text} {angle_diff:.0f}deg"
#         else:
#             align_label_full = align_text
#         (aw, _ah), _ = cv2.getTextSize(align_label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
#         ax = scx - aw // 2
#         ay = scy - 14
#         cv2.rectangle(frame, (ax - 3, ay - 14), (ax + aw + 4, ay + 4), (10, 10, 10), -1)
#         cv2.putText(frame, align_label_full, (ax, ay),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.44, align_color_cv, 1, cv2.LINE_AA)

#     # ── Solar Plant Shadow HUD (camera IS on solar plant) ──
#     if cloud_type != "ClearSky":
#         solar = compute_solar_shadow_forecast(
#             cloud_type, height_m, speed_mps, speed_kmh,
#             direction, OUT_W * 0.05,   # small pixel_disp proxy for HUD
#             OUT_W, fov, coverage_pct=50.0
#         )
#         status = solar["status"]
#         if status == "now" or status == "stationary":
#             s_line1 = f"SHADOW ON SOLAR PLANT NOW!"
#             s_line2 = f"Power drop: {solar['power_drop_pct']}%"
#             box_col  = (0, 60, 220)   # red-orange
#             txt_col1 = (60, 80, 255)
#             txt_col2 = (60, 255, 160)
#         elif status == "incoming":
#             arr = solar["shadow_time_min"]
#             arr_str = f"{arr:.1f}min" if arr < 60 else f"{int(arr//60)}h{int(arr%60)}m"
#             s_line1 = f"Shadow arrives: {arr_str}"
#             s_line2 = f"Power drop: {solar['power_drop_pct']}%"
#             box_col  = (0, 160, 240)
#             txt_col1 = (80, 220, 255)
#             txt_col2 = (80, 255, 160)
#         elif status == "miss":
#             s_line1 = "Shadow will NOT hit solar plant"
#             s_line2 = "Power drop: 0%  [SAFE]"
#             box_col  = (0, 130, 40)
#             txt_col1 = (80, 255, 120)
#             txt_col2 = (80, 255, 120)
#         else:
#             s_line1 = "Clear sky — solar plant safe"
#             s_line2 = "No shadow expected"
#             box_col  = (0, 130, 40)
#             txt_col1 = (80, 255, 120)
#             txt_col2 = (80, 255, 120)

#         (sw1, _), _ = cv2.getTextSize(s_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
#         (sw2, _), _ = cv2.getTextSize(s_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
#         box_w = max(sw1, sw2) + 22
#         box_h = 58
#         bx, by = OUT_W - box_w - 8, 8

#         sol_ov = frame.copy()
#         cv2.rectangle(sol_ov, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
#         cv2.addWeighted(sol_ov, 0.62, frame, 0.38, 0, frame)
#         cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), box_col, 1)
#         cv2.putText(frame, s_line1, (bx + 8, by + 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col1, 1, cv2.LINE_AA)
#         cv2.putText(frame, s_line2, (bx + 8, by + 44),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col2, 1, cv2.LINE_AA)

#     # ── Depth Legend (colorbar) — bottom right ──
#     bar_x, bar_y, bar_w, bar_h = OUT_W - 120, OUT_H - 90, 18, 70
#     for i in range(bar_h):
#         val   = int(255 * (1.0 - i / bar_h))
#         color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
#         cv2.line(frame, (bar_x, bar_y + i), (bar_x + bar_w, bar_y + i), color, 1)
#     cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
#     cv2.putText(frame, "Near", (bar_x + bar_w + 4, bar_y + 8),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 80, 80), 1, cv2.LINE_AA)
#     cv2.putText(frame, "Far",  (bar_x + bar_w + 4, bar_y + bar_h),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 200), 1, cv2.LINE_AA)
#     cv2.putText(frame, "Depth", (bar_x - 2, bar_y - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

#     return frame


# def generate_boxed_video(input_path, output_path, speed_kmh, speed_mps,
#                           direction, cloud_type, height_m, dist_5, dist_15, fov=75,
#                           time_to_exit_min=999):
#     cap     = cv2.VideoCapture(input_path)
#     fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     OUT_W, OUT_H = 960, 540
#     delta_t = 1.0 / fps

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out    = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, OUT_H))

#     prev_gray = None
#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame   = cv2.resize(frame, (OUT_W, OUT_H))
#         elapsed = frame_idx / fps
#         gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame   = draw_boxes_on_frame(
#             frame, speed_kmh, direction, cloud_type,
#             height_m, dist_5, dist_15, elapsed,
#             prev_gray=prev_gray, delta_t=delta_t, fov=fov,
#             time_to_exit_min=time_to_exit_min
#         )
#         out.write(frame)
#         prev_gray = gray
#         frame_idx += 1

#     cap.release()
#     out.release()

#     # Re-encode to H.264 so browser can play it in st.video()
#     if shutil.which("ffmpeg"):
#         tmp_h264 = output_path.replace(".mp4", "_h264.mp4")
#         subprocess.run([
#             "ffmpeg", "-y", "-i", output_path,
#             "-vcodec", "libx264", "-crf", "23",
#             "-preset", "fast", "-pix_fmt", "yuv420p",
#             "-movflags", "+faststart",
#             tmp_h264
#         ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
#             os.replace(tmp_h264, output_path)


# # ─────────────────────────── UI HEADER ────────────────────────
# st.markdown("""
# <div class="cv-header">
#   <div class="cv-logo">☁️</div>
#   <div>
#     <div class="cv-title">CloudVision AI</div>
#     <div class="cv-sub">Cloud Classification &amp; Motion Prediction System</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

# tab1, tab2 = st.tabs(["🎬 Video Analysis", "🖼️ Multi Image Analysis"])

# # ── Solar location inputs (shared across tabs) ──
# with st.sidebar:
#     st.markdown("### ☀️ Solar Location")
#     st.caption("Enter your location for real-time sun position tracking")
#     user_lat = st.number_input("Latitude",  value=28.6, min_value=-90.0,  max_value=90.0,  step=0.1, format="%.4f", key="user_lat")
#     user_lon = st.number_input("Longitude", value=77.2, min_value=-180.0, max_value=180.0, step=0.1, format="%.4f", key="user_lon")
#     st.caption("🇮🇳 Default: New Delhi")

#     st.markdown("---")
#     st.markdown("### 🕐 Media Timestamp")
#     st.caption("App auto-reads EXIF/video metadata. Override manually if needed.")
#     use_manual_time = st.checkbox("✏️ Override timestamp manually", value=False, key="use_manual_time")
#     if use_manual_time:
#         _today = datetime.date.today()
#         manual_date = st.date_input("Date", value=_today, key="manual_date")
#         manual_time = st.time_input("Time (local)", value=datetime.time(12, 0), key="manual_time")
#         tz_offset   = st.number_input("Timezone offset (hrs from UTC)", value=5.5,
#                                        min_value=-12.0, max_value=14.0, step=0.5, key="tz_offset")
#         # Convert to UTC datetime
#         local_dt = datetime.datetime.combine(manual_date, manual_time)
#         manual_utc = local_dt - datetime.timedelta(hours=tz_offset)
#         st.session_state["manual_utc"] = manual_utc
#         st.caption(f"UTC: {manual_utc.strftime('%Y-%m-%d %H:%M')}")
#     else:
#         st.session_state["manual_utc"] = None

#     if pvlib is not None:
#         _ts_sb = st.session_state.get("manual_utc") or datetime.datetime.utcnow()
#         _az, _el = get_solar_position(user_lat, user_lon, _ts_sb)
#         if _az is not None:
#             _sun_dir = sun_azimuth_to_direction(_az)
#             if _el < 0:
#                 _sun_status = "🌙 Sun below horizon"
#                 _sun_color  = "#4a6580"
#             elif _el < 15:
#                 _sun_status = "🌅 Sun near horizon"
#                 _sun_color  = "#f59e0b"
#             else:
#                 _sun_status = "☀️ Sun above horizon"
#                 _sun_color  = "#fbbf24"
#             st.markdown(f"""
# <div style='background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:14px;margin-top:8px;'>
#   <div style='font-size:0.68rem;font-family:monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Live Sun Position</div>
#   <div style='font-size:0.92rem;font-weight:600;color:{_sun_color};margin-bottom:6px;'>{_sun_status}</div>
#   <div style='font-size:0.78rem;color:#94b8d4;'>Azimuth: <b style='color:#e2ecf6;'>{_az:.1f}°</b> ({_sun_dir})</div>
#   <div style='font-size:0.78rem;color:#94b8d4;'>Elevation: <b style='color:#e2ecf6;'>{_el:.1f}°</b></div>
# </div>
# """, unsafe_allow_html=True)
#     else:
#         st.info("Install pvlib + pandas for live sun tracking:\n`pip install pvlib pandas`")

#     st.markdown("---")
#     st.markdown("### ℹ️ About")
#     st.caption(
#         "**CloudVision AI** — Cloud classification, motion analysis & "
#         "solar shadow forecasting.\n\n"
#         "Model: Keras CNN · Classes: Cumulus, Altocumulus, Cirrus, "
#         "ClearSky, Stratocumulus, Cumulonimbus, Mixed"
#     )

# # ══════════════════════════ VIDEO TAB ══════════════════════════
# with tab1:
#     st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🎬 Video Analysis</div>', unsafe_allow_html=True)
#     st.subheader("Upload a Sky Video")

#     col1, col2 = st.columns(2)

#     with col1:
#         uploaded_video_cam1 = st.file_uploader(
#             "Camera 1 Video",
#             type=["mp4","avi","mov"],
#             key="video_upload_cam1"
#         )

#     with col2:
#         uploaded_video_cam2 = st.file_uploader(
#             "Camera 2 Video",
#             type=["mp4","avi","mov"],
#             key="video_upload_cam2"
#         )

#     uploaded_video = uploaded_video_cam1
#     fov_video = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
#                           help="Phone: 70-80° | Wide angle: 90-120° | Telephoto: 30-50°")

#     # ── Preview uploaded videos immediately ──
#     if uploaded_video_cam1 is not None or uploaded_video_cam2 is not None:
#         prev_col1, prev_col2 = st.columns(2)
#         with prev_col1:
#             if uploaded_video_cam1 is not None:
#                 st.markdown('<div class="cv-eyebrow" style="margin-top:16px;margin-bottom:8px;">📹 Camera 1 — Uploaded Video</div>', unsafe_allow_html=True)
#                 uploaded_video_cam1.seek(0)
#                 st.video(uploaded_video_cam1)
#         with prev_col2:
#             if uploaded_video_cam2 is not None:
#                 st.markdown('<div class="cv-eyebrow" style="margin-top:16px;margin-bottom:8px;">📹 Camera 2 — Uploaded Video</div>', unsafe_allow_html=True)
#                 uploaded_video_cam2.seek(0)
#                 st.video(uploaded_video_cam2)
#         st.divider()

#     if uploaded_video is not None:

#         uploaded_video.seek(0)
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         tfile.write(uploaded_video.read())
#         tfile.flush()
#         tfile.close()

#         with st.spinner("🔍 Analysing video..."):
#             cap          = cv2.VideoCapture(tfile.name)
#             fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#             sample_frames = []
#             for pt in [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pt))
#                 ret, frm = cap.read()
#                 if ret:
#                     sample_frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

#             cloud_type, avg_conf = predict_cloud_type(sample_frames)

#             frame_gap   = max(1, int(fps))
#             delta_t_sec = frame_gap / fps
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0);         ret1, f1 = cap.read()
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_gap); ret2, f2 = cap.read()
#             cap.release()

#         if ret1 and ret2:
#             fw = f1.shape[1]
#             g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
#             g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
#             pixel_disp, avg_angle = compute_optical_flow(g1, g2)
#             direction = angle_to_direction(np.degrees(avg_angle))

#             speed_mps, speed_kmh, deg_per_px, theta_deg, distance_m, height_m = \
#                 pixels_to_kmh(pixel_disp, delta_t_sec, cloud_type, fw, fov_video)
#             pixel_speed = pixel_disp / delta_t_sec
#             dist_5  = speed_kmh * (5  / 60)
#             dist_15 = speed_kmh * (15 / 60)

#             # Density & Visibility compute
#             sample_bgr = cv2.cvtColor(sample_frames[len(sample_frames)//2], cv2.COLOR_RGB2BGR)
#             sky_h_sample = int(sample_bgr.shape[0] * 0.78)
#             cov_pct, den_label, den_color = compute_cloud_density(sample_bgr, sky_h_sample)
#             vis_verdict, vis_reason, vis_color, time_to_exit_min = predict_visibility(
#                 cloud_type, speed_kmh, direction, cov_pct, fov_video)

#             # ── Timestamp resolution (priority: manual > ffprobe > now) ──
#             _manual_utc = st.session_state.get("manual_utc")
#             if _manual_utc is not None:
#                 media_ts      = _manual_utc
#                 ts_source     = "manual"
#             else:
#                 media_ts = extract_video_datetime(tfile.name)
#                 ts_source = "ffprobe" if media_ts is not None else "now"
#                 if media_ts is None:
#                     media_ts = datetime.datetime.utcnow()

#             # ── Image-based sun elevation (for frames where sun not visible) ──
#             img_el_est, img_el_conf, img_el_note = estimate_sun_elevation_from_image(sample_bgr)

#             # ── Sun detection from frame (if sun is visible in sky) ──
#             sun_x, sun_y, sun_visible = detect_sun_in_frame(sample_bgr)
#             sun_az_from_frame = None
#             if sun_visible:
#                 sun_az_from_frame = sun_pixel_to_azimuth(sun_x, sample_bgr.shape[1], fov_video)
#                 img_el_note = (f"☀️ Sun detected in frame at pixel ({sun_x}, {sun_y}). "
#                                f"Estimated azimuth from camera: {sun_az_from_frame:.1f}°")

#             show_metrics(cloud_type, avg_conf, direction, height_m, fov_video,
#                          fw, pixel_disp, delta_t_sec, deg_per_px, theta_deg,
#                          distance_m, speed_mps, speed_kmh, dist_5, dist_15,
#                          coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
#                          vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
#                          time_to_exit_min=time_to_exit_min,
#                          media_timestamp=media_ts, timestamp_source=ts_source,
#                          image_elevation_est=img_el_est, image_elevation_conf=img_el_conf,
#                          image_elevation_note=img_el_note)
            
#                         # ── Detection video — shown RIGHT HERE under uploader ──
#             st.markdown('<div class="cv-eyebrow">📦 Cloud Detection Video</div>', unsafe_allow_html=True)
#             with st.spinner("Generating detection video…"):
#                 tmp_box = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 tmp_box.close()
#                 generate_boxed_video(
#                     tfile.name, tmp_box.name,
#                     speed_kmh, speed_mps, direction,
#                     cloud_type, height_m, dist_5, dist_15, fov=fov_video,
#                     time_to_exit_min=999
#                 )
#                 with open(tmp_box.name, "rb") as f:
#                     vdata = f.read()
#             st.video(vdata)
#             st.download_button("📥 Download Detection Video", data=vdata,
#                                file_name=f"cloud_{cloud_type}_boxes.mp4",
#                                mime="video/mp4", key="dl_box")
#             try: os.unlink(tmp_box.name)
#             except: pass

#             st.divider()
#             st.markdown('<div class="cv-eyebrow">📊 Analysis Results</div>', unsafe_allow_html=True)


#             # ── Prediction video ──
#             st.divider()
#             st.markdown('<div class="cv-eyebrow">🔮 Motion Prediction Video</div>', unsafe_allow_html=True)
#             with st.spinner("Generating prediction video…"):
#                 viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
#                                             direction=direction, pixel_speed=pixel_speed)
#                 tmp_pred = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 tmp_pred.close()
#                 viz.save_video_with_prediction(tmp_pred.name, prediction_minutes=15)
#                 # Re-encode to H.264 for browser playback
#                 if shutil.which("ffmpeg"):
#                     tmp_h264 = tmp_pred.name.replace(".mp4", "_h264.mp4")
#                     subprocess.run([
#                         "ffmpeg", "-y", "-i", tmp_pred.name,
#                         "-vcodec", "libx264", "-crf", "23",
#                         "-preset", "fast", "-pix_fmt", "yuv420p",
#                         "-movflags", "+faststart", tmp_h264
#                     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                     if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
#                         os.replace(tmp_h264, tmp_pred.name)
#                 with open(tmp_pred.name, "rb") as f:
#                     vdata2 = f.read()
#             st.video(vdata2)
#             st.download_button("📥 Download Prediction Video", data=vdata2,
#                                file_name=f"cloud_{cloud_type}_prediction.mp4",
#                                mime="video/mp4", key="dl_pred")
#             try: os.unlink(tmp_pred.name)
#             except: pass

#         try: os.unlink(tfile.name)
#         except: pass

# # ══════════════════════ MULTI IMAGE TAB ════════════════════════
# with tab2:
#     st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🖼️ Image Analysis</div>', unsafe_allow_html=True)
#     st.subheader("Upload Sky Images")

#     uploaded_images = st.file_uploader("Upload 2 or more images taken at a fixed time interval",
#                                         type=["jpg","jpeg","png"],
#                                         accept_multiple_files=True, key="img_upload")
#     interval   = st.number_input("⏱️ Time Between Images (seconds)", min_value=1, value=60)
#     fov_images = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
#                            help="Phone: 70-80° | Wide angle: 90-120°", key="fov_images")

#     if uploaded_images:
#         st.success(f"{len(uploaded_images)} image(s) uploaded.")

#         with st.expander("🖼️ Preview Uploaded Images"):
#             img_cols = st.columns(min(len(uploaded_images), 4))
#             for i, img_file in enumerate(uploaded_images):
#                 img_file.seek(0)
#                 with img_cols[i % 4]:
#                     st.image(img_file, caption=img_file.name, use_container_width=True)

#         if len(uploaded_images) >= 2:
#             with st.spinner("🔍 Analysing images..."):
#                 dirs_deg, px_disps = [], []
#                 for i in range(len(uploaded_images) - 1):
#                     uploaded_images[i].seek(0); uploaded_images[i+1].seek(0)
#                     img1 = np.array(Image.open(uploaded_images[i]).convert("RGB").resize((640,480)))
#                     img2 = np.array(Image.open(uploaded_images[i+1]).convert("RGB").resize((640,480)))
#                     med, ang = compute_optical_flow(
#                         cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
#                         cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
#                     )
#                     px_disps.append(med); dirs_deg.append(np.degrees(ang))

#                 avg_disp    = float(np.mean(px_disps))
#                 direction   = angle_to_direction(float(np.mean(dirs_deg)))
#                 pil_imgs    = []
#                 for f in uploaded_images:
#                     f.seek(0); pil_imgs.append(Image.open(f))
#                 cloud_type, avg_conf = predict_cloud_type(pil_imgs)

#             fw = 640
#             speed_mps, speed_kmh, deg_per_px, theta_deg, distance_m, height_m = \
#                 pixels_to_kmh(avg_disp, interval, cloud_type, fw, fov_images)
#             pixel_speed = avg_disp / interval
#             dist_5  = speed_kmh * (5  / 60)
#             dist_15 = speed_kmh * (15 / 60)

#             st.divider()
#             st.markdown('<div class="cv-eyebrow">📊 Analysis Results</div>', unsafe_allow_html=True)

#             # Density & Visibility compute from first image
#             uploaded_images[0].seek(0)
#             first_bgr = cv2.resize(
#                 cv2.cvtColor(np.array(Image.open(uploaded_images[0]).convert("RGB")), cv2.COLOR_RGB2BGR),
#                 (640, 480)
#             )
#             sky_h_img = int(first_bgr.shape[0] * 0.78)
#             cov_pct, den_label, den_color = compute_cloud_density(first_bgr, sky_h_img)
#             vis_verdict, vis_reason, vis_color, time_to_exit_min = predict_visibility(
#                 cloud_type, speed_kmh, direction, cov_pct, fov_images)

#             # ── Timestamp resolution for images (manual > EXIF > now) ──
#             _manual_utc_i = st.session_state.get("manual_utc")
#             if _manual_utc_i is not None:
#                 media_ts_i  = _manual_utc_i
#                 ts_source_i = "manual"
#             else:
#                 uploaded_images[0].seek(0)
#                 _pil_first = Image.open(uploaded_images[0])
#                 media_ts_i = extract_exif_datetime(_pil_first)
#                 ts_source_i = "exif" if media_ts_i is not None else "now"
#                 if media_ts_i is None:
#                     media_ts_i = datetime.datetime.utcnow()

#             # ── Image-based elevation estimate from first uploaded image ──
#             img_el_est_i, img_el_conf_i, img_el_note_i = estimate_sun_elevation_from_image(first_bgr)

#             # ── Sun detection from image (if sun is visible in sky) ──
#             sun_x_i, sun_y_i, sun_visible_i = detect_sun_in_frame(first_bgr)
#             sun_az_from_img = None
#             if sun_visible_i:
#                 sun_az_from_img = sun_pixel_to_azimuth(sun_x_i, first_bgr.shape[1], fov_images)
#                 img_el_note_i = (f"☀️ Sun detected in image at pixel ({sun_x_i}, {sun_y_i}). "
#                                  f"Estimated azimuth from camera: {sun_az_from_img:.1f}°")

#             show_metrics(cloud_type, avg_conf, direction, height_m, fov_images,
#                          fw, avg_disp, interval, deg_per_px, theta_deg,
#                          distance_m, speed_mps, speed_kmh, dist_5, dist_15,
#                          coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
#                          vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
#                          time_to_exit_min=time_to_exit_min,
#                          media_timestamp=media_ts_i, timestamp_source=ts_source_i,
#                          image_elevation_est=img_el_est_i, image_elevation_conf=img_el_conf_i,
#                          image_elevation_note=img_el_note_i)

#             st.divider()
#             st.markdown('<div class="cv-eyebrow">🎬 Export</div>', unsafe_allow_html=True)
#             st.subheader("Generate Output")

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.markdown("**📦 Cloud Detection on Images**")
#                 st.caption("Bounding boxes with speed and depth overlay on each uploaded image")
#                 if st.button("Show Detection Boxes", key="img_box"):
#                     cols3 = st.columns(min(len(uploaded_images), 3))
#                     for i, img_file in enumerate(uploaded_images):
#                         img_file.seek(0)
#                         arr = cv2.resize(
#                             cv2.cvtColor(np.array(Image.open(img_file).convert("RGB")),
#                                          cv2.COLOR_RGB2BGR), (640, 480)
#                         )
#                         prev_arr = None
#                         if i > 0:
#                             uploaded_images[i-1].seek(0)
#                             prev_arr_bgr = cv2.resize(
#                                 cv2.cvtColor(np.array(Image.open(uploaded_images[i-1]).convert("RGB")),
#                                              cv2.COLOR_RGB2BGR), (640, 480)
#                             )
#                             prev_arr = cv2.cvtColor(prev_arr_bgr, cv2.COLOR_BGR2GRAY)
#                         arr = draw_boxes_on_frame(arr, speed_kmh, direction, cloud_type,
#                                                    height_m, dist_5, dist_15, i * interval,
#                                                    prev_gray=prev_arr, delta_t=float(interval),
#                                                    fov=fov_images,
#                                                    time_to_exit_min=time_to_exit_min)
#                         with cols3[i % 3]:
#                             st.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),
#                                      caption=f"Image {i+1}", use_container_width=True)

#             with col2:
#                 st.markdown("**🔮 Motion Prediction Video**")
#                 st.caption("Simulated animation — +5 min and +15 min forecast")
#                 if st.button("Generate Prediction Video", key="img_pred"):
#                     with st.spinner("Simulating cloud motion…"):
#                         viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
#                                                     direction=direction, pixel_speed=pixel_speed)
#                         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                         tmp.close()
#                         viz.save_video_with_prediction(tmp.name, prediction_minutes=15)
#                         if shutil.which("ffmpeg"):
#                             tmp_h264 = tmp.name.replace(".mp4", "_h264.mp4")
#                             subprocess.run([
#                                 "ffmpeg", "-y", "-i", tmp.name,
#                                 "-vcodec", "libx264", "-crf", "23",
#                                 "-preset", "fast", "-pix_fmt", "yuv420p",
#                                 "-movflags", "+faststart", tmp_h264
#                             ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                             if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
#                                 os.replace(tmp_h264, tmp.name)
#                         with open(tmp.name, "rb") as f:
#                             vdata = f.read()
#                         st.success("Prediction video ready.")
#                         st.video(vdata)
#                         st.download_button("📥 Download Prediction Video", data=vdata,
#                                            file_name=f"cloud_{cloud_type}_prediction.mp4",
#                                            mime="video/mp4")
#                         try: os.unlink(tmp.name)
#                         except: pass
#         else:
#             st.warning("Upload at least 2 images to run analysis.")







# code imp





# import streamlit as st
# import numpy as np
# import cv2
# import tempfile
# import os
# import math
# import subprocess
# import shutil
# import json
# import datetime
# from collections import Counter
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from motion_visualizer import CloudMotionVisualizer

# # ===== SOLAR SHADOW & SUN TRACKING =====
# # pip install pvlib pandas
# try:
#     import pvlib
#     import pandas as pd
# except ImportError:
#     pvlib = None
#     pd = None

# def get_solar_position(lat, lon, timestamp):
#     if pvlib is None:
#         return None, None
#     times = pd.DatetimeIndex([timestamp])
#     sol = pvlib.solarposition.get_solarposition(times, lat, lon)
#     return float(sol["azimuth"].iloc[0]), float(sol["elevation"].iloc[0])


# def detect_clouds_binary(frame_bgr, threshold=150):
#     """
#     STEP 2: Cloud Detection (FindClouds Module)
    
#     Convert image to grayscale and create binary cloud mask.
#     Pixels > threshold = cloud (white), pixels < threshold = sky (black)
    
#     Returns: binary mask (0-255)
#     """
#     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
#     return binary_mask


# def mirror_cloud_mask(mask):
#     """
#     STEP 3: Mirror Images (Flip Vertically)
    
#     Flip the cloud mask vertically for overlay alignment.
#     Ground → Clouds becomes Clouds → Ground
    
#     Returns: mirrored mask
#     """
#     return cv2.flip(mask, 0)  # 0 = flip vertically


# def calculate_cbh_by_overlay(mask1_mirrored, mask2_original, baseline_km=2.0, 
#                              fov_deg=70, frame_width=640, height_range=(0, 12000, 100)):
#     """
#     STEP 4: Cloud Base Height (CBH) Calculation by Overlay
    
#     Loop through different heights and find the height at which 
#     the two cloud masks align perfectly (lowest matching error).
    
#     Args:
#         mask1_mirrored: Camera 1 cloud mask (flipped vertically)
#         mask2_original: Camera 2 cloud mask (original)
#         baseline_km: Distance between cameras in km
#         fov_deg: Camera field of view in degrees
#         frame_width: Image width in pixels
#         height_range: (start_m, end_m, step_m) tuple for height search
    
#     Returns:
#         best_height_m: CBH in meters
#         error_by_height: dict of {height: error}
#     """
#     start_h, end_h, step_h = height_range
#     error_by_height = {}
    
#     baseline_px = (baseline_km * 1000) * (frame_width / (2 * fov_deg * 111.32))  # pixels
    
#     for height_m in range(start_h, end_h, step_h):
#         if height_m == 0:
#             continue  # Skip 0 height
        
#         # Parallax shift in pixels = baseline / height * focal_length_px
#         focal_length_px = (frame_width * fov_deg) / (2 * np.tan(np.radians(fov_deg/2)))
#         shift_px = int((baseline_px * focal_length_px) / height_m)
        
#         # Shift Camera 1 mask
#         if shift_px > 0:
#             mask1_shifted = np.roll(mask1_mirrored, shift_px, axis=1)
#         else:
#             mask1_shifted = mask1_mirrored
        
#         # Calculate matching error (normalized cross-correlation)
#         mask1_shifted_f = mask1_shifted.astype(np.float32) / 255.0
#         mask2_f = mask2_original.astype(np.float32) / 255.0
        
#         # Error = 1 - correlation
#         correlation = np.corrcoef(mask1_shifted_f.flatten(), mask2_f.flatten())[0, 1]
#         error = 1.0 - (correlation if not np.isnan(correlation) else 0.0)
        
#         error_by_height[height_m] = error
    
#     # Find height with lowest error
#     best_height_m = min(error_by_height, key=error_by_height.get)
    
#     return best_height_m, error_by_height


# def calculate_cloud_speed_optical_flow(frame_t0_gray, frame_t1_gray, 
#                                       cbh_m, fov_deg=70, frame_width=640, 
#                                       time_delta_sec=60):
#     """
#     STEP 5: Calculate Cloud Speed using Optical Flow
    
#     Track cloud movement between two frames and convert pixel motion to km/h.
    
#     Args:
#         frame_t0_gray: Grayscale frame at time T0
#         frame_t1_gray: Grayscale frame at time T1
#         cbh_m: Cloud base height in meters (from CBH calculation)
#         fov_deg: Camera FOV in degrees
#         frame_width: Image width in pixels
#         time_delta_sec: Time between T0 and T1 in seconds
    
#     Returns:
#         speed_kmh: Speed in km/h
#         speed_mps: Speed in m/s
#         displacement_px: Pixel displacement
#         direction_deg: Direction angle in degrees
#     """
#     # Compute optical flow
#     displacement_px, angle_rad = compute_optical_flow(frame_t0_gray, frame_t1_gray)
    
#     # Convert pixel displacement to ground distance
#     # Ground resolution = cbh_m * tan(fov_deg/2) / (frame_width/2)
#     ground_resolution_m_per_px = (cbh_m * np.tan(np.radians(fov_deg/2))) / (frame_width/2)
    
#     # Distance traveled in meters
#     distance_m = displacement_px * ground_resolution_m_per_px
    
#     # Speed in m/s and km/h
#     speed_mps = distance_m / time_delta_sec
#     speed_kmh = speed_mps * 3.6
    
#     direction_deg = np.degrees(angle_rad)
    
#     return speed_kmh, speed_mps, displacement_px, direction_deg


# def forecast_solar_impact(cbh_m, speed_kmh, direction_deg, distance_to_plant_m, 
#                          solar_plant_location="origin"):
#     """
#     STEP 6: Calculate Direction + Forecast Solar Impact
    
#     Predict if/when cloud will reach solar plant and cause shadow.
    
#     Args:
#         cbh_m: Cloud base height in meters
#         speed_kmh: Cloud speed in km/h
#         direction_deg: Cloud direction in degrees (0=North, 90=East, etc)
#         distance_to_plant_m: Distance from cameras to plant in meters
#         solar_plant_location: Plant location reference
    
#     Returns:
#         will_impact: Boolean if cloud will hit plant
#         time_to_impact_min: Minutes until cloud reaches plant
#         forecast_30min: String describing 30-min forecast
#     """
#     if speed_kmh < 0.1:
#         return False, float('inf'), "Cloud stationary - no immediate impact"
    
#     # Speed in m/min
#     speed_m_per_min = speed_kmh * 1000 / 60
    
#     # Time to reach plant
#     time_to_plant_min = distance_to_plant_m / speed_m_per_min
    
#     # Determine if cloud is moving toward plant (rough check)
#     # This would need proper bearing calculation in production
#     will_impact = time_to_plant_min < 60  # Within 60 minutes
    
#     if time_to_plant_min < 5:
#         forecast = f"⚠️ CRITICAL: Cloud reaches plant in {time_to_plant_min:.1f} min - SOLAR SHADOW LIKELY"
#     elif time_to_plant_min < 15:
#         forecast = f"🟡 WARNING: Cloud reaches plant in {time_to_plant_min:.1f} min - Prepare for shade"
#     elif time_to_plant_min < 30:
#         forecast = f"🟢 CAUTION: Cloud reaches plant in {time_to_plant_min:.1f} min - Monitor closely"
#     else:
#         forecast = f"🟢 OK: Cloud reaches plant in {time_to_plant_min:.1f} min - No immediate concern"
    
#     return will_impact, time_to_plant_min, forecast


# def generate_cbh_visualization(error_by_height):
#     """
#     Generate visualization of CBH calculation showing error at each height.
#     Returns a chart-ready dictionary.
#     """
#     heights = sorted(error_by_height.keys())
#     errors = [error_by_height[h] for h in heights]
    
#     return {"Height (m)": heights, "Matching Error": errors}



#     """Extract capture datetime from EXIF. Returns datetime or None."""
#     try:
#         exif_data = pil_image._getexif()
#         if exif_data is None:
#             return None
#         for tag_id in (36867, 36868, 306):
#             if tag_id in exif_data:
#                 return datetime.datetime.strptime(exif_data[tag_id], "%Y:%m:%d %H:%M:%S")
#     except Exception:
#         pass
#     return None


# def extract_video_datetime(video_path):
#     """Extract creation_time from MP4/MOV via ffprobe. Returns datetime or None."""
#     if not shutil.which("ffprobe"):
#         return None
#     try:
#         result = subprocess.run(
#             ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
#             capture_output=True, text=True, timeout=10
#         )
#         meta = json.loads(result.stdout)
#         tags = meta.get("format", {}).get("tags", {})
#         for key in ("creation_time", "com.apple.quicktime.creationdate"):
#             val = tags.get(key)
#             if val:
#                 val = val.rstrip("Z").split(".")[0]
#                 return datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
#     except Exception:
#         pass
#     return None


# def estimate_sun_elevation_from_image(frame_bgr):
#     """
#     Estimate sun elevation from sky brightness when sun is NOT visible in frame.

#     Uses 4 monocular cues:
#       1. Overall sky brightness  → proxy for solar irradiance
#       2. Horizon glow ratio      → low ratio = high sun, high ratio = low sun
#       3. Blue channel dominance  → clear vs overcast sky
#       4. Saturation              → confidence modifier

#     Returns: (elevation_deg, confidence, method_note)
#     """
#     sky_h = int(frame_bgr.shape[0] * 0.78)
#     sky   = frame_bgr[:sky_h, :]
#     H, W  = sky.shape[:2]

#     sky_f  = sky.astype(np.float32)
#     hsv    = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV).astype(np.float32)
#     bright = hsv[:, :, 2] / 255.0

#     mean_bright  = float(np.mean(bright))
#     horizon_mean = float(np.mean(bright[int(H * 0.80):, :]))
#     top_mean     = float(np.mean(bright[:int(H * 0.20), :]))
#     horizon_ratio = horizon_mean / (top_mean + 1e-6)

#     b_ch = sky_f[:, :, 0] / 255.0
#     r_ch = sky_f[:, :, 2] / 255.0
#     blue_ratio = float(np.mean(b_ch)) / (float(np.mean(r_ch)) + 1e-6)
#     mean_sat   = float(np.mean(hsv[:, :, 1] / 255.0))

#     if mean_bright < 0.15:
#         el_base = 2.0
#         note = "Very dark sky — sun likely below horizon or nighttime"
#         conf = "low"
#     elif mean_bright < 0.30:
#         if horizon_ratio > 1.3:
#             el_base = 8.0 + (horizon_ratio - 1.3) * 10
#             note = "Horizon glow detected — estimated sunrise/sunset angle"
#             conf = "medium"
#         else:
#             el_base = 15.0 + mean_bright * 40
#             note = "Dim sky — low sun elevation estimated from brightness"
#             conf = "low"
#     elif mean_bright < 0.55:
#         el_base = 25.0 + (mean_bright - 0.30) / 0.25 * 30
#         if horizon_ratio > 1.15:
#             el_base -= 10
#         note = "Moderate brightness — mid-range elevation estimated"
#         conf = "medium"
#     else:
#         el_base = 55.0 + (mean_bright - 0.55) / 0.45 * 25
#         note = "Bright sky — high elevation estimated (near noon)"
#         conf = "medium" if blue_ratio > 1.1 else "low"

#     if mean_sat < 0.10 and blue_ratio < 1.05:
#         conf = "low"
#         note += " (overcast — estimate less reliable)"

#     return round(float(np.clip(el_base, 0.0, 85.0)), 1), conf, note


# def detect_sun_in_frame(frame_bgr):
#     """
#     Detect the sun by finding the brightest spot in the sky region.

#     Looks at the top 78% of the frame (sky), finds the pixel with maximum
#     brightness. If that brightness exceeds 240 → sun is visible.

#     Args:
#         frame_bgr: BGR image (numpy array from cv2).

#     Returns:
#         (sun_x, sun_y, True)    — pixel position of sun if detected
#         (None,  None,  False)   — if sun is not visible
#     """
#     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     sky_h = int(frame_bgr.shape[0] * 0.78)
#     sky = gray[:sky_h, :]

#     # Sabse bright pixel dhundho
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sky)

#     # Agar brightness bahut high hai = sun visible
#     if max_val > 240:
#         sun_x, sun_y = max_loc
#         return sun_x, sun_y, True   # pixel position
#     return None, None, False


# def sun_pixel_to_azimuth(sun_x, frame_width, fov_deg, device_heading=0):
#     """
#     Convert sun's pixel X position to an estimated azimuth (compass bearing).

#     Calculates the angular offset of the sun from the frame center using
#     the camera's field of view, then adds the device heading to get true azimuth.

#     Args:
#         sun_x:          Horizontal pixel position of the sun in the frame.
#         frame_width:    Total width of the frame in pixels.
#         fov_deg:        Camera horizontal field of view in degrees.
#         device_heading: Compass heading the camera is pointing (0=N, 90=E, etc.).

#     Returns:
#         Estimated sun azimuth in degrees [0, 360).
#     """
#     # Center se kitna door hai sun
#     offset_px = sun_x - frame_width / 2
#     offset_deg = offset_px * (fov_deg / frame_width)
#     azimuth = (device_heading + offset_deg) % 360
#     return azimuth


# def sun_azimuth_to_direction(azimuth_deg):
#     """Convert sun azimuth (0=N, 90=E, 180=S, 270=W) to compass label."""
#     if azimuth_deg is None:
#         return "Unknown"
#     a = azimuth_deg % 360
#     if   a < 22.5 or a >= 337.5: return "North"
#     elif a < 67.5:  return "NE"
#     elif a < 112.5: return "East"
#     elif a < 157.5: return "SE"
#     elif a < 202.5: return "South"
#     elif a < 247.5: return "SW"
#     elif a < 292.5: return "West"
#     else:           return "NW"



# def get_cloud_sun_alignment(cloud_direction, sun_azimuth_deg, flow_angle_deg=None):
#     if sun_azimuth_deg is None:
#         return "unknown", None, "Sun position unavailable (set location or timestamp)."

#     if flow_angle_deg is not None:
#         cloud_az = flow_angle_deg % 360
#         diff = abs((cloud_az - sun_azimuth_deg + 180) % 360 - 180)
#     else:
#         dir_to_az = {"North": 0, "NE": 45, "East": 90, "SE": 135, "South": 180, "SW": 225, "West": 270, "NW": 315}
#         cloud_az = dir_to_az.get(cloud_direction, 0)
#         diff = abs((cloud_az - sun_azimuth_deg + 180) % 360 - 180)

#     if diff < 20:
#         return "toward_sun", diff, f"Cloud motion is nearly aligned with the sun ({diff:.0f}° offset). Shadow risk is high."
#     elif diff < 60:
#         return "glancing", diff, f"Cloud motion is partly aligned with sun ({diff:.0f}° offset). Shadow may partially affect the panel."
#     elif diff < 120:
#         return "crossing", diff, f"Cloud path is crossing the sun direction ({diff:.0f}° offset). Shadow may be brief."
#     else:
#         return "away_from_sun", diff, f"Cloud is moving away from sun direction ({diff:.0f}° offset). Shadow risk is lower."

# # ═══════════════════════════════════════════════════════════════



# # ─────────────────────────── CONFIG ────────────────────────────
# st.set_page_config(page_title="CloudVision AI", page_icon="☁️", layout="wide")

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

# /* ── Base ── */
# html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

# .stApp {
#     background: #070d14;
# }

# /* ── Hide default streamlit chrome ── */
# #MainMenu, footer { visibility: hidden; }
# .block-container { padding-top: 1.5rem !important; max-width: 1280px; }

# /* ── Header ── */
# .cv-header {
#     display: flex; align-items: center; gap: 16px;
#     padding: 28px 0 8px 0; border-bottom: 1px solid #1a2d44;
#     margin-bottom: 24px;
# }
# .cv-logo {
#     width: 48px; height: 48px; border-radius: 12px;
#     background: linear-gradient(135deg, #0ea5e9, #6366f1);
#     display: flex; align-items: center; justify-content: center;
#     font-size: 24px; flex-shrink: 0;
#     box-shadow: 0 0 24px rgba(14,165,233,0.35);
# }
# .cv-title { font-size: 1.75rem; font-weight: 700; color: #f0f6ff; letter-spacing: -0.02em; }
# .cv-sub   { font-size: 0.82rem; color: #4a6580; font-family: 'JetBrains Mono', monospace;
#             text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

# /* ── Tabs ── */
# .stTabs [data-baseweb="tab-list"] {
#     background: transparent;
#     border-bottom: 1px solid #1a2d44;
#     gap: 4px;
# }
# .stTabs [data-baseweb="tab"] {
#     background: transparent;
#     border: 1px solid transparent;
#     border-radius: 8px 8px 0 0;
#     color: #4a6580;
#     padding: 10px 22px;
#     font-weight: 500;
#     font-size: 0.88rem;
#     transition: all 0.15s;
# }
# .stTabs [data-baseweb="tab"]:hover { color: #94b8d4; background: #0d1a27; }
# .stTabs [aria-selected="true"] {
#     background: #0d1a27 !important;
#     color: #38bdf8 !important;
#     border-color: #1a2d44 #1a2d44 transparent !important;
# }
# .stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }

# /* ── Metric cards ── */
# [data-testid="metric-container"] {
#     background: #0d1a27;
#     border: 1px solid #1a2d44;
#     border-radius: 12px;
#     padding: 18px 20px !important;
#     transition: border-color 0.2s;
# }
# [data-testid="metric-container"]:hover { border-color: #2a4a64; }
# [data-testid="stMetricLabel"] {
#     font-size: 0.75rem !important;
#     color: #4a6580 !important;
#     text-transform: uppercase;
#     letter-spacing: 0.08em;
#     font-family: 'JetBrains Mono', monospace;
# }
# [data-testid="stMetricValue"] {
#     font-size: 1.35rem !important;
#     font-weight: 600 !important;
#     color: #e2ecf6 !important;
# }
# [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

# /* ── Buttons ── */
# .stButton > button {
#     background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
#     color: #fff;
#     border: none;
#     border-radius: 8px;
#     padding: 10px 22px;
#     font-weight: 600;
#     font-size: 0.875rem;
#     letter-spacing: 0.01em;
#     transition: opacity 0.15s, transform 0.1s;
#     width: 100%;
# }
# .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
# .stButton > button:active { transform: translateY(0); }

# /* ── Upload area ── */
# [data-testid="stFileUploader"] {
#     background: #0d1a27;
#     border: 1.5px dashed #1e3650;
#     border-radius: 12px;
#     padding: 12px;
#     transition: border-color 0.2s;
# }
# [data-testid="stFileUploader"]:hover { border-color: #0ea5e9; }

# /* ── Sliders ── */
# [data-testid="stSlider"] > div > div > div > div {
#     background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
# }

# /* ── Expander ── */
# [data-testid="stExpander"] {
#     background: #0d1a27;
#     border: 1px solid #1a2d44;
#     border-radius: 10px;
# }
# [data-testid="stExpander"] summary {
#     color: #94b8d4 !important;
#     font-size: 0.85rem;
#     font-weight: 500;
# }

# /* ── Spinner ── */
# [data-testid="stSpinner"] { color: #38bdf8 !important; }

# /* ── Divider ── */
# hr { border-color: #1a2d44 !important; margin: 20px 0 !important; }

# /* ── Subheader ── */
# h2, h3 { color: #c8dff0 !important; font-weight: 600 !important; }

# /* ── Number input / selectbox ── */
# [data-baseweb="input"], [data-baseweb="select"] {
#     background: #0d1a27 !important;
#     border-color: #1a2d44 !important;
#     border-radius: 8px !important;
#     color: #e2ecf6 !important;
# }

# /* ── Markdown tables ── */
# table { width: 100%; border-collapse: collapse; }
# th { background: #0d1a27; color: #4a6580; font-family: 'JetBrains Mono', monospace;
#      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
#      padding: 10px 14px; border-bottom: 1px solid #1a2d44; }
# td { color: #c8dff0; padding: 9px 14px; font-family: 'JetBrains Mono', monospace;
#      font-size: 0.82rem; border-bottom: 1px solid #0f1e2d; }
# tr:last-child td { border-bottom: none; }

# /* ── Success / Warning / Info ── */
# [data-testid="stAlert"] { border-radius: 10px !important; border-width: 1px !important; }

# /* ── Video ── */
# video { border-radius: 10px; border: 1px solid #1a2d44; }

# /* ── Caption ── */
# .stCaption { color: #4a6580 !important; font-size: 0.78rem !important; }

# /* ── Stat pill used in custom cards ── */
# .cv-pill {
#     display: inline-block;
#     padding: 3px 10px;
#     border-radius: 999px;
#     font-size: 0.72rem;
#     font-family: 'JetBrains Mono', monospace;
#     font-weight: 500;
#     letter-spacing: 0.04em;
#     background: #0a1929;
#     border: 1px solid #1a2d44;
#     color: #94b8d4;
#     margin-right: 4px;
# }

# /* ── Section label eyebrow ── */
# .cv-eyebrow {
#     font-size: 0.7rem;
#     font-family: 'JetBrains Mono', monospace;
#     text-transform: uppercase;
#     letter-spacing: 0.12em;
#     color: #4a6580;
#     margin-bottom: 10px;
# }

# /* ── Download button ── */
# [data-testid="stDownloadButton"] > button {
#     background: #0d1a27 !important;
#     border: 1px solid #1a2d44 !important;
#     color: #38bdf8 !important;
#     font-weight: 500 !important;
# }
# [data-testid="stDownloadButton"] > button:hover {
#     border-color: #38bdf8 !important;
#     background: #0a2035 !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # ─────────────────────────── MODEL ─────────────────────────────
# @st.cache_resource
# def load_cloud_model():
#     return load_model("cloud_model.keras")

# model = load_cloud_model()

# class_names  = ["Cumulus","Altocumulus","Cirrus","ClearSky","Stratocumulus","Cumulonimbus","Mixed"]
# cloud_height = {"Cumulus":1500,"Altocumulus":4500,"Cirrus":9000,
#                 "ClearSky":0,"Stratocumulus":1200,"Cumulonimbus":6000,"Mixed":3500}
# cloud_emoji  = {"Cumulus":"⛅","Altocumulus":"🌤️","Cirrus":"🌬️",
#                 "ClearSky":"☀️","Stratocumulus":"🌥️","Cumulonimbus":"⛈️","Mixed":"🌦️"}

# # ─────────────────────────── HELPERS ───────────────────────────
# def angle_to_direction(angle_deg):
#     a = angle_deg % 360
#     if 22.5 <= a < 67.5:
#         return "NE"
#     elif 67.5 <= a < 112.5:
#         return "East"
#     elif 112.5 <= a < 157.5:
#         return "SE"
#     elif 157.5 <= a < 202.5:
#         return "South"
#     elif 202.5 <= a < 247.5:
#         return "SW"
#     elif 247.5 <= a < 292.5:
#         return "West"
#     elif 292.5 <= a < 337.5:
#         return "NW"
#     return "North"

# def compute_optical_flow(gray1, gray2):
#     flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
#     return float(np.median(mag)), float(np.median(ang))

# def pixels_to_kmh(pixel_displacement, delta_t_sec, cloud_type, frame_width, fov):
#     height_m = float(cloud_height.get(cloud_type, 2000))
#     degree_per_px = fov / max(frame_width, 1)
#     theta_deg = abs(pixel_displacement) * degree_per_px
#     theta_rad = math.radians(theta_deg)
#     distance_m = height_m * math.tan(theta_rad)
#     speed_mps = distance_m / max(delta_t_sec, 1e-6)
#     speed_kmh = speed_mps * 3.6
#     return speed_mps, speed_kmh, degree_per_px, theta_deg, distance_m, height_m

# def compute_cloud_density(frame, sky_h):
#     """
#     Calculates cloud coverage percentage within the sky region.
#     Returns: coverage_percent (0-100), density_label, density_color
#     """
#     gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     sky_gray = gray[:sky_h, :]
#     sky_hsv  = hsv[:sky_h, :]

#     _, bright_mask = cv2.threshold(sky_gray, 140, 255, cv2.THRESH_BINARY)
#     sat = sky_hsv[:, :, 1]
#     _, sat_mask = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)
#     hue = sky_hsv[:, :, 0]
#     blue_sky = cv2.inRange(hue, 95, 135)
#     not_blue = cv2.bitwise_not(blue_sky)
#     combined = cv2.bitwise_or(bright_mask, cv2.bitwise_and(sat_mask, not_blue))

#     total_pixels = sky_gray.shape[0] * sky_gray.shape[1]
#     cloud_pixels = int(cv2.countNonZero(combined))
#     coverage = min(100.0, (cloud_pixels / total_pixels) * 100)

#     if coverage < 20:
#         label, color = "Low ☀️", "#22c55e"
#     elif coverage < 55:
#         label, color = "Medium 🌤️", "#f59e0b"
#     else:
#         label, color = "High ⛅", "#ef4444"

#     return round(coverage, 1), label, color


# def predict_visibility(cloud_type, speed_kmh, direction, coverage_pct, fov_deg):
#     """
#     Estimates how long the cloud will remain visible in the sky (until dissipation).
#     Estimate is based on cloud type's atmospheric lifetime, speed, and coverage.
#     Returns: verdict, reason, color, lifetime_min
#     """

#     # ── Typical atmospheric lifetimes by cloud type (minutes) ──
#     # Based on meteorological averages:
#     # Cumulus: 10–60 min (convective, quickly form/dissipate)
#     # Altocumulus: 30–120 min (mid-level, moderate lifetime)
#     # Cirrus: 60–360 min (high-altitude ice, very long lasting)
#     # ClearSky: no cloud
#     # Stratocumulus: 60–480 min (layer cloud, very persistent)
#     # Cumulonimbus: 30–90 min (active storm cell, intense but burns out)
#     # Mixed: 20–90 min (varied)
#     cloud_lifetime_range = {
#         "Cumulus": (10, 60),
#         "Altocumulus": (30, 120),
#         "Cirrus": (60, 360),
#         "ClearSky": (0, 0),
#         "Stratocumulus": (60, 480),
#         "Cumulonimbus": (30, 90),
#         "Mixed": (20, 90),
#     }

#     if cloud_type == "ClearSky":
#         return (
#             "☀️ Clear Sky — No Cloud",
#             "Sky is currently clear. No clouds are visible.",
#             "#22c55e",
#             999,
#         )

#     if speed_kmh < 0.5:
#         lo, hi = cloud_lifetime_range.get(cloud_type, (20, 90))
#         mid = (lo + hi) // 2
#         return (
#             f"🟡 Stationary — ~{mid} min remaining",
#             (f"{cloud_type} clouds typically last {lo}–{hi} min. "
#              f"This cloud is currently stationary and will remain visible for approximately {mid} more minutes."),
#             "#f59e0b",
#             float(mid),
#         )

#     lo, hi = cloud_lifetime_range.get(cloud_type, (20, 90))

#     # High coverage = more moisture/mass = longer lifetime
#     coverage_bonus = (coverage_pct / 100.0) * (hi - lo) * 0.3

#     # Fast-moving clouds dissipate faster (turbulence, mixing)
#     # Normalize: >60 km/h = fast, cuts lifetime by up to 30%
#     speed_factor = max(0.7, 1.0 - (speed_kmh / 200.0))

#     lifetime_min = ((lo + hi) / 2 + coverage_bonus) * speed_factor
#     lifetime_min = round(max(lo, min(hi, lifetime_min)))

#     # ── Verdict tiers ──
#     if lifetime_min > 60:
#         hrs = lifetime_min // 60
#         mins = lifetime_min % 60
#         time_str = f"~{hrs}h {mins}min" if mins else f"~{hrs}h"
#         verdict = f"🟢 Will Stay — {time_str} remaining"
#         reason  = (f"{cloud_type} clouds are long-lasting (typically {lo}–{hi} min). "
#                    f"With {coverage_pct}% coverage and a speed of {speed_kmh:.1f} km/h, "
#                    f"this cloud is expected to remain visible for approximately {time_str}.")
#         color   = "#22c55e"
#     elif lifetime_min > 20:
#         verdict = f"🟡 Moderate — ~{lifetime_min} min remaining"
#         reason  = (f"{cloud_type} clouds typically last {lo}–{hi} min. "
#                    f"Based on the current speed ({speed_kmh:.1f} km/h) and coverage ({coverage_pct}%), "
#                    f"this cloud is expected to remain visible for approximately {lifetime_min} more minutes.")
#         color   = "#f59e0b"
#     else:
#         verdict = f"🔴 Dissipating — ~{lifetime_min} min remaining"
#         reason  = (f"{cloud_type} clouds dissipate quickly (lifetime {lo}–{hi} min). "
#                    f"The high speed ({speed_kmh:.1f} km/h) and low coverage ({coverage_pct}%) suggest "
#                    f"this cloud will disappear from the sky in approximately {lifetime_min} minutes.")
#         color   = "#ef4444"

#     return verdict, reason, color, float(lifetime_min)
# def classify_cloud_scenario(cloud_type, coverage_pct, speed_kmh, power_drop_pct):
#     if cloud_type == "ClearSky" or coverage_pct < 10:
#         return "Clear Sky", "Highest and most stable generation."
#     if cloud_type == "Cumulonimbus" or coverage_pct >= 80:
#         return "Storm Clouds", "Very low output with high ramp risk."
#     if power_drop_pct >= 60 or coverage_pct >= 65:
#         return "Dense Overcast", "Low but steady output with strong irradiance suppression."
#     if speed_kmh >= 25:
#         return "Fast-Moving Clouds", "Rapid ramps and high short-term forecast uncertainty."
#     if power_drop_pct >= 25 or coverage_pct >= 25:
#         return "Partial Cloud Cover", "Variable output with drops, recoveries, and possible edge enhancement."
#     return "Mixed Cloud Conditions", "Moderate variability with short-lived changes in output."

# def estimate_ramp_risk(speed_kmh, coverage_pct, cloud_type):
#     score = 0
#     if speed_kmh >= 30:
#         score += 3
#     elif speed_kmh >= 15:
#         score += 2
#     elif speed_kmh >= 5:
#         score += 1
#     if coverage_pct >= 70:
#         score += 3
#     elif coverage_pct >= 40:
#         score += 2
#     elif coverage_pct >= 15:
#         score += 1
#     if cloud_type in ["Cumulonimbus", "Cumulus", "Stratocumulus"]:
#         score += 1
#     if score >= 6:
#         return "High", score
#     if score >= 3:
#         return "Medium", score
#     return "Low", score

# def cloud_enhancement_flag(coverage_pct, cloud_type, sun_visible=False):
#     if cloud_type in ["Cirrus", "Cumulus", "Altocumulus"] and 5 <= coverage_pct <= 45:
#         return True, "Cloud-edge enhancement possible; brief power spike may exceed clear-sky expectation."
#     if sun_visible and coverage_pct <= 30:
#         return True, "Sun-visible partial cloud scene; short enhancement spike is possible near cloud edges."
#     return False, "No strong cloud-enhancement signal."

# def build_scenario_text(scenario, impact_text, ramp_level, enh_text, cloud_sun_text):
#     return f"Scenario: {scenario}. {impact_text} Ramp risk: {ramp_level}. {enh_text} {cloud_sun_text}"

# def compute_weather_context(weather_json=None):
#     if not isinstance(weather_json, dict):
#         return {"temp_c": None, "humidity": None, "wind_ms": None, "pressure_hpa": None, "cloud_cover": None, "rain_mm": None}
#     return {
#         "temp_c": weather_json.get("temp_c"),
#         "humidity": weather_json.get("humidity"),
#         "wind_ms": weather_json.get("wind_ms"),
#         "pressure_hpa": weather_json.get("pressure_hpa"),
#         "cloud_cover": weather_json.get("cloud_cover"),
#         "rain_mm": weather_json.get("rain_mm"),
#     }

# def predict_cloud_type(frames_or_images):
#     """Batch-predict cloud type from a list of frames or PIL images."""
#     batch = []
#     for img in frames_or_images:
#         img_pil = Image.fromarray(img).resize((224, 224)) if isinstance(img, np.ndarray) else img.resize((224, 224))
#         batch.append(image.img_to_array(img_pil) / 255.0)

#     batch_arr = np.stack(batch, axis=0)          # shape: (N, 224, 224, 3)
#     preds_all = model.predict(batch_arr, verbose=0)  # single forward pass

#     preds = [class_names[np.argmax(p)] for p in preds_all]
#     confs = [float(np.max(p)) * 100 for p in preds_all]
#     return Counter(preds).most_common(1)[0][0], float(np.mean(confs))

# def compute_solar_shadow_forecast(cloud_type, height_m, speed_mps, speed_kmh,
#                                    direction, pixel_disp, frame_width, fov_deg,
#                                    coverage_pct):
#     """
#     Camera is mounted on or near the solar plant.
#     When a cloud enters the field of view, its shadow falls somewhere on the ground.

#     Physics:
#       - Cloud's current angular position from frame center → ground_offset
#         ground_offset = height_m * tan(angle_from_zenith)
#       - Shadow ground offset = cloud's horizontal distance from directly overhead
#       - If cloud is moving toward the camera center → shadow WILL hit the panel
#       - Time to shadow = ground_offset / speed_mps
#       - If cloud is moving away from center → shadow WON'T hit the panel

#     Returns dict with all forecast info.
#     """
#     cloud_power_factor = {
#         "Cumulus": 0.55, "Altocumulus": 0.45, "Cirrus": 0.18,
#         "ClearSky": 0.0, "Stratocumulus": 0.72, "Cumulonimbus": 0.85, "Mixed": 0.50
#     }

#     if cloud_type == "ClearSky":
#         return {
#             "will_hit": False,
#             "reason": "☀️ Sky is completely clear — no clouds, no shadow.",
#             "status": "clear",
#             "shadow_time_min": None,
#             "power_drop_pct": 0.0,
#             "ground_offset_m": 0.0,
#         }

#     # Angular position of cloud from frame center (pixels → degrees → radians)
#     deg_per_px     = fov_deg / frame_width
#     # pixel_disp is displacement magnitude; we use half-frame as rough center-offset
#     # Center offset: assume cloud centroid is ~frame center (worst case / mean case)
#     # More accurate: use half FOV = cloud is within FOV, so it's within ±fov/2 of zenith
#     # We use the actual angle the cloud has already traveled (theta) as proxy for offset
#     half_fov_deg   = fov_deg / 2.0
#     # Angle from zenith to cloud edge (conservative: use half-FOV)
#     angle_rad      = math.radians(half_fov_deg * 0.5)   # ~centre of visible sky arc
#     ground_offset_m = height_m * math.tan(angle_rad)    # metres from panel

#     # Direction logic: is cloud moving TOWARD overhead (will shadow hit) or AWAY?
#     # "toward overhead" = cloud is off to one side and moving toward center
#     # Simplified: if cloud is IN the FOV it is either overhead now or will cross overhead
#     # based on direction + frame position.
#     # We check: does the cloud's travel path cross the zenith column?
#     # Heuristic: if cloud is moving and it's within FOV → it will pass overhead → shadow hits
#     # UNLESS cloud is already past center and moving further away.

#     # We estimate current cloud X position from optical flow direction
#     # North/South movement means cloud crosses overhead (shadow hits)
#     # East/West also crosses — it just depends on whether it already passed
#     # Use pixel_disp relative to frame to estimate if approaching or receding
    
#     # Simple model: cloud in FOV = overhead within half-FOV → shadow WILL hit
#     # Time for shadow to reach panel = ground_offset / speed_mps
    
#     if speed_mps < 0.05:
#         # Nearly stationary
#         return {
#             "will_hit": True,
#             "reason": f"Cloud is still in the sky and is practically stationary. Shadow may already be present on the solar panel or has stalled overhead.",
#             "status": "stationary",
#             "shadow_time_min": 0.0,
#             "power_drop_pct": _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor),
#             "ground_offset_m": ground_offset_m,
#         }

#     time_to_shadow_sec = ground_offset_m / speed_mps
#     time_to_shadow_min = time_to_shadow_sec / 60.0

#     # If time is very small (< 0.5 min) → shadow is basically overhead now
#     # If cloud is moving away (already past zenith), offset increases over time → no hit
#     # Proxy: use pixel displacement direction vs frame center
#     # If the optical flow vector points toward frame center → cloud approaching
#     # Rough approximation: if time_to_shadow_min < 0 conceptually, cloud already passed
#     # We use: if time_to_shadow_min > time_to_exit_min-equivalent → won't cross
    
#     # Practical cutoff: if it takes longer than cloud lifetime to arrive → won't hit
#     cloud_lifetime = {
#         "Cumulus": 35, "Altocumulus": 75, "Cirrus": 200,
#         "ClearSky": 0, "Stratocumulus": 270, "Cumulonimbus": 60, "Mixed": 55
#     }
#     lifetime_min = cloud_lifetime.get(cloud_type, 60)

#     power_drop = _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor)

#     if time_to_shadow_min <= 0.3:
#         return {
#             "will_hit": True,
#             "reason": f"Cloud is almost directly overhead — shadow is currently falling on the solar panel or is about to.",
#             "status": "now",
#             "shadow_time_min": 0.0,
#             "power_drop_pct": power_drop,
#             "ground_offset_m": ground_offset_m,
#         }
#     elif time_to_shadow_min <= lifetime_min:
#         return {
#             "will_hit": True,
#             "reason": f"{cloud_type} cloud is offset by {ground_offset_m/1000:.2f} km. At {speed_kmh:.1f} km/h, the shadow will reach the solar panel in {time_to_shadow_min:.1f} minutes.",
#             "status": "incoming",
#             "shadow_time_min": time_to_shadow_min,
#             "power_drop_pct": power_drop,
#             "ground_offset_m": ground_offset_m,
#         }
#     else:
#         return {
#             "will_hit": False,
#             "reason": (f"The estimated shadow arrival time for this {cloud_type} cloud is ~{time_to_shadow_min:.0f} min, "
#                        f"but the cloud's expected lifetime is only ~{lifetime_min} min. "
#                        f"The cloud will dissipate or exit the field of view before the shadow reaches the panel."),
#             "status": "miss",
#             "shadow_time_min": time_to_shadow_min,
#             "power_drop_pct": 0.0,
#             "ground_offset_m": ground_offset_m,
#         }


# def _calc_power_drop(cloud_type, coverage_pct, factor_map):
#     # Updated power-drop heuristic with optional confidence weighting
#     base = float(factor_map.get(cloud_type, 0.50))
#     cov = max(0.0, min(1.0, (coverage_pct or 0.0) / 100.0))
#     conf = 1.0
#     drop = (base * (0.50 + 0.50 * cov) * 100.0) * conf
#     return round(min(95.0, drop), 1)
# def _calc_power_drop(cloud_type, coverage_pct, factor_map, confidence_pct=100.0):
#     base = float(factor_map.get(cloud_type, 0.50))
#     cov = max(0.0, min(1.0, (coverage_pct or 0.0) / 100.0))
#     conf = max(0.35, min(1.0, confidence_pct / 100.0))
#     drop = (base * (0.50 + 0.50 * cov) * 100.0) * conf
#     return round(min(95.0, drop), 1)


# def compute_second_plant_forecast(cloud_type, speed_kmh, speed_mps, direction,
#                                    coverage_pct, plant2_dist_km=20.0,
#                                    plant2_bearing_deg=270.0):
#     """
#     Forecast when (and how much) a cloud's shadow will reach a SECOND solar plant
#     located at a fixed distance and bearing from Plant 1 (camera location).

#     Args:
#         cloud_type:         Detected cloud type string.
#         speed_kmh:          Cloud ground speed in km/h.
#         speed_mps:          Cloud ground speed in m/s.
#         direction:          Cloud movement direction string (e.g. 'West', 'NW', ...).
#         coverage_pct:       Sky cloud coverage percent (0–100).
#         plant2_dist_km:     Distance from Plant 1 to Plant 2 in km (default 20).
#         plant2_bearing_deg: Compass bearing of Plant 2 from Plant 1 in degrees
#                             (0=N, 90=E, 180=S, 270=W). Default 270 = West.

#     Returns dict:
#         shadow_arrives_min  – estimated minutes until shadow hits Plant 2 (or None)
#         power_drop_pct      – expected power drop at Plant 2
#         status              – 'incoming' | 'safe' | 'clear' | 'slow'
#         reason              – human-readable explanation string
#         effective_dist_km   – effective distance cloud must travel to reach Plant 2
#     """
#     cloud_power_factor = {
#         "Cumulus": 0.55, "Altocumulus": 0.45, "Cirrus": 0.18,
#         "ClearSky": 0.0, "Stratocumulus": 0.72, "Cumulonimbus": 0.85, "Mixed": 0.50
#     }
#     cloud_lifetime_min = {
#         "Cumulus": 35, "Altocumulus": 75, "Cirrus": 200,
#         "ClearSky": 0, "Stratocumulus": 270, "Cumulonimbus": 60, "Mixed": 55
#     }

#     if cloud_type == "ClearSky":
#         return {
#             "shadow_arrives_min": None,
#             "power_drop_pct": 0.0,
#             "status": "clear",
#             "reason": "☀️ Clear sky — no shadow risk for either plant.",
#             "effective_dist_km": plant2_dist_km,
#         }

#     if speed_kmh < 0.5:
#         return {
#             "shadow_arrives_min": None,
#             "power_drop_pct": 0.0,
#             "status": "slow",
#             "reason": "Cloud is nearly stationary. It is unlikely to reach the second plant.",
#             "effective_dist_km": plant2_dist_km,
#         }

#     # Convert cloud direction string → bearing in degrees
#     dir_to_bearing = {
#         "North": 0, "NE": 45, "East": 90, "SE": 135,
#         "South": 180, "SW": 225, "West": 270, "NW": 315
#     }
#     cloud_bearing = dir_to_bearing.get(direction, 0)

#     # Compute angle between cloud travel direction and bearing toward Plant 2
#     # delta = 0° means cloud is heading straight toward Plant 2
#     delta_deg = abs((cloud_bearing - plant2_bearing_deg + 180) % 360 - 180)

#     # Effective distance cloud must travel = plant2_dist_km / cos(delta)
#     # If delta >= 90°, cloud is moving away — it will never reach Plant 2
#     if delta_deg >= 85:
#         return {
#             "shadow_arrives_min": None,
#             "power_drop_pct": 0.0,
#             "status": "safe",
#             "reason": (
#                 f"Cloud is moving {direction} ({cloud_bearing:.0f}°), "
#                 f"which is {delta_deg:.0f}° away from Plant 2 direction ({plant2_bearing_deg:.0f}°). "
#                 f"The shadow will not reach Plant 2."
#             ),
#             "effective_dist_km": None,
#         }

#     delta_rad = math.radians(delta_deg)
#     effective_dist_km = plant2_dist_km / math.cos(delta_rad)

#     # Time for cloud shadow to travel that effective distance
#     travel_time_min = (effective_dist_km / speed_kmh) * 60.0

#     # Check against cloud lifetime — will it still be alive when it arrives?
#     lifetime = cloud_lifetime_min.get(cloud_type, 60)
#     if travel_time_min > lifetime:
#         return {
#             "shadow_arrives_min": travel_time_min,
#             "power_drop_pct": 0.0,
#             "status": "safe",
#             "reason": (
#                 f"Cloud would reach Plant 2 in ~{travel_time_min:.0f} min, "
#                 f"but {cloud_type} clouds only last ~{lifetime} min. "
#                 f"The cloud will dissipate before reaching Plant 2."
#             ),
#             "effective_dist_km": effective_dist_km,
#         }

#     power_drop = _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor)

#     direction_note = (
#         f"Cloud is moving {direction} ({cloud_bearing:.0f}°) — "
#         f"{delta_deg:.0f}° offset from Plant 2 direction ({plant2_bearing_deg:.0f}°)."
#     )

#     return {
#         "shadow_arrives_min": round(travel_time_min, 1),
#         "power_drop_pct": power_drop,
#         "status": "incoming",
#         "reason": (
#             f"{direction_note} "
#             f"Effective travel distance to Plant 2: {effective_dist_km:.1f} km. "
#             f"At {speed_kmh:.1f} km/h, shadow will reach Plant 2 in "
#             f"~{travel_time_min:.1f} minutes with an expected power drop of {power_drop}%."
#         ),
#         "effective_dist_km": effective_dist_km,
#     }


# def show_metrics(cloud_type, confidence, direction, height_m, fov,
#                  frame_width, pixel_disp, delta_t, deg_per_px,
#                  theta_deg, distance_m, speed_mps, speed_kmh, dist_5, dist_15,
#                  coverage_pct=None, density_label=None, density_color=None,
#                  vis_verdict=None, vis_reason=None, vis_color=None,
#                  time_to_exit_min=999, solar_dist_km=1.0,
#                  media_timestamp=None, timestamp_source="now",
#                  image_elevation_est=None, image_elevation_conf=None,
#                  image_elevation_note=None,
#                  plant2_dist_km=20.0, plant2_bearing_deg=270.0):
#     emoji = cloud_emoji.get(cloud_type, "☁️")

#     # ── Low confidence warning ──
#     if confidence < 60:
#         st.markdown(
#             f"⚠️ **Low Model Confidence ({confidence:.1f}%)** — The model is uncertain about this cloud type. Results may be less accurate.",
#             unsafe_allow_html=True,
#         )

#     # ── Section divider helper ──
#     def section_header(icon, title):
#         st.markdown(
#             f"<div style='font-size:0.9rem;font-weight:700;color:#94b8d4;margin:8px 0 6px 0;'>{icon} {title}</div>",
#             unsafe_allow_html=True,
#         )

#     # ── Section 1: Cloud Classification ──
#     section_header("☁", "Cloud Classification")
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric(f"{emoji} Cloud Type",    cloud_type)
#     c2.metric("🎯 Model Confidence",    f"{confidence:.1f}%")
#     c3.metric("🧭 Cloud Direction",      direction)
#     c4.metric("📍 Estimated Altitude",  f"{height_m:,} m")

#     # ── Section 2: Motion & Displacement ──
#     section_header("⚡", "Motion & Displacement")
#     label_5  = f"~{dist_5:.2f} km"  if dist_5  > 0 else "—"
#     label_15 = f"~{dist_15:.2f} km" if dist_15 > 0 else "—"

#     c5, c6, c7, c8 = st.columns(4)
#     c5.metric("Cloud Speed",            f"{speed_kmh:.1f} km/h")
#     c6.metric("Cloud Speed (m/s)",      f"{speed_mps:.2f} m/s")
#     c7.metric("Projected Dist. +5 min",  label_5,  delta_color="off")
#     c8.metric("Projected Dist. +15 min", label_15, delta_color="off")

#     # ── Section 3: Atmospheric Analysis ──
#     if coverage_pct is not None:
#         section_header("📡", "Atmospheric Analysis")
#         d1, d2 = st.columns(2)

#         with d1:
#             bar_filled = int(coverage_pct)
#             st.markdown(f"""
# <div style="background:#0d1a27;border:1px solid {density_color}44;border-radius:12px;padding:22px;">
#   <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#               letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">Sky Coverage</div>
#   <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:16px;">
#     <span style="font-size:2.2rem;font-weight:700;color:{density_color};font-family:'Inter',sans-serif;line-height:1;">{coverage_pct}%</span>
#     <span style="font-size:0.82rem;color:{density_color};font-weight:600;padding:3px 10px;
#                  background:{density_color}18;border-radius:999px;border:1px solid {density_color}33;">{density_label}</span>
#   </div>
#   <div style="background:#050c14;border-radius:8px;height:10px;width:100%;overflow:hidden;margin-bottom:8px;">
#     <div style="background:linear-gradient(90deg,{density_color}77,{density_color});
#                 width:{bar_filled}%;height:10px;border-radius:8px;"></div>
#   </div>
#   <div style="display:flex;justify-content:space-between;margin-top:6px;">
#     <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">Low — &lt;20%</span>
#     <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">Medium — 20–55%</span>
#     <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#2e4a64;">High — &gt;55%</span>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#         with d2:
#             st.markdown(f"""
# <div style="background:#0d1a27;border:1px solid {vis_color}44;border-radius:12px;padding:22px;height:100%;">
#   <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#               letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">Cloud Lifetime Forecast</div>
#   <div style="font-size:1rem;font-weight:700;color:{vis_color};margin-bottom:12px;
#               padding:8px 12px;background:{vis_color}14;border-radius:8px;border-left:3px solid {vis_color};">{vis_verdict}</div>
#   <div style="font-size:0.82rem;color:#7a9ab4;line-height:1.65;">{vis_reason}</div>
# </div>
# """, unsafe_allow_html=True)

#     # ── Section 4: Solar Plant Impact Forecast ──
#     section_header("☀️", "Solar Plant Impact Forecast")

#     solar = compute_solar_shadow_forecast(
#         cloud_type, height_m, speed_mps, speed_kmh,
#         direction, pixel_disp, frame_width, fov,
#         coverage_pct if coverage_pct is not None else 50.0
#     )

#     status = solar["status"]

#     if status == "clear":
#         st.markdown("""
# <div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:12px;padding:24px;display:flex;align-items:center;gap:20px;">
#   <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
#               display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">☀️</div>
#   <div>
#     <div style="font-size:1.05rem;font-weight:700;color:#22c55e;margin-bottom:6px;">Clear Sky — Solar Plant Fully Safe</div>
#     <div style="font-size:0.82rem;color:#4a8060;line-height:1.5;">No clouds detected. No shadow risk. Solar plant is operating at full capacity.</div>
#   </div>
#   <div style="margin-left:auto;text-align:right;flex-shrink:0;">
#     <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Power Status</div>
#     <div style="font-size:1.1rem;font-weight:700;color:#22c55e;">🟢 100%</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif status == "now":
#         pdrop = solar["power_drop_pct"]
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #ef444455;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:#ef444418;border:1px solid #ef444433;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">⚠️</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:#ef4444;margin-bottom:5px;">Shadow Currently Falling on Solar Panel</div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; {speed_kmh:.1f} km/h
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
#     <div style="background:#100505;border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#ef4444;line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#100505;border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
#       <div style="font-size:1rem;font-weight:700;color:#ff6b6b;margin-top:4px;">🔴 Active Now</div>
#     </div>
#     <div style="background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 2;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif status == "stationary":
#         pdrop = solar["power_drop_pct"]
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #f59e0b55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:#f59e0b18;border:1px solid #f59e0b33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🟡</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:#f59e0b;margin-bottom:5px;">Cloud Stationary — Shadow Present on Panel</div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; Nearly stationary
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
#     <div style="background:#100a00;border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#f59e0b;line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#100a00;border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Movement</div>
#       <div style="font-size:1rem;font-weight:700;color:#f59e0b;margin-top:4px;">⏸ Stationary</div>
#     </div>
#     <div style="background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 2;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif status == "incoming":
#         pdrop    = solar["power_drop_pct"]
#         arr_min  = solar["shadow_time_min"]
#         off_km   = solar["ground_offset_m"] / 1000.0
#         if arr_min < 10:
#             urg_color = "#ef4444"; urg_icon = "🔴"
#         elif arr_min < 30:
#             urg_color = "#f59e0b"; urg_icon = "🟡"
#         else:
#             urg_color = "#22c55e"; urg_icon = "🟢"
#         arr_str = f"{arr_min:.1f} min" if arr_min < 60 else f"{int(arr_min//60)}h {int(arr_min%60)}m"
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid {urg_color}55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:{urg_color}18;border:1px solid {urg_color}33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">☁️</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:{urg_color};margin-bottom:5px;">
#         {urg_icon} Shadow Will Reach Solar Plant in {arr_str}
#       </div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;">
#     <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Shadow Arrives In</div>
#       <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{arr_str}</div>
#     </div>
#     <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1e3a50;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Ground Offset</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#38bdf8;line-height:1;">{off_km:.2f} km</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 3;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Forecast Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{solar["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     else:  # status == "miss"
#         arr_min = solar["shadow_time_min"]
#         arr_str = f"{arr_min:.0f} min" if arr_min is not None else "N/A"
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">✅</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:#22c55e;margin-bottom:5px;">Shadow Will Not Reach the Solar Plant</div>
#       <div style="font-size:0.78rem;color:#4a6580;font-family:'JetBrains Mono',monospace;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
#       </div>
#     </div>
#   </div>
#   <div style="background:#050c14;border:1px solid #22c55e22;border-radius:10px;padding:16px 18px;margin-bottom:12px;">
#     <div style="font-size:0.82rem;color:#6aaa84;line-height:1.7;">{solar["reason"]}</div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Power Drop</div>
#       <div style="font-size:2.1rem;font-weight:700;color:#22c55e;line-height:1;">0%</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
#       <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-top:4px;">🟢 Safe</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
#     # ── Section 5: Second Solar Plant Forecast ──
#     section_header("🏭", "Second Solar Plant — Remote Shadow Forecast")

#     p2 = compute_second_plant_forecast(
#         cloud_type, speed_kmh, speed_mps, direction,
#         coverage_pct if coverage_pct is not None else 50.0,
#         plant2_dist_km=plant2_dist_km,
#         plant2_bearing_deg=plant2_bearing_deg,
#     )

#     p2_status = p2["status"]

#     # Direction label for Plant 2 bearing
#     _bear_labels = {0:"North",45:"NE",90:"East",135:"SE",180:"South",225:"SW",270:"West",315:"NW"}
#     _nearest_bear = min(_bear_labels.keys(), key=lambda k: abs(k - plant2_bearing_deg))
#     plant2_dir_label = _bear_labels[_nearest_bear]

#     if p2_status == "clear":
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:12px;padding:20px;display:flex;align-items:center;gap:20px;">
#   <div style="font-size:1.6rem;">🌿</div>
#   <div>
#     <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-bottom:4px;">Plant 2 — Clear Sky, No Risk</div>
#     <div style="font-size:0.82rem;color:#4a8060;">No clouds detected. Plant 2 ({plant2_dir_label}, {plant2_dist_km:.0f} km away) is fully safe.</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     elif p2_status == "slow":
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #4a6580;border-radius:12px;padding:20px;">
#   <div style="font-size:1rem;font-weight:700;color:#94b8d4;margin-bottom:6px;">⏸ Cloud Too Slow — Plant 2 Likely Safe</div>
#   <div style="font-size:0.82rem;color:#7a9ab4;">{p2["reason"]}</div>
# </div>
# """, unsafe_allow_html=True)

#     elif p2_status == "safe":
#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid #22c55e55;border-radius:12px;padding:20px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">
#     <div style="width:48px;height:48px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;">✅</div>
#     <div>
#       <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-bottom:4px;">Plant 2 — Shadow Will NOT Arrive</div>
#       <div style="font-size:0.75rem;font-family:'JetBrains Mono',monospace;color:#4a6580;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:3px 10px;display:inline-block;">
#         Plant 2: {plant2_dir_label} &nbsp;·&nbsp; {plant2_dist_km:.0f} km &nbsp;·&nbsp; Cloud: {direction}
#       </div>
#     </div>
#   </div>
#   <div style="background:#050c14;border:1px solid #22c55e22;border-radius:8px;padding:14px 16px;">
#     <div style="font-size:0.82rem;color:#6aaa84;line-height:1.7;">{p2["reason"]}</div>
#   </div>
#   <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;">
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:8px;padding:14px;">
#       <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Expected Power Drop</div>
#       <div style="font-size:2rem;font-weight:700;color:#22c55e;line-height:1;">0%</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:8px;padding:14px;">
#       <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Plant 2 Status</div>
#       <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-top:4px;">🟢 Safe</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     else:  # incoming
#         arr_min  = p2["shadow_arrives_min"]
#         pdrop    = p2["power_drop_pct"]
#         eff_dist = p2.get("effective_dist_km", plant2_dist_km)
#         arr_str  = f"{arr_min:.1f} min" if arr_min < 60 else f"{int(arr_min//60)}h {int(arr_min%60)}m"
#         if arr_min < 15:
#             urg_color = "#ef4444"; urg_icon = "🔴"
#         elif arr_min < 40:
#             urg_color = "#f59e0b"; urg_icon = "🟡"
#         else:
#             urg_color = "#22c55e"; urg_icon = "🟢"

#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid {urg_color}55;border-radius:14px;padding:24px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
#     <div style="width:52px;height:52px;border-radius:12px;background:{urg_color}18;border:1px solid {urg_color}33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🏭</div>
#     <div>
#       <div style="font-size:1.1rem;font-weight:700;color:{urg_color};margin-bottom:5px;">
#         {urg_icon} Shadow Reaches Plant 2 in {arr_str}
#       </div>
#       <div style="font-size:0.75rem;font-family:'JetBrains Mono',monospace;color:#4a6580;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:4px 10px;display:inline-block;">
#         Plant 2: {plant2_dir_label} &nbsp;·&nbsp; {plant2_dist_km:.0f} km &nbsp;·&nbsp; Cloud: {direction} @ {speed_kmh:.1f} km/h
#       </div>
#     </div>
#   </div>
#   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;">
#     <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Shadow Arrives In</div>
#       <div style="font-size:2rem;font-weight:700;color:{urg_color};line-height:1;">{arr_str}</div>
#     </div>
#     <div style="background:#050c14;border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
#       <div style="font-size:2rem;font-weight:700;color:{urg_color};line-height:1;">{pdrop}%</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1e3a50;border-radius:10px;padding:16px 18px;">
#       <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Travel Distance</div>
#       <div style="font-size:2rem;font-weight:700;color:#38bdf8;line-height:1;">{eff_dist:.1f} km</div>
#     </div>
#     <div style="background:#050c14;border:1px solid #1a2d44;border-radius:10px;padding:16px 18px;grid-column:span 3;">
#       <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Forecast Analysis</div>
#       <div style="font-size:0.82rem;color:#94b8d4;line-height:1.7;">{p2["reason"]}</div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     # ── Section 6: Sun Position & Cloud Alignment ──
#     _lat = st.session_state.get("user_lat", 28.6)
#     _lon = st.session_state.get("user_lon", 77.2)

#     # Use media timestamp if available, else current time
#     _ts = media_timestamp if media_timestamp is not None else datetime.datetime.utcnow()
#     sun_az, sun_el = get_solar_position(_lat, _lon, _ts)

#     # Timestamp source badge
#     _ts_badge_map = {
#         "exif":    ("📷 From EXIF",        "#22c55e"),
#         "ffprobe": ("🎬 From Video Meta",  "#22c55e"),
#         "manual":  ("🕐 Manual Input",      "#f59e0b"),
#         "now":     ("⏱️ Current Time",      "#4a6580"),
#     }
#     _ts_label, _ts_color = _ts_badge_map.get(timestamp_source, ("⏱️ Current Time", "#4a6580"))

#     section_header("🌞", "Sun Position & Cloud Alignment — Plant 1")

#     if sun_az is None:
#         st.info("📍 Install pvlib + pandas and set your location in the sidebar for live sun tracking.")
#     else:
#         sun_dir  = sun_azimuth_to_direction(sun_az)
#         align_status, angle_diff, align_desc = get_cloud_sun_alignment(direction, sun_az)

#         align_color_map = {
#             "toward_sun":    "#ef4444",
#             "glancing":      "#f59e0b",
#             "crossing":      "#38bdf8",
#             "away_from_sun": "#22c55e",
#             "unknown":       "#4a6580",
#         }
#         align_icon_map = {
#             "toward_sun":    "🔴 Heading Toward Sun",
#             "glancing":      "🟡 Glancing Sun",
#             "crossing":      "🔵 Crossing Sun Path",
#             "away_from_sun": "🟢 Moving Away from Sun",
#             "unknown":       "❓ Unknown",
#         }
#         a_color = align_color_map.get(align_status, "#4a6580")
#         a_label = align_icon_map.get(align_status, "")

#         if sun_el < 0:
#             sun_status_label = "🌙 Below Horizon"
#             sun_el_color = "#4a6580"
#         elif sun_el < 15:
#             sun_status_label = "🌅 Near Horizon"
#             sun_el_color = "#f59e0b"
#         else:
#             sun_status_label = "☀️ Above Horizon"
#             sun_el_color = "#fbbf24"

#         # ── Determine which elevation to show ──
#         # pvlib = authoritative; image estimate = fallback shown alongside
#         show_img_est = (image_elevation_est is not None)
#         img_conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(
#             image_elevation_conf, "#4a6580")

#         sc1, sc2, sc3, sc4 = st.columns(4)
#         sc1.metric("☀️ Sun Azimuth",   f"{sun_az:.1f}°")
#         sc2.metric("📐 Sun Elevation (pvlib)", f"{sun_el:.1f}°")
#         sc3.metric("🧭 Sun Direction", sun_dir)
#         sc4.metric("☁️ Cloud Moving",  direction)

#         # Timestamp badge + image estimate row
#         badge_html = f"""
# <div style="display:flex;align-items:center;gap:10px;margin:10px 0 14px 0;flex-wrap:wrap;">
#   <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;padding:3px 10px;
#                border-radius:999px;background:{_ts_color}18;border:1px solid {_ts_color}44;
#                color:{_ts_color};">{_ts_label}: {_ts.strftime('%Y-%m-%d %H:%M UTC')}</span>"""

#         if show_img_est:
#             badge_html += f"""
#   <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;padding:3px 10px;
#                border-radius:999px;background:{img_conf_color}18;border:1px solid {img_conf_color}44;
#                color:{img_conf_color};">
#     📸 Image Estimate: {image_elevation_est}° ({image_elevation_conf} confidence)
#   </span>"""

#         badge_html += "</div>"
#         st.markdown(badge_html, unsafe_allow_html=True)

#         # Image elevation detail card (only when sun not visible in frame)
#         if show_img_est:
#             diff_el = abs(sun_el - image_elevation_est)
#             st.markdown(f"""
# <div style="background:#0d1a27;border:1px solid {img_conf_color}44;border-radius:12px;
#             padding:16px 20px;margin-bottom:14px;">
#   <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
#               letter-spacing:0.12em;color:#4a6580;margin-bottom:10px;">
#     📸 Image-Based Sun Elevation Estimate (Sun not visible in frame)
#   </div>
#   <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;">
#     <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
#       <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:6px;">Image Estimate</div>
#       <div style="font-size:1.6rem;font-weight:700;color:{img_conf_color};line-height:1;">
#         {image_elevation_est}°</div>
#     </div>
#     <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
#       <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:6px;">pvlib (Location+Time)</div>
#       <div style="font-size:1.6rem;font-weight:700;color:#fbbf24;line-height:1;">{sun_el:.1f}°</div>
#     </div>
#     <div style="background:#050c14;border-radius:8px;padding:12px 14px;">
#       <div style="font-size:0.6rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#                   letter-spacing:0.1em;margin-bottom:6px;">Difference</div>
#       <div style="font-size:1.6rem;font-weight:700;color:{'#22c55e' if diff_el < 10 else '#f59e0b' if diff_el < 25 else '#ef4444'};line-height:1;">
#         ±{diff_el:.1f}°</div>
#     </div>
#   </div>
#   <div style="font-size:0.8rem;color:#7a9ab4;line-height:1.6;font-style:italic;">{image_elevation_note}</div>
# </div>
# """, unsafe_allow_html=True)

#         st.markdown(f"""
# <div style="background:#0d1a27;border:1.5px solid {a_color}55;border-radius:14px;padding:22px;margin-top:4px;">
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
#     <div style="width:48px;height:48px;border-radius:12px;background:{a_color}18;border:1px solid {a_color}33;
#                 display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;">🌞</div>
#     <div>
#       <div style="font-size:1rem;font-weight:700;color:{a_color};margin-bottom:4px;">
#         Cloud–Sun Alignment: {a_label}
#       </div>
#       <div style="font-size:0.78rem;font-family:'JetBrains Mono',monospace;color:#4a6580;
#                   background:#0a0f16;border:1px solid #1a2d44;border-radius:6px;padding:3px 10px;display:inline-block;">
#         {sun_status_label} &nbsp;·&nbsp; Azimuth {sun_az:.1f}° &nbsp;·&nbsp; Elevation {sun_el:.1f}°
#         {'&nbsp;·&nbsp; ' + str(round(angle_diff)) + '° offset' if angle_diff is not None else ''}
#       </div>
#     </div>
#   </div>
#   <div style="font-size:0.84rem;color:#94b8d4;line-height:1.7;background:#050c14;
#               border-radius:10px;padding:14px 18px;border:1px solid #1a2d44;">
#     {align_desc}
#   </div>
# </div>
# """, unsafe_allow_html=True)

#     st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

#     # ── Export Analysis Results ──
#     with st.expander("📤 Export Analysis Report"):
#         import csv, io
#         report_data = {
#             "cloud_type": cloud_type,
#             "confidence_pct": round(confidence, 1),
#             "direction": direction,
#             "altitude_m": height_m,
#             "speed_kmh": round(speed_kmh, 1),
#             "speed_mps": round(speed_mps, 2),
#             "projected_dist_5min_km": round(dist_5, 2),
#             "projected_dist_15min_km": round(dist_15, 2),
#             "sky_coverage_pct": coverage_pct,
#             "density_label": density_label,
#             "visibility_forecast": vis_verdict,
#             "timestamp_utc": _ts.strftime("%Y-%m-%d %H:%M UTC") if media_timestamp is not None else datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
#             "timestamp_source": timestamp_source,
#         }
#         # JSON download
#         import json as _json
#         json_str = _json.dumps(report_data, indent=2)
#         st.download_button(
#             "📥 Download JSON Report",
#             data=json_str,
#             file_name=f"cloudvision_report_{cloud_type}.json",
#             mime="application/json",
#             key="dl_json_report"
#         )
#         # CSV download
#         csv_buf = io.StringIO()
#         writer = csv.DictWriter(csv_buf, fieldnames=report_data.keys())
#         writer.writeheader()
#         writer.writerow(report_data)
#         st.download_button(
#             "📥 Download CSV Report",
#             data=csv_buf.getvalue(),
#             file_name=f"cloudvision_report_{cloud_type}.csv",
#             mime="text/csv",
#             key="dl_csv_report"
#         )
#         st.code(json_str, language="json")

#     with st.expander("🔬 Optical Flow — Calculation Details"):
#         st.markdown(f"""
# | Parameter | Value |
# |---|---|
# | Camera FOV | {fov}° |
# | Frame Width | {frame_width} px |
# | Degrees per Pixel | {deg_per_px:.4f} °/px |
# | Pixel Displacement | {pixel_disp:.2f} px over {delta_t:.2f} s |
# | Angular Displacement (θ) | {theta_deg:.4f}° |
# | Horizontal Distance (tan formula) | {distance_m:.2f} m |
# | Derived Speed | {speed_mps:.2f} m/s → {speed_kmh:.1f} km/h |
# """)

# # ─────────────────────────── CLOUD DETECTION ───────────────────
# def detect_clouds(frame, sky_h):
#     """
#     Multi-method cloud detection:
#     1. Brightness threshold (white clouds)
#     2. HSV saturation (low saturation = cloud/white)
#     3. Combine both masks
#     Uses watershed-style distance-based separation to assign distinct
#     bounding boxes to individual cloud regions.
#     """
#     OUT_W = frame.shape[1]

#     gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     sky_gray = gray[:sky_h, :]
#     sky_hsv  = hsv[:sky_h, :]

#     # Method 1: brightness — lower threshold to catch grey clouds too
#     _, bright_mask = cv2.threshold(sky_gray, 140, 255, cv2.THRESH_BINARY)

#     # Method 2: low saturation = white/grey cloud (not blue sky)
#     sat = sky_hsv[:, :, 1]
#     _, sat_mask = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY_INV)

#     # Method 3: not-blue sky — blue sky has high hue (100-130)
#     hue = sky_hsv[:, :, 0]
#     blue_sky = cv2.inRange(hue, 95, 135)
#     not_blue = cv2.bitwise_not(blue_sky)

#     # Combine: bright OR (low-sat AND not-blue-sky)
#     combined = cv2.bitwise_or(bright_mask,
#                 cv2.bitwise_and(sat_mask, not_blue))

#     # Morphology — smaller kernels to keep clouds separate
#     k_close = np.ones((12, 12), np.uint8)
#     k_open  = np.ones((6,  6),  np.uint8)
#     combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
#     combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)

#     # --- Watershed separation to split merged clouds ---
#     dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
#     cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
#     _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
#     sure_fg    = np.uint8(sure_fg)

#     sure_bg    = cv2.dilate(combined, np.ones((3,3), np.uint8), iterations=3)
#     unknown    = cv2.subtract(sure_bg, sure_fg)

#     _, markers = cv2.connectedComponents(sure_fg)
#     markers    = markers + 1
#     markers[unknown == 255] = 0

#     # Watershed needs 3-channel BGR image
#     sky_bgr = frame[:sky_h, :].copy()
#     markers = cv2.watershed(sky_bgr, markers)

#     # Extract bounding boxes from each watershed region
#     boxes = []
#     unique_labels = np.unique(markers)
#     for lbl in unique_labels:
#         if lbl <= 1:   # background or border
#             continue
#         mask_lbl = np.zeros_like(combined)
#         mask_lbl[markers == lbl] = 255
#         cnts, _ = cv2.findContours(mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in cnts:
#             area = cv2.contourArea(cnt)
#             if area < 600:   # ignore tiny noise
#                 continue
#             x, y, w, h = cv2.boundingRect(cnt)
#             boxes.append((x, y, w, h, mask_lbl))

#     # Fallback: if watershed gave nothing, use simple contours
#     if not boxes:
#         cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in cnts:
#             if cv2.contourArea(cnt) < 600:
#                 continue
#             x, y, w, h = cv2.boundingRect(cnt)
#             boxes.append((x, y, w, h, None))

#     return boxes, gray

# # ─────────────────────── STEREO DEPTH VISION ───────────────────
# def compute_pseudo_depth_map(frame_bgr, sky_h):
#     """
#     Single-image pseudo stereo depth map for cloud regions.

#     Physics cues used (all monocular):
#       1. Brightness  — brighter cloud core = optically thicker = visually 'closer'
#       2. Texture     — high-freq detail = nearer; smooth/hazy = farther
#       3. Saturation  — desaturated (grey/white) regions = cloud mass present
#       4. Vertical pos— lower in sky frame ≈ closer horizon clouds

#     Output: depth_map (H x W float32, 0=far / blue … 1=near / red)
#             depth_color (H x W x 3 uint8, COLORMAP_JET applied)
#     """
#     sky = frame_bgr[:sky_h, :].copy()
#     H, W = sky.shape[:2]

#     gray = cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
#     hsv  = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV).astype(np.float32)
#     sat  = hsv[:, :, 1] / 255.0
#     val  = hsv[:, :, 2] / 255.0

#     # Cue 1: brightness (brighter = more cloud mass = closer)
#     bright_cue = val

#     # Cue 2: texture energy via Laplacian (sharp edges = nearer)
#     lap = cv2.Laplacian(gray, cv2.CV_32F)
#     tex_cue = np.abs(lap)
#     tex_cue = cv2.GaussianBlur(tex_cue, (15, 15), 0)
#     tex_max = tex_cue.max()
#     if tex_max > 0:
#         tex_cue /= tex_max

#     # Cue 3: low saturation = cloud (not blue sky) → weight up
#     cloud_presence = 1.0 - np.clip(sat, 0, 1)   # white/grey = high weight

#     # Cue 4: vertical position — lower row = closer (nearer horizon)
#     row_idx  = np.linspace(1.0, 0.0, H, dtype=np.float32)   # top=far, bottom=near
#     vert_cue = np.tile(row_idx[:, None], (1, W))

#     # Weighted fusion
#     depth = (0.40 * bright_cue +
#              0.25 * tex_cue    +
#              0.20 * cloud_presence +
#              0.15 * vert_cue)

#     # Smooth for clean visualization
#     depth = cv2.GaussianBlur(depth, (21, 21), 0)
#     cv2.normalize(depth, depth, 0, 1, cv2.NORM_MINMAX)

#     # Colorize: COLORMAP_JET  blue=far → green=mid → red=near
#     depth_u8    = (depth * 255).astype(np.uint8)
#     depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

#     return depth, depth_color


# def depth_to_distance_km(depth_val, cloud_height_m, fov_deg):
#     """
#     Depth value (0–1) → estimated slant distance in km.
#     Uses trigonometry: closer clouds (higher depth) are nearer to cloud_height_m;
#     farther (lower depth) are assumed to be 1.5–3x that height away (oblique angle).
#     """
#     # depth=1 → distance = cloud_height_m (directly overhead)
#     # depth=0 → distance = 3 * cloud_height_m (far horizon, shallow angle)
#     distance_m = cloud_height_m * (1.0 + 2.0 * (1.0 - float(depth_val)))
#     return round(distance_m / 1000.0, 2)


# # ─────────────────────────── BOUNDING BOX FUNCTION ─────────────
# def draw_boxes_on_frame(frame, speed_kmh, direction, cloud_type, height_m,
#                          dist_5, dist_15, elapsed_sec, prev_gray=None, delta_t=None,
#                          fov=75, time_to_exit_min=999):
#     OUT_W = frame.shape[1]
#     OUT_H = frame.shape[0]
#     sky_h = int(OUT_H * 0.78)   # slightly more sky area

#     # Pre-compute full dense optical flow if prev frame available
#     gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     sky_gray = gray[:sky_h, :]
#     full_flow = None
#     if prev_gray is not None and delta_t is not None and delta_t > 0:
#         prev_sky = prev_gray[:sky_h, :]
#         full_flow = cv2.calcOpticalFlowFarneback(
#             prev_sky, sky_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
#         )

#     # Detect clouds with improved method
#     boxes, _ = detect_clouds(frame, sky_h)

#     # ── Compute pseudo stereo depth map for full sky region ──
#     depth_map, depth_color = compute_pseudo_depth_map(frame, sky_h)

#     for (x, y, w, h, _) in boxes:
#         pad = 8
#         x1 = max(0,       x - pad);    y1 = max(0,       y - pad)
#         x2 = min(OUT_W-1, x+w + pad);  y2 = min(sky_h,   y+h + pad)

#         # ── Per-cloud speed from optical flow ROI ──
#         if full_flow is not None:
#             roi_flow = full_flow[y1:y2, x1:x2]
#             if roi_flow.size > 0:
#                 mag, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
#                 roi_pixel_disp = float(np.median(mag))
#                 if roi_pixel_disp > 0.1:
#                     _, cloud_speed_kmh, _, _, _, _ = pixels_to_kmh(
#                         roi_pixel_disp, delta_t, cloud_type, OUT_W, fov
#                     )
#                 else:
#                     cloud_speed_kmh = 0.0
#             else:
#                 cloud_speed_kmh = speed_kmh
#         else:
#             cloud_speed_kmh = speed_kmh

#         # ── Stereo Depth Overlay inside box ──
#         roi_depth_color = depth_color[y1:y2, x1:x2]
#         roi_frame       = frame[y1:y2, x1:x2]
#         if roi_depth_color.shape == roi_frame.shape and roi_frame.size > 0:
#             # Blend depth colormap (40%) with original frame (60%)
#             cv2.addWeighted(roi_depth_color, 0.40, roi_frame, 0.60, 0,
#                             frame[y1:y2, x1:x2])

#         # Estimated distance from depth at box center
#         cy_box = min((y1 + y2) // 2, depth_map.shape[0] - 1)
#         cx_box = min((x1 + x2) // 2, depth_map.shape[1] - 1)
#         center_depth = float(depth_map[cy_box, cx_box])
#         est_dist_km  = depth_to_distance_km(center_depth, height_m, fov)

#         # Glow effect
#         glow = frame.copy()
#         cv2.rectangle(glow, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 100), 4)
#         cv2.addWeighted(glow, 0.3, frame, 0.7, 0, frame)

#         # Main box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

#         # Corner ticks
#         t = 14
#         for (px_, py_, sdx, sdy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
#             cv2.line(frame, (px_, py_), (px_+sdx*t, py_),    (0, 255, 60), 2)
#             cv2.line(frame, (px_, py_), (px_, py_+sdy*t),    (0, 255, 60), 2)

#         # Per-cloud speed + depth distance label
#         label = f"{cloud_speed_kmh:.1f} km/h  |  D:{est_dist_km:.1f}km"
#         fs    = 0.46
#         (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
#         lx = x1
#         ly = y1 - 5 if y1 - 5 - th > 2 else y1 + th + 6
#         cv2.rectangle(frame, (lx-2, ly-th-3), (lx+tw+6, ly+bl+1), (0, 150, 55), -1)
#         cv2.rectangle(frame, (lx-2, ly-th-3), (lx+tw+6, ly+bl+1), (0, 255, 100), 1)
#         cv2.putText(frame, label, (lx+3, ly),
#                     cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

#     # ── HUD top-left ──
#     ov = frame.copy()
#     cv2.rectangle(ov, (0,0), (360,125), (0,0,0), -1)
#     cv2.addWeighted(ov, 0.58, frame, 0.42, 0, frame)
#     cv2.rectangle(frame, (0,0), (360,125), (0,200,80), 1)

#     def txt(t, y, sc=0.52, c=(255,255,255), b=1):
#         cv2.putText(frame, t, (12,y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, b, cv2.LINE_AA)

#     txt(f"Cloud  : {cloud_type}",   22, c=(140,230,255), b=2)
#     txt(f"Height : {height_m:,} m", 43)
#     txt(f"Speed  : {speed_kmh:.1f} km/h  ({speed_kmh/3.6:.2f} m/s)", 64, c=(80,255,160))

#     # Smart +5 / +15 min — show "OUT OF FRAME" if cloud will have exited by then
#     lbl_5  = f"~{dist_5:.2f} km"
#     lbl_15 = f"~{dist_15:.2f} km"
#     txt(f"Dir:{direction}  +5m:{lbl_5}  +15m:{lbl_15}", 86, sc=0.40, c=(200, 200, 200))

#     mins = int(elapsed_sec)//60;  secs = int(elapsed_sec)%60
#     cv2.putText(frame, f"T+ {mins:02d}:{secs:02d}",
#                 (OUT_W-155, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,180), 2, cv2.LINE_AA)

#     # Direction arrow
#     dir_vec = {"East":(1,0),"West":(-1,0),"North":(0,-1),"South":(0,1)}.get(direction,(1,0))
#     cx, cy  = OUT_W//2, OUT_H - 35
#     cv2.arrowedLine(frame, (cx,cy),
#                     (int(cx+dir_vec[0]*55), int(cy+dir_vec[1]*55)),
#                     (255,255,255), 3, tipLength=0.35)
#     cv2.putText(frame, direction, (cx-30, cy+20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

#     # ── Sun Position Arrow ──
#     try:
#         _lat2 = st.session_state.get("user_lat", 28.6)
#         _lon2 = st.session_state.get("user_lon", 77.2)
#         _sun_az2, _sun_el2 = get_solar_position(_lat2, _lon2, datetime.datetime.utcnow())
#     except Exception:
#         _sun_az2, _sun_el2 = None, None

#     if _sun_az2 is not None and _sun_el2 > 0:
#         # Draw sun arrow at bottom-center right of cloud arrow
#         sun_az_rad = math.radians(_sun_az2)   # 0=N, 90=E
#         # Convert azimuth to screen vector (x right=East, y down=South)
#         sun_dx = math.sin(sun_az_rad)   # East component
#         sun_dy = -math.cos(sun_az_rad)  # North component (inverted for screen)
#         scx, scy = OUT_W//2 + 120, OUT_H - 35
#         cv2.arrowedLine(frame, (scx, scy),
#                         (int(scx + sun_dx * 50), int(scy + sun_dy * 50)),
#                         (30, 220, 255), 2, tipLength=0.4)
#         cv2.putText(frame, f"Sun {_sun_az2:.0f}", (scx - 28, scy + 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 220, 255), 1, cv2.LINE_AA)

#         # Cloud-Sun alignment indicator
#         align_status, angle_diff, _ = get_cloud_sun_alignment(direction, _sun_az2)
#         align_color_cv = {
#             "toward_sun":    (60, 80, 255),
#             "glancing":      (60, 180, 255),
#             "crossing":      (255, 200, 60),
#             "away_from_sun": (60, 220, 100),
#             "unknown":       (150, 150, 150),
#         }.get(align_status, (150, 150, 150))
#         align_text = {
#             "toward_sun":    "TO SUN",
#             "glancing":      "GLANCING",
#             "crossing":      "CROSSING",
#             "away_from_sun": "FROM SUN",
#             "unknown":       "?",
#         }.get(align_status, "?")
#         if angle_diff is not None:
#             align_label_full = f"{align_text} {angle_diff:.0f}deg"
#         else:
#             align_label_full = align_text
#         (aw, _ah), _ = cv2.getTextSize(align_label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
#         ax = scx - aw // 2
#         ay = scy - 14
#         cv2.rectangle(frame, (ax - 3, ay - 14), (ax + aw + 4, ay + 4), (10, 10, 10), -1)
#         cv2.putText(frame, align_label_full, (ax, ay),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.44, align_color_cv, 1, cv2.LINE_AA)

#     # ── Solar Plant Shadow HUD (camera IS on solar plant) ──
#     if cloud_type != "ClearSky":
#         solar = compute_solar_shadow_forecast(
#             cloud_type, height_m, speed_mps, speed_kmh,
#             direction, OUT_W * 0.05,   # small pixel_disp proxy for HUD
#             OUT_W, fov, coverage_pct=50.0
#         )
#         status = solar["status"]
#         if status == "now" or status == "stationary":
#             s_line1 = f"SHADOW ON SOLAR PLANT NOW!"
#             s_line2 = f"Power drop: {solar['power_drop_pct']}%"
#             box_col  = (0, 60, 220)   # red-orange
#             txt_col1 = (60, 80, 255)
#             txt_col2 = (60, 255, 160)
#         elif status == "incoming":
#             arr = solar["shadow_time_min"]
#             arr_str = f"{arr:.1f}min" if arr < 60 else f"{int(arr//60)}h{int(arr%60)}m"
#             s_line1 = f"Shadow arrives: {arr_str}"
#             s_line2 = f"Power drop: {solar['power_drop_pct']}%"
#             box_col  = (0, 160, 240)
#             txt_col1 = (80, 220, 255)
#             txt_col2 = (80, 255, 160)
#         elif status == "miss":
#             s_line1 = "Shadow will NOT hit solar plant"
#             s_line2 = "Power drop: 0%  [SAFE]"
#             box_col  = (0, 130, 40)
#             txt_col1 = (80, 255, 120)
#             txt_col2 = (80, 255, 120)
#         else:
#             s_line1 = "Clear sky — solar plant safe"
#             s_line2 = "No shadow expected"
#             box_col  = (0, 130, 40)
#             txt_col1 = (80, 255, 120)
#             txt_col2 = (80, 255, 120)

#         (sw1, _), _ = cv2.getTextSize(s_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
#         (sw2, _), _ = cv2.getTextSize(s_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
#         box_w = max(sw1, sw2) + 22
#         box_h = 58
#         bx, by = OUT_W - box_w - 8, 8

#         sol_ov = frame.copy()
#         cv2.rectangle(sol_ov, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
#         cv2.addWeighted(sol_ov, 0.62, frame, 0.38, 0, frame)
#         cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), box_col, 1)
#         cv2.putText(frame, s_line1, (bx + 8, by + 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col1, 1, cv2.LINE_AA)
#         cv2.putText(frame, s_line2, (bx + 8, by + 44),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col2, 1, cv2.LINE_AA)

#     # ── Depth Legend (colorbar) — bottom right ──
#     bar_x, bar_y, bar_w, bar_h = OUT_W - 120, OUT_H - 90, 18, 70
#     for i in range(bar_h):
#         val   = int(255 * (1.0 - i / bar_h))
#         color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
#         cv2.line(frame, (bar_x, bar_y + i), (bar_x + bar_w, bar_y + i), color, 1)
#     cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
#     cv2.putText(frame, "Near", (bar_x + bar_w + 4, bar_y + 8),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 80, 80), 1, cv2.LINE_AA)
#     cv2.putText(frame, "Far",  (bar_x + bar_w + 4, bar_y + bar_h),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 200), 1, cv2.LINE_AA)
#     cv2.putText(frame, "Depth", (bar_x - 2, bar_y - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

#     return frame


# def generate_boxed_video(input_path, output_path, speed_kmh, speed_mps,
#                           direction, cloud_type, height_m, dist_5, dist_15, fov=75,
#                           time_to_exit_min=999):
#     cap     = cv2.VideoCapture(input_path)
#     fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     OUT_W, OUT_H = 960, 540
#     delta_t = 1.0 / fps

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out    = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, OUT_H))

#     prev_gray = None
#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame   = cv2.resize(frame, (OUT_W, OUT_H))
#         elapsed = frame_idx / fps
#         gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame   = draw_boxes_on_frame(
#             frame, speed_kmh, direction, cloud_type,
#             height_m, dist_5, dist_15, elapsed,
#             prev_gray=prev_gray, delta_t=delta_t, fov=fov,
#             time_to_exit_min=time_to_exit_min
#         )
#         out.write(frame)
#         prev_gray = gray
#         frame_idx += 1

#     cap.release()
#     out.release()

#     # Re-encode to H.264 so browser can play it in st.video()
#     if shutil.which("ffmpeg"):
#         tmp_h264 = output_path.replace(".mp4", "_h264.mp4")
#         subprocess.run([
#             "ffmpeg", "-y", "-i", output_path,
#             "-vcodec", "libx264", "-crf", "23",
#             "-preset", "fast", "-pix_fmt", "yuv420p",
#             "-movflags", "+faststart",
#             tmp_h264
#         ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
#             os.replace(tmp_h264, output_path)


# # ─────────────────────────── UI HEADER ────────────────────────
# st.markdown("""
# <div class="cv-header">
#   <div class="cv-logo">☁️</div>
#   <div>
#     <div class="cv-title">CloudVision AI</div>
#     <div class="cv-sub">Cloud Classification &amp; Motion Prediction System</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

# tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎬 Video Analysis", "🖼️ Multi Image Analysis", "📸 Stereo Cameras", "🎥 Live Tracking", "📡 Live IP Cameras"])

# # ── Solar location inputs (shared across tabs) ──
# with st.sidebar:
#     st.markdown("### ☀️ Solar Location")
#     st.caption("Enter your location for real-time sun position tracking")
#     user_lat = st.number_input("Latitude",  value=28.6, min_value=-90.0,  max_value=90.0,  step=0.1, format="%.4f", key="user_lat")
#     user_lon = st.number_input("Longitude", value=77.2, min_value=-180.0, max_value=180.0, step=0.1, format="%.4f", key="user_lon")
#     st.caption("🇮🇳 Default: New Delhi")

#     st.markdown("---")
#     st.markdown("### 🕐 Media Timestamp")
#     st.caption("App auto-reads EXIF/video metadata. Override manually if needed.")
#     use_manual_time = st.checkbox("✏️ Override timestamp manually", value=False, key="use_manual_time")
#     if use_manual_time:
#         _today = datetime.date.today()
#         manual_date = st.date_input("Date", value=_today, key="manual_date")
#         manual_time = st.time_input("Time (local)", value=datetime.time(12, 0), key="manual_time")
#         tz_offset   = st.number_input("Timezone offset (hrs from UTC)", value=5.5,
#                                        min_value=-12.0, max_value=14.0, step=0.5, key="tz_offset")
#         # Convert to UTC datetime
#         local_dt = datetime.datetime.combine(manual_date, manual_time)
#         manual_utc = local_dt - datetime.timedelta(hours=tz_offset)
#         st.session_state["manual_utc"] = manual_utc
#         st.caption(f"UTC: {manual_utc.strftime('%Y-%m-%d %H:%M')}")
#     else:
#         st.session_state["manual_utc"] = None

#     if pvlib is not None:
#         _ts_sb = st.session_state.get("manual_utc") or datetime.datetime.utcnow()
#         _az, _el = get_solar_position(user_lat, user_lon, _ts_sb)
#         if _az is not None:
#             _sun_dir = sun_azimuth_to_direction(_az)
#             if _el < 0:
#                 _sun_status = "🌙 Sun below horizon"
#                 _sun_color  = "#4a6580"
#             elif _el < 15:
#                 _sun_status = "🌅 Sun near horizon"
#                 _sun_color  = "#f59e0b"
#             else:
#                 _sun_status = "☀️ Sun above horizon"
#                 _sun_color  = "#fbbf24"
#             st.markdown(f"""
# <div style='background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:14px;margin-top:8px;'>
#   <div style='font-size:0.68rem;font-family:monospace;color:#4a6580;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Live Sun Position</div>
#   <div style='font-size:0.92rem;font-weight:600;color:{_sun_color};margin-bottom:6px;'>{_sun_status}</div>
#   <div style='font-size:0.78rem;color:#94b8d4;'>Azimuth: <b style='color:#e2ecf6;'>{_az:.1f}°</b> ({_sun_dir})</div>
#   <div style='font-size:0.78rem;color:#94b8d4;'>Elevation: <b style='color:#e2ecf6;'>{_el:.1f}°</b></div>
# </div>
# """, unsafe_allow_html=True)
#     else:
#         st.info("Install pvlib + pandas for live sun tracking:\n`pip install pvlib pandas`")

#     st.markdown("---")
#     st.markdown("### 🏭 Second Solar Plant (Remote)")
#     st.caption("Set location of your second plant relative to Plant 1 (camera location)")

#     plant2_dist_km = st.number_input(
#         "Distance from Plant 1 (km)", min_value=0.1, max_value=500.0,
#         value=20.0, step=0.5, format="%.1f", key="plant2_dist_km"
#     )

#     _bearing_options = {
#         "West (270°)": 270.0, "East (90°)": 90.0,
#         "North (0°)": 0.0, "South (180°)": 180.0,
#         "NW (315°)": 315.0, "NE (45°)": 45.0,
#         "SW (225°)": 225.0, "SE (135°)": 135.0,
#     }
#     _bearing_label = st.selectbox(
#         "Direction of Plant 2 from Plant 1",
#         list(_bearing_options.keys()),
#         index=0, key="plant2_direction_label"
#     )
#     plant2_bearing_deg = _bearing_options[_bearing_label]
#     st.session_state["plant2_bearing_deg"] = plant2_bearing_deg

#     # Mini info card
#     st.markdown(f"""
# <div style='background:#0d1a27;border:1px solid #1a2d44;border-radius:10px;padding:12px;margin-top:6px;'>
#   <div style='font-size:0.68rem;font-family:monospace;color:#4a6580;text-transform:uppercase;
#               letter-spacing:0.1em;margin-bottom:6px;'>Plant 2 Config</div>
#   <div style='font-size:0.82rem;color:#94b8d4;'>📍 {plant2_dist_km:.1f} km {_bearing_label.split(" ")[0]} of Plant 1</div>
#   <div style='font-size:0.75rem;color:#4a6580;margin-top:4px;'>Bearing: {plant2_bearing_deg:.0f}°</div>
# </div>
# """, unsafe_allow_html=True)

#     st.markdown("---")
#     st.markdown("### ℹ️ About")
#     st.caption(
#         "**CloudVision AI** — Cloud classification, motion analysis & "
#         "solar shadow forecasting.\n\n"
#         "Model: Keras CNN · Classes: Cumulus, Altocumulus, Cirrus, "
#         "ClearSky, Stratocumulus, Cumulonimbus, Mixed"
#     )

# # ══════════════════════════ VIDEO TAB ══════════════════════════
# with tab1:
#     st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🎬 Video Analysis</div>', unsafe_allow_html=True)
#     st.subheader("Upload a Sky Video")

#     uploaded_video = st.file_uploader("Drop an MP4 / AVI / MOV file here",
#                                        type=["mp4","avi","mov"], key="video_upload")
#     fov_video = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
#                           help="Phone: 70-80° | Wide angle: 90-120° | Telephoto: 30-50°")

#     if uploaded_video is not None:

#         uploaded_video.seek(0)
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         tfile.write(uploaded_video.read())
#         tfile.flush()
#         tfile.close()

#         with st.spinner("🔍 Analysing video..."):
#             cap          = cv2.VideoCapture(tfile.name)
#             fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#             sample_frames = []
#             for pt in [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pt))
#                 ret, frm = cap.read()
#                 if ret:
#                     sample_frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

#             cloud_type, avg_conf = predict_cloud_type(sample_frames)

#             frame_gap   = max(1, int(fps))
#             delta_t_sec = frame_gap / fps
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0);         ret1, f1 = cap.read()
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_gap); ret2, f2 = cap.read()
#             cap.release()

#         if ret1 and ret2:
#             fw = f1.shape[1]
#             g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
#             g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
#             pixel_disp, avg_angle = compute_optical_flow(g1, g2)
#             direction = angle_to_direction(np.degrees(avg_angle))

#             speed_mps, speed_kmh, deg_per_px, theta_deg, distance_m, height_m = \
#                 pixels_to_kmh(pixel_disp, delta_t_sec, cloud_type, fw, fov_video)
#             pixel_speed = pixel_disp / delta_t_sec
#             dist_5  = speed_kmh * (5  / 60)
#             dist_15 = speed_kmh * (15 / 60)

#             # Density & Visibility compute
#             sample_bgr = cv2.cvtColor(sample_frames[len(sample_frames)//2], cv2.COLOR_RGB2BGR)
#             sky_h_sample = int(sample_bgr.shape[0] * 0.78)
#             cov_pct, den_label, den_color = compute_cloud_density(sample_bgr, sky_h_sample)
#             vis_verdict, vis_reason, vis_color, time_to_exit_min = predict_visibility(
#                 cloud_type, speed_kmh, direction, cov_pct, fov_video)

#             # ── Timestamp resolution (priority: manual > ffprobe > now) ──
#             _manual_utc = st.session_state.get("manual_utc")
#             if _manual_utc is not None:
#                 media_ts      = _manual_utc
#                 ts_source     = "manual"
#             else:
#                 media_ts = extract_video_datetime(tfile.name)
#                 ts_source = "ffprobe" if media_ts is not None else "now"
#                 if media_ts is None:
#                     media_ts = datetime.datetime.utcnow()

#             # ── Image-based sun elevation (for frames where sun not visible) ──
#             img_el_est, img_el_conf, img_el_note = estimate_sun_elevation_from_image(sample_bgr)

#             # ── Sun detection from frame (if sun is visible in sky) ──
#             sun_x, sun_y, sun_visible = detect_sun_in_frame(sample_bgr)
#             sun_az_from_frame = None
#             if sun_visible:
#                 sun_az_from_frame = sun_pixel_to_azimuth(sun_x, sample_bgr.shape[1], fov_video)
#                 img_el_note = (f"☀️ Sun detected in frame at pixel ({sun_x}, {sun_y}). "
#                                f"Estimated azimuth from camera: {sun_az_from_frame:.1f}°")

#             show_metrics(cloud_type, avg_conf, direction, height_m, fov_video,
#                          fw, pixel_disp, delta_t_sec, deg_per_px, theta_deg,
#                          distance_m, speed_mps, speed_kmh, dist_5, dist_15,
#                          coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
#                          vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
#                          time_to_exit_min=time_to_exit_min,
#                          media_timestamp=media_ts, timestamp_source=ts_source,
#                          image_elevation_est=img_el_est, image_elevation_conf=img_el_conf,
#                          image_elevation_note=img_el_note,
#                          plant2_dist_km=st.session_state.get("plant2_dist_km", 20.0),
#                          plant2_bearing_deg=st.session_state.get("plant2_bearing_deg", 270.0))
            
#                         # ── Detection video — shown RIGHT HERE under uploader ──
#             st.markdown('<div class="cv-eyebrow">📦 Cloud Detection Video</div>', unsafe_allow_html=True)
#             with st.spinner("Generating detection video…"):
#                 tmp_box = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 tmp_box.close()
#                 generate_boxed_video(
#                     tfile.name, tmp_box.name,
#                     speed_kmh, speed_mps, direction,
#                     cloud_type, height_m, dist_5, dist_15, fov=fov_video,
#                     time_to_exit_min=999
#                 )
#                 with open(tmp_box.name, "rb") as f:
#                     vdata = f.read()
#             st.video(vdata)
#             st.download_button("📥 Download Detection Video", data=vdata,
#                                file_name=f"cloud_{cloud_type}_boxes.mp4",
#                                mime="video/mp4", key="dl_box")
#             try: os.unlink(tmp_box.name)
#             except: pass

#             st.divider()
#             st.markdown('<div class="cv-eyebrow">📊 Analysis Results</div>', unsafe_allow_html=True)


#             # ── Prediction video ──
#             st.divider()
#             st.markdown('<div class="cv-eyebrow">🔮 Motion Prediction Video</div>', unsafe_allow_html=True)
#             with st.spinner("Generating prediction video…"):
#                 viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
#                                             direction=direction, pixel_speed=pixel_speed)
#                 tmp_pred = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 tmp_pred.close()
#                 viz.save_video_with_prediction(tmp_pred.name, prediction_minutes=15)
#                 # Re-encode to H.264 for browser playback
#                 if shutil.which("ffmpeg"):
#                     tmp_h264 = tmp_pred.name.replace(".mp4", "_h264.mp4")
#                     subprocess.run([
#                         "ffmpeg", "-y", "-i", tmp_pred.name,
#                         "-vcodec", "libx264", "-crf", "23",
#                         "-preset", "fast", "-pix_fmt", "yuv420p",
#                         "-movflags", "+faststart", tmp_h264
#                     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                     if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
#                         os.replace(tmp_h264, tmp_pred.name)
#                 with open(tmp_pred.name, "rb") as f:
#                     vdata2 = f.read()
#             st.video(vdata2)
#             st.download_button("📥 Download Prediction Video", data=vdata2,
#                                file_name=f"cloud_{cloud_type}_prediction.mp4",
#                                mime="video/mp4", key="dl_pred")
#             try: os.unlink(tmp_pred.name)
#             except: pass

#         try: os.unlink(tfile.name)
#         except: pass

# # ══════════════════════ MULTI IMAGE TAB ════════════════════════
# with tab2:
#     st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🖼️ Image Analysis</div>', unsafe_allow_html=True)
#     st.subheader("Upload Sky Images")

#     uploaded_images = st.file_uploader("Upload 2 or more images taken at a fixed time interval",
#                                         type=["jpg","jpeg","png"],
#                                         accept_multiple_files=True, key="img_upload")
#     interval   = st.number_input("⏱️ Time Between Images (seconds)", min_value=1, value=60)
#     fov_images = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
#                            help="Phone: 70-80° | Wide angle: 90-120°", key="fov_images")

#     if uploaded_images:
#         st.success(f"{len(uploaded_images)} image(s) uploaded.")

#         with st.expander("🖼️ Preview Uploaded Images"):
#             img_cols = st.columns(min(len(uploaded_images), 4))
#             for i, img_file in enumerate(uploaded_images):
#                 img_file.seek(0)
#                 with img_cols[i % 4]:
#                     st.image(img_file, caption=img_file.name, use_container_width=True)

#         if len(uploaded_images) >= 2:
#             with st.spinner("🔍 Analysing images..."):
#                 dirs_deg, px_disps = [], []
#                 for i in range(len(uploaded_images) - 1):
#                     uploaded_images[i].seek(0); uploaded_images[i+1].seek(0)
#                     img1 = np.array(Image.open(uploaded_images[i]).convert("RGB").resize((640,480)))
#                     img2 = np.array(Image.open(uploaded_images[i+1]).convert("RGB").resize((640,480)))
#                     med, ang = compute_optical_flow(
#                         cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
#                         cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
#                     )
#                     px_disps.append(med); dirs_deg.append(np.degrees(ang))

#                 avg_disp    = float(np.mean(px_disps))
#                 direction   = angle_to_direction(float(np.mean(dirs_deg)))
#                 pil_imgs    = []
#                 for f in uploaded_images:
#                     f.seek(0); pil_imgs.append(Image.open(f))
#                 cloud_type, avg_conf = predict_cloud_type(pil_imgs)

#             fw = 640
#             speed_mps, speed_kmh, deg_per_px, theta_deg, distance_m, height_m = \
#                 pixels_to_kmh(avg_disp, interval, cloud_type, fw, fov_images)
#             pixel_speed = avg_disp / interval
#             dist_5  = speed_kmh * (5  / 60)
#             dist_15 = speed_kmh * (15 / 60)

#             st.divider()
#             st.markdown('<div class="cv-eyebrow">📊 Analysis Results</div>', unsafe_allow_html=True)

#             # Density & Visibility compute from first image
#             uploaded_images[0].seek(0)
#             first_bgr = cv2.resize(
#                 cv2.cvtColor(np.array(Image.open(uploaded_images[0]).convert("RGB")), cv2.COLOR_RGB2BGR),
#                 (640, 480)
#             )
#             sky_h_img = int(first_bgr.shape[0] * 0.78)
#             cov_pct, den_label, den_color = compute_cloud_density(first_bgr, sky_h_img)
#             vis_verdict, vis_reason, vis_color, time_to_exit_min = predict_visibility(
#                 cloud_type, speed_kmh, direction, cov_pct, fov_images)

#             # ── Timestamp resolution for images (manual > EXIF > now) ──
#             _manual_utc_i = st.session_state.get("manual_utc")
#             if _manual_utc_i is not None:
#                 media_ts_i  = _manual_utc_i
#                 ts_source_i = "manual"
#             else:
#                 uploaded_images[0].seek(0)
#                 _pil_first = Image.open(uploaded_images[0])
#                 media_ts_i = extract_exif_datetime(_pil_first)
#                 ts_source_i = "exif" if media_ts_i is not None else "now"
#                 if media_ts_i is None:
#                     media_ts_i = datetime.datetime.utcnow()

#             # ── Image-based elevation estimate from first uploaded image ──
#             img_el_est_i, img_el_conf_i, img_el_note_i = estimate_sun_elevation_from_image(first_bgr)

#             # ── Sun detection from image (if sun is visible in sky) ──
#             sun_x_i, sun_y_i, sun_visible_i = detect_sun_in_frame(first_bgr)
#             sun_az_from_img = None
#             if sun_visible_i:
#                 sun_az_from_img = sun_pixel_to_azimuth(sun_x_i, first_bgr.shape[1], fov_images)
#                 img_el_note_i = (f"☀️ Sun detected in image at pixel ({sun_x_i}, {sun_y_i}). "
#                                  f"Estimated azimuth from camera: {sun_az_from_img:.1f}°")

#             show_metrics(cloud_type, avg_conf, direction, height_m, fov_images,
#                          fw, avg_disp, interval, deg_per_px, theta_deg,
#                          distance_m, speed_mps, speed_kmh, dist_5, dist_15,
#                          coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
#                          vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
#                          time_to_exit_min=time_to_exit_min,
#                          media_timestamp=media_ts_i, timestamp_source=ts_source_i,
#                          image_elevation_est=img_el_est_i, image_elevation_conf=img_el_conf_i,
#                          image_elevation_note=img_el_note_i,
#                          plant2_dist_km=st.session_state.get("plant2_dist_km", 20.0),
#                          plant2_bearing_deg=st.session_state.get("plant2_bearing_deg", 270.0))

#             st.divider()
#             st.markdown('<div class="cv-eyebrow">🎬 Export</div>', unsafe_allow_html=True)
#             st.subheader("Generate Output")

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.markdown("**📦 Cloud Detection on Images**")
#                 st.caption("Bounding boxes with speed and depth overlay on each uploaded image")
#                 if st.button("Show Detection Boxes", key="img_box"):
#                     cols3 = st.columns(min(len(uploaded_images), 3))
#                     for i, img_file in enumerate(uploaded_images):
#                         img_file.seek(0)
#                         arr = cv2.resize(
#                             cv2.cvtColor(np.array(Image.open(img_file).convert("RGB")),
#                                          cv2.COLOR_RGB2BGR), (640, 480)
#                         )
#                         prev_arr = None
#                         if i > 0:
#                             uploaded_images[i-1].seek(0)
#                             prev_arr_bgr = cv2.resize(
#                                 cv2.cvtColor(np.array(Image.open(uploaded_images[i-1]).convert("RGB")),
#                                              cv2.COLOR_RGB2BGR), (640, 480)
#                             )
#                             prev_arr = cv2.cvtColor(prev_arr_bgr, cv2.COLOR_BGR2GRAY)
#                         arr = draw_boxes_on_frame(arr, speed_kmh, direction, cloud_type,
#                                                    height_m, dist_5, dist_15, i * interval,
#                                                    prev_gray=prev_arr, delta_t=float(interval),
#                                                    fov=fov_images,
#                                                    time_to_exit_min=time_to_exit_min)
#                         with cols3[i % 3]:
#                             st.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),
#                                      caption=f"Image {i+1}", use_container_width=True)

#             with col2:
#                 st.markdown("**🔮 Motion Prediction Video**")
#                 st.caption("Simulated animation — +5 min and +15 min forecast")
#                 if st.button("Generate Prediction Video", key="img_pred"):
#                     with st.spinner("Simulating cloud motion…"):
#                         viz = CloudMotionVisualizer(cloud_type=cloud_type, height_m=height_m,
#                                                     direction=direction, pixel_speed=pixel_speed)
#                         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                         tmp.close()
#                         viz.save_video_with_prediction(tmp.name, prediction_minutes=15)
#                         if shutil.which("ffmpeg"):
#                             tmp_h264 = tmp.name.replace(".mp4", "_h264.mp4")
#                             subprocess.run([
#                                 "ffmpeg", "-y", "-i", tmp.name,
#                                 "-vcodec", "libx264", "-crf", "23",
#                                 "-preset", "fast", "-pix_fmt", "yuv420p",
#                                 "-movflags", "+faststart", tmp_h264
#                             ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                             if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
#                                 os.replace(tmp_h264, tmp.name)
#                         with open(tmp.name, "rb") as f:
#                             vdata = f.read()
#                         st.success("Prediction video ready.")
#                         st.video(vdata)
#                         st.download_button("📥 Download Prediction Video", data=vdata,
#                                            file_name=f"cloud_{cloud_type}_prediction.mp4",
#                                            mime="video/mp4")
#                         try: os.unlink(tmp.name)
#                         except: pass
#         else:
#             st.warning("Upload at least 2 images to run analysis.")

# # ══════════════════════ STEREO DUAL CAMERA TAB (Tab 3) ════════════════════════
# with tab3:
#     st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">📸 Stereo Cloud Base Height (CBH) Analysis</div>', unsafe_allow_html=True)
#     st.subheader("Synchronized Dual Camera Analysis - 2 km Baseline")
    
#     st.info("""
#     🌞 **Workflow:**
#     1. Upload synchronized images from Camera 1 & Camera 2 (same timestamp)
#     2. Detect clouds using binary threshold (pixels > 150 = cloud)
#     3. Mirror Camera 1 image for proper overlay alignment
#     4. Find CBH by overlaying masks at different heights (lowest error = CBH)
#     5. Calculate cloud speed from optical flow
#     6. Forecast solar plant impact
#     """)

#     col_c1, col_c2 = st.columns(2)
    
#     with col_c1:
#         st.markdown("**📷 Camera 1 (Position A)**")
#         camera1_t0 = st.file_uploader("Image T0 (12:00:00)", type=["jpg","jpeg","png"], key="cam1_t0")
#         camera1_t1 = st.file_uploader("Image T1 (12:01:00)", type=["jpg","jpeg","png"], key="cam1_t1")
    
#     with col_c2:
#         st.markdown("**📷 Camera 2 (Position B)**")
#         camera2_t0 = st.file_uploader("Image T0 (12:00:00)", type=["jpg","jpeg","png"], key="cam2_t0")
#         camera2_t1 = st.file_uploader("Image T1 (12:01:00)", type=["jpg","jpeg","png"], key="cam2_t1")

#     if camera1_t0 and camera2_t0 and camera1_t1 and camera2_t1:
#         st.success("✅ All 4 images uploaded")
        
#         baseline_km = st.number_input("📏 Baseline Distance (km)", min_value=0.1, value=2.0, step=0.1,
#                                       help="Distance between Camera 1 and Camera 2")
#         fov_deg = st.slider("📷 Camera FOV (degrees)", 30, 120, 70, key="stereo_fov")
#         threshold = st.slider("☁️ Cloud Threshold (pixel intensity)", 100, 200, 150,
#                              help="Pixels above threshold = cloud")
#         dist_to_plant_m = st.number_input("📍 Distance from cameras to solar plant (meters)", 
#                                           min_value=100, value=712, step=100)

#         # Preview section
#         with st.expander("🔍 Preview Images"):
#             prev_cols = st.columns(2)
            
#             with prev_cols[0]:
#                 st.markdown("**Camera 1**")
#                 c1_row = st.columns(2)
#                 with c1_row[0]:
#                     camera1_t0.seek(0)
#                     st.image(camera1_t0, caption="T0 (12:00:00)", use_container_width=True)
#                 with c1_row[1]:
#                     camera1_t1.seek(0)
#                     st.image(camera1_t1, caption="T1 (12:01:00)", use_container_width=True)
            
#             with prev_cols[1]:
#                 st.markdown("**Camera 2**")
#                 c2_row = st.columns(2)
#                 with c2_row[0]:
#                     camera2_t0.seek(0)
#                     st.image(camera2_t0, caption="T0 (12:00:00)", use_container_width=True)
#                 with c2_row[1]:
#                     camera2_t1.seek(0)
#                     st.image(camera2_t1, caption="T1 (12:01:00)", use_container_width=True)

#         if st.button("🔬 Run CBH Analysis Workflow", key="stereo_analyze"):
#             with st.spinner("Processing stereo analysis..."):
#                 # Load images
#                 camera1_t0.seek(0)
#                 camera2_t0.seek(0)
#                 camera1_t1.seek(0)
#                 camera2_t1.seek(0)
                
#                 img_c1_t0 = cv2.cvtColor(np.array(Image.open(camera1_t0).convert("RGB")), cv2.COLOR_RGB2BGR)
#                 img_c2_t0 = cv2.cvtColor(np.array(Image.open(camera2_t0).convert("RGB")), cv2.COLOR_RGB2BGR)
#                 img_c1_t1 = cv2.cvtColor(np.array(Image.open(camera1_t1).convert("RGB")), cv2.COLOR_RGB2BGR)
#                 img_c2_t1 = cv2.cvtColor(np.array(Image.open(camera2_t1).convert("RGB")), cv2.COLOR_RGB2BGR)
                
#                 # Resize to standard size
#                 h, w = 480, 640
#                 img_c1_t0 = cv2.resize(img_c1_t0, (w, h))
#                 img_c2_t0 = cv2.resize(img_c2_t0, (w, h))
#                 img_c1_t1 = cv2.resize(img_c1_t1, (w, h))
#                 img_c2_t1 = cv2.resize(img_c2_t1, (w, h))
                
#                 st.divider()
#                 st.markdown('<div class="cv-eyebrow">STEP 1-2: Cloud Detection</div>', unsafe_allow_html=True)
                
#                 # STEP 2: Cloud Detection (Binary masks)
#                 mask_c1_t0 = detect_clouds_binary(img_c1_t0, threshold=threshold)
#                 mask_c2_t0 = detect_clouds_binary(img_c2_t0, threshold=threshold)
                
#                 # Display binary masks
#                 mask_cols = st.columns(2)
#                 with mask_cols[0]:
#                     st.image(mask_c1_t0, caption="Camera 1 - Binary Cloud Mask (T0)", use_container_width=True)
#                 with mask_cols[1]:
#                     st.image(mask_c2_t0, caption="Camera 2 - Binary Cloud Mask (T0)", use_container_width=True)
                
#                 st.divider()
#                 st.markdown('<div class="cv-eyebrow">STEP 3: Mirror Images for Alignment</div>', unsafe_allow_html=True)
                
#                 # STEP 3: Mirror Camera 1 mask
#                 mask_c1_t0_mirrored = mirror_cloud_mask(mask_c1_t0)
                
#                 mirror_cols = st.columns(2)
#                 with mirror_cols[0]:
#                     st.image(mask_c1_t0, caption="Camera 1 - Original Mask", use_container_width=True)
#                 with mirror_cols[1]:
#                     st.image(mask_c1_t0_mirrored, caption="Camera 1 - Mirrored Mask", use_container_width=True)
                
#                 st.divider()
#                 st.markdown('<div class="cv-eyebrow">STEP 4: Cloud Base Height (CBH) Calculation</div>', unsafe_allow_html=True)
                
#                 # STEP 4: Calculate CBH by overlay
#                 with st.spinner("Testing heights 0m → 12000m..."):
#                     cbh_m, error_by_height = calculate_cbh_by_overlay(
#                         mask_c1_t0_mirrored, mask_c2_t0,
#                         baseline_km=baseline_km,
#                         fov_deg=fov_deg,
#                         frame_width=w,
#                         height_range=(0, 12000, 100)
#                     )
                
#                 st.success(f"✅ **Cloud Base Height (CBH): {cbh_m:.0f} meters**")
                
#                 # Visualize error curve
#                 cbh_viz_data = generate_cbh_visualization(error_by_height)
#                 st.line_chart(cbh_viz_data)
#                 st.caption(f"Height with lowest matching error = {cbh_m}m (optimal match between cameras)")
                
#                 st.divider()
#                 st.markdown('<div class="cv-eyebrow">STEP 5: Cloud Speed Calculation</div>', unsafe_allow_html=True)
                
#                 # STEP 5: Calculate speed from optical flow
#                 gray_c1_t0 = cv2.cvtColor(img_c1_t0, cv2.COLOR_BGR2GRAY)
#                 gray_c1_t1 = cv2.cvtColor(img_c1_t1, cv2.COLOR_BGR2GRAY)
                
#                 speed_kmh, speed_mps, disp_px, direction_deg = calculate_cloud_speed_optical_flow(
#                     gray_c1_t0, gray_c1_t1,
#                     cbh_m=cbh_m,
#                     fov_deg=fov_deg,
#                     frame_width=w,
#                     time_delta_sec=60  # 1 minute between T0 and T1
#                 )
                
#                 direction_compass = angle_to_direction(direction_deg)
                
#                 speed_cols = st.columns(3)
#                 with speed_cols[0]:
#                     st.metric("Speed", f"{speed_kmh:.2f} km/h", f"{speed_mps:.2f} m/s")
#                 with speed_cols[1]:
#                     st.metric("Direction", direction_compass, f"{direction_deg:.1f}°")
#                 with speed_cols[2]:
#                     st.metric("Pixel Displacement", f"{disp_px:.2f} px", "in 60 sec")
                
#                 st.divider()
#                 st.markdown('<div class="cv-eyebrow">STEP 6: Solar Plant Impact Forecast</div>', unsafe_allow_html=True)
                
#                 # STEP 6: Forecast solar impact
#                 will_impact, time_to_impact_min, forecast_msg = forecast_solar_impact(
#                     cbh_m, speed_kmh, direction_deg, dist_to_plant_m
#                 )
                
#                 # Impact status
#                 if will_impact and time_to_impact_min < 5:
#                     st.error(forecast_msg)
#                 elif will_impact and time_to_impact_min < 15:
#                     st.warning(forecast_msg)
#                 else:
#                     st.info(forecast_msg)
                
#                 st.metric("Time to Solar Plant", f"{time_to_impact_min:.1f} minutes", 
#                          f"At {speed_kmh:.2f} km/h toward {dist_to_plant_m}m")
                
#                 st.divider()
#                 st.markdown('<div class="cv-eyebrow">📊 FINAL RESULTS SUMMARY</div>', unsafe_allow_html=True)
                
#                 # Final metrics
#                 result_cols = st.columns(4)
#                 with result_cols[0]:
#                     st.metric("CBH", f"{cbh_m:.0f} m", "Cloud base height")
#                 with result_cols[1]:
#                     st.metric("Speed", f"{speed_kmh:.1f} km/h", "Cloud velocity")
#                 with result_cols[2]:
#                     st.metric("Direction", direction_compass, "Compass bearing")
#                 with result_cols[3]:
#                     impact_status = "🔴 YES" if will_impact else "🟢 NO"
#                     st.metric("Will Shadow Plant?", impact_status, f"in {time_to_impact_min:.0f} min")
#     else:
#         st.info("📤 Upload all 4 images: Camera 1 (T0 & T1) + Camera 2 (T0 & T1)")



# # ══════════════════════ LIVE VIDEO DUAL CAMERA TAB (Tab 4) ════════════════════════
# with tab4:
#     st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🎥 Live Cloud Tracking (Camera 1 & 2)</div>', unsafe_allow_html=True)
#     st.subheader("Real-time Cloud Direction, Speed & Height")
#     st.info("📹 Upload video feeds from both cameras to track cloud motion in real-time with speed, direction, and height estimation.")

#     live_cols = st.columns([1, 1])
    
#     with live_cols[0]:
#         st.markdown("**🎥 Video 1 (Camera 1)**")
#         video1 = st.file_uploader("Upload video from Camera 1", type=["mp4", "mov", "avi"], key="video1_live")
    
#     with live_cols[1]:
#         st.markdown("**🎥 Video 2 (Camera 2)**")
#         video2 = st.file_uploader("Upload video from Camera 2", type=["mp4", "mov", "avi"], key="video2_live")

#     if video1 and video2:
#         st.success("✅ Both videos uploaded")
        
#         live_fov = st.slider("📷 Camera FOV (degrees)", 30, 120, 70, key="live_fov")
#         frame_interval = st.number_input("⏱️ Process every N frames", min_value=1, value=5, step=1,
#                                         help="Process every Nth frame to speed up analysis")
#         time_frame_input = st.number_input("⏳ Time Frame for Analysis (seconds)", min_value=1, value=30, step=1,
#                                           help="Analyze first N seconds of video")

#         if st.button("▶️ Start Live Analysis", key="live_analyze"):
#             with st.spinner("🔄 Processing video streams..."):
#                 # Read Video 1
#                 cap1 = cv2.VideoCapture(video1.name if hasattr(video1, 'name') else video1)
#                 cap2 = cv2.VideoCapture(video2.name if hasattr(video2, 'name') else video2)
                
#                 # Get video properties
#                 fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30
#                 fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30
                
#                 max_frames_1 = int(time_frame_input * fps1)
#                 max_frames_2 = int(time_frame_input * fps2)
                
#                 frame_count_1 = 0
#                 frame_count_2 = 0
#                 all_dirs_1, all_speeds_1, all_heights_1 = [], [], []
#                 all_dirs_2, all_speeds_2, all_heights_2 = [], [], []
#                 sample_frames_1, sample_frames_2 = [], []
                
#                 # Extract frames from Video 1
#                 prev_frame_1 = None
#                 while frame_count_1 < max_frames_1:
#                     ret1, frame1 = cap1.read()
#                     if not ret1:
#                         break
                    
#                     if frame_count_1 % frame_interval == 0:
#                         frame1_resized = cv2.resize(frame1, (640, 480))
#                         gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
                        
#                         if prev_frame_1 is not None:
#                             med, ang = compute_optical_flow(prev_frame_1, gray1)
#                             cloud_type_live_1 = "Cumulus"  # Default, can refine
#                             speed_mps_1, speed_kmh_1, _, _, _, height_1 = \
#                                 pixels_to_kmh(med, frame_interval / fps1, cloud_type_live_1, 640, live_fov)
                            
#                             all_dirs_1.append(np.degrees(ang))
#                             all_speeds_1.append(speed_kmh_1)
#                             all_heights_1.append(height_1)
                        
#                         prev_frame_1 = gray1
#                         sample_frames_1.append(frame1_resized)
                    
#                     frame_count_1 += 1
                
#                 # Extract frames from Video 2
#                 prev_frame_2 = None
#                 while frame_count_2 < max_frames_2:
#                     ret2, frame2 = cap2.read()
#                     if not ret2:
#                         break
                    
#                     if frame_count_2 % frame_interval == 0:
#                         frame2_resized = cv2.resize(frame2, (640, 480))
#                         gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
                        
#                         if prev_frame_2 is not None:
#                             med, ang = compute_optical_flow(prev_frame_2, gray2)
#                             cloud_type_live_2 = "Cumulus"  # Default
#                             speed_mps_2, speed_kmh_2, _, _, _, height_2 = \
#                                 pixels_to_kmh(med, frame_interval / fps2, cloud_type_live_2, 640, live_fov)
                            
#                             all_dirs_2.append(np.degrees(ang))
#                             all_speeds_2.append(speed_kmh_2)
#                             all_heights_2.append(height_2)
                        
#                         prev_frame_2 = gray2
#                         sample_frames_2.append(frame2_resized)
                    
#                     frame_count_2 += 1
                
#                 cap1.release()
#                 cap2.release()
                
#                 # Compute statistics
#                 if all_dirs_1 and all_dirs_2:
#                     avg_dir_1 = angle_to_direction(np.mean(all_dirs_1))
#                     avg_dir_2 = angle_to_direction(np.mean(all_dirs_2))
#                     avg_speed_1 = np.mean(all_speeds_1) if all_speeds_1 else 0
#                     avg_speed_2 = np.mean(all_speeds_2) if all_speeds_2 else 0
#                     avg_height_1 = np.mean(all_heights_1) if all_heights_1 else 500
#                     avg_height_2 = np.mean(all_heights_2) if all_heights_2 else 500
                    
#                     st.divider()
#                     st.markdown('<div class="cv-eyebrow">📊 Live Tracking Results</div>', unsafe_allow_html=True)
                    
#                     live_res_cols = st.columns(2)
                    
#                     with live_res_cols[0]:
#                         st.markdown("### 🎥 Video 1 (Camera 1)")
#                         st.metric("Avg Direction", avg_dir_1, f"{np.mean(all_dirs_1):.1f}°")
#                         st.metric("Avg Speed", f"{avg_speed_1:.2f} km/h", f"{avg_speed_1 * 0.278:.2f} m/s")
#                         st.metric("Est. Height", f"{avg_height_1:.0f} m")
#                         st.metric("Frames Analyzed", len(sample_frames_1))
                    
#                     with live_res_cols[1]:
#                         st.markdown("### 🎥 Video 2 (Camera 2)")
#                         st.metric("Avg Direction", avg_dir_2, f"{np.mean(all_dirs_2):.1f}°")
#                         st.metric("Avg Speed", f"{avg_speed_2:.2f} km/h", f"{avg_speed_2 * 0.278:.2f} m/s")
#                         st.metric("Est. Height", f"{avg_height_2:.0f} m")
#                         st.metric("Frames Analyzed", len(sample_frames_2))
                    
#                     # Sample frames with overlay
#                     st.divider()
#                     st.markdown('<div class="cv-eyebrow">📸 Sample Frames with Tracking</div>', unsafe_allow_html=True)
                    
#                     frame_disp_cols = st.columns(2)
                    
#                     with frame_disp_cols[0]:
#                         if sample_frames_1:
#                             sample_frame_1 = sample_frames_1[min(len(sample_frames_1)//2, len(sample_frames_1)-1)]
#                             arr_1 = draw_boxes_on_frame(sample_frame_1, avg_speed_1, avg_dir_1, "Cumulus",
#                                                         avg_height_1, 0, 0, 0, fov=live_fov)
#                             st.image(cv2.cvtColor(arr_1, cv2.COLOR_BGR2RGB), caption="Video 1 Tracking Sample",
#                                     use_container_width=True)
                    
#                     with frame_disp_cols[1]:
#                         if sample_frames_2:
#                             sample_frame_2 = sample_frames_2[min(len(sample_frames_2)//2, len(sample_frames_2)-1)]
#                             arr_2 = draw_boxes_on_frame(sample_frame_2, avg_speed_2, avg_dir_2, "Cumulus",
#                                                         avg_height_2, 0, 0, 0, fov=live_fov)
#                             st.image(cv2.cvtColor(arr_2, cv2.COLOR_BGR2RGB), caption="Video 2 Tracking Sample",
#                                     use_container_width=True)
                    
#                     # Trends
#                     st.divider()
#                     st.markdown('<div class="cv-eyebrow">📈 Speed & Height Trends</div>', unsafe_allow_html=True)
                    
#                     trend_cols = st.columns(2)
                    
#                     with trend_cols[0]:
#                         st.write("**Speed Trend - Camera 1**")
#                         if all_speeds_1:
#                             trend_data_1 = {"Frame": range(len(all_speeds_1)), "Speed (km/h)": all_speeds_1}
#                             st.line_chart(trend_data_1)
                    
#                     with trend_cols[1]:
#                         st.write("**Speed Trend - Camera 2**")
#                         if all_speeds_2:
#                             trend_data_2 = {"Frame": range(len(all_speeds_2)), "Speed (km/h)": all_speeds_2}
#                             st.line_chart(trend_data_2)
#                 else:
#                     st.warning("⚠️ Could not extract motion data from videos. Check video format and content.")
#     else:
#         st.info("📹 Upload both video feeds to start live tracking")

# # ══════════════════════ LIVE IP CAMERA STREAMING TAB (Tab 5) ════════════════════════
# with tab5:
#     st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">📡 Live IP Camera Streaming - Real-time CBH & Speed</div>', unsafe_allow_html=True)
#     st.subheader("2 IP Cameras (Stereo Setup) - On-Demand Analysis")
    
#     st.info("""
#     🎥 **Real-time Analysis Features:**
#     • Connect to 2 IP cameras (RTSP/HTTP streams)
#     • Capture frames simultaneously
#     • Calculate CBH by overlay matching
#     • Track cloud speed from frame sequence
#     • Forecast solar plant impact in real-time
#     • Display live metrics dashboard
#     """)

#     # Camera configuration section
#     st.markdown("### 📸 IP Camera Configuration")
    
#     config_cols = st.columns([1, 1])
    
#     with config_cols[0]:
#         st.markdown("**Camera 1 Settings**")
#         camera1_url = st.text_input("Camera 1 URL (RTSP/HTTP)", 
#                                    value="rtsp://192.168.1.100:554/stream",
#                                    placeholder="rtsp://ip:port/stream",
#                                    key="cam1_url_live")
#         camera1_enabled = st.checkbox("Enable Camera 1", value=True, key="cam1_enabled")
    
#     with config_cols[1]:
#         st.markdown("**Camera 2 Settings**")
#         camera2_url = st.text_input("Camera 2 URL (RTSP/HTTP)", 
#                                    value="rtsp://192.168.1.101:554/stream",
#                                    placeholder="rtsp://ip:port/stream",
#                                    key="cam2_url_live")
#         camera2_enabled = st.checkbox("Enable Camera 2", value=True, key="cam2_enabled")

#     st.divider()
#     st.markdown("### ⚙️ Analysis Parameters")
    
#     param_cols = st.columns(4)
    
#     with param_cols[0]:
#         baseline_km_live = st.number_input("📏 Baseline (km)", min_value=0.1, value=2.0, step=0.1, key="baseline_live")
    
#     with param_cols[1]:
#         fov_deg_live = st.slider("📷 FOV (°)", 30, 120, 70, key="fov_live")
    
#     with param_cols[2]:
#         threshold_live = st.slider("☁️ Threshold", 100, 200, 150, key="threshold_live")
    
#     with param_cols[3]:
#         dist_to_plant_live = st.number_input("📍 Plant Distance (m)", min_value=100, value=712, step=100, key="dist_plant_live")

#     st.divider()
    
#     # Live streaming and analysis section
#     if camera1_enabled and camera2_enabled:
#         if st.button("▶️ START LIVE ANALYSIS", key="start_live_streaming"):
#             st.markdown('<div class="cv-eyebrow" style="margin-top:20px;">🔴 LIVE STREAM STATUS</div>', unsafe_allow_html=True)
            
#             # Create placeholder for live metrics
#             metric_placeholder = st.empty()
#             frame_placeholder_c1 = st.empty()
#             frame_placeholder_c2 = st.empty()
#             info_placeholder = st.empty()
            
#             try:
#                 # Open camera streams
#                 cap1 = cv2.VideoCapture(camera1_url)
#                 cap2 = cv2.VideoCapture(camera2_url)
                
#                 # Check if cameras opened successfully
#                 if not cap1.isOpened() or not cap2.isOpened():
#                     st.error("❌ Cannot connect to cameras. Check URLs and network connectivity.")
#                     st.write("**Debug Info:**")
#                     st.code(f"Camera 1: {camera1_url}\nCamera 2: {camera2_url}")
#                 else:
#                     st.success("✅ Connected to both IP cameras")
                    
#                     # Frame buffers for analysis
#                     frame_buffer_c1 = []
#                     frame_buffer_c2 = []
#                     max_buffer_size = 3  # Keep last 3 frames for motion analysis
                    
#                     # Storage for metrics history
#                     metrics_history = {
#                         "timestamp": [],
#                         "cbh": [],
#                         "speed_kmh": [],
#                         "direction": [],
#                         "will_impact": []
#                     }
                    
#                     analysis_count = 0
#                     max_analyses = 5  # Run analysis 5 times then stop
                    
#                     while analysis_count < max_analyses and st.session_state.get("live_streaming_active", True):
#                         # Capture frames from both cameras
#                         ret1, frame1 = cap1.read()
#                         ret2, frame2 = cap2.read()
                        
#                         if not ret1 or not ret2:
#                             st.warning("⚠️ Lost connection to one or both cameras")
#                             break
                        
#                         # Resize frames
#                         h, w = 480, 640
#                         frame1 = cv2.resize(frame1, (w, h))
#                         frame2 = cv2.resize(frame2, (w, h))
                        
#                         # Store frames in buffer
#                         frame_buffer_c1.append(frame1)
#                         frame_buffer_c2.append(frame2)
                        
#                         if len(frame_buffer_c1) > max_buffer_size:
#                             frame_buffer_c1.pop(0)
#                         if len(frame_buffer_c2) > max_buffer_size:
#                             frame_buffer_c2.pop(0)
                        
#                         # Display live frames
#                         with frame_placeholder_c1:
#                             st.markdown("**Live Feed - Camera 1**")
#                             st.image(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
#                         with frame_placeholder_c2:
#                             st.markdown("**Live Feed - Camera 2**")
#                             st.image(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
#                         # Run analysis every 3 frames (to allow buffer to fill)
#                         if len(frame_buffer_c1) >= 2 and len(frame_buffer_c2) >= 2:
#                             analysis_count += 1
                            
#                             with info_placeholder.container():
#                                 with st.spinner(f"⚙️ Analysis {analysis_count}/5..."):
#                                     # Get T0 and T1 frames
#                                     frame_c1_t0 = frame_buffer_c1[-2]
#                                     frame_c1_t1 = frame_buffer_c1[-1]
#                                     frame_c2_t0 = frame_buffer_c2[-2]
#                                     frame_c2_t1 = frame_buffer_c2[-1]
                                    
#                                     # STEP 2: Cloud Detection
#                                     mask_c1_t0 = detect_clouds_binary(frame_c1_t0, threshold=threshold_live)
#                                     mask_c2_t0 = detect_clouds_binary(frame_c2_t0, threshold=threshold_live)
                                    
#                                     # STEP 3: Mirror
#                                     mask_c1_t0_mirrored = mirror_cloud_mask(mask_c1_t0)
                                    
#                                     # STEP 4: CBH Calculation
#                                     cbh_m_live, error_by_height_live = calculate_cbh_by_overlay(
#                                         mask_c1_t0_mirrored, mask_c2_t0,
#                                         baseline_km=baseline_km_live,
#                                         fov_deg=fov_deg_live,
#                                         frame_width=w,
#                                         height_range=(0, 12000, 200)  # Faster: 200m steps
#                                     )
                                    
#                                     # STEP 5: Speed Calculation
#                                     gray_c1_t0 = cv2.cvtColor(frame_c1_t0, cv2.COLOR_BGR2GRAY)
#                                     gray_c1_t1 = cv2.cvtColor(frame_c1_t1, cv2.COLOR_BGR2GRAY)
                                    
#                                     speed_kmh_live, speed_mps_live, disp_px_live, direction_deg_live = calculate_cloud_speed_optical_flow(
#                                         gray_c1_t0, gray_c1_t1,
#                                         cbh_m=cbh_m_live,
#                                         fov_deg=fov_deg_live,
#                                         frame_width=w,
#                                         time_delta_sec=0.033  # ~30ms between frames at 30fps
#                                     )
                                    
#                                     direction_compass_live = angle_to_direction(direction_deg_live)
                                    
#                                     # STEP 6: Solar Impact Forecast
#                                     will_impact_live, time_to_impact_live, forecast_live = forecast_solar_impact(
#                                         cbh_m_live, speed_kmh_live, direction_deg_live, dist_to_plant_live
#                                     )
                                    
#                                     # Store in history
#                                     metrics_history["timestamp"].append(analysis_count)
#                                     metrics_history["cbh"].append(cbh_m_live)
#                                     metrics_history["speed_kmh"].append(speed_kmh_live)
#                                     metrics_history["direction"].append(direction_deg_live)
#                                     metrics_history["will_impact"].append(will_impact_live)
                                    
#                                     # Display metrics
#                                     with metric_placeholder.container():
#                                         st.markdown('<div class="cv-eyebrow">📊 LIVE METRICS - Analysis #{}</div>'.format(analysis_count), unsafe_allow_html=True)
                                        
#                                         live_metric_cols = st.columns(5)
                                        
#                                         with live_metric_cols[0]:
#                                             st.metric("CBH", f"{cbh_m_live:.0f} m", f"±{min(error_by_height_live.values())*100:.1f}%")
                                        
#                                         with live_metric_cols[1]:
#                                             st.metric("Speed", f"{speed_kmh_live:.2f} km/h", f"{speed_mps_live:.2f} m/s")
                                        
#                                         with live_metric_cols[2]:
#                                             st.metric("Direction", direction_compass_live, f"{direction_deg_live:.1f}°")
                                        
#                                         with live_metric_cols[3]:
#                                             impact_icon = "🔴" if will_impact_live else "🟢"
#                                             st.metric("Impact", f"{impact_icon} {time_to_impact_live:.1f}m", "to plant")
                                        
#                                         with live_metric_cols[4]:
#                                             st.metric("Status", "ACTIVE", "streaming")
                    
#                     cap1.release()
#                     cap2.release()
                    
#                     st.divider()
#                     st.markdown('<div class="cv-eyebrow">📈 ANALYSIS SUMMARY</div>', unsafe_allow_html=True)
                    
#                     if metrics_history["cbh"]:
#                         summary_cols = st.columns(4)
                        
#                         with summary_cols[0]:
#                             avg_cbh = np.mean(metrics_history["cbh"])
#                             st.metric("Avg CBH", f"{avg_cbh:.0f} m", f"min: {min(metrics_history['cbh']):.0f}m, max: {max(metrics_history['cbh']):.0f}m")
                        
#                         with summary_cols[1]:
#                             avg_speed = np.mean(metrics_history["speed_kmh"])
#                             st.metric("Avg Speed", f"{avg_speed:.2f} km/h", f"min: {min(metrics_history['speed_kmh']):.1f}, max: {max(metrics_history['speed_kmh']):.1f}")
                        
#                         with summary_cols[2]:
#                             avg_dir = np.mean(metrics_history["direction"])
#                             st.metric("Avg Direction", angle_to_direction(avg_dir), f"{avg_dir:.1f}°")
                        
#                         with summary_cols[3]:
#                             impact_count = sum(metrics_history["will_impact"])
#                             st.metric("Impact Events", impact_count, f"out of {len(metrics_history['will_impact'])}")
                        
#                         # Trends
#                         st.markdown('<div class="cv-eyebrow">📊 TRENDS</div>', unsafe_allow_html=True)
                        
#                         trend_cols = st.columns(2)
                        
#                         with trend_cols[0]:
#                             st.write("**CBH Trend**")
#                             trend_cbh_data = {"Analysis": metrics_history["timestamp"], "CBH (m)": metrics_history["cbh"]}
#                             st.line_chart(trend_cbh_data)
                        
#                         with trend_cols[1]:
#                             st.write("**Speed Trend**")
#                             trend_speed_data = {"Analysis": metrics_history["timestamp"], "Speed (km/h)": metrics_history["speed_kmh"]}
#                             st.line_chart(trend_speed_data)
                        
#                         st.success("✅ Live analysis completed successfully!")
            
#             except Exception as e:
#                 st.error(f"❌ Error during live streaming: {str(e)}")
#                 st.write("**Troubleshooting:**")
#                 st.write("""
#                 - Check if camera URLs are correct (RTSP/HTTP format)
#                 - Verify cameras are on same network
#                 - Check firewall settings
#                 - Try accessing camera directly in VLC: File → Open Network Stream
#                 """)
#     else:
#         st.warning("⚠️ Enable both cameras to start live analysis")
#         st.info("""
#         **How to get IP Camera URLs:**
        
#         1. **Hikvision/Dahua**: `rtsp://username:password@ip:554/stream1`
#         2. **Axis**: `rtsp://username:password@ip/axis-media/media.amp`
#         3. **Generic RTSP**: `rtsp://ip:554/stream`
#         4. **HTTP Stream**: `http://ip:8080/video` (check manual)
        
#         **Test URL with VLC:**
#         - File → Open Network Stream → Paste URL
#         """)

# # PATCH NOTE: Stereo CBH fix block not auto-located.