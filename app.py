import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import math
import subprocess
import shutil
import json
import urllib.parse
import urllib.request
import datetime
from collections import Counter
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from motion_visualizer import CloudMotionVisualizer

# ===== YOUTUBE VIDEO DOWNLOAD (for "paste a YouTube link" support) =====
# pip install yt-dlp
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Seconds to record from a YouTube LIVE stream (analysis only needs a short clip).
_YT_LIVE_RECORD_SECONDS = 60


def _resolve_ffmpeg_exe():
    """Path to ffmpeg on PATH, or the imageio-ffmpeg bundle if installed."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def encode_mp4_for_mobile(input_path, output_path=None, max_size_mb=64):
    """
    Re-encode OpenCV mp4v output to H.264 MP4 for WhatsApp / mobile sharing.

    WhatsApp requires H.264 (AVC) + yuv420p — NOT the default OpenCV 'mp4v' codec.
    Uses imageio-ffmpeg fallback when ffmpeg is not on PATH.
    Returns (final_path, success_bool).
    """
    ffmpeg_exe = _resolve_ffmpeg_exe()
    if not ffmpeg_exe or not os.path.isfile(input_path):
        return input_path, False

    replace_in_place = output_path is None or os.path.abspath(output_path) == os.path.abspath(input_path)
    tmp_out = input_path.replace(".mp4", "_h264.mp4") if replace_in_place else output_path

    def _run_encode(src, dst, crf="26"):
        return subprocess.run([
            ffmpeg_exe, "-y", "-i", src,
            "-an",
            "-vcodec", "libx264",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-tag:v", "avc1",
            "-crf", str(crf),
            "-preset", "fast",
            dst,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    result = _run_encode(input_path, tmp_out, crf="26")
    if result.returncode != 0 or not os.path.exists(tmp_out) or os.path.getsize(tmp_out) == 0:
        return input_path, False

    final = tmp_out
    size_mb = os.path.getsize(final) / (1024 * 1024)
    if size_mb > max_size_mb:
        smaller = final.replace(".mp4", "_small.mp4")
        if _run_encode(final, smaller, crf="32").returncode == 0 and os.path.getsize(smaller) > 0:
            try:
                os.remove(final)
            except OSError:
                pass
            final = smaller

    if replace_in_place:
        os.replace(final, input_path)
        return input_path, True
    if final != output_path:
        os.replace(final, output_path)
    return output_path, True


def _resolve_ffmpeg_dir_for_ytdlp():
    """
    yt-dlp expects a directory containing ffmpeg.exe (Windows). imageio-ffmpeg
    ships a differently named binary, so we copy it into a small cache folder.
    """
    exe = _resolve_ffmpeg_exe()
    if not exe:
        return None
    base = os.path.basename(exe).lower()
    if base in ("ffmpeg", "ffmpeg.exe"):
        return os.path.dirname(exe)
    cache_dir = os.path.join(tempfile.gettempdir(), "cloud_speed_ffmpeg")
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, "ffmpeg.exe")
    try:
        if not os.path.isfile(target) or os.path.getmtime(exe) > os.path.getmtime(target):
            shutil.copy2(exe, target)
    except OSError:
        return None
    return cache_dir


def _is_youtube_url(url):
    u = (url or "").strip().lower()
    return "youtube.com/" in u or "youtu.be/" in u


@st.cache_data(ttl=1800, show_spinner=False)
def _resolve_youtube_stream_url(youtube_url, max_height=720):
    """Resolve a YouTube page link to a direct HLS/HTTP stream URL for live view."""
    if yt_dlp is None:
        return None
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": f"best[height<={max_height}]/best",
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(youtube_url.strip(), download=False)
            return info.get("url")
    except Exception:
        return None


def _normalize_live_stream_url(url):
    """RTSP/HTTP URLs pass through; YouTube page links are resolved to stream URLs."""
    url = (url or "").strip()
    if not url:
        return ""
    if _is_youtube_url(url):
        resolved = _resolve_youtube_stream_url(url)
        return resolved or ""
    return url


def download_youtube_video(url, max_height=720):
    """
    Downloads a YouTube (or any yt-dlp supported site) video to a local
    temp .mp4 file and returns its path. Returns (path, error_message);
    error_message is None on success.
    """
    if yt_dlp is None:
        return None, (
            "YouTube downloads need the `yt-dlp` package. "
            "Install it with: pip install yt-dlp"
        )

    out_dir = tempfile.mkdtemp(prefix="cv_yt_")
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")
    ffmpeg_dir = _resolve_ffmpeg_dir_for_ytdlp()
    info_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}

    try:
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        return None, f"Couldn't read that video: {e}"
    except Exception as e:
        return None, f"Unexpected error while reading video info: {e}"

    video_id = info.get("id") or "video"
    is_live = bool(info.get("is_live"))

    # LIVE streams are HLS-only — record a short clip with ffmpeg instead of a full download.
    if is_live:
        ffmpeg_exe = _resolve_ffmpeg_exe()
        if not ffmpeg_exe:
            return None, (
                "This is a LIVE YouTube stream. Recording it needs ffmpeg. "
                "Install ffmpeg, or run: pip install imageio-ffmpeg"
            )
        fmt_opts = {
            **info_opts,
            "format": f"best[height<={max_height}]/best",
        }
        try:
            with yt_dlp.YoutubeDL(fmt_opts) as ydl:
                fmt_info = ydl.extract_info(url, download=False)
                stream_url = fmt_info.get("url")
        except Exception as e:
            return None, f"Couldn't resolve live stream URL: {e}"
        if not stream_url:
            return None, "Couldn't resolve live stream URL."

        out_path = os.path.join(out_dir, f"{video_id}.mp4")
        cmd = [
            ffmpeg_exe, "-y",
            "-t", str(_YT_LIVE_RECORD_SECONDS),
            "-i", stream_url,
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            out_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_YT_LIVE_RECORD_SECONDS + 120,
            )
        except subprocess.TimeoutExpired:
            return None, "Timed out while recording the live stream."
        if result.returncode != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 300:
                detail = detail[-300:]
            return None, f"Couldn't record live stream: {detail or 'unknown error'}"
        return out_path, None

    ydl_opts = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={max_height}][ext=mp4]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "max_filesize": 300 * 1024 * 1024,  # 300 MB safety cap
    }
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info)
            if not os.path.exists(downloaded_path):
                base, _ = os.path.splitext(downloaded_path)
                candidate = base + ".mp4"
                if os.path.exists(candidate):
                    downloaded_path = candidate
            if not os.path.exists(downloaded_path):
                return None, "Download finished but the output file could not be found."
            return downloaded_path, None
    except yt_dlp.utils.DownloadError as e:
        err = str(e)
        if "ffmpeg could not be found" in err.lower():
            return None, (
                "This video needs ffmpeg to download. Install ffmpeg, or run: "
                "pip install imageio-ffmpeg"
            )
        return None, f"Couldn't download that video: {e}"
    except Exception as e:
        return None, f"Unexpected error while downloading: {e}"

# ===== LIVE CAMERA AUTO-REFRESH COMPATIBILITY =====
# st.fragment(run_every=...) was promoted from experimental in newer Streamlit
# versions. This shim makes the live dual-camera tab work either way.    python -m streamlit run app.py
if hasattr(st, "fragment"):
    st_fragment = st.fragment
else:
    st_fragment = st.experimental_fragment

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
    except Exception:
        return None

    for tag_id in (36867, 36868, 306):
        if tag_id in exif_data:
            try:
                return datetime.datetime.strptime(exif_data[tag_id], "%Y:%m:%d %H:%M:%S")
            except Exception:
                continue  # this tag was malformed — try the next one instead of giving up
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



def get_cloud_sun_alignment(cloud_direction, sun_azimuth_deg, flow_angle_deg=None):
    if sun_azimuth_deg is None:
        return "unknown", None, "Sun position unavailable (set location or timestamp)."

    if flow_angle_deg is not None:
        cloud_az = flow_angle_deg % 360
        diff = abs((cloud_az - sun_azimuth_deg + 180) % 360 - 180)
    else:
        dir_to_az = {"North": 0, "NE": 45, "East": 90, "SE": 135, "South": 180, "SW": 225, "West": 270, "NW": 315}
        cloud_az = dir_to_az.get(cloud_direction, 0)
        diff = abs((cloud_az - sun_azimuth_deg + 180) % 360 - 180)

    if diff < 20:
        return "toward_sun", diff, f"Cloud motion is nearly aligned with the sun ({diff:.0f}° offset). Shadow risk is high."
    elif diff < 60:
        return "glancing", diff, f"Cloud motion is partly aligned with sun ({diff:.0f}° offset). Shadow may partially affect the panel."
    elif diff < 120:
        return "crossing", diff, f"Cloud path is crossing the sun direction ({diff:.0f}° offset). Shadow may be brief."
    else:
        return "away_from_sun", diff, f"Cloud is moving away from sun direction ({diff:.0f}° offset). Shadow risk is lower."

# ═══════════════════════════════════════════════════════════════



# ─────────────────────────── CONFIG ────────────────────────────
st.set_page_config(page_title="CloudVision AI", page_icon="☁️", layout="wide")

# ── Theme state (Dark / Light) ──
if "cv_theme" not in st.session_state:
    st.session_state["cv_theme"] = "dark"

_THEME = st.session_state["cv_theme"]
_IS_LIGHT = (_THEME == "light")

# ── Background photo (embedded as base64 so the app stays single-file) ──
import base64 as _b64

@st.cache_resource
def _load_bg_image_b64(_cache_key):
    bg_path = os.path.join(os.path.dirname(__file__), "sky_bg.jpg")
    with open(bg_path, "rb") as f:
        return _b64.b64encode(f.read()).decode("ascii")

_bg_path_for_mtime = os.path.join(os.path.dirname(__file__), "sky_bg.jpg")
_bg_mtime = os.path.getmtime(_bg_path_for_mtime) if os.path.exists(_bg_path_for_mtime) else 0
_BG_IMG_B64 = _load_bg_image_b64(_bg_mtime)

# ── Theme variable values ──
# Status colors (green/amber/red/blue) stay identical in both themes —
# only the neutral surface/border/text scale changes.
if _IS_LIGHT:
    _CV_VARS = {
        "--cv-card-bg":        "#f3f6fa",
        "--cv-card-bg-inset":  "#e7edf3",
        "--cv-card-bg-deep":   "#eaeff5",
        "--cv-card-bg-red":    "#fdeeee",
        "--cv-card-bg-amber":  "#fdf3e3",
        "--cv-border":         "#c3d0de",
        "--cv-text-primary":   "#11202e",
        "--cv-text-secondary": "#33495d",
        "--cv-text-muted":     "#5b7186",
        "--cv-text-dim":       "#6c8198",
        "--cv-photo-overlay-top":    "rgba(255,255,255,0.40)",
        "--cv-photo-overlay-mid":    "rgba(255,255,255,0.10)",
        "--cv-photo-overlay-bottom": "rgba(255,255,255,0.45)",
        "--cv-photo-vignette":       "rgba(20,30,42,0.18)",
        "--cv-photo-filter":   "saturate(1.05) contrast(1.0) brightness(1.06)",
        "--cv-text-shadow":    "0 1px 2px rgba(255,255,255,0.65), 0 1px 6px rgba(255,255,255,0.4)",
        "--cv-alert-bg":       "rgba(255,255,255,0.82)",
        "--cv-sidebar-bg":     "rgba(255,255,255,0.88)",
        "--cv-title-gradient": "linear-gradient(135deg, #0c4a6e 0%, #0369a1 40%, #4338ca 100%)",
        "--cv-chrome-bg":        "rgba(255,255,255,0.55)",
        "--cv-chrome-bg-strong": "rgba(255,255,255,0.75)",
        "--cv-chrome-border":    "rgba(120,140,165,0.35)",
        "--cv-table-row-line":   "rgba(120,140,165,0.25)",
    }
else:
    _CV_VARS = {
        "--cv-card-bg":        "#0d1a27",
        "--cv-card-bg-inset":  "#0a0f16",
        "--cv-card-bg-deep":   "#050c14",
        "--cv-card-bg-red":    "#1a0d0d",
        "--cv-card-bg-amber":  "#1a160d",
        "--cv-border":         "#1a2d44",
        "--cv-text-primary":   "#e2ecf6",
        "--cv-text-secondary": "#94b8d4",
        "--cv-text-muted":     "#4a6580",
        "--cv-text-dim":       "#3d5a73",
        "--cv-photo-overlay-top":    "rgba(2,6,12,0.62)",
        "--cv-photo-overlay-mid":    "rgba(2,6,12,0.08)",
        "--cv-photo-overlay-bottom": "rgba(2,6,12,0.72)",
        "--cv-photo-vignette":       "rgba(2,6,12,0.55)",
        "--cv-photo-filter":   "saturate(1.08) contrast(1.05) brightness(0.92)",
        "--cv-text-shadow":    "0 1px 3px rgba(0,0,0,0.55), 0 1px 12px rgba(0,0,0,0.35)",
        "--cv-alert-bg":       "rgba(8,16,26,0.78)",
        "--cv-sidebar-bg":     "rgba(13,26,39,0.85)",
        "--cv-title-gradient": "linear-gradient(135deg, #e0f2fe 0%, #38bdf8 40%, #818cf8 100%)",
        "--cv-chrome-bg":        "rgba(13,26,39,0.65)",
        "--cv-chrome-bg-strong": "rgba(13,26,39,0.85)",
        "--cv-chrome-border":    "rgba(26,45,68,0.8)",
        "--cv-table-row-line":   "rgba(15,30,45,0.8)",
    }

_CV_VARS_CSS = "\n".join(f"    {k}: {v};" for k, v in _CV_VARS.items())

st.markdown(
    """
    <style>
    :root {
""" + _CV_VARS_CSS + """
    }

    /* ── Photographic sky background, layered for a 3D/parallax feel ── */

    /* Layer 1 (back): the actual photo, slightly oversized + slow drifting
       zoom/pan so the clouds feel like they have real depth instead of
       sitting flat behind the UI */
    .cv-bg-photo {
        position: fixed;
        inset: -2% -2%;
        z-index: -4;
        background-image: url('data:image/jpeg;base64,__BG_IMG_B64__');
        background-size: cover;
        background-position: center 42%;
        background-repeat: no-repeat;
        filter: var(--cv-photo-filter);
        transform: scale(1.04);
        animation: cvBgParallax 36s ease-in-out infinite alternate;
        will-change: transform;
        pointer-events: none;
    }

    /* Layer 2: vignette / depth shading — darkens or lightens the rim of
       the frame (theme-dependent) so foreground cards visually separate
       (pop) from the photo behind them */
    .cv-bg-depth {
        position: fixed;
        inset: 0;
        z-index: -3;
        background:
            radial-gradient(ellipse 140% 90% at 50% 45%, transparent 35%, var(--cv-photo-vignette) 100%),
            linear-gradient(180deg, var(--cv-photo-overlay-top) 0%, var(--cv-photo-overlay-mid) 24%, var(--cv-photo-overlay-mid) 60%, var(--cv-photo-overlay-bottom) 100%);
        pointer-events: none;
    }

    @keyframes cvBgParallax {
        0%   { transform: scale(1.04) translate3d(0, 0, 0); }
        50%  { transform: scale(1.07) translate3d(-0.6%, -0.8%, 0); }
        100% { transform: scale(1.05) translate3d(0.5%, 0.4%, 0); }
    }
    </style>
    <div class="cv-bg-photo"></div>
    <div class="cv-bg-depth"></div>
    """.replace("__BG_IMG_B64__", _BG_IMG_B64),
    unsafe_allow_html=True
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Animated Sky Background ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: transparent;
    position: relative;
    min-height: 100vh;
}

/* Animated aurora/cloud shimmer layer — sits above the photo (z-index -2)
   adding a subtle drifting color glow for extra atmospheric depth */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: -2;
    background:
        radial-gradient(ellipse 120% 60% at 20% 10%, rgba(14,165,233,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 80% 50% at 80% 5%, rgba(99,102,241,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 100% 40% at 50% 20%, rgba(56,189,248,0.04) 0%, transparent 50%),
        radial-gradient(ellipse 60% 30% at 10% 60%, rgba(14,165,233,0.03) 0%, transparent 40%);
    animation: auroraShift 18s ease-in-out infinite alternate;
}

@keyframes auroraShift {
    0%   { opacity: 0.6; transform: scale(1) translateY(0px); }
    33%  { opacity: 0.9; transform: scale(1.02) translateY(-8px); }
    66%  { opacity: 0.7; transform: scale(0.99) translateY(4px); }
    100% { opacity: 1.0; transform: scale(1.01) translateY(-4px); }
}

/* Drifting cloud layers */
.cloud-bg {
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    overflow: hidden;
}
.cloud-bg::before {
    content: '';
    position: absolute;
    width: 200%;
    height: 100%;
    background-image:
        radial-gradient(ellipse 180px 60px at 15% 18%, rgba(255,255,255,0.025) 0%, transparent 70%),
        radial-gradient(ellipse 260px 80px at 45% 12%, rgba(255,255,255,0.02) 0%, transparent 70%),
        radial-gradient(ellipse 150px 50px at 72% 22%, rgba(255,255,255,0.018) 0%, transparent 70%),
        radial-gradient(ellipse 320px 90px at 88% 8%,  rgba(255,255,255,0.015) 0%, transparent 70%),
        radial-gradient(ellipse 200px 65px at 30% 35%, rgba(14,165,233,0.03) 0%, transparent 70%),
        radial-gradient(ellipse 280px 75px at 60% 30%, rgba(99,102,241,0.025) 0%, transparent 70%);
    animation: cloudDrift 40s linear infinite;
}
.cloud-bg::after {
    content: '';
    position: absolute;
    width: 200%;
    height: 100%;
    background-image:
        radial-gradient(ellipse 220px 70px at 25% 55%, rgba(255,255,255,0.015) 0%, transparent 70%),
        radial-gradient(ellipse 180px 55px at 65% 48%, rgba(255,255,255,0.02) 0%, transparent 70%),
        radial-gradient(ellipse 300px 85px at 10% 70%, rgba(14,165,233,0.02) 0%, transparent 70%),
        radial-gradient(ellipse 240px 72px at 80% 65%, rgba(99,102,241,0.018) 0%, transparent 70%);
    animation: cloudDrift2 55s linear infinite;
}

@keyframes cloudDrift  { from { transform: translateX(0); }    to { transform: translateX(-50%); } }
@keyframes cloudDrift2 { from { transform: translateX(-30%); } to { transform: translateX(-80%); } }

/* Stars layer — disabled: the real photo background is a daytime/stormy
   sky, so star twinkles would look out of place over it */
.star-field {
    display: none;
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background-image:
        radial-gradient(1px 1px at 10% 5%,  rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 22% 14%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 38% 8%,  rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 3%,  rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 11%, rgba(255,255,255,0.45) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 83% 6%,  rgba(255,255,255,0.55) 0%, transparent 100%),
        radial-gradient(1px 1px at 93% 15%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 5%  25%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 48% 20%, rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 62% 28%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(2px 2px at 76% 18%, rgba(56,189,248,0.5) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 15% 33%, rgba(255,255,255,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 38%, rgba(255,255,255,0.3) 0%, transparent 100%);
    animation: twinkle 6s ease-in-out infinite alternate;
}

@keyframes twinkle {
    0%   { opacity: 0.6; }
    50%  { opacity: 1.0; }
    100% { opacity: 0.7; }
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1280px; }

/* ── Top toolbar strip (Deploy button / hamburger menu) ──
   This sits above everything as a solid opaque bar by default,
   blocking the photo background. Make it transparent so the photo
   continues underneath; keep the controls themselves visible. */
[data-testid="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
    backdrop-filter: blur(6px);
}
[data-testid="stToolbar"] {
    background: transparent !important;
}
[data-testid="stDecoration"] {
    background: transparent !important;
    display: none;
}
[data-testid="stHeader"] * {
    text-shadow: var(--cv-text-shadow);
}
[data-testid="stHeader"] svg {
    filter: drop-shadow(0 1px 3px rgba(0,0,0,0.5));
}

/* ── Header ── */
.cv-header {
    display: flex; align-items: center; gap: 20px;
    padding: 32px 0 16px 0;
    border-bottom: 1px solid rgba(56,189,248,0.15);
    margin-bottom: 28px;
    position: relative;
}
.cv-header::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(14,165,233,0.5), rgba(99,102,241,0.5), transparent);
}

.cv-logo {
    width: 56px; height: 56px; border-radius: 16px;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; flex-shrink: 0;
    box-shadow: 0 0 32px rgba(14,165,233,0.4), 0 0 64px rgba(99,102,241,0.2);
    animation: logoPulse 4s ease-in-out infinite;
}
@keyframes logoPulse {
    0%, 100% { box-shadow: 0 0 32px rgba(14,165,233,0.4), 0 0 64px rgba(99,102,241,0.2); }
    50%       { box-shadow: 0 0 48px rgba(14,165,233,0.6), 0 0 80px rgba(99,102,241,0.35); }
}

.cv-title {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.03em;
    background: var(--cv-title-gradient);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.cv-sub {
    font-size: 0.78rem; color: var(--cv-text-dim);
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.12em; margin-top: 4px;
}
.cv-badge {
    margin-left: auto;
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: flex;
    align-items: center;
    gap: 6px;
}
.cv-badge::before {
    content: '●';
    color: #22c55e;
    font-size: 0.6rem;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--cv-sidebar-bg) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid var(--cv-chrome-border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--cv-chrome-bg);
    backdrop-filter: blur(8px);
    border: 1px solid var(--cv-chrome-border);
    border-radius: 12px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--cv-text-dim);
    padding: 10px 24px;
    font-weight: 500;
    font-size: 0.87rem;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--cv-text-secondary); background: rgba(14,165,233,0.06); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(99,102,241,0.15)) !important;
    color: #38bdf8 !important;
    box-shadow: inset 0 0 0 1px rgba(56,189,248,0.25) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 28px !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--cv-chrome-bg-strong);
    backdrop-filter: blur(8px);
    border: 1px solid var(--cv-chrome-border);
    border-radius: 14px;
    padding: 20px 22px !important;
    transition: border-color 0.25s, transform 0.2s, box-shadow 0.25s;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
    opacity: 0;
    transition: opacity 0.25s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(56,189,248,0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(14,165,233,0.12);
}
[data-testid="metric-container"]:hover::before { opacity: 1; }
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: var(--cv-text-dim) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--cv-text-primary) !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 11px 24px;
    font-weight: 600;
    font-size: 0.875rem;
    letter-spacing: 0.01em;
    transition: opacity 0.15s, transform 0.15s, box-shadow 0.15s;
    width: 100%;
    box-shadow: 0 4px 16px rgba(14,165,233,0.25);
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(14,165,233,0.35);
}
.stButton > button:active { transform: translateY(0); }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: var(--cv-chrome-bg);
    backdrop-filter: blur(6px);
    border: 1.5px dashed var(--cv-chrome-border);
    border-radius: 14px;
    padding: 16px;
    transition: border-color 0.25s, background 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #0ea5e9;
    background: rgba(14,165,233,0.04);
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--cv-chrome-bg);
    backdrop-filter: blur(6px);
    border: 1px solid var(--cv-chrome-border);
    border-radius: 12px;
}
[data-testid="stExpander"] summary {
    color: var(--cv-text-secondary) !important;
    font-size: 0.85rem;
    font-weight: 500;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #38bdf8 !important; }

/* ── Divider ── */
hr { border-color: var(--cv-chrome-border) !important; margin: 24px 0 !important; }

/* ── Subheader ── */
h2, h3 { color: var(--cv-text-primary) !important; font-weight: 600 !important; }

/* ── Number input / selectbox ── */
[data-baseweb="input"], [data-baseweb="select"] {
    background: var(--cv-chrome-bg-strong) !important;
    border-color: var(--cv-chrome-border) !important;
    border-radius: 8px !important;
    color: var(--cv-text-primary) !important;
}

/* ── Markdown tables ── */
table { width: 100%; border-collapse: collapse; }
th {
    background: var(--cv-chrome-bg-strong);
    color: var(--cv-text-dim);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
    padding: 11px 16px; border-bottom: 1px solid var(--cv-chrome-border);
}
td {
    color: var(--cv-text-primary); padding: 10px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; border-bottom: 1px solid var(--cv-table-row-line);
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(14,165,233,0.04); }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 1px !important;
    backdrop-filter: blur(4px);
}

/* ── Video ── */
video {
    border-radius: 12px;
    border: 1px solid var(--cv-chrome-border);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

/* ── Caption ── */
.stCaption { color: var(--cv-text-dim) !important; font-size: 0.78rem !important; }

/* ── Stat pill ── */
.cv-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.04em;
    background: var(--cv-chrome-bg-strong);
    border: 1px solid var(--cv-chrome-border);
    color: var(--cv-text-secondary);
    margin-right: 4px;
    backdrop-filter: blur(4px);
}

/* ── Section eyebrow label ── */
.cv-eyebrow {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--cv-text-dim);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.cv-eyebrow::before {
    content: '';
    display: inline-block;
    width: 18px; height: 1px;
    background: linear-gradient(90deg, #0ea5e9, transparent);
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: var(--cv-chrome-bg-strong) !important;
    border: 1px solid var(--cv-chrome-border) !important;
    color: #38bdf8 !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    backdrop-filter: blur(4px);
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #38bdf8 !important;
    background: rgba(14,165,233,0.08) !important;
    box-shadow: 0 0 16px rgba(56,189,248,0.15) !important;
}

/* ── Info / card boxes in sidebar ── */
[data-testid="stSidebar"] [style*="background:var(--cv-card-bg)"],
[data-testid="stSidebar"] div[style*="--cv-card-bg"] {
    backdrop-filter: blur(8px);
}
[data-testid="stSidebar"] {
    background: var(--cv-sidebar-bg) !important;
    backdrop-filter: blur(10px);
}

/* ══════════════════════════════════════════════════════════════
   TEXT READABILITY OVER PHOTO BACKGROUND
   Everything below exists because text now sits on a real photo
   (bright clouds, sun glare) instead of a flat dark gradient —
   so every piece of text gets a soft shadow + a brightness floor.
   Shadow direction flips automatically between themes via
   --cv-text-shadow (dark shadow on Dark theme, light glow on Light).
   ══════════════════════════════════════════════════════════════ */

/* Universal soft shadow: keeps every glyph legible over bright sky/
   cloud/sun regions without looking like a harsh drop-shadow */
.stApp, .stApp p, .stApp span, .stApp div, .stApp label,
.stApp li, .stApp td, .stApp th, h1, h2, h3, h4, h5, h6 {
    text-shadow: var(--cv-text-shadow);
}

/* Default Streamlit body / markdown text — was inheriting theme
   default with no contrast guarantee against the photo */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
.stApp p {
    color: var(--cv-text-primary) !important;
}

/* Widget labels (selectbox, slider, radio, number input, file
   uploader, text input) — previously unstyled, easy to lose over
   bright patches of sky */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stSlider label, .stSelectbox label, .stRadio label,
.stNumberInput label, .stTextInput label, .stFileUploader label {
    color: var(--cv-text-primary) !important;
    font-weight: 500 !important;
    text-shadow: var(--cv-text-shadow);
}

/* Captions / fine print — bumped up from a near-invisible var(--cv-text-dim)
   so secondary text still reads clearly */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--cv-text-secondary) !important;
    text-shadow: var(--cv-text-shadow);
}

/* st.info / st.warning / st.error / st.success banners — give them
   a frosted backdrop so their text has a guaranteed surface under it,
   no matter how bright the photo is behind that spot */
[data-testid="stAlert"] {
    background: var(--cv-alert-bg) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(56,189,248,0.18) !important;
}
[data-testid="stAlert"] p, [data-testid="stAlert"] span {
    color: var(--cv-text-primary) !important;
    text-shadow: none !important;
}

/* Sub-header eyebrow text + header subtitle were tuned for a near-
   black gradient; lighten them for the photo */
.cv-sub { color: var(--cv-text-secondary) !important; }

/* Tabs label text */
.stTabs [data-baseweb="tab"] p { text-shadow: var(--cv-text-shadow); }
</style>

<!-- Animated sky layers injected into DOM -->
<div class="cloud-bg"></div>
<div class="star-field"></div>
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

MIN_CLOUD_SPEED_KMH = 11.0

# ─────────────────────────── HELPERS ───────────────────────────
def clamp_cloud_speed(speed_mps, speed_kmh, delta_t_sec=1.0, distance_m=None):
    """Floor cloud speed when motion is detected but estimate is unrealistically low."""
    if speed_kmh <= 0:
        return speed_mps, speed_kmh, distance_m
    if speed_kmh >= MIN_CLOUD_SPEED_KMH:
        return speed_mps, speed_kmh, distance_m
    speed_kmh = MIN_CLOUD_SPEED_KMH
    speed_mps = speed_kmh / 3.6
    if distance_m is not None and delta_t_sec > 0:
        distance_m = speed_mps * delta_t_sec
    return speed_mps, speed_kmh, distance_m

def render_speed_formula_breakdown(
    cloud_type, fov, frame_width, pixel_disp, delta_t,
    focal_length_px, distance_m, speed_mps, speed_kmh, height_m,
    dist_5, dist_15, direction=None, motion_method="Farneback",
):
    """Show step-by-step speed formulas with plugged-in values."""
    fov_half_rad = math.radians(fov / 2.0)
    focal_recalc = (max(frame_width, 1) / 2.0) / math.tan(fov_half_rad)
    dist_recalc = (abs(pixel_disp) * height_m) / max(focal_length_px, 1e-6)
    speed_mps_raw = dist_recalc / max(delta_t, 1e-6)
    speed_kmh_raw = speed_mps_raw * 3.6
    pixel_speed = abs(pixel_disp) / max(delta_t, 1e-6)
    was_clamped = speed_kmh_raw > 0 and speed_kmh_raw < MIN_CLOUD_SPEED_KMH and speed_kmh >= MIN_CLOUD_SPEED_KMH

    st.markdown("#### 📐 Formulas Used")
    st.latex(r"f_{px} = \frac{W/2}{\tan(\mathrm{FOV}/2)}")
    st.latex(r"d = \frac{|\Delta p| \times h}{f_{px}}")
    st.latex(r"v_{m/s} = \frac{d}{\Delta t}")
    st.latex(r"v_{km/h} = v_{m/s} \times 3.6")
    st.latex(r"\text{Projected distance} = v_{km/h} \times \frac{t_{min}}{60}")

    st.markdown("#### 🔢 Step-by-Step Calculation")
    st.markdown(f"""
| Step | Formula | Calculation | Result |
|------|---------|-------------|--------|
| **0** | Optical flow ({motion_method}) | median pixel shift over `{delta_t:.3f}` s | **{abs(pixel_disp):.3f} px** |
| **0b** | Pixel speed | `{abs(pixel_disp):.3f} ÷ {delta_t:.3f}` | **{pixel_speed:.3f} px/s** |
| **1** | Focal length | `({frame_width} ÷ 2) ÷ tan({fov}° ÷ 2)` | **{focal_length_px:.2f} px** |
| **2** | Cloud height | lookup for `{cloud_type}` | **{height_m:,.0f} m** |
| **3** | Real distance | `({abs(pixel_disp):.3f} × {height_m:,.0f}) ÷ {focal_length_px:.2f}` | **{distance_m:.2f} m** |
| **4** | Speed (m/s) | `{distance_m:.2f} ÷ {delta_t:.3f}` | **{speed_mps_raw:.3f} m/s** |
| **5** | Speed (km/h) | `{speed_mps_raw:.3f} × 3.6` | **{speed_kmh_raw:.2f} km/h** |
""")

    if was_clamped:
        st.info(
            f"**Minimum speed floor applied:** `{speed_kmh_raw:.2f} km/h` → "
            f"**{speed_kmh:.1f} km/h** (minimum = {MIN_CLOUD_SPEED_KMH} km/h when motion detected)"
        )
    elif speed_kmh_raw > 0:
        st.success(f"**Final cloud speed:** **{speed_kmh:.1f} km/h** ({speed_mps:.2f} m/s)")

    st.markdown("#### ⏱️ Distance Projections")
    st.markdown(f"""
| Horizon | Formula | Result |
|---------|---------|--------|
| **+5 min** | `{speed_kmh:.1f} × (5 ÷ 60)` | **~{dist_5:.2f} km** |
| **+15 min** | `{speed_kmh:.1f} × (15 ÷ 60)` | **~{dist_15:.2f} km** |
""")
    if direction:
        st.caption(f"Direction from optical flow angle → **{direction}**")

    with st.expander("View raw parameters"):
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Motion method | {motion_method} |
| Camera FOV | {fov}° |
| Frame width | {frame_width} px |
| Focal length (recalc) | {focal_recalc:.2f} px |
| Pixel displacement | {pixel_disp:.4f} px |
| Time interval Δt | {delta_t:.4f} s |
| Cloud type → height | {cloud_type} → {height_m:,.0f} m |
| Raw speed (before floor) | {speed_kmh_raw:.2f} km/h |
| Final speed | {speed_kmh:.1f} km/h |
| Direction | {direction or '—'} |
""")

def angle_to_direction(angle_deg):
    a = angle_deg % 360
    if 22.5 <= a < 67.5:
        return "NE"
    elif 67.5 <= a < 112.5:
        return "East"
    elif 112.5 <= a < 157.5:
        return "SE"
    elif 157.5 <= a < 202.5:
        return "South"
    elif 202.5 <= a < 247.5:
        return "SW"
    elif 247.5 <= a < 292.5:
        return "West"
    elif 292.5 <= a < 337.5:
        return "NW"
    return "North"

def compute_optical_flow(gray1, gray2):
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    return float(np.median(mag)), float(np.median(ang))

def pixels_to_kmh(pixel_displacement, delta_t_sec, cloud_type, frame_width, fov):
    """
    Convert optical-flow pixel displacement to real cloud speed
    using camera focal length (better than tan approximation).
    """
    height_m = float(cloud_height.get(cloud_type, 2000))

    # Focal length in pixels
    focal_length_px = (max(frame_width, 1) / 2.0) / math.tan(math.radians(fov / 2.0))

    # Real distance travelled by cloud
    distance_m = (abs(pixel_displacement) * height_m) / focal_length_px

    # Speed
    speed_mps = distance_m / max(delta_t_sec, 1e-6)
    speed_kmh = speed_mps * 3.6
    speed_mps, speed_kmh, distance_m = clamp_cloud_speed(
        speed_mps, speed_kmh, delta_t_sec, distance_m
    )

    return (
        speed_mps,
        speed_kmh,
        focal_length_px,
        distance_m,
        height_m,
    )


# ═══════════════════════════════════════════════════════════════
# TWO-CAMERA TRIANGULATION — accurate cloud height + speed
# ═══════════════════════════════════════════════════════════════

def triangulate_cloud_height(alpha_deg, beta_deg, baseline_m):
    """
    Calculate accurate cloud height using two cameras (triangulation).

    Two cameras are placed at a known distance apart (baseline).
    Each camera measures the elevation angle to the same cloud feature.

    Formula:
        h = baseline / (1/tan(alpha) + 1/tan(beta))

    Where:
        alpha  = elevation angle from Camera 1 (degrees above horizon)
        beta   = elevation angle from Camera 2 (degrees above horizon)
        h      = cloud base height in metres

    Args:
        alpha_deg   : Elevation angle from Camera 1 (degrees).
        beta_deg    : Elevation angle from Camera 2 (degrees).
        baseline_m  : Distance between the two cameras (metres).

    Returns:
        height_m (float) : Triangulated cloud height in metres.
                           Returns None if angles are invalid.
    """
    alpha_rad = math.radians(max(alpha_deg, 0.5))   # avoid div/0 near horizon
    beta_rad  = math.radians(max(beta_deg,  0.5))

    tan_a = math.tan(alpha_rad)
    tan_b = math.tan(beta_rad)

    denominator = (1.0 / tan_a) + (1.0 / tan_b)
    if denominator <= 0:
        return None

    height_m = baseline_m / denominator
    return round(height_m, 1)


def elevation_angle_from_pixel(cloud_y_px, frame_height, fov_vertical_deg,
                                horizon_y_fraction=0.75):
    """
    Convert a cloud's Y pixel position to elevation angle above the horizon.

    Assumes the horizon sits at `horizon_y_fraction` of the frame height
    (default 75% down from top — typical for a ground-level sky camera).

    Args:
        cloud_y_px          : Y pixel coordinate of the tracked cloud feature.
        frame_height        : Total height of the frame in pixels.
        fov_vertical_deg    : Vertical field of view of the camera in degrees.
        horizon_y_fraction  : Fraction of frame height where horizon sits (0–1).

    Returns:
        elevation_deg (float): Angle above horizon in degrees (positive = above).
    """
    horizon_y    = frame_height * horizon_y_fraction
    deg_per_px   = fov_vertical_deg / frame_height
    pixel_offset = horizon_y - cloud_y_px        # positive when cloud is above horizon
    elevation    = pixel_offset * deg_per_px
    return round(float(elevation), 2)


def two_camera_speed(
    cam1_frame1, cam1_frame2,
    cam2_frame1, cam2_frame2,
    delta_t_sec,
    baseline_m,
    fov_horizontal_deg,
    fov_vertical_deg,
    horizon_y_fraction=0.75,
):
    """
    Full two-camera pipeline: triangulate height then compute real cloud speed.

    Steps:
      1. Run optical flow on each camera pair to get pixel displacement + angle.
      2. Detect a prominent cloud feature in cam1_frame1 and cam2_frame1
         and compute their elevation angles.
      3. Triangulate height from the two elevation angles + baseline.
      4. Convert pixel displacement → real-world speed using triangulated height.

    Args:
        cam1_frame1/2   : Grayscale frames from Camera 1 at t=0 and t=Δt.
        cam2_frame1/2   : Grayscale frames from Camera 2 at t=0 and t=Δt.
        delta_t_sec     : Time interval between frames (seconds).
        baseline_m      : Distance between Camera 1 and Camera 2 (metres).
        fov_horizontal_deg : Horizontal FOV of both cameras (degrees).
        fov_vertical_deg   : Vertical FOV of both cameras (degrees).
        horizon_y_fraction : Where horizon sits in frame (default 0.75).

    Returns:
        dict with keys:
            height_m        — triangulated cloud height (metres)
            alpha_deg       — elevation angle from Camera 1
            beta_deg        — elevation angle from Camera 2
            speed_mps       — real cloud speed (m/s)
            speed_kmh       — real cloud speed (km/h)
            direction       — compass direction string
            pixel_disp_cam1 — median pixel displacement from Camera 1
            pixel_disp_cam2 — median pixel displacement from Camera 2
            distance_m      — real distance cloud moved between frames (metres)
            method          — always "two_camera_triangulation"
    """
    H1, W1 = cam1_frame1.shape[:2]
    H2, W2 = cam2_frame1.shape[:2]

    # ── Step 1: Optical flow on both cameras ──
    disp_c1, ang_c1 = compute_optical_flow(cam1_frame1, cam1_frame2)
    disp_c2, ang_c2 = compute_optical_flow(cam2_frame1, cam2_frame2)

    # ── Step 2: Find cloud Y position for elevation angle ──
    # Use the region of highest optical flow magnitude as the cloud feature
    flow1  = cv2.calcOpticalFlowFarneback(
        cam1_frame1, cam1_frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag1, _ = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
    cy1, cx1 = np.unravel_index(np.argmax(mag1), mag1.shape)

    flow2  = cv2.calcOpticalFlowFarneback(
        cam2_frame1, cam2_frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag2, _ = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
    cy2, cx2 = np.unravel_index(np.argmax(mag2), mag2.shape)

    # ── Step 3: Elevation angles → triangulated height ──
    alpha_deg = elevation_angle_from_pixel(cy1, H1, fov_vertical_deg, horizon_y_fraction)
    beta_deg  = elevation_angle_from_pixel(cy2, H2, fov_vertical_deg, horizon_y_fraction)

    alpha_deg = max(alpha_deg, 5.0)   # clamp: < 5° = too near horizon, unreliable
    beta_deg  = max(beta_deg,  5.0)

    height_m = triangulate_cloud_height(alpha_deg, beta_deg, baseline_m)
    if height_m is None or height_m <= 0:
        height_m = 1500.0  # safe fallback

    # ── Step 4: Pixel displacement → real distance → speed ──
    # Use Camera 1 as the primary motion measurement
    deg_per_px   = fov_horizontal_deg / max(W1, 1)
    theta_deg    = abs(disp_c1) * deg_per_px
    theta_rad    = math.radians(theta_deg)
    distance_m   = height_m * math.tan(theta_rad)
    speed_mps    = distance_m / max(delta_t_sec, 1e-6)
    speed_kmh    = speed_mps * 3.6
    speed_mps, speed_kmh, distance_m = clamp_cloud_speed(
        speed_mps, speed_kmh, delta_t_sec, distance_m
    )
    direction    = angle_to_direction(float(np.degrees(ang_c1)))

    return {
        "height_m"        : round(height_m, 1),
        "alpha_deg"       : round(alpha_deg, 2),
        "beta_deg"        : round(beta_deg,  2),
        "speed_mps"       : round(speed_mps, 3),
        "speed_kmh"       : round(speed_kmh, 2),
        "direction"       : direction,
        "pixel_disp_cam1" : round(disp_c1, 2),
        "pixel_disp_cam2" : round(disp_c2, 2),
        "distance_m"      : round(distance_m, 1),
        "method"          : "two_camera_triangulation",
    }


def render_two_camera_ui(tab):
    """
    Streamlit UI block for the Two-Camera Triangulation tab.
    Call this inside a `with tab:` block.
    """
    st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">📐 Two-Camera Triangulation</div>',
                unsafe_allow_html=True)
    st.subheader("Accurate Cloud Height & Speed")
    st.info(
        "Upload one image **pair** from each camera taken at the same time interval. "
        "The two cameras must point at the same cloud from different positions. "
        "Measure the distance between them (baseline) accurately for best results.",
        icon="ℹ️"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**📷 Camera 1 — Images**")
        cam1_files = st.file_uploader("Camera 1: upload 2 images (t=0 and t=Δt)",
                                       type=["jpg","jpeg","png"],
                                       accept_multiple_files=True, key="tc_cam1")
    with col_b:
        st.markdown("**📷 Camera 2 — Images**")
        cam2_files = st.file_uploader("Camera 2: upload 2 images (t=0 and t=Δt)",
                                       type=["jpg","jpeg","png"],
                                       accept_multiple_files=True, key="tc_cam2")

    st.divider()
    st.markdown("**⚙️ Camera & Site Parameters**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        baseline_m     = st.number_input("Baseline distance (m)", min_value=10.0,
                                          max_value=5000.0, value=200.0, step=10.0,
                                          help="Distance between Camera 1 and Camera 2")
    with col2:
        tc_interval    = st.number_input("Time between images (s)", min_value=1,
                                          max_value=300, value=30, key="tc_interval")
    with col3:
        tc_fov_h       = st.slider("Horizontal FOV (°)", 30, 180, 90, key="tc_fov_h",
                                    help="Camera horizontal field of view")
    with col4:
        tc_fov_v       = st.slider("Vertical FOV (°)", 20, 120, 60, key="tc_fov_v",
                                    help="Camera vertical field of view")

    horizon_frac = st.slider("Horizon position in frame (%)", 50, 90, 75, key="tc_horizon",
                              help="How far down from top the horizon appears (75% = typical)") / 100.0

    st.divider()

    if st.button("🔺 Calculate Height & Speed", key="tc_run"):
        if not cam1_files or not cam2_files:
            st.error("Please upload images for both cameras.")
            return
        if len(cam1_files) < 2 or len(cam2_files) < 2:
            st.error("Each camera needs at least 2 images (t=0 and t=Δt).")
            return

        with st.spinner("Running two-camera triangulation…"):
            def load_gray(f):
                f.seek(0)
                arr = np.array(Image.open(f).convert("RGB").resize((640, 480)))
                return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

            c1f1 = load_gray(cam1_files[0])
            c1f2 = load_gray(cam1_files[1])
            c2f1 = load_gray(cam2_files[0])
            c2f2 = load_gray(cam2_files[1])

            result = two_camera_speed(
                c1f1, c1f2, c2f1, c2f2,
                delta_t_sec        = float(tc_interval),
                baseline_m         = baseline_m,
                fov_horizontal_deg = tc_fov_h,
                fov_vertical_deg   = tc_fov_v,
                horizon_y_fraction = horizon_frac,
            )

        st.success("✅ Triangulation complete — height calculated from geometry, not cloud type lookup.")
        st.divider()
        st.markdown('<div class="cv-eyebrow">📊 Triangulation Results</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("☁️ Cloud Height (triangulated)", f"{result['height_m']:,.0f} m",
                  help="Derived from elevation angles of both cameras — no lookup table used")
        m2.metric("💨 Cloud Speed", f"{result['speed_kmh']:.1f} km/h",
                  delta=f"{result['speed_mps']:.2f} m/s")
        m3.metric("🧭 Direction", result["direction"])

        m4, m5, m6 = st.columns(3)
        m4.metric("📐 Elevation α (Cam 1)", f"{result['alpha_deg']}°",
                  help="Angle above horizon to cloud from Camera 1")
        m5.metric("📐 Elevation β (Cam 2)", f"{result['beta_deg']}°",
                  help="Angle above horizon to cloud from Camera 2")
        m6.metric("📏 Distance moved", f"{result['distance_m']:.1f} m",
                  help="Real-world distance the cloud moved between the two frames")

        with st.expander("🔬 Full triangulation details"):
            st.markdown(f"""
| Parameter | Value |
|---|---|
| Baseline (d) | {baseline_m:.1f} m |
| Elevation α (Camera 1) | {result['alpha_deg']}° |
| Elevation β (Camera 2) | {result['beta_deg']}° |
| Formula | h = d / (1/tan α + 1/tan β) |
| **Cloud height (h)** | **{result['height_m']:,.0f} m** |
| Pixel displacement Cam 1 | {result['pixel_disp_cam1']} px |
| Pixel displacement Cam 2 | {result['pixel_disp_cam2']} px |
| Time interval | {tc_interval} s |
| **Speed** | **{result['speed_kmh']:.1f} km/h  ({result['speed_mps']:.2f} m/s)** |
| Method | {result['method']} |
""")

        st.info(
            f"💡 **Tip:** With baseline = {baseline_m:.0f} m, "
            f"you can achieve ±{max(30, int(baseline_m * 0.15)):.0f} m height accuracy. "
            "Increase baseline distance for higher clouds.",
            icon="💡"
        )


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


@st.cache_resource
def load_sky_checker_model():
    """
    Loads a small pretrained MobileNetV2 (ImageNet weights) used purely as a
    zero-shot "does this look like an outdoor sky scene" checker. This is a
    DIFFERENT model from the cloud-type classifier (cloud_model.keras) — it
    is never used to pick a cloud type, only as a gatekeeper that runs first.

    Why a model instead of pixel heuristics: simple color/edge/texture
    heuristics were tried first and failed in BOTH directions — a plain
    bright white business card scored as "sky-like" (false accept), while a
    dense, high-contrast real cumulus cloud photo scored as "not sky"
    (false reject). Pixel statistics alone just aren't a reliable signal
    here. A general-purpose pretrained classifier has actually learned what
    skies, clouds, and everyday objects look like, so it makes this call
    far more reliably than any hand-tuned formula.
    """
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights="imagenet", include_top=True)


# ImageNet class names that correspond to outdoor sky/weather/atmosphere
# scenes. Used to score how much of the model's attention is on sky-like
# content versus an everyday object (card, document, furniture, person...).
_SKY_RELATED_IMAGENET_TERMS = (
    "sky", "cloud", "horizon", "sunset", "sunrise", "cumulus", "cirrus",
    "rainbow", "lightning", "storm", "fog", "mist", "haze", "alp",
    "volcano", "balloon", "kite", "airliner", "airship", "parachute",
    "seashore", "cliff", "promontory", "valley", "geyser", "iceberg",
    "glacier", "coral_reef", "lakeside", "ocean",
)


def is_outdoor_sky_scene(pil_image, min_confidence=0.08, top_k=10):
    """
    Gatekeeper: asks a general-purpose pretrained model (MobileNetV2 /
    ImageNet) what it thinks the image contains, and checks whether
    sky/outdoor/weather-related concepts make up a meaningful share of its
    top predictions. This runs BEFORE the cloud-type classifier, so things
    like business cards, documents, faces, or random objects get rejected
    with a clear message instead of being confidently mislabeled as a cloud
    type (the cloud model itself has no "not a sky photo" class, so left
    unchecked it will always force-pick one of its 7 labels).

    The threshold is intentionally lenient (low min_confidence) because the
    cost of a false reject (turning away a real cloud photo) is worse here
    than the cost of a false accept on a genuinely ambiguous image — and
    real outdoor sky photos almost always pull at least *some* weight on
    sky/cloud/horizon/outdoor ImageNet classes even when the top-1 label is
    something else (e.g. "umbrella" for a sky photo with an object in frame).

    Returns: (passes: bool, sky_related_score: float 0-1, top_label: str)
    """
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input, decode_predictions
    )

    checker = load_sky_checker_model()
    img = pil_image.resize((224, 224)).convert("RGB")
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = checker.predict(arr, verbose=0)
    decoded = decode_predictions(preds, top=top_k)[0]  # list of (id, label, prob)

    top_label = decoded[0][1].replace("_", " ")
    sky_related_score = sum(
        float(prob) for (_id, label, prob) in decoded
        if any(term in label.lower() for term in _SKY_RELATED_IMAGENET_TERMS)
    )

    passes = sky_related_score >= min_confidence
    return passes, round(float(sky_related_score), 4), top_label


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
        "Cumulus": (10, 60),
        "Altocumulus": (30, 120),
        "Cirrus": (60, 360),
        "ClearSky": (0, 0),
        "Stratocumulus": (60, 480),
        "Cumulonimbus": (30, 90),
        "Mixed": (20, 90),
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
def classify_cloud_scenario(cloud_type, coverage_pct, speed_kmh, power_drop_pct):
    if cloud_type == "ClearSky" or coverage_pct < 10:
        return "Clear Sky", "Highest and most stable generation."
    if cloud_type == "Cumulonimbus" or coverage_pct >= 80:
        return "Storm Clouds", "Very low output with high ramp risk."
    if power_drop_pct >= 60 or coverage_pct >= 65:
        return "Dense Overcast", "Low but steady output with strong irradiance suppression."
    if speed_kmh >= 25:
        return "Fast-Moving Clouds", "Rapid ramps and high short-term forecast uncertainty."
    if power_drop_pct >= 25 or coverage_pct >= 25:
        return "Partial Cloud Cover", "Variable output with drops, recoveries, and possible edge enhancement."
    return "Mixed Cloud Conditions", "Moderate variability with short-lived changes in output."

def estimate_ramp_risk(speed_kmh, coverage_pct, cloud_type):
    score = 0
    if speed_kmh >= 30:
        score += 3
    elif speed_kmh >= 15:
        score += 2
    elif speed_kmh >= 5:
        score += 1
    if coverage_pct >= 70:
        score += 3
    elif coverage_pct >= 40:
        score += 2
    elif coverage_pct >= 15:
        score += 1
    if cloud_type in ["Cumulonimbus", "Cumulus", "Stratocumulus"]:
        score += 1
    if score >= 6:
        return "High", score
    if score >= 3:
        return "Medium", score
    return "Low", score

def cloud_enhancement_flag(coverage_pct, cloud_type, sun_visible=False):
    if cloud_type in ["Cirrus", "Cumulus", "Altocumulus"] and 5 <= coverage_pct <= 45:
        return True, "Cloud-edge enhancement possible; brief power spike may exceed clear-sky expectation."
    if sun_visible and coverage_pct <= 30:
        return True, "Sun-visible partial cloud scene; short enhancement spike is possible near cloud edges."
    return False, "No strong cloud-enhancement signal."

def build_scenario_text(scenario, impact_text, ramp_level, enh_text, cloud_sun_text):
    return f"Scenario: {scenario}. {impact_text} Ramp risk: {ramp_level}. {enh_text} {cloud_sun_text}"

def compute_weather_context(weather_json=None):
    if not isinstance(weather_json, dict):
        return {"temp_c": None, "humidity": None, "wind_ms": None, "pressure_hpa": None, "cloud_cover": None, "rain_mm": None}
    return {
        "temp_c": weather_json.get("temp_c"),
        "humidity": weather_json.get("humidity"),
        "wind_ms": weather_json.get("wind_ms"),
        "pressure_hpa": weather_json.get("pressure_hpa"),
        "cloud_cover": weather_json.get("cloud_cover"),
        "rain_mm": weather_json.get("rain_mm"),
    }


# ─────────────────────────── LIVE WIND (Open-Meteo) ─────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_wind(lat, lon):
    """
    Fetch near-real-time surface wind from Open-Meteo (free, no API key).
    Not second-by-second live — typically refreshed every ~15 minutes.
    """
    params = urllib.parse.urlencode({
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "current": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,cloud_cover",
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        cur = payload.get("current") or {}
        return {
            "wind_speed_kmh": cur.get("wind_speed_10m"),
            "wind_gusts_kmh": cur.get("wind_gusts_10m"),
            "wind_from_deg": cur.get("wind_direction_10m"),
            "temp_c": cur.get("temperature_2m"),
            "cloud_cover_pct": cur.get("cloud_cover"),
            "observed_at": cur.get("time"),
            "update_interval_sec": cur.get("interval") or 900,
            "timezone": payload.get("timezone"),
            "error": None,
        }
    except Exception as exc:
        return {
            "wind_speed_kmh": None,
            "wind_gusts_kmh": None,
            "wind_from_deg": None,
            "temp_c": None,
            "cloud_cover_pct": None,
            "observed_at": None,
            "update_interval_sec": 900,
            "timezone": None,
            "error": str(exc),
        }


def wind_freshness_label(wind_data):
    """Explain how old the Open-Meteo reading is (not instant live)."""
    observed_at = wind_data.get("observed_at")
    interval_sec = int(wind_data.get("update_interval_sec") or 900)
    interval_min = max(1, interval_sec // 60)
    if not observed_at:
        return f"Near real-time model data · updates every ~{interval_min} min"

    try:
        obs_dt = datetime.datetime.fromisoformat(str(observed_at))
        age_min = int(max(0, (datetime.datetime.now() - obs_dt).total_seconds() // 60))
    except ValueError:
        return f"Reading time: {observed_at} · updates every ~{interval_min} min"

    tz = wind_data.get("timezone") or "local"
    if age_min == 0:
        age_txt = "current (this hour)"
    elif age_min < interval_min + 5:
        age_txt = f"~{age_min} min old"
    else:
        age_txt = f"~{age_min} min old (slightly delayed)"

    return (
        f"{age_txt} · new data every ~{interval_min} min · {tz} time · "
        "not a second-by-second live station — weather model data"
    )


def met_wind_to_compass(wind_from_deg):
    """Meteorological wind-from degrees → 8-point compass."""
    if wind_from_deg is None:
        return "—"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((float(wind_from_deg) % 360 + 22.5) // 45) % 8
    return dirs[idx]


def cloud_drift_from_wind(wind_from_deg):
    """Wind blows FROM this bearing; clouds drift in the opposite direction."""
    if wind_from_deg is None:
        return "—", None
    drift_deg = (float(wind_from_deg) + 180.0) % 360.0
    return met_wind_to_compass(drift_deg), drift_deg


def estimate_wind_at_cloud_height(surface_wind_kmh, height_m, z_ref=10.0, alpha=0.15):
    """Power-law wind profile: stronger winds aloft than at 10 m."""
    if surface_wind_kmh is None or surface_wind_kmh <= 0:
        return None
    return float(surface_wind_kmh) * ((max(float(height_m), 1.0) / z_ref) ** alpha)


def compare_cloud_speed_to_wind(measured_kmh, wind_est_kmh):
    if wind_est_kmh is None or wind_est_kmh < 0.3:
        return "unknown", 0.0, "Live wind data unavailable — speed comparison skipped."
    diff_pct = abs(measured_kmh - wind_est_kmh) / wind_est_kmh * 100.0
    if diff_pct <= 40.0:
        return "aligned", diff_pct, (
            f"Measured cloud speed ({measured_kmh:.1f} km/h) is close to the live "
            f"wind estimate at cloud height (~{wind_est_kmh:.1f} km/h). "
            "Cloud motion is consistent with current wind conditions."
        )
    if measured_kmh < wind_est_kmh * 0.55:
        return "slower", diff_pct, (
            f"Video speed ({measured_kmh:.1f} km/h) is lower than live wind "
            f"({wind_est_kmh:.1f} km/h at altitude). Cloud may be moving partly "
            "toward/away from the camera, or the clip has limited motion."
        )
    return "faster", diff_pct, (
        f"Video speed ({measured_kmh:.1f} km/h) exceeds the live wind estimate "
        f"({wind_est_kmh:.1f} km/h). Stronger winds aloft or gusts may be present."
    )


def wind_adjusted_cloud_speed(measured_kmh, wind_est_kmh, status):
    """Blend video speed with live wind when they agree."""
    if wind_est_kmh is None:
        blended = measured_kmh
    elif status == "aligned":
        blended = 0.6 * measured_kmh + 0.4 * wind_est_kmh
    elif status == "slower" and measured_kmh < wind_est_kmh:
        blended = 0.45 * measured_kmh + 0.55 * wind_est_kmh
    else:
        blended = measured_kmh
    if blended > 0:
        blended = max(blended, MIN_CLOUD_SPEED_KMH)
    return blended

def predict_cloud_type(frames_or_images, return_diagnostics=False):
    """
    Batch-predict cloud type from a list of frames or PIL images.

    If return_diagnostics=True, also returns a diagnostics dict with the
    raw softmax distribution's top-2 gap and entropy. This is used as a
    second safeguard (on top of is_outdoor_sky_scene) to catch cases where a
    non-sky image still slips through the sky-scene gate but the cloud model
    itself isn't really sure / isn't seeing a clean class signal.
    """
    batch = []
    for img in frames_or_images:
        img_pil = Image.fromarray(img).resize((224, 224)) if isinstance(img, np.ndarray) else img.resize((224, 224))
        batch.append(image.img_to_array(img_pil) / 255.0)

    batch_arr = np.stack(batch, axis=0)          # shape: (N, 224, 224, 3)
    preds_all = model.predict(batch_arr, verbose=0)  # single forward pass

    preds = [class_names[np.argmax(p)] for p in preds_all]
    confs = [float(np.max(p)) * 100 for p in preds_all]
    top_type, top_conf = Counter(preds).most_common(1)[0][0], float(np.mean(confs))

    if not return_diagnostics:
        return top_type, top_conf

    # Diagnostics computed on the mean probability distribution across the batch
    mean_probs = np.mean(preds_all, axis=0)
    sorted_probs = np.sort(mean_probs)[::-1]
    top1, top2 = float(sorted_probs[0]), float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    top2_gap = top1 - top2  # large gap = model is decisive; small gap = model is guessing
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-9)))
    max_entropy = float(np.log(len(class_names)))
    normalized_entropy = entropy / max_entropy  # 0 = very confident, 1 = totally uniform/unsure

    diagnostics = {
        "top2_gap": round(top2_gap, 4),
        "normalized_entropy": round(normalized_entropy, 4),
    }
    return top_type, top_conf, diagnostics

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


def _calc_power_drop(cloud_type, coverage_pct, factor_map, confidence_pct=100.0):
    base = float(factor_map.get(cloud_type, 0.50))
    cov = max(0.0, min(1.0, (coverage_pct or 0.0) / 100.0))
    conf = max(0.35, min(1.0, confidence_pct / 100.0))
    drop = (base * (0.50 + 0.50 * cov) * 100.0) * conf
    return round(min(95.0, drop), 1)


def compute_second_plant_forecast(cloud_type, speed_kmh, speed_mps, direction,
                                   coverage_pct, plant2_dist_km=20.0,
                                   plant2_bearing_deg=270.0):
    """
    Forecast when (and how much) a cloud's shadow will reach a SECOND solar plant
    located at a fixed distance and bearing from Plant 1 (camera location).

    Args:
        cloud_type:         Detected cloud type string.
        speed_kmh:          Cloud ground speed in km/h.
        speed_mps:          Cloud ground speed in m/s.
        direction:          Cloud movement direction string (e.g. 'West', 'NW', ...).
        coverage_pct:       Sky cloud coverage percent (0–100).
        plant2_dist_km:     Distance from Plant 1 to Plant 2 in km (default 20).
        plant2_bearing_deg: Compass bearing of Plant 2 from Plant 1 in degrees
                            (0=N, 90=E, 180=S, 270=W). Default 270 = West.

    Returns dict:
        shadow_arrives_min  – estimated minutes until shadow hits Plant 2 (or None)
        power_drop_pct      – expected power drop at Plant 2
        status              – 'incoming' | 'safe' | 'clear' | 'slow'
        reason              – human-readable explanation string
        effective_dist_km   – effective distance cloud must travel to reach Plant 2
    """
    cloud_power_factor = {
        "Cumulus": 0.55, "Altocumulus": 0.45, "Cirrus": 0.18,
        "ClearSky": 0.0, "Stratocumulus": 0.72, "Cumulonimbus": 0.85, "Mixed": 0.50
    }
    cloud_lifetime_min = {
        "Cumulus": 35, "Altocumulus": 75, "Cirrus": 200,
        "ClearSky": 0, "Stratocumulus": 270, "Cumulonimbus": 60, "Mixed": 55
    }

    if cloud_type == "ClearSky":
        return {
            "shadow_arrives_min": None,
            "power_drop_pct": 0.0,
            "status": "clear",
            "reason": "☀️ Clear sky — no shadow risk for either plant.",
            "effective_dist_km": plant2_dist_km,
        }

    if speed_kmh < 0.5:
        return {
            "shadow_arrives_min": None,
            "power_drop_pct": 0.0,
            "status": "slow",
            "reason": "Cloud is nearly stationary. It is unlikely to reach the second plant.",
            "effective_dist_km": plant2_dist_km,
        }

    # Convert cloud direction string → bearing in degrees
    dir_to_bearing = {
        "North": 0, "NE": 45, "East": 90, "SE": 135,
        "South": 180, "SW": 225, "West": 270, "NW": 315
    }
    cloud_bearing = dir_to_bearing.get(direction, 0)

    # Compute angle between cloud travel direction and bearing toward Plant 2
    # delta = 0° means cloud is heading straight toward Plant 2
    delta_deg = abs((cloud_bearing - plant2_bearing_deg + 180) % 360 - 180)

    # Effective distance cloud must travel = plant2_dist_km / cos(delta)
    # If delta >= 90°, cloud is moving away — it will never reach Plant 2
    if delta_deg >= 85:
        return {
            "shadow_arrives_min": None,
            "power_drop_pct": 0.0,
            "status": "safe",
            "reason": (
                f"Cloud is moving {direction} ({cloud_bearing:.0f}°), "
                f"which is {delta_deg:.0f}° away from Plant 2 direction ({plant2_bearing_deg:.0f}°). "
                f"The shadow will not reach Plant 2."
            ),
            "effective_dist_km": None,
        }

    delta_rad = math.radians(delta_deg)
    effective_dist_km = plant2_dist_km / math.cos(delta_rad)

    # Time for cloud shadow to travel that effective distance
    travel_time_min = (effective_dist_km / speed_kmh) * 60.0

    # Check against cloud lifetime — will it still be alive when it arrives?
    lifetime = cloud_lifetime_min.get(cloud_type, 60)
    if travel_time_min > lifetime:
        return {
            "shadow_arrives_min": travel_time_min,
            "power_drop_pct": 0.0,
            "status": "safe",
            "reason": (
                f"Cloud would reach Plant 2 in ~{travel_time_min:.0f} min, "
                f"but {cloud_type} clouds only last ~{lifetime} min. "
                f"The cloud will dissipate before reaching Plant 2."
            ),
            "effective_dist_km": effective_dist_km,
        }

    power_drop = _calc_power_drop(cloud_type, coverage_pct, cloud_power_factor)

    direction_note = (
        f"Cloud is moving {direction} ({cloud_bearing:.0f}°) — "
        f"{delta_deg:.0f}° offset from Plant 2 direction ({plant2_bearing_deg:.0f}°)."
    )

    return {
        "shadow_arrives_min": round(travel_time_min, 1),
        "power_drop_pct": power_drop,
        "status": "incoming",
        "reason": (
            f"{direction_note} "
            f"Effective travel distance to Plant 2: {effective_dist_km:.1f} km. "
            f"At {speed_kmh:.1f} km/h, shadow will reach Plant 2 in "
            f"~{travel_time_min:.1f} minutes with an expected power drop of {power_drop}%."
        ),
        "effective_dist_km": effective_dist_km,
    }


def show_metrics(cloud_type, confidence, direction, height_m, fov,
                 frame_width, pixel_disp, delta_t, focal_length_px,
                 distance_m, speed_mps, speed_kmh, dist_5, dist_15,
                 coverage_pct=None, density_label=None, density_color=None,
                 vis_verdict=None, vis_reason=None, vis_color=None,
                 time_to_exit_min=999, solar_dist_km=1.0,
                 media_timestamp=None, timestamp_source="now",
                 image_elevation_est=None, image_elevation_conf=None,
                 image_elevation_note=None,
                 plant2_dist_km=20.0, plant2_bearing_deg=270.0,
                 user_lat=None, user_lon=None, use_wind_blend=True):
    emoji = cloud_emoji.get(cloud_type, "☁️")

    # ── Low confidence warning ──
    if confidence < 60:
        st.markdown(
            f"⚠️ **Low Model Confidence ({confidence:.1f}%)** — The model is uncertain about this cloud type. Results may be less accurate.",
            unsafe_allow_html=True,
        )

    # ── Section divider helper ──
    def section_header(icon, title):
        st.markdown(
            f"<div style='font-size:0.9rem;font-weight:700;color:var(--cv-text-secondary);margin:8px 0 6px 0;'>{icon} {title}</div>",
            unsafe_allow_html=True,
        )

    # ── Section 1: Cloud Classification ──
    section_header("☁", "Cloud Classification")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{emoji} Cloud Type",    cloud_type)
    c2.metric("🎯 Model Confidence",    f"{confidence:.1f}%")
    c3.metric("🧭 Cloud Direction",      direction)
    c4.metric("📍 Estimated Altitude",  f"{height_m:,} m")

    # ── Section 2: Motion & Displacement ──
    section_header("⚡", "Motion & Displacement")
    label_5  = f"~{dist_5:.2f} km"  if dist_5  > 0 else "—"
    label_15 = f"~{dist_15:.2f} km" if dist_15 > 0 else "—"

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Cloud Speed",            f"{speed_kmh:.1f} km/h")
    c6.metric("Cloud Speed (m/s)",      f"{speed_mps:.2f} m/s")
    c7.metric("Projected Dist. +5 min", label_5,  delta_color="off")
    c8.metric("Projected Dist. +15 min", label_15, delta_color="off")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Focal Length (px)", f"{focal_length_px:.2f}")
    with m2:
        st.metric("Distance (m)", f"{distance_m:.2f}")

    # ── Section 2b: Live Wind Validation ──
    if user_lat is not None and user_lon is not None:
        section_header("🌬️", "Wind Validation (Near Real-Time)")
        wind = fetch_live_wind(user_lat, user_lon)
        if wind.get("error"):
            st.warning(f"Could not fetch live wind data: {wind['error']}")
        elif wind.get("wind_speed_kmh") is not None:
            surface_kmh = float(wind["wind_speed_kmh"])
            wind_from = met_wind_to_compass(wind["wind_from_deg"])
            drift_lbl, _ = cloud_drift_from_wind(wind["wind_from_deg"])
            est_aloft_kmh = estimate_wind_at_cloud_height(surface_kmh, height_m)
            match_status, match_pct, match_note = compare_cloud_speed_to_wind(
                speed_kmh, est_aloft_kmh
            )
            fresh_note = wind_freshness_label(wind)
            status_colors = {
                "aligned": "#22c55e",
                "slower": "#f59e0b",
                "faster": "#38bdf8",
                "unknown": "var(--cv-text-muted)",
            }
            status_labels = {
                "aligned": "✅ Matches live wind",
                "slower": "⚠️ Slower than wind",
                "faster": "💨 Faster than wind",
                "unknown": "—",
            }
            match_color = status_colors.get(match_status, "var(--cv-text-muted)")

            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Surface Wind (10m)", f"{surface_kmh:.1f} km/h")
            w2.metric("Wind From", f"{wind_from}")
            w3.metric("Est. Wind @ Cloud Height", f"{est_aloft_kmh:.1f} km/h" if est_aloft_kmh else "—")
            w4.metric("Expected Drift", drift_lbl)

            if use_wind_blend and est_aloft_kmh is not None:
                blended_kmh = wind_adjusted_cloud_speed(speed_kmh, est_aloft_kmh, match_status)
                if abs(blended_kmh - speed_kmh) > 0.3:
                    st.info(
                        f"**Wind-adjusted estimate:** {blended_kmh:.1f} km/h "
                        f"(video {speed_kmh:.1f} km/h + live wind {est_aloft_kmh:.1f} km/h blend)"
                    )

            gust_txt = ""
            if wind.get("wind_gusts_kmh") is not None:
                gust_txt = f" · Gusts up to {float(wind['wind_gusts_kmh']):.0f} km/h"

            st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1px solid {match_color}44;border-radius:12px;padding:18px;margin-top:8px;">
  <div style="font-size:0.95rem;font-weight:700;color:{match_color};margin-bottom:8px;">
    {status_labels.get(match_status, "Wind check")} · {match_pct:.0f}% difference
  </div>
  <div style="font-size:0.82rem;color:var(--cv-text-muted);line-height:1.65;margin-bottom:10px;">{match_note}</div>
  <div style="font-size:0.75rem;color:var(--cv-text-dim);font-family:'JetBrains Mono',monospace;">
    Camera cloud direction: <b style="color:var(--cv-text-secondary);">{direction}</b>
    &nbsp;·&nbsp; Wind drift (model): <b style="color:var(--cv-text-secondary);">{drift_lbl}</b>
    {gust_txt}
    <br/>{fresh_note}
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.caption("Live wind data not available for this location.")

    # ── Section 3: Atmospheric Analysis ──
    if coverage_pct is not None:
        section_header("📡", "Atmospheric Analysis")
        d1, d2 = st.columns(2)

        with d1:
            bar_filled = int(coverage_pct)
            st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1px solid {density_color}44;border-radius:12px;padding:22px;">
  <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
              letter-spacing:0.12em;color:var(--cv-text-muted);margin-bottom:10px;">Sky Coverage</div>
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:16px;">
    <span style="font-size:2.2rem;font-weight:700;color:{density_color};font-family:'Inter',sans-serif;line-height:1;">{coverage_pct}%</span>
    <span style="font-size:0.82rem;color:{density_color};font-weight:600;padding:3px 10px;
                 background:{density_color}18;border-radius:999px;border:1px solid {density_color}33;">{density_label}</span>
  </div>
  <div style="background:var(--cv-card-bg-deep);border-radius:8px;height:10px;width:100%;overflow:hidden;margin-bottom:8px;">
    <div style="background:linear-gradient(90deg,{density_color}77,{density_color});
                width:{bar_filled}%;height:10px;border-radius:8px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:6px;">
    <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-dim);">Low — &lt;20%</span>
    <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-dim);">Medium — 20–55%</span>
    <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-dim);">High — &gt;55%</span>
  </div>
</div>
""", unsafe_allow_html=True)

        with d2:
            st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1px solid {vis_color}44;border-radius:12px;padding:22px;height:100%;">
  <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
              letter-spacing:0.12em;color:var(--cv-text-muted);margin-bottom:10px;">Cloud Lifetime Forecast</div>
  <div style="font-size:1rem;font-weight:700;color:{vis_color};margin-bottom:12px;
              padding:8px 12px;background:{vis_color}14;border-radius:8px;border-left:3px solid {vis_color};">{vis_verdict}</div>
  <div style="font-size:0.82rem;color:var(--cv-text-muted);line-height:1.65;">{vis_reason}</div>
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
<div style="background:var(--cv-card-bg);border:1.5px solid #22c55e55;border-radius:12px;padding:24px;display:flex;align-items:center;gap:20px;">
  <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
              display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">☀️</div>
  <div>
    <div style="font-size:1.05rem;font-weight:700;color:#22c55e;margin-bottom:6px;">Clear Sky — Solar Plant Fully Safe</div>
    <div style="font-size:0.82rem;color:#4a8060;line-height:1.5;">No clouds detected. No shadow risk. Solar plant is operating at full capacity.</div>
  </div>
  <div style="margin-left:auto;text-align:right;flex-shrink:0;">
    <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Power Status</div>
    <div style="font-size:1.1rem;font-weight:700;color:#22c55e;">🟢 100%</div>
  </div>
</div>
""", unsafe_allow_html=True)

    elif status == "now":
        pdrop = solar["power_drop_pct"]
        st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1.5px solid #ef444455;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:#ef444418;border:1px solid #ef444433;
                display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">⚠️</div>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:#ef4444;margin-bottom:5px;">Shadow Currently Falling on Solar Panel</div>
      <div style="font-size:0.78rem;color:var(--cv-text-muted);font-family:'JetBrains Mono',monospace;
                  background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:6px;padding:4px 10px;display:inline-block;">
        {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; {speed_kmh:.1f} km/h
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
    <div style="background:var(--cv-card-bg-red);border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
      <div style="font-size:2.1rem;font-weight:700;color:#ef4444;line-height:1;">{pdrop}%</div>
    </div>
    <div style="background:var(--cv-card-bg-red);border:1px solid #ef444430;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
      <div style="font-size:1rem;font-weight:700;color:#ff6b6b;margin-top:4px;">🔴 Active Now</div>
    </div>
    <div style="background:var(--cv-card-bg);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;grid-column:span 2;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
      <div style="font-size:0.82rem;color:var(--cv-text-secondary);line-height:1.7;">{solar["reason"]}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    elif status == "stationary":
        pdrop = solar["power_drop_pct"]
        st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1.5px solid #f59e0b55;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:#f59e0b18;border:1px solid #f59e0b33;
                display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🟡</div>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:#f59e0b;margin-bottom:5px;">Cloud Stationary — Shadow Present on Panel</div>
      <div style="font-size:0.78rem;color:var(--cv-text-muted);font-family:'JetBrains Mono',monospace;
                  background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:6px;padding:4px 10px;display:inline-block;">
        {cloud_type} &nbsp;·&nbsp; {height_m:,} m altitude &nbsp;·&nbsp; Nearly stationary
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
    <div style="background:var(--cv-card-bg-amber);border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
      <div style="font-size:2.1rem;font-weight:700;color:#f59e0b;line-height:1;">{pdrop}%</div>
    </div>
    <div style="background:var(--cv-card-bg-amber);border:1px solid #f59e0b30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Movement</div>
      <div style="font-size:1rem;font-weight:700;color:#f59e0b;margin-top:4px;">⏸ Stationary</div>
    </div>
    <div style="background:var(--cv-card-bg);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;grid-column:span 2;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Analysis</div>
      <div style="font-size:0.82rem;color:var(--cv-text-secondary);line-height:1.7;">{solar["reason"]}</div>
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
<div style="background:var(--cv-card-bg);border:1.5px solid {urg_color}55;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:{urg_color}18;border:1px solid {urg_color}33;
                display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">☁️</div>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:{urg_color};margin-bottom:5px;">
        {urg_icon} Shadow Will Reach Solar Plant in {arr_str}
      </div>
      <div style="font-size:0.78rem;color:var(--cv-text-muted);font-family:'JetBrains Mono',monospace;
                  background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:6px;padding:4px 10px;display:inline-block;">
        {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;">
    <div style="background:var(--cv-card-bg-deep);border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Shadow Arrives In</div>
      <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{arr_str}</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
      <div style="font-size:2.1rem;font-weight:700;color:{urg_color};line-height:1;">{pdrop}%</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Ground Offset</div>
      <div style="font-size:2.1rem;font-weight:700;color:#38bdf8;line-height:1;">{off_km:.2f} km</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;grid-column:span 3;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Forecast Analysis</div>
      <div style="font-size:0.82rem;color:var(--cv-text-secondary);line-height:1.7;">{solar["reason"]}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    else:  # status == "miss"
        arr_min = solar["shadow_time_min"]
        arr_str = f"{arr_min:.0f} min" if arr_min is not None else "N/A"
        st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1.5px solid #22c55e55;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
                display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">✅</div>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:#22c55e;margin-bottom:5px;">Shadow Will Not Reach the Solar Plant</div>
      <div style="font-size:0.78rem;color:var(--cv-text-muted);font-family:'JetBrains Mono',monospace;
                  background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:6px;padding:4px 10px;display:inline-block;">
        {cloud_type} &nbsp;·&nbsp; {height_m:,} m &nbsp;·&nbsp; {speed_kmh:.1f} km/h &nbsp;·&nbsp; {direction}
      </div>
    </div>
  </div>
  <div style="background:var(--cv-card-bg-deep);border:1px solid #22c55e22;border-radius:10px;padding:16px 18px;margin-bottom:12px;">
    <div style="font-size:0.82rem;color:#6aaa84;line-height:1.7;">{solar["reason"]}</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
    <div style="background:var(--cv-card-bg-deep);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Power Drop</div>
      <div style="font-size:2.1rem;font-weight:700;color:#22c55e;line-height:1;">0%</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Shadow Status</div>
      <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-top:4px;">🟢 Safe</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    # ── Section 5: Second Solar Plant Forecast ──
    section_header("🏭", "Second Solar Plant — Remote Shadow Forecast")

    p2 = compute_second_plant_forecast(
        cloud_type, speed_kmh, speed_mps, direction,
        coverage_pct if coverage_pct is not None else 50.0,
        plant2_dist_km=plant2_dist_km,
        plant2_bearing_deg=plant2_bearing_deg,
    )

    p2_status = p2["status"]

    # Direction label for Plant 2 bearing
    _bear_labels = {0:"North",45:"NE",90:"East",135:"SE",180:"South",225:"SW",270:"West",315:"NW"}
    _nearest_bear = min(_bear_labels.keys(), key=lambda k: abs(k - plant2_bearing_deg))
    plant2_dir_label = _bear_labels[_nearest_bear]

    if p2_status == "clear":
        st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1.5px solid #22c55e55;border-radius:12px;padding:20px;display:flex;align-items:center;gap:20px;">
  <div style="font-size:1.6rem;">🌿</div>
  <div>
    <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-bottom:4px;">Plant 2 — Clear Sky, No Risk</div>
    <div style="font-size:0.82rem;color:#4a8060;">No clouds detected. Plant 2 ({plant2_dir_label}, {plant2_dist_km:.0f} km away) is fully safe.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    elif p2_status == "slow":
        st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1.5px solid var(--cv-text-muted);border-radius:12px;padding:20px;">
  <div style="font-size:1rem;font-weight:700;color:var(--cv-text-secondary);margin-bottom:6px;">⏸ Cloud Too Slow — Plant 2 Likely Safe</div>
  <div style="font-size:0.82rem;color:var(--cv-text-muted);">{p2["reason"]}</div>
</div>
""", unsafe_allow_html=True)

    elif p2_status == "safe":
        st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1.5px solid #22c55e55;border-radius:12px;padding:20px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">
    <div style="width:48px;height:48px;border-radius:12px;background:#22c55e18;border:1px solid #22c55e33;
                display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;">✅</div>
    <div>
      <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-bottom:4px;">Plant 2 — Shadow Will NOT Arrive</div>
      <div style="font-size:0.75rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);
                  background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:6px;padding:3px 10px;display:inline-block;">
        Plant 2: {plant2_dir_label} &nbsp;·&nbsp; {plant2_dist_km:.0f} km &nbsp;·&nbsp; Cloud: {direction}
      </div>
    </div>
  </div>
  <div style="background:var(--cv-card-bg-deep);border:1px solid #22c55e22;border-radius:8px;padding:14px 16px;">
    <div style="font-size:0.82rem;color:#6aaa84;line-height:1.7;">{p2["reason"]}</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;">
    <div style="background:var(--cv-card-bg-deep);border:1px solid var(--cv-border);border-radius:8px;padding:14px;">
      <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Expected Power Drop</div>
      <div style="font-size:2rem;font-weight:700;color:#22c55e;line-height:1;">0%</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border:1px solid var(--cv-border);border-radius:8px;padding:14px;">
      <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Plant 2 Status</div>
      <div style="font-size:1rem;font-weight:700;color:#22c55e;margin-top:4px;">🟢 Safe</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    else:  # incoming
        arr_min  = p2["shadow_arrives_min"]
        pdrop    = p2["power_drop_pct"]
        eff_dist = p2.get("effective_dist_km", plant2_dist_km)
        arr_str  = f"{arr_min:.1f} min" if arr_min < 60 else f"{int(arr_min//60)}h {int(arr_min%60)}m"
        if arr_min < 15:
            urg_color = "#ef4444"; urg_icon = "🔴"
        elif arr_min < 40:
            urg_color = "#f59e0b"; urg_icon = "🟡"
        else:
            urg_color = "#22c55e"; urg_icon = "🟢"

        st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1.5px solid {urg_color}55;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:{urg_color}18;border:1px solid {urg_color}33;
                display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🏭</div>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:{urg_color};margin-bottom:5px;">
        {urg_icon} Shadow Reaches Plant 2 in {arr_str}
      </div>
      <div style="font-size:0.75rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);
                  background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:6px;padding:4px 10px;display:inline-block;">
        Plant 2: {plant2_dir_label} &nbsp;·&nbsp; {plant2_dist_km:.0f} km &nbsp;·&nbsp; Cloud: {direction} @ {speed_kmh:.1f} km/h
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;">
    <div style="background:var(--cv-card-bg-deep);border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Shadow Arrives In</div>
      <div style="font-size:2rem;font-weight:700;color:{urg_color};line-height:1;">{arr_str}</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border:1px solid {urg_color}30;border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Expected Power Drop</div>
      <div style="font-size:2rem;font-weight:700;color:{urg_color};line-height:1;">{pdrop}%</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Travel Distance</div>
      <div style="font-size:2rem;font-weight:700;color:#38bdf8;line-height:1;">{eff_dist:.1f} km</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;grid-column:span 3;">
      <div style="font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Forecast Analysis</div>
      <div style="font-size:0.82rem;color:var(--cv-text-secondary);line-height:1.7;">{p2["reason"]}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Section 6: Sun Position & Cloud Alignment ──
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
        "now":     ("⏱️ Current Time",      "var(--cv-text-muted)"),
    }
    _ts_label, _ts_color = _ts_badge_map.get(timestamp_source, ("⏱️ Current Time", "var(--cv-text-muted)"))

    section_header("🌞", "Sun Position & Cloud Alignment — Plant 1")

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
            "unknown":       "var(--cv-text-muted)",
        }
        align_icon_map = {
            "toward_sun":    "🔴 Heading Toward Sun",
            "glancing":      "🟡 Glancing Sun",
            "crossing":      "🔵 Crossing Sun Path",
            "away_from_sun": "🟢 Moving Away from Sun",
            "unknown":       "❓ Unknown",
        }
        a_color = align_color_map.get(align_status, "var(--cv-text-muted)")
        a_label = align_icon_map.get(align_status, "")

        if sun_el < 0:
            sun_status_label = "🌙 Below Horizon"
            sun_el_color = "var(--cv-text-muted)"
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
            image_elevation_conf, "var(--cv-text-muted)")

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
<div style="background:var(--cv-card-bg);border:1px solid {img_conf_color}44;border-radius:12px;
            padding:16px 20px;margin-bottom:14px;">
  <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;text-transform:uppercase;
              letter-spacing:0.12em;color:var(--cv-text-muted);margin-bottom:10px;">
    📸 Image-Based Sun Elevation Estimate (Sun not visible in frame)
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;">
    <div style="background:var(--cv-card-bg-deep);border-radius:8px;padding:12px 14px;">
      <div style="font-size:0.6rem;font-family:monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:6px;">Image Estimate</div>
      <div style="font-size:1.6rem;font-weight:700;color:{img_conf_color};line-height:1;">
        {image_elevation_est}°</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border-radius:8px;padding:12px 14px;">
      <div style="font-size:0.6rem;font-family:monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:6px;">pvlib (Location+Time)</div>
      <div style="font-size:1.6rem;font-weight:700;color:#fbbf24;line-height:1;">{sun_el:.1f}°</div>
    </div>
    <div style="background:var(--cv-card-bg-deep);border-radius:8px;padding:12px 14px;">
      <div style="font-size:0.6rem;font-family:monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:6px;">Difference</div>
      <div style="font-size:1.6rem;font-weight:700;color:{'#22c55e' if diff_el < 10 else '#f59e0b' if diff_el < 25 else '#ef4444'};line-height:1;">
        ±{diff_el:.1f}°</div>
    </div>
  </div>
  <div style="font-size:0.8rem;color:var(--cv-text-muted);line-height:1.6;font-style:italic;">{image_elevation_note}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1.5px solid {a_color}55;border-radius:14px;padding:22px;margin-top:4px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
    <div style="width:48px;height:48px;border-radius:12px;background:{a_color}18;border:1px solid {a_color}33;
                display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;">🌞</div>
    <div>
      <div style="font-size:1rem;font-weight:700;color:{a_color};margin-bottom:4px;">
        Cloud–Sun Alignment: {a_label}
      </div>
      <div style="font-size:0.78rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);
                  background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:6px;padding:3px 10px;display:inline-block;">
        {sun_status_label} &nbsp;·&nbsp; Azimuth {sun_az:.1f}° &nbsp;·&nbsp; Elevation {sun_el:.1f}°
        {'&nbsp;·&nbsp; ' + str(round(angle_diff)) + '° offset' if angle_diff is not None else ''}
      </div>
    </div>
  </div>
  <div style="font-size:0.84rem;color:var(--cv-text-secondary);line-height:1.7;background:var(--cv-card-bg-deep);
              border-radius:10px;padding:14px 18px;border:1px solid var(--cv-border);">
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

    with st.expander("🔬 Optical Flow — Calculation Details", expanded=False):
        motion_method = st.session_state.get("cv_motion_method", "Farneback").title()
        render_speed_formula_breakdown(
            cloud_type, fov, frame_width, pixel_disp, delta_t,
            focal_length_px, distance_m, speed_mps, speed_kmh, height_m,
            dist_5, dist_15, direction=direction, motion_method=motion_method,
        )

# ─────────────────────────── CLOUD DETECTION ───────────────────
def detect_clouds(frame, sky_h, use_efficientnet=None):
    """Realistic tight cloud boxes (CLAHE mask + contour + EfficientNet refine)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        variant = st.session_state.get("efficientnet_variant", "B0")
        if use_efficientnet is None:
            use_en = st.session_state.get("use_efficientnet_detect", True)
        else:
            use_en = use_efficientnet
    except Exception:
        variant, use_en = "B0", use_efficientnet if use_efficientnet is not None else True

    try:
        from cloud_box_detector import detect_clouds_realistic
        boxes = detect_clouds_realistic(frame, sky_h, use_efficientnet=use_en, variant=variant)
        return boxes, gray
    except Exception:
        pass

    # Legacy fallback
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

    # EfficientNet refine (optional — skip on video frames for speed)
    try:
        variant = st.session_state.get("efficientnet_variant", "B0")
        if use_efficientnet is None:
            use_en = st.session_state.get("use_efficientnet_detect", True)
        else:
            use_en = use_efficientnet
    except Exception:
        variant, use_en = "B0", use_efficientnet if use_efficientnet is not None else True

    if use_en:
        try:
            from efficientnet_cloud_detector import merge_and_refine_detections
            boxes = merge_and_refine_detections(
                frame, sky_h, boxes, variant=variant, use_efficientnet=True
            )
        except Exception:
            pass
    else:
        try:
            from efficientnet_cloud_detector import merge_and_refine_detections
            boxes = merge_and_refine_detections(
                frame, sky_h, boxes, variant=variant, use_efficientnet=False
            )
        except Exception:
            pass

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
                         fov=75, time_to_exit_min=999, speed_mps=None,
                         precomputed_boxes=None, use_efficientnet=None,
                         skip_depth=False, precomputed_depth=None):
    OUT_W = frame.shape[1]
    OUT_H = frame.shape[0]
    sky_h = int(OUT_H * 0.85)   # include lower horizon clouds
    if speed_mps is None:
        speed_mps = speed_kmh / 3.6

    # Pre-compute full dense optical flow if prev frame available
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sky_gray = gray[:sky_h, :]
    full_flow = None
    if prev_gray is not None and delta_t is not None and delta_t > 0:
        prev_sky = prev_gray[:sky_h, :]
        full_flow = cv2.calcOpticalFlowFarneback(
            prev_sky, sky_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

    # Detect clouds (or reuse cached boxes for video speed)
    if precomputed_boxes is not None:
        boxes = precomputed_boxes
    else:
        boxes, _ = detect_clouds(frame, sky_h, use_efficientnet=use_efficientnet)

    # ── Compute pseudo stereo depth map for full sky region ──
    if precomputed_depth is not None:
        depth_map, depth_color = precomputed_depth
    elif skip_depth:
        depth_map = np.zeros((sky_h, OUT_W), dtype=np.float32)
        depth_color = np.zeros((sky_h, OUT_W, 3), dtype=np.uint8)
    else:
        depth_map, depth_color = compute_pseudo_depth_map(frame, sky_h)

    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        mask_lbl = box[4] if len(box) > 4 else None
        pad = 4
        x1 = max(0,       x - pad);    y1 = max(0,       y - pad)
        x2 = min(OUT_W-1, x+w + pad);  y2 = min(sky_h,   y+h + pad)

        # ── Per-cloud speed from optical flow ROI (smoothed vs global) ──
        cloud_speed_kmh = speed_kmh
        if full_flow is not None and mask_lbl is not None:
            roi_flow = full_flow[y:y+h, x:x+w]
            mroi = mask_lbl[y:y+h, x:x+w] if mask_lbl.shape[:2] == (sky_h, OUT_W) else None
            if mroi is not None and mroi.shape == roi_flow.shape[:2]:
                valid = mroi > 0
                if valid.sum() > 20:
                    fx = roi_flow[..., 0][valid]
                    fy = roi_flow[..., 1][valid]
                    mag = np.sqrt(fx * fx + fy * fy)
                    roi_pixel_disp = float(np.median(mag))
                else:
                    roi_pixel_disp = 0.0
            elif roi_flow.size > 0:
                mag, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
                roi_pixel_disp = float(np.median(mag))
            else:
                roi_pixel_disp = 0.0
            if roi_pixel_disp > 0.1:
                _, raw_kmh, _, _, _ = pixels_to_kmh(
                    roi_pixel_disp, delta_t, cloud_type, OUT_W, fov
                )
                try:
                    from cloud_box_detector import smooth_box_speed
                    cloud_speed_kmh = smooth_box_speed(raw_kmh, speed_kmh)
                except Exception:
                    cloud_speed_kmh = speed_kmh

        # ── Depth overlay only on cloud mask pixels ──
        roi_frame = frame[y1:y2, x1:x2]
        if roi_frame.size > 0:
            roi_depth = depth_color[y1:y2, x1:x2]
            if mask_lbl is not None and mask_lbl.shape[:2] == (sky_h, OUT_W):
                m = mask_lbl[y1:y2, x1:x2]
                if m.shape == roi_frame.shape[:2]:
                    cloud_px = m > 0
                    if cloud_px.any():
                        blended = roi_frame.copy()
                        blended[cloud_px] = cv2.addWeighted(
                            roi_depth, 0.40, roi_frame, 0.60, 0
                        )[cloud_px]
                        frame[y1:y2, x1:x2] = blended
            elif roi_depth.shape == roi_frame.shape:
                cv2.addWeighted(roi_depth, 0.35, roi_frame, 0.65, 0, frame[y1:y2, x1:x2])

        cy_box = min((y1 + y2) // 2, depth_map.shape[0] - 1)
        cx_box = min((x1 + x2) // 2, depth_map.shape[1] - 1)
        center_depth = float(depth_map[cy_box, cx_box])
        est_dist_km  = depth_to_distance_km(center_depth, height_m, fov)

        # Contour outline (tight) + box
        if mask_lbl is not None and mask_lbl.shape[:2] == (sky_h, OUT_W):
            cnts, _ = cv2.findContours(mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(frame, cnts, -1, (0, 255, 80), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

        t = 10
        for (px_, py_, sdx, sdy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px_, py_), (px_+sdx*t, py_),    (0, 255, 60), 2)
            cv2.line(frame, (px_, py_), (px_, py_+sdy*t),    (0, 255, 60), 2)

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

    # Smart +5 / +15 min projections
    lbl_5  = f"~{dist_5:.2f} km"
    lbl_15 = f"~{dist_15:.2f} km"
    txt(f"+5m:{lbl_5}  +15m:{lbl_15}", 86, sc=0.40, c=(200, 200, 200))

    mins = int(elapsed_sec)//60;  secs = int(elapsed_sec)%60
    cv2.putText(frame, f"T+ {mins:02d}:{secs:02d}",
                (OUT_W-155, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,180), 2, cv2.LINE_AA)

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
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    OUT_W, OUT_H = 960, 540
    delta_t = 1.0 / fps

    # Cap length — long videos are very slow to process
    max_frames = min(total if total > 0 else 99999, int(fps * 45))
    box_refresh = max(1, int(fps // 3))     # full re-detect ~every 0.33s
    depth_refresh = max(1, int(fps * 2))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, OUT_H))

    prev_gray = None
    frame_idx = 0
    cached_boxes = None
    cached_depth = None
    sky_h = int(OUT_H * 0.85)

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame   = cv2.resize(frame, (OUT_W, OUT_H))
        elapsed = frame_idx / fps
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        run_en = (frame_idx == 0) or (frame_idx % int(fps) == 0)
        if frame_idx % box_refresh == 0 or cached_boxes is None:
            cached_boxes, _ = detect_clouds(frame, sky_h, use_efficientnet=run_en)
        elif prev_gray is not None and cached_boxes:
            try:
                from cloud_box_detector import track_boxes_optical_flow
                cached_boxes = track_boxes_optical_flow(prev_gray, gray, cached_boxes, sky_h)
            except Exception:
                pass

        if frame_idx % depth_refresh == 0 or cached_depth is None:
            cached_depth = compute_pseudo_depth_map(frame, sky_h)

        frame   = draw_boxes_on_frame(
            frame, speed_kmh, direction, cloud_type,
            height_m, dist_5, dist_15, elapsed,
            prev_gray=prev_gray, delta_t=delta_t, fov=fov,
            time_to_exit_min=time_to_exit_min,
            speed_mps=speed_mps,
            precomputed_boxes=cached_boxes,
            use_efficientnet=False,
            skip_depth=False,
            precomputed_depth=cached_depth,
        )
        out.write(frame)
        prev_gray = gray
        frame_idx += 1

    cap.release()
    out.release()

    encode_mp4_for_mobile(output_path)


# ═══════════════════════════════════════════════════════════════
# LIVE DUAL-CAMERA TRACKING (RTSP / IP camera streams)
# ═══════════════════════════════════════════════════════════════
LIGHT_GREEN_BGR = (144, 238, 144)  # "lightgreen" — symmetric R/B so BGR==RGB


def _get_session_id():
    """Best-effort unique id for the current browser session/tab."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is not None:
            return ctx.session_id
    except Exception:
        pass
    return "default"


@st.cache_resource(show_spinner=False)
def _open_live_capture(session_id, url):
    """
    Open (and cache) an RTSP/HTTP camera stream, keyed by (session_id, url).
    Cached so the same connection is reused across fragment auto-refreshes
    instead of being re-opened every cycle. Keying by session_id means
    st.cache_resource — which is normally shared across ALL viewers of the
    app — no longer lets one user's Stop button release another user's
    camera stream; each session gets its own cached capture.
    """
    stream_url = _normalize_live_stream_url(url)
    if not stream_url:
        return cv2.VideoCapture()
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def get_live_capture(url):
    """Session-scoped wrapper around _open_live_capture."""
    return _open_live_capture(_get_session_id(), url)


def release_live_capture(url):
    """Release and clear the cached capture for the CURRENT session only."""
    session_id = _get_session_id()
    try:
        cap = _open_live_capture(session_id, url)
        cap.release()
    except Exception:
        pass
    try:
        _open_live_capture.clear(session_id, url)
    except Exception:
        pass
    if _is_youtube_url(url):
        try:
            _resolve_youtube_stream_url.clear(url.strip())
        except Exception:
            pass


def read_fresh_live_frame(cap):
    """Drop a couple of stale buffered frames, then read the latest one."""
    if cap is None or not cap.isOpened():
        return False, None
    for _ in range(2):
        cap.grab()
    return cap.read()


def draw_density_boxes_live(frame_bgr, prev_gray=None, sky_frac=0.78,
                             min_thickness=1, max_thickness=9):
    """
    Detect cloud regions in a live frame and draw light-green bounding boxes
    whose LINE THICKNESS scales with how dense each individual cloud is
    (thicker border = denser cloud, thinner border = sparse/thin cloud).
    If a previous frame's grayscale image is supplied, also overlays sparse
    light-green motion arrows from optical flow to show live cloud movement.

    Returns: annotated_frame, gray (pass back in next call as prev_gray),
             overall_coverage_pct, box_infos (list of dicts), flow_speed_px,
             flow_dir_deg
    """
    H, W = frame_bgr.shape[:2]
    sky_h = int(H * sky_frac)
    out = frame_bgr.copy()

    boxes, gray = detect_clouds(frame_bgr, sky_h)

    total_sky_px = max(sky_h * W, 1)
    total_cloud_px = 0
    box_infos = []

    for (x, y, w, h, mask_lbl) in boxes:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(sky_h, y + h)
        box_area = max((x2 - x1) * (y2 - y1), 1)

        if mask_lbl is not None and mask_lbl.shape[:2] == (sky_h, W):
            cloud_px = int(cv2.countNonZero(mask_lbl[y1:y2, x1:x2]))
        else:
            cloud_px = box_area  # fallback contour: assume box is solid cloud

        density_pct = float(min(100.0, (cloud_px / box_area) * 100))
        total_cloud_px += cloud_px

        thickness = int(round(np.interp(density_pct, [0, 100],
                                         [min_thickness, max_thickness])))
        thickness = max(min_thickness, thickness)

        cv2.rectangle(out, (x1, y1), (x2, y2), LIGHT_GREEN_BGR, thickness)

        label = f"{density_pct:.0f}% dense"
        (tw, th_), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ly = y1 - 6 if y1 - 6 - th_ > 2 else y2 + th_ + 6
        cv2.rectangle(out, (x1 - 1, ly - th_ - 3), (x1 + tw + 5, ly + bl + 1),
                      (20, 50, 20), -1)
        cv2.putText(out, label, (x1 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    LIGHT_GREEN_BGR, 1, cv2.LINE_AA)

        box_infos.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                           "density_pct": round(density_pct, 1)})

    coverage_pct = round((total_cloud_px / total_sky_px) * 100, 1)

    # ── Live motion arrows from optical flow vs previous frame ──
    flow_speed_px, flow_dir_deg = 0.0, None
    if prev_gray is not None and prev_gray.shape == gray.shape:
        prev_sky = prev_gray[:sky_h, :]
        sky_gray = gray[:sky_h, :]
        flow = cv2.calcOpticalFlowFarneback(prev_sky, sky_gray, None,
                                             0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        moving = mag > 0.5
        if np.any(moving):
            flow_speed_px = float(np.median(mag[moving]))
            flow_dir_deg = float(np.median(ang[moving]))

        step = 36
        for yy in range(0, sky_h, step):
            for xx in range(0, W, step):
                dx, dy = flow[yy, xx]
                if math.hypot(dx, dy) < 0.5:
                    continue
                p1 = (xx, yy)
                p2 = (int(xx + dx * 5), int(yy + dy * 5))
                cv2.arrowedLine(out, p1, p2, LIGHT_GREEN_BGR, 1, tipLength=0.45)

    # ── HUD ──
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (230, 58), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)
    cv2.rectangle(out, (0, 0), (230, 58), LIGHT_GREEN_BGR, 1)
    cv2.putText(out, f"LIVE  Coverage: {coverage_pct:.1f}%", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, datetime.datetime.now().strftime("%H:%M:%S"), (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return out, gray, coverage_pct, box_infos, flow_speed_px, flow_dir_deg


def render_live_dual_camera_view(cam1_url, cam2_url, refresh_sec, sky_frac, max_thick):
    """
    Auto-refreshing Streamlit fragment: pulls one frame from each RTSP/IP
    camera every `refresh_sec`, draws density-scaled bounding boxes + motion
    arrows, and shows both feeds side by side. Only this fragment reruns on
    its own — the rest of the page (Start/Stop buttons, settings) stays put.
    """
    @st_fragment(run_every=refresh_sec)
    def _live_fragment():
        cam_specs = [("cam1", cam1_url, "📷 Camera 1")]
        if (cam2_url or "").strip():
            cam_specs.append(("cam2", cam2_url, "📷 Camera 2"))

        if len(cam_specs) == 2:
            cols = st.columns(2)
            items = [
                (key, url, label, cols[i])
                for i, (key, url, label) in enumerate(cam_specs)
            ]
        else:
            items = [(cam_specs[0][0], cam_specs[0][1], cam_specs[0][2], st.container())]

        for key, url, label, col in items:
            with col:
                st.markdown(f"**{label}**")
                cap = get_live_capture(url)
                if cap is None or not cap.isOpened():
                    if _is_youtube_url(url):
                        st.error(
                            "Could not open this YouTube stream. "
                            "Check the link, ensure yt-dlp is installed, then click Start again."
                        )
                    else:
                        st.error(
                            "Could not open this stream. Check the URL and click Start again."
                        )
                    continue
                ret, frame = read_fresh_live_frame(cap)
                if not ret or frame is None:
                    st.warning("No frame received — stream may be buffering or down.")
                    continue

                frame = cv2.resize(frame, (640, 480))
                prev_gray = st.session_state.get(f"live_prev_gray_{key}")
                annotated, gray, cov_pct, boxes, spd_px, dir_deg = draw_density_boxes_live(
                    frame, prev_gray, sky_frac, 1, max_thick
                )
                st.session_state[f"live_prev_gray_{key}"] = gray

                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                direction_txt = angle_to_direction(dir_deg) if dir_deg is not None else "—"
                mm1, mm2, mm3 = st.columns(3)
                mm1.metric("☁️ Coverage", f"{cov_pct:.1f}%")
                mm2.metric("💨 Motion", f"{spd_px:.1f} px" if dir_deg is not None else "—")
                mm3.metric("🧭 Dir", direction_txt)

                if boxes:
                    densest = max(b["density_pct"] for b in boxes)
                    st.caption(f"{len(boxes)} cloud region(s) · densest box: {densest:.0f}%")
                else:
                    st.caption("No distinct cloud regions detected right now.")

        st.caption(f"⏱️ Auto-refreshing every {refresh_sec:.1f}s · last update "
                   f"{datetime.datetime.now().strftime('%H:%M:%S')}")

    _live_fragment()


# ─────────────────────────── UI HEADER ────────────────────────
st.markdown("""
<div class="cv-header">
  <div class="cv-logo">☁️</div>
  <div>
    <div class="cv-title">CloudVision AI</div>
    <div class="cv-sub">Cloud Classification &amp; Motion Prediction System</div>
  </div>
  <div class="cv-badge">AI Model Active</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎬 Video Analysis", "🖼️ Multi Image Analysis",
    "📐 Two-Camera Triangulation", "🔴 Live Dual-Camera",
    "🔎 Quick Identify", "📡 Live Single Camera"
])

# ── Solar location inputs (shared across tabs) ──
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    _theme_choice = st.radio(
        "Theme",
        options=["dark", "light"],
        format_func=lambda x: "🌙 Dark" if x == "dark" else "☀️ Light",
        index=0 if st.session_state["cv_theme"] == "dark" else 1,
        horizontal=True,
        key="cv_theme_radio",
        label_visibility="collapsed",
    )
    if _theme_choice != st.session_state["cv_theme"]:
        st.session_state["cv_theme"] = _theme_choice
        st.rerun()

    st.markdown("---")
    st.markdown("### ☀️ Solar Location")
    st.caption("Enter your location for real-time sun position tracking")
    user_lat = st.number_input("Latitude",  value=28.6, min_value=-90.0,  max_value=90.0,  step=0.1, format="%.4f", key="user_lat")
    user_lon = st.number_input("Longitude", value=77.2, min_value=-180.0, max_value=180.0, step=0.1, format="%.4f", key="user_lon")
    st.caption("🇮🇳 Default: New Delhi")

    use_wind_blend = st.checkbox(
        "🌬️ Use live wind to refine speed estimate",
        value=True,
        key="use_wind_blend",
        help="Fetches near-real-time wind from Open-Meteo (~15 min updates) and compares with cloud speed.",
    )

    st.markdown("---")
    st.markdown("### 🧠 Cloud Detection (EfficientNet)")
    st.checkbox(
        "Use EfficientNet cloud filter",
        value=True,
        key="use_efficientnet_detect",
        help="Removes edge false boxes; detects lower horizon clouds better.",
    )
    st.selectbox(
        "EfficientNet model",
        ["B0", "B3"],
        index=0,
        key="efficientnet_variant",
        help="B0 = faster · B3 = more accurate (slower first load)",
    )
    st.caption("Video: EfficientNet runs on 1st frame only; boxes refresh ~1×/sec for speed.")

    _wind_sb = fetch_live_wind(user_lat, user_lon)
    if _wind_sb.get("error"):
        st.caption(f"Wind feed: unavailable ({_wind_sb['error'][:60]}…)")
    elif _wind_sb.get("wind_speed_kmh") is not None:
        _wf = met_wind_to_compass(_wind_sb["wind_from_deg"])
        _drift_sb, _ = cloud_drift_from_wind(_wind_sb["wind_from_deg"])
        _fresh_sb = wind_freshness_label(_wind_sb)
        st.markdown(f"""
<div style='background:var(--cv-card-bg);border:1px solid var(--cv-border);border-radius:10px;padding:14px;margin-top:8px;'>
  <div style='font-size:0.68rem;font-family:monospace;color:var(--cv-text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Near Real-Time Wind (Open-Meteo)</div>
  <div style='font-size:0.92rem;font-weight:600;color:#38bdf8;margin-bottom:6px;'>{float(_wind_sb["wind_speed_kmh"]):.1f} km/h from {_wf}</div>
  <div style='font-size:0.78rem;color:var(--cv-text-secondary);'>Clouds drift toward: <b>{_drift_sb}</b></div>
  <div style='font-size:0.75rem;color:var(--cv-text-muted);margin-top:6px;line-height:1.5;'>{_fresh_sb}</div>
</div>
""", unsafe_allow_html=True)

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
                _sun_color  = "var(--cv-text-muted)"
            elif _el < 15:
                _sun_status = "🌅 Sun near horizon"
                _sun_color  = "#f59e0b"
            else:
                _sun_status = "☀️ Sun above horizon"
                _sun_color  = "#fbbf24"
            st.markdown(f"""
<div style='background:var(--cv-card-bg);border:1px solid var(--cv-border);border-radius:10px;padding:14px;margin-top:8px;'>
  <div style='font-size:0.68rem;font-family:monospace;color:var(--cv-text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Live Sun Position</div>
  <div style='font-size:0.92rem;font-weight:600;color:{_sun_color};margin-bottom:6px;'>{_sun_status}</div>
  <div style='font-size:0.78rem;color:var(--cv-text-secondary);'>Azimuth: <b style='color:var(--cv-text-primary);'>{_az:.1f}°</b> ({_sun_dir})</div>
  <div style='font-size:0.78rem;color:var(--cv-text-secondary);'>Elevation: <b style='color:var(--cv-text-primary);'>{_el:.1f}°</b></div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Install pvlib + pandas for live sun tracking:\n`pip install pvlib pandas`")

    st.markdown("---")
    st.markdown("### 🏭 Second Solar Plant (Remote)")
    st.caption("Set location of your second plant relative to Plant 1 (camera location)")

    plant2_dist_km = st.number_input(
        "Distance from Plant 1 (km)", min_value=0.1, max_value=500.0,
        value=20.0, step=0.5, format="%.1f", key="plant2_dist_km"
    )

    _bearing_options = {
        "West (270°)": 270.0, "East (90°)": 90.0,
        "North (0°)": 0.0, "South (180°)": 180.0,
        "NW (315°)": 315.0, "NE (45°)": 45.0,
        "SW (225°)": 225.0, "SE (135°)": 135.0,
    }
    _bearing_label = st.selectbox(
        "Direction of Plant 2 from Plant 1",
        list(_bearing_options.keys()),
        index=0, key="plant2_direction_label"
    )
    plant2_bearing_deg = _bearing_options[_bearing_label]
    st.session_state["plant2_bearing_deg"] = plant2_bearing_deg

    # Mini info card
    st.markdown(f"""
<div style='background:var(--cv-card-bg);border:1px solid var(--cv-border);border-radius:10px;padding:12px;margin-top:6px;'>
  <div style='font-size:0.68rem;font-family:monospace;color:var(--cv-text-muted);text-transform:uppercase;
              letter-spacing:0.1em;margin-bottom:6px;'>Plant 2 Config</div>
  <div style='font-size:0.82rem;color:var(--cv-text-secondary);'>📍 {plant2_dist_km:.1f} km {_bearing_label.split(" ")[0]} of Plant 1</div>
  <div style='font-size:0.75rem;color:var(--cv-text-muted);margin-top:4px;'>Bearing: {plant2_bearing_deg:.0f}°</div>
</div>
""", unsafe_allow_html=True)

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

    video_source = st.radio(
        "Source",
        options=["Upload a file", "YouTube link"],
        horizontal=True,
        key="video_source_choice",
        label_visibility="collapsed",
    )

    video_path = None      # local file path that the pipeline below will read
    uploaded_video = None

    if video_source == "Upload a file":
        uploaded_video = st.file_uploader("Drop an MP4 / AVI / MOV file here",
                                           type=["mp4","avi","mov"], key="video_upload")
        if uploaded_video is not None:
            uploaded_video.seek(0)
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()
            tfile.close()
            video_path = tfile.name

    else:  # YouTube link
        yt_url = st.text_input(
            "Paste a YouTube link",
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url_input",
        )
        if yt_url:
            if yt_dlp is None:
                st.error(
                    "⚠️ YouTube downloads need the `yt-dlp` package, which isn't "
                    "installed yet. Run `pip install yt-dlp` and restart the app."
                )
            else:
                cache_key = f"yt_dl_{yt_url}"
                if st.session_state.get(cache_key) is None:
                    with st.spinner("⬇️ Fetching video from YouTube… (live streams take ~1 min)"):
                        dl_path, dl_error = download_youtube_video(yt_url)
                    if dl_error:
                        st.error(f"⚠️ {dl_error}")
                        st.session_state[cache_key] = False
                    else:
                        st.session_state[cache_key] = dl_path
                cached = st.session_state.get(cache_key)
                if cached:
                    video_path = cached
                    st.success("✅ Video downloaded — analysing below.")
                elif cached is False:
                    if st.button("🔄 Retry download", key="yt_retry"):
                        del st.session_state[cache_key]
                        st.rerun()

    fov_video = st.slider("📷 Camera FOV (degrees)", 30, 120, 75,
                          help="Phone: 70-80° | Wide angle: 90-120° | Telephoto: 30-50°")

    # ── Preview uploaded / downloaded video immediately ──
    if video_source == "Upload a file" and uploaded_video is not None:
        st.markdown('<div class="cv-eyebrow" style="margin-top:16px;margin-bottom:8px;">📹 Uploaded Video</div>', unsafe_allow_html=True)
        uploaded_video.seek(0)
        st.video(uploaded_video)
        st.divider()
    elif video_source == "YouTube link" and video_path:
        st.markdown('<div class="cv-eyebrow" style="margin-top:16px;margin-bottom:8px;">📹 Downloaded Video</div>', unsafe_allow_html=True)
        with open(video_path, "rb") as f:
            st.video(f.read())
        st.divider()

    if video_path is not None:

        with st.spinner("🔍 Analysing video..."):
            cap          = cv2.VideoCapture(video_path)
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

            speed_mps, speed_kmh, focal_length_px, distance_m, height_m = \
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
                media_ts = extract_video_datetime(video_path)
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
                         fw, pixel_disp, delta_t_sec, focal_length_px,
                         distance_m, speed_mps, speed_kmh, dist_5, dist_15,
                         coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
                         vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
                         time_to_exit_min=time_to_exit_min,
                         media_timestamp=media_ts, timestamp_source=ts_source,
                         image_elevation_est=img_el_est, image_elevation_conf=img_el_conf,
                         image_elevation_note=img_el_note,
                         plant2_dist_km=st.session_state.get("plant2_dist_km", 20.0),
                         plant2_bearing_deg=st.session_state.get("plant2_bearing_deg", 270.0),
                         user_lat=st.session_state.get("user_lat"),
                         user_lon=st.session_state.get("user_lon"),
                         use_wind_blend=st.session_state.get("use_wind_blend", True))
            
                        # ── Detection video — shown RIGHT HERE under uploader ──
            st.markdown('<div class="cv-eyebrow">📦 Cloud Detection Video</div>', unsafe_allow_html=True)
            with st.spinner("Generating detection video…"):
                tmp_box = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_box.close()
                generate_boxed_video(
                    video_path, tmp_box.name,
                    speed_kmh, speed_mps, direction,
                    cloud_type, height_m, dist_5, dist_15, fov=fov_video,
                    time_to_exit_min=999
                )
                with open(tmp_box.name, "rb") as f:
                    vdata = f.read()
                _box_mb = len(vdata) / (1024 * 1024)
                _h264_ok = _resolve_ffmpeg_exe() is not None
            st.video(vdata)
            if _h264_ok:
                st.caption(
                    f"✅ WhatsApp-ready H.264 MP4 ({_box_mb:.1f} MB) — "
                    "Download karo, phir phone gallery se WhatsApp pe share karo."
                )
            else:
                st.warning(
                    "⚠️ ffmpeg not found — video may not upload on WhatsApp. "
                    "Run: `pip install imageio-ffmpeg` then restart the app."
                )
            if _box_mb > 64:
                st.caption("💡 File 64 MB se badi hai — WhatsApp ke liye chhota video use karo ya clip trim karo.")
            st.download_button("📥 Download Detection Video (WhatsApp)", data=vdata,
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
                encode_mp4_for_mobile(tmp_pred.name)
                with open(tmp_pred.name, "rb") as f:
                    vdata2 = f.read()
            st.video(vdata2)
            st.download_button("📥 Download Prediction Video", data=vdata2,
                               file_name=f"cloud_{cloud_type}_prediction.mp4",
                               mime="video/mp4", key="dl_pred")
            try: os.unlink(tmp_pred.name)
            except: pass

        if video_source == "Upload a file":
            try: os.unlink(video_path)
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
            speed_mps, speed_kmh, focal_length_px, distance_m, height_m = \
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
                         fw, avg_disp, interval, focal_length_px,
                         distance_m, speed_mps, speed_kmh, dist_5, dist_15,
                         coverage_pct=cov_pct, density_label=den_label, density_color=den_color,
                         vis_verdict=vis_verdict, vis_reason=vis_reason, vis_color=vis_color,
                         time_to_exit_min=time_to_exit_min,
                         media_timestamp=media_ts_i, timestamp_source=ts_source_i,
                         image_elevation_est=img_el_est_i, image_elevation_conf=img_el_conf_i,
                         image_elevation_note=img_el_note_i,
                         plant2_dist_km=st.session_state.get("plant2_dist_km", 20.0),
                         plant2_bearing_deg=st.session_state.get("plant2_bearing_deg", 270.0),
                         user_lat=st.session_state.get("user_lat"),
                         user_lon=st.session_state.get("user_lon"),
                         use_wind_blend=st.session_state.get("use_wind_blend", True))

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
                        encode_mp4_for_mobile(tmp.name)
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

# ══════════════════════ TWO-CAMERA TAB ═════════════════════════
with tab3:
    render_two_camera_ui(tab3)

# ══════════════════════ LIVE DUAL-CAMERA TAB ═══════════════════
with tab4:
    st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🔴 Live Dual-Camera Cloud Tracking</div>',
                unsafe_allow_html=True)
    st.subheader("Real-Time Cloud Movement — Both Cameras")
    st.info(
        "Enter an RTSP/HTTP camera URL **or a YouTube link** (live streams supported), "
        "then click **Start Live View**. Camera 2 is optional. "
        "Each detected cloud gets a light-green bounding box — the **thicker** the border, "
        "the **denser** that cloud is. Arrows show live motion direction between refreshes.",
        icon="🛰️"
    )

    lc1, lc2 = st.columns(2)
    with lc1:
        live_cam1_url = st.text_input(
            "📷 Camera 1 stream URL",
            placeholder="rtsp://... or https://youtu.be/...",
            key="live_cam1_url"
        )
    with lc2:
        live_cam2_url = st.text_input(
            "📷 Camera 2 stream URL (optional)",
            placeholder="rtsp://... or leave empty for single camera",
            key="live_cam2_url"
        )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        live_refresh_sec = st.slider("Refresh interval (sec)", 0.5, 5.0, 1.5, 0.5,
                                      key="live_refresh_sec",
                                      help="Lower = smoother but more load on the camera/network")
    with sc2:
        live_sky_pct = st.slider("Horizon position (%)", 50, 90, 78, key="live_sky_pct",
                                  help="How far down from the top of frame the sky region ends")
    with sc3:
        live_max_thick = st.slider("Max box thickness (densest cloud)", 3, 15, 9,
                                    key="live_max_thick")

    bcol1, bcol2, bcol3 = st.columns([1, 1, 2])
    with bcol1:
        live_start = st.button("▶️ Start Live View", key="live_start_btn", use_container_width=True)
    with bcol2:
        live_stop = st.button("⏹️ Stop", key="live_stop_btn", use_container_width=True)
    with bcol3:
        st.caption("Keep this tab open — the feeds auto-refresh on their own.")

    if live_start:
        if not live_cam1_url.strip():
            st.error("Please enter a stream URL for Camera 1.")
        else:
            st.session_state["live_running"] = True

    if live_stop:
        for _u in (live_cam1_url, live_cam2_url):
            if _u and _u.strip():
                release_live_capture(_u)
        st.session_state["live_running"] = False
        st.session_state.pop("live_prev_gray_cam1", None)
        st.session_state.pop("live_prev_gray_cam2", None)
        st.info("Live view stopped and streams disconnected.")

    st.divider()

    if st.session_state.get("live_running", False):
        render_live_dual_camera_view(
            live_cam1_url, live_cam2_url,
            live_refresh_sec, live_sky_pct / 100.0, live_max_thick
        )
    else:
        st.warning(
            "Live view is stopped. Enter a Camera 1 URL (Camera 2 optional) "
            "and click **Start Live View**."
        )

# ══════════════════════ QUICK IDENTIFY TAB ═════════════════════
with tab5:
    st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">🔎 Quick Identify</div>',
                unsafe_allow_html=True)
    st.subheader("Single Image — Cloud Type & Height")
    st.info(
        "Upload **one** sky photo. No speed/direction/shadow analysis here — "
        "just cloud type, estimated base height, model confidence, and sky coverage.",
        icon="ℹ️"
    )

    _EXAMPLE_SKY_SVG_B64 = (
        "PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDMyMCAxODAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGRlZnM+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InNreUdyYWQiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMCUiIHkyPSIxMDAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzFlNmZiOCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjU1JSIgc3RvcC1jb2xvcj0iIzNhOWJkOSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiM3ZWMzZWMiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8cmFkaWFsR3JhZGllbnQgaWQ9ImNsb3VkU2hhZGUiIGN4PSI1MCUiIGN5PSIzNSUiIHI9IjY1JSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNmZmZmZmYiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjY2ZlMGVjIi8+CiAgICA8L3JhZGlhbEdyYWRpZW50PgogIDwvZGVmcz4KCiAgPHJlY3QgeD0iMCIgeT0iMCIgd2lkdGg9IjMyMCIgaGVpZ2h0PSIxODAiIHJ4PSIxMCIgZmlsbD0idXJsKCNza3lHcmFkKSIvPgoKICA8ZyBvcGFjaXR5PSIwLjkiPgogICAgPGVsbGlwc2UgY3g9IjcwIiBjeT0iMTUwIiByeD0iMTUwIiByeT0iNDAiIGZpbGw9IiNiY2RjZWYiIG9wYWNpdHk9IjAuMzUiLz4KICA8L2c+CgogIDxnPgogICAgPGVsbGlwc2UgY3g9IjEyMCIgY3k9Ijk1IiByeD0iNTgiIHJ5PSIzNCIgZmlsbD0idXJsKCNjbG91ZFNoYWRlKSIvPgogICAgPGVsbGlwc2UgY3g9IjkwIiAgY3k9IjEwNSIgcng9IjM2IiByeT0iMjQiIGZpbGw9InVybCgjY2xvdWRTaGFkZSkiLz4KICAgIDxlbGxpcHNlIGN4PSIxNjAiIGN5PSIxMDAiIHJ4PSI0MiIgcnk9IjI2IiBmaWxsPSJ1cmwoI2Nsb3VkU2hhZGUpIi8+CiAgICA8ZWxsaXBzZSBjeD0iMTM1IiBjeT0iODAiICByeD0iMzQiIHJ5PSIyMiIgZmlsbD0iI2ZmZmZmZiIvPgogICAgPGVsbGlwc2UgY3g9IjEwNSIgY3k9Ijg1IiAgcng9IjI2IiByeT0iMTgiIGZpbGw9IiNmZmZmZmYiLz4KCiAgICA8ZWxsaXBzZSBjeD0iMTE1IiBjeT0iMTEwIiByeD0iNDgiIHJ5PSIxNCIgZmlsbD0iI2FlYmZkNCIgb3BhY2l0eT0iMC41NSIvPgogIDwvZz4KCiAgPGc+CiAgICA8ZWxsaXBzZSBjeD0iMjM1IiBjeT0iNjAiIHJ4PSIzNCIgcnk9IjE4IiBmaWxsPSJ1cmwoI2Nsb3VkU2hhZGUpIi8+CiAgICA8ZWxsaXBzZSBjeD0iMjE1IiBjeT0iNjYiIHJ4PSIyMiIgcnk9IjEzIiBmaWxsPSJ1cmwoI2Nsb3VkU2hhZGUpIi8+CiAgICA8ZWxsaXBzZSBjeD0iMjUwIiBjeT0iNjgiIHJ4PSIyMCIgcnk9IjEyIiBmaWxsPSIjZmZmZmZmIi8+CiAgICA8ZWxsaXBzZSBjeD0iMjMyIiBjeT0iNzIiIHJ4PSIzMCIgcnk9IjkiIGZpbGw9IiNhZWJmZDQiIG9wYWNpdHk9IjAuNSIvPgogIDwvZz4KCiAgPGcgb3BhY2l0eT0iMC44NSI+CiAgICA8ZWxsaXBzZSBjeD0iNDgiIGN5PSI0OCIgcng9IjIwIiByeT0iMTAiIGZpbGw9IiNlN2YxZjgiLz4KICAgIDxlbGxpcHNlIGN4PSIzNCIgY3k9IjUyIiByeD0iMTMiIHJ5PSI3IiBmaWxsPSIjZTdmMWY4Ii8+CiAgPC9nPgoKICA8Y2lyY2xlIGN4PSIyNzAiIGN5PSIzMiIgcj0iMTQiIGZpbGw9IiNmZmY0ZDYiIG9wYWNpdHk9IjAuOSIvPgo8L3N2Zz4K"
    )
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;background:var(--cv-card-bg);border:1px solid var(--cv-border);
            border-radius:14px;padding:14px 18px;margin-bottom:18px;">
  <img src="data:image/svg+xml;base64,{_EXAMPLE_SKY_SVG_B64}"
       style="width:140px;height:auto;border-radius:8px;flex-shrink:0;" />
  <div>
    <div style="font-size:0.78rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Good example</div>
    <div style="font-size:0.85rem;color:var(--cv-text-secondary);line-height:1.5;">
      Open sky with clouds clearly visible, like this — minimal buildings, trees, or
      indoor clutter in frame.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    quick_image = st.file_uploader(
        "Drop a single sky photo here",
        type=["jpg", "jpeg", "png"], key="quick_img_upload"
    )

    if quick_image is not None:
        quick_image.seek(0)
        quick_pil = Image.open(quick_image).convert("RGB")

        qc1, qc2 = st.columns([1, 1.4])
        with qc1:
            st.image(quick_pil, caption=quick_image.name, use_container_width=True)

        with qc2:
            with st.spinner("🔍 Checking image..."):
                # ── Gate: is this actually an outdoor sky/cloud scene? ──
                outdoor_ok, sky_related_score, top_label = is_outdoor_sky_scene(quick_pil)

            if not outdoor_ok:
                st.markdown(f"""
<div style="background:var(--cv-card-bg-red);border:1.5px solid #ef444455;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">
    <div style="width:52px;height:52px;border-radius:12px;background:#ef444418;border:1px solid #ef444433;
                display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">🚫</div>
    <div>
      <div style="font-size:1.15rem;font-weight:700;color:var(--cv-text-primary);margin-bottom:4px;">Not a sky photo</div>
      <div style="font-size:0.78rem;color:var(--cv-text-muted);font-family:'JetBrains Mono',monospace;">
        This looks more like: {top_label} · Sky-related score: {sky_related_score:.2f}
      </div>
    </div>
  </div>
  <div style="font-size:0.88rem;color:var(--cv-text-secondary);line-height:1.5;">
    A general image check says this isn't an outdoor sky/cloud scene.
  </div>
  <div style="margin-top:14px;font-size:0.82rem;color:var(--cv-text-muted);">
    This app only classifies cloud types from photos of the sky. Please upload a photo
    where the sky/clouds are clearly visible.
  </div>
</div>
""", unsafe_allow_html=True)
            else:
                with st.spinner("🔍 Identifying cloud type..."):
                    q_type, q_conf, q_diag = predict_cloud_type([quick_pil], return_diagnostics=True)

                    q_bgr = cv2.cvtColor(
                        np.array(quick_pil.resize((640, 480))), cv2.COLOR_RGB2BGR
                    )
                    q_sky_h = int(q_bgr.shape[0] * 0.78)
                    q_cov_pct, q_den_label, q_den_color = compute_cloud_density(q_bgr, q_sky_h)
                    q_height_m = cloud_height.get(q_type, 0)
                    q_emoji = cloud_emoji.get(q_type, "☁️")

                # ── Gate 2: even though it looked sky-like, is the model
                # itself actually confident and decisive about a class? ──
                model_unsure = (q_conf < 55.0) or (q_diag["top2_gap"] < 0.12) or (q_diag["normalized_entropy"] > 0.75)

                if model_unsure:
                    st.markdown(f"""
<div style="background:var(--cv-card-bg-amber);border:1.5px solid #f59e0b55;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">
    <div style="width:52px;height:52px;border-radius:12px;background:#f59e0b18;border:1px solid #f59e0b33;
                display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">⚠️</div>
    <div>
      <div style="font-size:1.15rem;font-weight:700;color:var(--cv-text-primary);margin-bottom:4px;">Uncertain — best guess was "{q_type}"</div>
      <div style="font-size:0.78rem;color:var(--cv-text-muted);font-family:'JetBrains Mono',monospace;">
        Confidence: {q_conf:.1f}% · Top-2 gap: {q_diag['top2_gap']:.2f} · Entropy: {q_diag['normalized_entropy']:.2f}
      </div>
    </div>
  </div>
  <div style="font-size:0.88rem;color:var(--cv-text-secondary);line-height:1.5;">
    This image passed the sky check, but the model isn't confident enough about a specific
    cloud type to report one reliably. Try a clearer photo with more of the sky/cloud
    visible and less clutter (buildings, trees, glare, etc.) near the frame edges.
  </div>
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style="background:var(--cv-card-bg);border:1.5px solid #38bdf855;border-radius:14px;padding:24px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
    <div style="width:52px;height:52px;border-radius:12px;background:#38bdf818;border:1px solid #38bdf833;
                display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">{q_emoji}</div>
    <div>
      <div style="font-size:1.2rem;font-weight:700;color:var(--cv-text-primary);margin-bottom:4px;">{q_type}</div>
      <div style="font-size:0.78rem;color:var(--cv-text-muted);font-family:'JetBrains Mono',monospace;">
        Model confidence: {q_conf:.1f}%
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;">
    <div style="background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Estimated Height</div>
      <div style="font-size:1.7rem;font-weight:700;color:#38bdf8;line-height:1;">{q_height_m:,} m</div>
    </div>
    <div style="background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Confidence</div>
      <div style="font-size:1.7rem;font-weight:700;color:var(--cv-text-primary);line-height:1;">{q_conf:.1f}%</div>
    </div>
    <div style="background:var(--cv-card-bg-inset);border:1px solid var(--cv-border);border-radius:10px;padding:16px 18px;">
      <div style="font-size:0.63rem;font-family:'JetBrains Mono',monospace;color:var(--cv-text-muted);text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">Sky Coverage</div>
      <div style="font-size:1.7rem;font-weight:700;color:{q_den_color};line-height:1;">{q_cov_pct}%</div>
    </div>
  </div>
  <div style="margin-top:14px;">
    <span class="cv-pill">Density: {q_den_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)

                    st.caption(
                        "Height is a typical reference altitude for the detected cloud type "
                        "(not measured from this image — single-image height requires the "
                        "Two-Camera Triangulation tab for an actual measurement)."
                    )
    else:
        st.warning("Upload an image to identify the cloud type.")

# ══════════════════════ LIVE SINGLE-CAMERA TAB ═══════════════════
with tab6:
    st.markdown('<div class="cv-eyebrow" style="margin-bottom:12px;">📡 Live Single-Camera Cloud Tracking</div>',
                unsafe_allow_html=True)
    st.subheader("Real-Time Cloud Movement — Single Camera (Full View)")
    st.info(
        "Enter an RTSP/HTTP camera URL **or a YouTube link** (live streams supported), "
        "then click **▶️ Start**. Cloud regions get light-green bounding boxes — "
        "**thicker border = denser cloud**. Arrows show live motion direction. "
        "Full-width single-camera view with expanded metrics.",
        icon="📡"
    )

    # ── URL input ──
    sc_cam_url = st.text_input(
        "📷 Camera stream URL",
        placeholder="rtsp://192.168.1.100:554/stream  or  https://youtu.be/...",
        key="sc_cam_url"
    )

    # ── Settings row ──
    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
    with sc_col1:
        sc_refresh_sec = st.slider(
            "Refresh interval (sec)", 0.5, 5.0, 1.5, 0.5,
            key="sc_refresh_sec",
            help="Lower = smoother but more network load"
        )
    with sc_col2:
        sc_sky_pct = st.slider(
            "Horizon position (%)", 50, 90, 78,
            key="sc_sky_pct",
            help="How far down from top of frame the sky region ends"
        )
    with sc_col3:
        sc_max_thick = st.slider(
            "Max box thickness (densest)", 3, 15, 9,
            key="sc_max_thick"
        )
    with sc_col4:
        sc_fov = st.slider(
            "Camera FOV (°)", 30, 120, 75,
            key="sc_fov",
            help="Phone: 70-80° | Wide: 90-120° | Telephoto: 30-50°"
        )

    # ── Start / Stop buttons ──
    bc1, bc2, bc3 = st.columns([1, 1, 3])
    with bc1:
        sc_start = st.button("▶️ Start", key="sc_start_btn", use_container_width=True)
    with bc2:
        sc_stop  = st.button("⏹️ Stop",  key="sc_stop_btn",  use_container_width=True)
    with bc3:
        st.caption("Keep this tab open — the feed auto-refreshes on its own.")

    if sc_start:
        if not sc_cam_url.strip():
            st.error("Please enter a camera stream URL first.")
        else:
            st.session_state["sc_live_running"] = True

    if sc_stop:
        if sc_cam_url.strip():
            release_live_capture(sc_cam_url)
        st.session_state["sc_live_running"] = False
        st.session_state.pop("sc_prev_gray", None)
        st.session_state.pop("sc_history_cov", None)
        st.session_state.pop("sc_history_spd", None)
        st.info("Live single-camera view stopped and stream disconnected.")

    st.divider()

    def _sc_detect_clouds(frame_bgr, sky_h, min_thick, max_thick):
        """
        Improved single-camera cloud detection.

        Problems fixed vs the shared detect_clouds():
        - Overcast / near-100% coverage sky gave ONE giant box covering everything.
        - Now uses CLAHE contrast enhancement + adaptive thresholding so individual
          cloud *clusters* are found even when the sky is nearly white.
        - Hard min/max area filters remove noise blobs AND sky-filling mega-boxes.
        - Returns (annotated_frame, gray, coverage_pct, box_info_list,
                   flow_speed_px, flow_dir_deg, prev_gray)
        """
        H, W = frame_bgr.shape[:2]
        out = frame_bgr.copy()

        # ── 1. Sky region only ──
        sky = frame_bgr[:sky_h, :]
        sky_gray = cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY)
        sky_hsv  = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV)

        # ── 2. CLAHE on L channel to enhance local contrast ──
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        sky_cl = clahe.apply(sky_gray)

        # ── 3. Adaptive threshold — finds local bright regions (clouds) ──
        adapt = cv2.adaptiveThreshold(
            sky_cl, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51, C=-8
        )

        # ── 4. Also remove deep-blue sky pixels (not clouds) ──
        hue = sky_hsv[:, :, 0]
        sat = sky_hsv[:, :, 1]
        blue_sky_mask = cv2.inRange(hue, 90, 140)          # blue hue
        clear_blue    = cv2.bitwise_and(blue_sky_mask,
                            cv2.threshold(sat, 55, 255, cv2.THRESH_BINARY)[1])
        adapt = cv2.bitwise_and(adapt, cv2.bitwise_not(clear_blue))

        # ── 5. Morphology: close small gaps, open noise ──
        k_close = np.ones((18, 18), np.uint8)
        k_open  = np.ones((8,  8),  np.uint8)
        mask = cv2.morphologyEx(adapt,  cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  k_open)

        # ── 6. Find contours — individual cloud blobs ──
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_sky_px   = max(sky_h * W, 1)
        total_cloud_px = 0
        box_infos      = []

        # Area limits: ignore tiny noise AND giant sky-filling blobs
        min_area = 800
        max_area = (sky_h * W) * 0.70   # never more than 70% of sky in one box

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + bw), min(sky_h, y + bh)

            # Cloud fill density inside this bounding box
            box_mask = np.zeros((sky_h, W), dtype=np.uint8)
            cv2.drawContours(box_mask, [cnt], -1, 255, -1)
            box_area   = max((x2 - x1) * (y2 - y1), 1)
            cloud_fill = int(cv2.countNonZero(box_mask[y1:y2, x1:x2]))
            density_pct = float(min(100.0, (cloud_fill / box_area) * 100))

            total_cloud_px += cloud_fill

            thickness = int(round(np.interp(density_pct, [0, 100],
                                             [min_thick, max_thick])))
            thickness = max(min_thick, thickness)

            cv2.rectangle(out, (x1, y1), (x2, y2), LIGHT_GREEN_BGR, thickness)

            # Density label above box (clipped to frame)
            lbl = f"{density_pct:.0f}% dense"
            (tw, th_), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ly = y1 - 6 if y1 - 6 - th_ > 2 else y2 + th_ + 6
            ly = max(th_ + 4, min(ly, sky_h - 4))
            cv2.rectangle(out, (x1 - 1, ly - th_ - 3), (x1 + tw + 5, ly + bl + 1),
                          (20, 50, 20), -1)
            cv2.putText(out, lbl, (x1 + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, LIGHT_GREEN_BGR, 1, cv2.LINE_AA)

            box_infos.append({
                "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                "density_pct": round(density_pct, 1)
            })

        coverage_pct = round((total_cloud_px / total_sky_px) * 100, 1)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return out, gray, coverage_pct, box_infos

    def render_live_single_camera_view(cam_url, refresh_sec, sky_frac, max_thick, fov_deg):
        """
        Auto-refreshing Streamlit fragment: single camera, full-width layout,
        with expanded metrics panel and a mini history chart.
        """
        @st_fragment(run_every=refresh_sec)
        def _sc_fragment():
            cap = get_live_capture(cam_url)

            if cap is None or not cap.isOpened():
                if _is_youtube_url(cam_url):
                    st.error(
                        "Could not open this YouTube stream. "
                        "Check the link, ensure yt-dlp is installed, then click Start again."
                    )
                else:
                    st.error(
                        "Could not open the stream. Check the URL and click Start again."
                    )
                return

            ret, frame = read_fresh_live_frame(cap)
            if not ret or frame is None:
                st.warning("No frame received — stream may be buffering or down.")
                return

            OUT_W, OUT_H = 960, 540
            frame    = cv2.resize(frame, (OUT_W, OUT_H))
            sky_h    = int(OUT_H * sky_frac)
            prev_gray = st.session_state.get("sc_prev_gray")

            # ── Better detection (custom, not the shared draw_density_boxes_live) ──
            annotated, gray, cov_pct, boxes = _sc_detect_clouds(
                frame, sky_h, 1, max_thick
            )
            st.session_state["sc_prev_gray"] = gray

            # ── Optical flow for speed & direction ──
            spd_px  = 0.0
            dir_deg = None
            if prev_gray is not None and prev_gray.shape == gray.shape:
                prev_sky = prev_gray[:sky_h, :]
                curr_sky = gray[:sky_h, :]
                flow = cv2.calcOpticalFlowFarneback(
                    prev_sky, curr_sky, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
                moving = mag > 0.5
                if np.any(moving):
                    spd_px  = float(np.median(mag[moving]))
                    dir_deg = float(np.median(ang[moving]))

                # Draw sparse flow arrows on annotated frame
                step = 36
                for yy in range(0, sky_h, step):
                    for xx in range(0, OUT_W, step):
                        dx, dy = flow[yy, xx]
                        if math.hypot(dx, dy) < 0.5:
                            continue
                        p2 = (int(xx + dx * 5), int(yy + dy * 5))
                        cv2.arrowedLine(annotated, (xx, yy), p2,
                                        LIGHT_GREEN_BGR, 1, tipLength=0.45)

            # ── HUD top-left: LIVE / Coverage / Time ──
            n_clouds = len(boxes)
            hud_lines = [
                f"LIVE  Coverage: {cov_pct:.1f}%",
                datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                f"Clouds detected: {n_clouds}",
            ]
            hud_h = 20 + len(hud_lines) * 22
            hud_w = 260
            ov = annotated.copy()
            cv2.rectangle(ov, (0, 0), (hud_w, hud_h), (10, 10, 10), -1)
            cv2.addWeighted(ov, 0.55, annotated, 0.45, 0, annotated)
            cv2.rectangle(annotated, (0, 0), (hud_w, hud_h), LIGHT_GREEN_BGR, 1)
            for li, txt in enumerate(hud_lines):
                cv2.putText(annotated, txt, (10, 18 + li * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            (255, 255, 255) if li == 0 else (200, 200, 200),
                            1, cv2.LINE_AA)

            # ── Horizon line ──
            cv2.line(annotated, (0, sky_h), (OUT_W, sky_h),
                     (100, 200, 100), 1)

            # ── Compass HUD — bottom-right (always inside frame) ──
            if dir_deg is not None:
                cx_ = OUT_W - 70
                cy_ = OUT_H - 70
                rad = math.radians(dir_deg)
                ex_ = int(cx_ + 40 * math.cos(rad))
                ey_ = int(cy_ + 40 * math.sin(rad))
                # clamp endpoint inside frame
                ex_ = max(0, min(OUT_W - 1, ex_))
                ey_ = max(0, min(OUT_H - 1, ey_))
                cov2 = annotated.copy()
                cv2.circle(cov2, (cx_, cy_), 46, (10, 10, 10), -1)
                cv2.addWeighted(cov2, 0.6, annotated, 0.4, 0, annotated)
                cv2.circle(annotated, (cx_, cy_), 46, LIGHT_GREEN_BGR, 1)
                cv2.arrowedLine(annotated, (cx_, cy_), (ex_, ey_),
                                LIGHT_GREEN_BGR, 2, tipLength=0.35)
                dir_lbl = angle_to_direction(dir_deg)
                (dw, _), _ = cv2.getTextSize(dir_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.putText(annotated, dir_lbl, (cx_ - dw // 2, OUT_H - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

            # ── Speed estimation ──
            speed_kmh_live = 0.0
            if spd_px > 0:
                _, speed_kmh_live, *_ = pixels_to_kmh(
                    spd_px * refresh_sec, refresh_sec, "Cumulus", OUT_W, fov_deg
                )

            # ── Display frame ──
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            # ── Metrics row ──
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("☁️ Coverage",      f"{cov_pct:.1f}%")
            m2.metric("📦 Cloud Regions", str(n_clouds))
            m3.metric("💨 Flow Speed",    f"{spd_px:.1f} px/s" if spd_px > 0 else "—")
            m4.metric("🧭 Direction",     angle_to_direction(dir_deg) if dir_deg is not None else "—")
            m5.metric("🚀 Speed Est.",    f"{speed_kmh_live:.1f} km/h" if speed_kmh_live > 0 else "—")
            densest_val = max((b["density_pct"] for b in boxes), default=0.0)
            m6.metric("🌫️ Densest Box",  f"{densest_val:.0f}%" if boxes else "—")

            # ── Per-box detail table ──
            if boxes:
                with st.expander(f"📋 Cloud region details ({n_clouds} detected)", expanded=False):
                    rows_html = ""
                    for i, b in enumerate(boxes, 1):
                        bx_   = b["x"]
                        by_   = b["y"]
                        bw_   = b["w"]
                        bh_px = b["h"]
                        bd_   = b["density_pct"]
                        bar_w = min(int(bd_), 100)
                        rows_html += (
                            f'<tr style="border-bottom:1px solid var(--cv-border);">'
                            f'<td style="padding:6px 10px;color:var(--cv-text-muted);font-family:monospace;">#{i}</td>'
                            f'<td style="padding:6px 10px;">{bx_}, {by_}</td>'
                            f'<td style="padding:6px 10px;">{bw_} × {bh_px}</td>'
                            f'<td style="padding:6px 10px;">'
                            f'<div style="display:flex;align-items:center;gap:8px;">'
                            f'<div style="width:{bar_w}px;height:8px;'
                            f'background:linear-gradient(90deg,#22c55e,#16a34a);'
                            f'border-radius:4px;flex-shrink:0;"></div>'
                            f'<span>{bd_:.1f}%</span>'
                            f'</div></td>'
                            f'</tr>'
                        )
                    st.markdown(
                        f'<table style="width:100%;border-collapse:collapse;font-size:0.83rem;'
                        f'color:var(--cv-text-secondary);">'
                        f'<thead><tr style="border-bottom:2px solid var(--cv-border);">'
                        f'<th style="padding:6px 10px;text-align:left;color:var(--cv-text-muted);">#</th>'
                        f'<th style="padding:6px 10px;text-align:left;">Position (x,y)</th>'
                        f'<th style="padding:6px 10px;text-align:left;">Size (px)</th>'
                        f'<th style="padding:6px 10px;text-align:left;">Density</th>'
                        f'</tr></thead><tbody>{rows_html}</tbody></table>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No distinct cloud regions detected in this frame.")

            # ── Rolling history charts ──
            hist_cov = st.session_state.get("sc_history_cov", [])
            hist_spd = st.session_state.get("sc_history_spd", [])
            hist_cov.append(round(cov_pct, 1))
            hist_spd.append(round(spd_px, 2))
            hist_cov = hist_cov[-20:]
            hist_spd = hist_spd[-20:]
            st.session_state["sc_history_cov"] = hist_cov
            st.session_state["sc_history_spd"] = hist_spd

            if len(hist_cov) >= 3:
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.markdown("**☁️ Coverage history (last 20 readings)**")
                    st.line_chart({"Coverage %": hist_cov}, height=120)
                with ch2:
                    st.markdown("**💨 Flow speed history (px/s)**")
                    st.line_chart({"Flow px/s": hist_spd}, height=120)

            st.caption(
                f"⏱️ Auto-refreshing every {refresh_sec:.1f}s · "
                f"last update {datetime.datetime.now().strftime('%H:%M:%S')}"
            )

        _sc_fragment()

    if st.session_state.get("sc_live_running", False):
        render_live_single_camera_view(
            sc_cam_url, sc_refresh_sec, sc_sky_pct / 100.0,
            sc_max_thick, sc_fov
        )
    else:
        st.warning(
            "Live view is stopped. Enter a Camera stream URL above "
            "and click **▶️ Start** to begin."
        )