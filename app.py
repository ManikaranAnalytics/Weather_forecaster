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
                 theta_deg, distance_m, speed_mps, speed_kmh, dist_5, dist_15):
    emoji = cloud_emoji.get(cloud_type, "☁️")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"{emoji} Cloud Type", cloud_type)
    c2.metric("🎯 Confidence",       f"{confidence:.1f}%")
    c3.metric("🧭 Direction",        direction)
    c4.metric("📍 Height",           f"{height_m:,} m")
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("⚡ Speed",            f"{speed_kmh:.1f} km/h")
    c6.metric("⚡ Speed (m/s)",      f"{speed_mps:.2f} m/s")
    c7.metric("⏳ +5 min",          f"~{dist_5:.2f} km")
    c8.metric("⏳ +15 min",         f"~{dist_15:.2f} km")
    with st.expander("🔬 Calculation Details"):
        st.markdown(f"""
| Parameter | Value |
|---|---|
| FOV | {fov}° | Frame Width | {frame_width} px |
| Degree/Pixel | {deg_per_px:.4f} °/px | Pixel Disp | {pixel_disp:.2f} px/{delta_t:.2f}s |
| Theta | {theta_deg:.4f}° | Tan Distance | {distance_m:.2f} m |
| Speed | {speed_mps:.2f} m/s → {speed_kmh:.1f} km/h |
""")

# ─────────────────────────── BOUNDING BOX FUNCTION ─────────────
def draw_boxes_on_frame(frame, speed_kmh, direction, cloud_type, height_m,
                         dist_5, dist_15, elapsed_sec, prev_gray=None, delta_t=None,
                         fov=75):
    """
    Real video frame pe cloud bounding boxes + per-cloud speed label draw karta hai.
    Agar prev_gray aur delta_t diye hain to har cloud ki individual speed calculate hoti hai.
    """
    OUT_W = frame.shape[1]
    OUT_H = frame.shape[0]
    sky_h = int(OUT_H * 0.75)

    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sky_gray = gray[:sky_h, :]

    # Clouds = bright white regions
    _, thresh = cv2.threshold(sky_gray, 175, 255, cv2.THRESH_BINARY)
    k_close   = np.ones((18, 18), np.uint8)
    k_open    = np.ones((9,  9),  np.uint8)
    thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
    thresh    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k_open)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pre-compute full dense optical flow if prev frame available
    full_flow = None
    if prev_gray is not None and delta_t is not None and delta_t > 0:
        prev_sky = prev_gray[:sky_h, :]
        full_flow = cv2.calcOpticalFlowFarneback(
            prev_sky, sky_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

    for cnt in contours:
        if cv2.contourArea(cnt) < 1200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        pad = 6
        x1 = max(0,       x - pad);    y1 = max(0,       y - pad)
        x2 = min(OUT_W-1, x+w + pad);  y2 = min(OUT_H-1, y+h + pad)

        # ── Per-cloud speed from optical flow ROI ──
        if full_flow is not None:
            roi_flow = full_flow[y1:y2, x1:x2]
            if roi_flow.size > 0:
                mag, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
                roi_pixel_disp = float(np.median(mag))
                _, cloud_speed_kmh, _, _, _, _ = pixels_to_kmh(
                    roi_pixel_disp, delta_t, cloud_type, OUT_W, fov
                )
            else:
                cloud_speed_kmh = speed_kmh
        else:
            cloud_speed_kmh = speed_kmh

        # Glow effect
        glow = frame.copy()
        cv2.rectangle(glow, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 100), 4)
        cv2.addWeighted(glow, 0.3, frame, 0.7, 0, frame)

        # Main box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

        # Corner ticks
        t = 14
        for (px_, py_, sdx, sdy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px_, py_), (px_+sdx*t, py_),       (0, 255, 60), 2)
            cv2.line(frame, (px_, py_), (px_, py_+sdy*t),       (0, 255, 60), 2)

        # Per-cloud speed label
        label = f"{cloud_speed_kmh:.1f} km/h"
        fs    = 0.46
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        lx = x1
        ly = y1 - 5 if y1 - 5 - th > 0 else y1 + th + 6
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
    txt(f"Dir    : {direction}  |  +5min ~{dist_5:.2f} km  |  +15min ~{dist_15:.2f} km",
        86, sc=0.43)

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

    return frame


def generate_boxed_video(input_path, output_path, speed_kmh, speed_mps,
                          direction, cloud_type, height_m, dist_5, dist_15, fov=75):
    """Uploaded video ke har frame pe bounding boxes draw karke naya video banata hai.
    Har cloud ki individual speed uske ROI optical flow se calculate hoti hai."""
    cap    = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
        frame = draw_boxes_on_frame(
            frame, speed_kmh, direction, cloud_type,
            height_m, dist_5, dist_15, elapsed,
            prev_gray=prev_gray, delta_t=delta_t, fov=fov
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
            show_metrics(cloud_type, avg_conf, direction, height_m, fov_video,
                         fw, pixel_disp, delta_t_sec, deg_per_px, theta_deg,
                         distance_m, speed_mps, speed_kmh, dist_5, dist_15)

            st.divider()
            st.subheader("🎬 Output Videos")

            col1, col2 = st.columns(2)

            # ── BUTTON 1: Real video + boxes ──
            with col1:
                st.markdown("**📦 Real Video with Cloud Boxes**")
                st.caption("Tumhara actual video — har cloud ke around box aur speed")
                if st.button("🟩 Generate Boxed Video", key="vid_box"):
                    with st.spinner("⏳ Har frame pe box draw ho raha hai..."):
                        tmp_box = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmp_box.close()
                        generate_boxed_video(
                            tfile.name, tmp_box.name,
                            speed_kmh, speed_mps, direction,
                            cloud_type, height_m, dist_5, dist_15, fov=fov_video
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

            # ── BUTTON 2: Simulated prediction video ──
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
            show_metrics(cloud_type, avg_conf, direction, height_m, fov_images,
                         fw, avg_disp, interval, deg_per_px, theta_deg,
                         distance_m, speed_mps, speed_kmh, dist_5, dist_15)

            st.divider()
            st.subheader("🎬 Output")

            col1, col2 = st.columns(2)

            # ── Images pe boxes ──
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
                                                   fov=fov_images)
                        with cols3[i % 3]:
                            st.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),
                                     caption=f"Image {i+1}", use_container_width=True)

            # ── Prediction video ──
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