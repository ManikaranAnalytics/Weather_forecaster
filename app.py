import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from motion_visualizer import CloudMotionVisualizer

# Load Model
model = load_model("cloud_model.keras")

class_names = [
    "Cumulus",
    "Altocumulus",
    "Cirrus",
    "ClearSky",
    "Stratocumulus",
    "Cumulonimbus",
    "Mixed"
]

st.set_page_config(page_title="CloudVision AI")

st.title("☁️ CloudVision AI")
st.subheader("Cloud Classification and Motion Prediction System")

tab1, tab2, tab3 = st.tabs(
    [
        "Image Analysis",
        "Video Analysis",
        "Multi Image Analysis"
    ]
)

# ---------------- IMAGE TAB ----------------

with tab1:

    uploaded_image = st.file_uploader(
        "Upload Cloud Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        img = Image.open(uploaded_image)

        st.image(img, caption="Uploaded Image")

        img_resized = img.resize((224, 224))

        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        st.success(
            f"Cloud Type: {class_names[predicted_class]}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )
        predicted_class = np.argmax(prediction)
        cloud_type = class_names[predicted_class]
        cloud_height = {
            "Cumulus": 1500,
            "Altocumulus": 4500,
            "Cirrus": 9000,
            "ClearSky": 0,
            "Stratocumulus": 1200,
            "Cumulonimbus": 6000,
            "Mixed": 3500
        }
        height_m = cloud_height.get(cloud_type, 2000)

        st.info(
        f"📍 Estimated Cloud Height: {height_m:,} m"
)

        # -------- GENERATE MOTION VISUALIZATION --------
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎬 Generate Motion Video"):
                with st.spinner("Video generate ho rahi hai..."):
                    visualizer = CloudMotionVisualizer(
                        cloud_type=cloud_type,
                        height_m=height_m,
                        direction="South",  # Default for image analysis
                        pixel_speed=2.0
                    )
                    temp_video = tempfile.NamedTemporaryFile(
                        suffix=".mp4", 
                        delete=False
                    )
                    visualizer.save_video(temp_video.name, duration_minutes=5)
                    
                    with open(temp_video.name, "rb") as f:
                        video_data = f.read()
                    
                    st.download_button(
                        label="📥 Download Video",
                        data=video_data,
                        file_name=f"cloud_{cloud_type}_motion.mp4",
                        mime="video/mp4"
                    )
        
        with col2:
            if st.button("🖼️ Generate Image Sequence"):
                with st.spinner("Images generate ho rahi hain..."):
                    visualizer = CloudMotionVisualizer(
                        cloud_type=cloud_type,
                        height_m=height_m,
                        direction="South",
                        pixel_speed=2.0
                    )
                    frames = visualizer.generate_frame_sequence(num_frames=9)
                    
                    cols = st.columns(3)
                    for idx, frame in enumerate(frames):
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        with cols[idx % 3]:
                            st.image(frame_rgb, use_column_width=True)


# ---------------- VIDEO TAB ----------------

with tab2:

    uploaded_video = st.file_uploader(
        "Upload Cloud Video",
        type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:

    from collections import Counter

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # -------- CLOUD TYPE USING MAJORITY VOTING --------

    sample_points = [
        0.05, 0.15, 0.25, 0.35, 0.45,
        0.55, 0.65, 0.75, 0.85, 0.95
    ]

    predictions_list = []
    confidence_list = []

    for point in sample_points:

        frame_no = int(
            total_frames * point
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_no
        )

        ret, frame = cap.read()

        if not ret:
            continue

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_pil = Image.fromarray(
            frame_rgb
        )

        frame_pil = frame_pil.resize(
            (224,224)
        )

        img_array = image.img_to_array(
            frame_pil
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = img_array / 255.0

        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_class = np.argmax(
            prediction
        )

        predictions_list.append(
            class_names[predicted_class]
        )

        confidence_list.append(
            np.max(prediction) * 100
        )

    cloud_type = Counter(
        predictions_list
    ).most_common(1)[0][0]

    avg_confidence = np.mean(
        confidence_list
    )

    # -------- SPEED & DIRECTION --------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    ret1, frame1 = cap.read()

    for _ in range(10):
        ret2, frame2 = cap.read()

    if ret1 and ret2:

        gray1 = cv2.cvtColor(
            frame1,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            frame2,
            cv2.COLOR_BGR2GRAY
        )

        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )

        avg_speed = np.mean(
            magnitude
        )

        avg_angle = np.mean(
            angle
        )

        pixel_speed = (
            avg_speed * fps
        )

        angle_deg = np.degrees(
            avg_angle
        )

        if 45 <= angle_deg < 135:
            direction = "North"
        elif 135 <= angle_deg < 225:
            direction = "West"
        elif 225 <= angle_deg < 315:
            direction = "South"
        else:
            direction = "East"

        pred_5 = pixel_speed * 300
        pred_15 = pixel_speed * 900

        st.success(
            f"Cloud Type: {cloud_type}"
        )

        st.info(
            f"Confidence: {avg_confidence:.2f}%"
        )

        st.success(
            f"Direction: {direction}"
        )

# Cloud-wise speed factor

        speed_factor = {
            "Cumulus": 0.18,
            "Altocumulus": 0.30,
            "Cirrus": 0.60,
            "ClearSky": 0.00,
            "Stratocumulus": 0.22,
            "Cumulonimbus": 0.45,
            "Mixed": 0.30
        }

        factor = speed_factor.get(
            cloud_type,
            0.20
        )

        estimated_speed_kmh = pixel_speed * factor

        distance_5_min = estimated_speed_kmh * (5/60)

        distance_15_min = estimated_speed_kmh * (15/60)

        st.info(
            f"⚡ Estimated Speed: {estimated_speed_kmh:.1f} km/h"
        )

        st.warning(
            f"⏳ After 5 Minutes: ~{distance_5_min:.2f} km travel"
        )

        st.warning(
            f"⏳ After 15 Minutes: ~{distance_15_min:.2f} km travel"
        )
        
        # -------- GENERATE MOTION VISUALIZATION --------
        st.divider()
        st.subheader("📊 Motion Prediction Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎬 Generate Prediction Video", key="video_tab"):
                with st.spinner("Video generate ho rahi hai..."):
                    visualizer = CloudMotionVisualizer(
                        cloud_type=cloud_type,
                        height_m=height_m,
                        direction=direction,
                        pixel_speed=pixel_speed
                    )
                    temp_video = tempfile.NamedTemporaryFile(
                        suffix=".mp4",
                        delete=False
                    )
                    visualizer.save_video_with_prediction(
                        temp_video.name,
                        prediction_minutes=15
                    )
                    
                    with open(temp_video.name, "rb") as f:
                        video_data = f.read()
                    
                    st.success("✅ Video ready!")
                    st.download_button(
                        label="📥 Download Video (5min + 15min prediction)",
                        data=video_data,
                        file_name=f"cloud_{cloud_type}_prediction.mp4",
                        mime="video/mp4"
                    )
                    os.unlink(temp_video.name)
        
        with col2:
            if st.button("🖼️ Generate Frame Sequence", key="frames_tab"):
                with st.spinner("Frames generate ho rahi hain..."):
                    visualizer = CloudMotionVisualizer(
                        cloud_type=cloud_type,
                        height_m=height_m,
                        direction=direction,
                        pixel_speed=pixel_speed
                    )
                    frames = visualizer.generate_frame_sequence(num_frames=12)
                    
                    cols = st.columns(3)
                    for idx, frame in enumerate(frames):
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        with cols[idx % 3]:
                            st.image(frame_rgb, use_column_width=True)


    cap.release()

with tab3:

    st.subheader(
        "Multi Image Cloud Motion Analysis"
    )

    uploaded_images = st.file_uploader(
        "Upload Multiple Cloud Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    interval = st.number_input(
        "Time Gap Between Images (seconds)",
        min_value=1,
        value=60
    )

    if uploaded_images:

        st.success(
            f"{len(uploaded_images)} Images Uploaded"
        )

        if len(uploaded_images) >= 2:

            directions = []
            speeds = []

            for i in range(len(uploaded_images) - 1):

                img1 = Image.open(
                    uploaded_images[i]
                )

                img2 = Image.open(
                    uploaded_images[i + 1]
                )

                img1 = np.array(
                    img1.resize((640, 480))
                )

                img2 = np.array(
                    img2.resize((640, 480))
                )

                gray1 = cv2.cvtColor(
                    img1,
                    cv2.COLOR_RGB2GRAY
                )

                gray2 = cv2.cvtColor(
                    img2,
                    cv2.COLOR_RGB2GRAY
                )

                flow = cv2.calcOpticalFlowFarneback(
                    gray1,
                    gray2,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0
                )

                magnitude, angle = cv2.cartToPolar(
                    flow[...,0],
                    flow[...,1]
                )

                avg_speed = np.mean(
                    magnitude
                )

                avg_angle = np.mean(
                    angle
                )

                speeds.append(avg_speed)

                directions.append(
                    np.degrees(avg_angle)
                )
            avg_speed = np.mean(speeds)

            avg_direction = np.mean(directions)

            pixel_speed = avg_speed / interval

            if 45 <= avg_direction < 135:
                direction = "North"

            elif 135 <= avg_direction < 225:
                direction = "West"

            elif 225 <= avg_direction < 315:
                direction = "South"

            else:
                direction = "East"

            # -------- CLOUD TYPE MAJORITY VOTING --------

            from collections import Counter

            predictions_list = []

            for img_file in uploaded_images:

                img = Image.open(img_file)

                img = img.resize((224,224))

                img_array = image.img_to_array(img)

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                img_array = img_array / 255.0

                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_class = np.argmax(
                    prediction
                )

                predictions_list.append(
                    class_names[predicted_class]
                )

            cloud_type = Counter(
                predictions_list
            ).most_common(1)[0][0]

            cloud_height = {
                "Cumulus": 1500,
                "Altocumulus": 4500,
                "Cirrus": 9000,
                "ClearSky": 0,
                "Stratocumulus": 1200,
                "Cumulonimbus": 6000,
                "Mixed": 3500
            }

            height_m = cloud_height.get(
                cloud_type,
                2000
            )

            meter_per_pixel = (
                height_m / 10000
            )

            speed_mps = (
                pixel_speed * meter_per_pixel
            )

            speed_kmh = (
                speed_mps * 3.6
            )

            distance_5 = (
                speed_kmh * (5/60)
            )

            distance_15 = (
                speed_kmh * (15/60)
            )

            st.success(
                f"☁ Cloud Type: {cloud_type}"
            )

            st.info(
                f"📍 Estimated Height: {height_m:,} m"
            )

            st.success(
                f"🧭 Direction: {direction}"
            )

            st.info(
                f"⚡ Estimated Speed: {speed_kmh:.2f} km/h"
            )

            st.warning(
                f"⏳ After 5 Minutes: ~{distance_5:.2f} km"
            )

            st.warning(
                f"⏳ After 15 Minutes: ~{distance_15:.2f} km"
            )
            
            # -------- GENERATE MOTION VISUALIZATION --------
            st.divider()
            st.subheader("📊 Motion Prediction Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎬 Generate Prediction Video", key="multi_video"):
                    with st.spinner("Video generate ho rahi hai..."):
                        visualizer = CloudMotionVisualizer(
                            cloud_type=cloud_type,
                            height_m=height_m,
                            direction=direction,
                            pixel_speed=pixel_speed
                        )
                        temp_video = tempfile.NamedTemporaryFile(
                            suffix=".mp4",
                            delete=False
                        )
                        visualizer.save_video_with_prediction(
                            temp_video.name,
                            prediction_minutes=15
                        )
                        
                        with open(temp_video.name, "rb") as f:
                            video_data = f.read()
                        
                        st.success("✅ Video ready!")
                        st.download_button(
                            label="📥 Download Video (5min + 15min prediction)",
                            data=video_data,
                            file_name=f"cloud_{cloud_type}_prediction.mp4",
                            mime="video/mp4"
                        )
                        os.unlink(temp_video.name)
            
            with col2:
                if st.button("🖼️ Generate Frame Sequence", key="multi_frames"):
                    with st.spinner("Frames generate ho rahi hain..."):
                        visualizer = CloudMotionVisualizer(
                            cloud_type=cloud_type,
                            height_m=height_m,
                            direction=direction,
                            pixel_speed=pixel_speed
                        )
                        frames = visualizer.generate_frame_sequence(num_frames=15)
                        
                        cols = st.columns(3)
                        for idx, frame in enumerate(frames):
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            with cols[idx % 3]:
                                st.image(frame_rgb, use_column_width=True)