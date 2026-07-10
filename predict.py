"""
predict.py  —  Standalone Cloud Type Predictor
Matches class_names order used in app.py.
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import sys
import os

# ─── CONFIG ────────────────────────────────────────────────────
MODEL_PATH = "cloud_model.keras"
IMG_SIZE   = 224

CLASS_NAMES = [
    "Cumulus", "Altocumulus", "Cirrus",
    "ClearSky", "Stratocumulus", "Cumulonimbus", "Mixed"
]

CLOUD_EMOJI = {
    "Cumulus":       "⛅",
    "Altocumulus":   "🌤️",
    "Cirrus":        "🌬️",
    "ClearSky":      "☀️",
    "Stratocumulus": "🌥️",
    "Cumulonimbus":  "⛈️",
    "Mixed":         "🌦️",
}

CLOUD_HEIGHT = {
    "Cumulus": 1500, "Altocumulus": 4500, "Cirrus": 9000,
    "ClearSky": 0,   "Stratocumulus": 1200,
    "Cumulonimbus": 6000, "Mixed": 3500,
}

# ─── LOAD MODEL ─────────────────────────────────────────────────
print(f"Loading model from '{MODEL_PATH}' ...")
model = load_model(MODEL_PATH)
print("Model loaded.\n")

# ─── PREDICT FUNCTION ───────────────────────────────────────────
def predict(img_path: str):
    """Load image, run inference, return class name + confidence."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img      = keras_image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr      = keras_image.img_to_array(img)
    arr      = np.expand_dims(arr, axis=0) / 255.0

    probs    = model.predict(arr, verbose=0)[0]   # shape: (NUM_CLASSES,)
    top_idx  = int(np.argmax(probs))
    top_name = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx]) * 100

    # Top-3 for richer output
    top3_idx  = np.argsort(probs)[::-1][:3]
    top3      = [(CLASS_NAMES[i], float(probs[i]) * 100) for i in top3_idx]

    return top_name, top_conf, top3

# ─── MAIN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    cloud_name, confidence, top3 = predict(img_path)
    emoji  = CLOUD_EMOJI.get(cloud_name, "☁️")
    height = CLOUD_HEIGHT.get(cloud_name, "—")

    print("=" * 40)
    print(f"Image       : {img_path}")
    print(f"Cloud Type  : {emoji}  {cloud_name}")
    print(f"Confidence  : {confidence:.1f}%")
    print(f"Approx Height : {height} m")
    print("\nTop-3 Predictions:")
    for rank, (name, conf) in enumerate(top3, 1):
        bar = "█" * int(conf / 5)
        print(f"  {rank}. {name:<15} {conf:5.1f}%  {bar}")
    print("=" * 40)