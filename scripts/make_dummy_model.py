"""
Run this once on your own machine (where TensorFlow is installed) to generate
a placeholder models/cloud_model.keras so app.py can load and run end-to-end.

    pip install tensorflow
    python scripts/make_dummy_model.py

This model is UNTRAINED — its cloud-type predictions are random/meaningless.
It exists only so the Streamlit app runs without crashing, so you can test
the rest of the pipeline (UI, motion math, video export, etc.). Once you have
a real trained model, replace models/cloud_model.keras with that one — no other
code changes are needed, since the class order matches app.py exactly.
"""

import os
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = ["Cumulus", "Altocumulus", "Cirrus", "ClearSky",
               "Stratocumulus", "Cumulonimbus", "Mixed"]

MODEL_PATH = os.path.join("models", "cloud_model.keras")
os.makedirs("models", exist_ok=True)

model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    layers.Conv2D(8, 3, activation="relu", padding="same"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(16, activation="relu"),
    layers.Dense(len(CLASS_NAMES), activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy")
model.save(MODEL_PATH)
print(f"Saved {MODEL_PATH} with classes: {CLASS_NAMES}")
