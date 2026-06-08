from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

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

img_path = "test.jpg"   # yahan image ka naam

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)

print("Cloud Type:", class_names[predicted_class])
print("Confidence:", np.max(prediction) * 100)