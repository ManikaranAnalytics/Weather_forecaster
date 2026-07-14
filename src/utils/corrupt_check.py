from PIL import Image
import os

bad_images = 0

for root, dirs, files in os.walk("data/dataset"):
    for file in files:
        try:
            img = Image.open(os.path.join(root, file))
            img.verify()
        except Exception:
            bad_images += 1
            print("Corrupt:", os.path.join(root, file))

print("Bad Images =", bad_images)
