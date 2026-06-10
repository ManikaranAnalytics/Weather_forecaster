from PIL import Image
import os

img_path = "dataset/1_cumulus/" + os.listdir("dataset/1_cumulus")[0]

img = Image.open(img_path)

print(img.size)