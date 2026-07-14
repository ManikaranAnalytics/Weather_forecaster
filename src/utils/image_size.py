from PIL import Image
import os

img_path = "data/dataset/1_cumulus/" + os.listdir("data/dataset/1_cumulus")[0]

img = Image.open(img_path)

print(img.size)
