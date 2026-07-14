import os

from PIL import Image
import matplotlib.pyplot as plt

dataset_path = "data/dataset"

total = 0

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        count = len(os.listdir(folder_path))
        total += count

        print(f"{folder} : {count}")

print("\nTotal Images :", total)

folder = "data/dataset/1_cumulus"

img_name = os.listdir(folder)[0]

img = Image.open(os.path.join(folder, img_name))

print(img.size)

plt.imshow(img)
plt.axis("off")
plt.show()
