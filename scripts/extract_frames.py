import cv2
import os

video = cv2.VideoCapture("data/videos/cloud.mp4")

os.makedirs("data/frames", exist_ok=True)

count = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    cv2.imwrite(f"data/frames/frame_{count}.jpg", frame)

    count += 1

video.release()

print("Frames Saved:", count)
