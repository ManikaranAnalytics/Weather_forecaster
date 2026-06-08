import cv2

video = cv2.VideoCapture("videos/cloud.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

print("FPS =", fps)

video.release()