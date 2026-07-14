import cv2
import numpy as np

img1 = cv2.imread("data/frames/frame_0.jpg")
img2 = cv2.imread("data/frames/frame_10.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

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

avg_speed = np.mean(magnitude)

avg_angle = np.mean(angle)

angle_deg = np.degrees(avg_angle)

print("Average Pixel Speed:", avg_speed)
print("Average Angle:", avg_angle)
print("Angle in Degree:", angle_deg)
