#!/usr/bin/env python
"""
Test script for Cloud Motion Visualization
Demo with user's cloud data
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.tracking.motion_visualizer import CloudMotionVisualizer

cloud_type = "Cumulus"
height_m = 1500
direction = "West"
pixel_speed = 8.4

print("=" * 60)
print("Cloud Motion Visualization Demo")
print("=" * 60)
print(f"Cloud Type: {cloud_type}")
print(f"Height: {height_m}m")
print(f"Direction: {direction}")
print(f"Motion Speed: {pixel_speed} px/sec")
print("=" * 60)

print("\nGenerating MP4 video...")
try:
    output_dir = os.path.join("data", "outputs", "cloud_motion_frames")
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "cloud_motion_prediction.mp4")
    visualizer = CloudMotionVisualizer(
        cloud_type=cloud_type,
        height_m=height_m,
        direction=direction,
        pixel_speed=pixel_speed,
    )
    visualizer.save_video_with_prediction(output_video, prediction_minutes=5)
    file_size = os.path.getsize(output_video) / (1024 * 1024)
    print(f"Video created: {output_video}")
    print(f"File size: {file_size:.2f} MB")
except Exception as e:
    print(f"Error: {e}")

print("\nMotion calculations")
pixel_speed_5min = pixel_speed * (5 * 60)
pixel_speed_15min = pixel_speed * (15 * 60)
print(f"   After 5 minutes:  {pixel_speed_5min:.0f} pixels")
print(f"   After 15 minutes: {pixel_speed_15min:.0f} pixels")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
