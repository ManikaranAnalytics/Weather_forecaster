#!/usr/bin/env python
"""
Test script for Cloud Motion Visualization
Demo with user's cloud data
"""

from motion_visualizer import CloudMotionVisualizer, generate_cloud_motion_video, generate_cloud_motion_images
import os

# आपके data के अनुसार:
cloud_type = "Cumulus"
height_m = 1500
direction = "West"
pixel_speed = 8.4

print("=" * 60)
print("☁️ Cloud Motion Visualization Demo")
print("=" * 60)
print(f"Cloud Type: {cloud_type}")
print(f"Height: {height_m}m")
print(f"Direction: {direction}")
print(f"Motion Speed: {pixel_speed} px/sec")
print("=" * 60)

# Test 1: Generate Image Sequence
print("\n🖼️  Test 1: Generating image sequence...")
try:
    output_dir = "cloud_motion_frames"
    image_paths = generate_cloud_motion_images(
        cloud_type=cloud_type,
        height_m=height_m,
        direction=direction,
        pixel_speed=pixel_speed,
        output_dir=output_dir
    )
    print(f"✅ Generated {len(image_paths)} images in '{output_dir}' folder")
    for i, path in enumerate(image_paths[:3], 1):
        print(f"   {i}. {os.path.basename(path)}")
    print(f"   ... और {len(image_paths) - 3} अन्य")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Generate Video
print("\n🎬 Test 2: Generating MP4 video...")
try:
    output_video = "cloud_motion_prediction.mp4"
    visualizer = CloudMotionVisualizer(
        cloud_type=cloud_type,
        height_m=height_m,
        direction=direction,
        pixel_speed=pixel_speed
    )
    visualizer.save_video(output_video, duration_minutes=5, fps=10)
    file_size = os.path.getsize(output_video) / (1024 * 1024)  # Convert to MB
    print(f"✅ Video created: {output_video}")
    print(f"   File size: {file_size:.2f} MB")
    print(f"   Duration: 5 minutes at 10 fps")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Show motion calculations
print("\n📊 Test 3: Motion calculations")
pixel_speed_5min = pixel_speed * (5 * 60)
pixel_speed_15min = pixel_speed * (15 * 60)
print(f"   After 5 minutes:  {pixel_speed_5min:.0f} pixels")
print(f"   After 15 minutes: {pixel_speed_15min:.0f} pixels")

print("\n" + "=" * 60)
print("✅ All tests completed!")
print("=" * 60)
