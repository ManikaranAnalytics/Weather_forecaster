"""
Minimal placeholder implementation of CloudMotionVisualizer.

This generates a simple animated MP4 showing a cloud-shaped blob drifting
across a sky-colored frame in the given direction, at a speed scaled from
pixel_speed. It exists so app.py has a working dependency to run against
while testing — it is NOT a physically accurate cloud simulation.

Required interface (as used in app.py):
    viz = CloudMotionVisualizer(cloud_type=str, height_m=float,
                                 direction=str or float, pixel_speed=float)
    viz.save_video_with_prediction(output_path: str, prediction_minutes: int)
"""

import math
import numpy as np
import cv2


DIRECTION_TO_ANGLE = {
    "North": 270, "NE": 315, "East": 0, "SE": 45,
    "South": 90, "SW": 135, "West": 180, "NW": 225,
}

CLOUD_TYPE_COLOR = {
    "Cumulus": (235, 235, 235),
    "Altocumulus": (220, 220, 225),
    "Cirrus": (245, 245, 250),
    "ClearSky": (255, 220, 130),
    "Stratocumulus": (200, 200, 205),
    "Cumulonimbus": (120, 120, 130),
    "Mixed": (210, 210, 215),
}


class CloudMotionVisualizer:
    def __init__(self, cloud_type="Cumulus", height_m=1500, direction="West",
                 pixel_speed=2.0, frame_size=(640, 480), fps=24):
        self.cloud_type = cloud_type
        self.height_m = height_m
        self.direction = direction
        self.pixel_speed = max(float(pixel_speed), 0.1)
        self.width, self.height = frame_size
        self.fps = fps

        if isinstance(direction, (int, float)):
            self.angle_deg = float(direction)
        else:
            self.angle_deg = DIRECTION_TO_ANGLE.get(direction, 180)

        self.dx = math.cos(math.radians(self.angle_deg))
        self.dy = math.sin(math.radians(self.angle_deg))
        self.cloud_color = CLOUD_TYPE_COLOR.get(cloud_type, (230, 230, 230))

    def _sky_background(self):
        sky_top = np.array([235, 178, 100], dtype=np.float32)
        sky_bottom = np.array([255, 230, 190], dtype=np.float32)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            t = y / max(self.height - 1, 1)
            color = sky_top * (1 - t) + sky_bottom * t
            frame[y, :] = color
        return frame

    def _draw_cloud(self, frame, cx, cy, scale=1.0):
        overlay = frame.copy()
        blob_specs = [(0, 0, 70), (45, -10, 50), (-45, -10, 50),
                      (90, 5, 35), (-90, 5, 35)]
        for ox, oy, r in blob_specs:
            center = (int(cx + ox * scale), int(cy + oy * scale))
            radius = max(int(r * scale), 4)
            cv2.circle(overlay, center, radius, self.cloud_color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, dst=frame)
        return frame

    def save_video_with_prediction(self, output_path, prediction_minutes=15):
        """
        Renders an MP4 showing the cloud drifting across the frame, with an
        on-screen label projecting forward by `prediction_minutes`.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        total_seconds = 6
        total_frames = total_seconds * self.fps
        start_x = -0.15 * self.width if self.dx >= 0 else 1.15 * self.width
        speed_px_per_frame = self.pixel_speed * 4

        for i in range(total_frames):
            frame = self._sky_background()
            cx = start_x + self.dx * speed_px_per_frame * i
            cy = self.height * 0.32 + self.dy * speed_px_per_frame * i * 0.3
            frame = self._draw_cloud(frame, cx, cy, scale=1.2)

            elapsed_minutes = (i / max(total_frames - 1, 1)) * prediction_minutes
            label = f"{self.cloud_type} | +{elapsed_minutes:.1f} min projected"
            cv2.putText(frame, label, (16, self.height - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)

            writer.write(frame)

        writer.release()
        return output_path
