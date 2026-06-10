import cv2
import numpy as np
import math


class CloudMotionVisualizer:
    def __init__(self, cloud_type, height_m, direction, pixel_speed):
        self.cloud_type  = cloud_type
        self.height_m    = height_m
        self.direction   = direction
        self.pixel_speed = pixel_speed

        self.width  = 960
        self.height = 540
        self.fps    = 30

        # BGR sky gradient (top, bottom)
        self.sky_color_map = {
            "Cumulus":       ((235, 180, 95),  (200, 140, 60)),
            "Altocumulus":   ((230, 185, 110), (195, 148, 75)),
            "Cirrus":        ((240, 200, 130), (210, 165, 95)),
            "ClearSky":      ((245, 175, 80),  (210, 135, 50)),
            "Stratocumulus": ((195, 155, 110), (165, 118, 75)),
            "Cumulonimbus":  ((105,  85,  75), (65,  48,  38)),
            "Mixed":         ((225, 180, 115), (190, 142, 80)),
        }

        self.direction_vector = {
            "East":  ( 1,  0),
            "West":  (-1,  0),
            "North": ( 0, -1),
            "South": ( 0,  1),
        }.get(direction, (1, 0))

        self.clouds = self._init_clouds(18)

    # ── INIT ──────────────────────────────────────────────────────
    def _init_clouds(self, count):
        rng = np.random.default_rng(99)
        clouds = []
        for _ in range(count):
            sz = int(rng.integers(65, 155))
            clouds.append({
                "x":     float(rng.integers(0, self.width)),
                "y":     float(rng.integers(40, self.height - 130)),
                "size":  sz,
                "layer": int(rng.integers(0, 3)),
                "puffs": self._make_puffs(sz, rng),
                # Each cloud gets a slightly varied shade of white/grey
                "shade": int(rng.integers(200, 245)),
            })
        clouds.sort(key=lambda c: c["layer"])
        return clouds

    def _make_puffs(self, size, rng):
        puffs = []
        for _ in range(int(rng.integers(4, 8))):
            ox = int(rng.integers(-size // 2, size // 2 + 1))
            oy = int(rng.integers(-size // 4, size // 4 + 1))
            r  = int(rng.integers(size // 3, size // 2 + 1))
            puffs.append((ox, oy, r))
        return puffs

    # ── SKY + GROUND ──────────────────────────────────────────────
    def _draw_sky(self, frame):
        top, bot = self.sky_color_map.get(
            self.cloud_type, ((235, 180, 95), (200, 140, 60))
        )
        for y in range(self.height):
            t = y / self.height
            frame[y, :] = (
                int(top[0] + (bot[0] - top[0]) * t),
                int(top[1] + (bot[1] - top[1]) * t),
                int(top[2] + (bot[2] - top[2]) * t),
            )

    def _draw_ground(self, frame):
        gy = self.height - 80
        cv2.rectangle(frame, (0, gy),      (self.width, self.height), (45, 100, 45), -1)
        cv2.rectangle(frame, (0, gy),      (self.width, gy + 7),      (30,  80, 30), -1)
        # horizon line
        cv2.line(frame, (0, gy), (self.width, gy), (60, 130, 60), 2)

    # ── CLOUD + BBOX + SPEED ──────────────────────────────────────
    def _draw_cloud(self, frame, cloud, speed_kmh):
        cx, cy = int(cloud["x"]), int(cloud["y"])
        shade  = cloud["shade"]

        # ── 1. Draw shadow first (slightly below, dark grey) ──
        shadow_color = (80, 80, 85)
        shadow_ov    = frame.copy()
        for (ox, oy, r) in cloud["puffs"]:
            px, py = cx + ox + 4, cy + oy + 6
            if 0 <= px < self.width and 0 <= py < self.height:
                cv2.ellipse(shadow_ov, (px, py),
                            (r, int(r * 0.65)), 0, 0, 360,
                            shadow_color, -1)
        cv2.addWeighted(shadow_ov, 0.28, frame, 0.72, 0, frame)

        # ── 2. Draw cloud body (solid, opaque) ──
        for (ox, oy, r) in cloud["puffs"]:
            px, py = cx + ox, cy + oy
            if 0 <= px < self.width and 0 <= py < self.height:
                # Base fill — solid white-grey
                cv2.ellipse(frame, (px, py),
                            (r, int(r * 0.68)), 0, 0, 360,
                            (shade, shade, shade), -1)
                # Highlight top edge
                cv2.ellipse(frame, (px - 2, py - 3),
                            (max(1, r - 6), max(1, int(r * 0.55))),
                            0, 0, 360,
                            (min(255, shade + 18), min(255, shade + 18), min(255, shade + 18)),
                            -1)
                # Dark bottom edge for depth
                cv2.ellipse(frame, (px, py + int(r * 0.18)),
                            (r, int(r * 0.28)), 0, 0, 360,
                            (max(0, shade - 40), max(0, shade - 40), max(0, shade - 40)),
                            -1)

        # ── 3. Bounding box via contour ──
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for (ox, oy, r) in cloud["puffs"]:
            px, py = cx + ox, cy + oy
            if 0 <= px < self.width and 0 <= py < self.height:
                cv2.ellipse(mask, (px, py),
                            (r, int(r * 0.68)), 0, 0, 360, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        all_pts        = np.concatenate(contours)
        bx, by, bw, bh = cv2.boundingRect(all_pts)

        # Padding
        pad = 7
        bx1 = max(0,             bx - pad)
        by1 = max(0,             by - pad)
        bx2 = min(self.width-1,  bx + bw + pad)
        by2 = min(self.height-1, by + bh + pad)

        # Outer soft glow
        glow = frame.copy()
        cv2.rectangle(glow, (bx1 - 3, by1 - 3), (bx2 + 3, by2 + 3),
                      (0, 255, 120), 4)
        cv2.addWeighted(glow, 0.35, frame, 0.65, 0, frame)

        # Main crisp box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 120), 2)

        # Corner ticks  ╔ style
        tick = 12
        for (px_, py_, dx_, dy_) in [
            (bx1, by1,  1,  1), (bx2, by1, -1,  1),
            (bx1, by2,  1, -1), (bx2, by2, -1, -1)
        ]:
            cv2.line(frame, (px_, py_), (px_ + dx_*tick, py_),           (0, 255, 80), 2)
            cv2.line(frame, (px_, py_), (px_,             py_ + dy_*tick),(0, 255, 80), 2)

        # ── 4. Speed label ──
        label   = f"{speed_kmh:.1f} km/h"
        font    = cv2.FONT_HERSHEY_SIMPLEX
        fscale  = 0.44
        fthick  = 1
        (tw, th), bl = cv2.getTextSize(label, font, fscale, fthick)

        lx = bx1
        ly = by1 - 5
        if ly - th < 2:          # too close to top? put inside
            ly = by1 + th + 6

        # Pill background
        cv2.rectangle(frame,
                      (lx - 2,      ly - th - 3),
                      (lx + tw + 6, ly + bl + 1),
                      (0, 160, 60), -1)
        cv2.rectangle(frame,
                      (lx - 2,      ly - th - 3),
                      (lx + tw + 6, ly + bl + 1),
                      (0, 255, 120), 1)
        cv2.putText(frame, label, (lx + 3, ly),
                    font, fscale, (255, 255, 255), fthick, cv2.LINE_AA)

    # ── DIRECTION ARROW ───────────────────────────────────────────
    def _draw_arrow(self, frame):
        cx, cy = self.width // 2, self.height - 42
        dx, dy = self.direction_vector
        cv2.arrowedLine(frame,
                        (cx, cy),
                        (int(cx + dx * 60), int(cy + dy * 60)),
                        (255, 255, 255), 3, tipLength=0.35)
        cv2.putText(frame, self.direction,
                    (cx - 30, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # ── HUD PANEL ─────────────────────────────────────────────────
    def _draw_hud(self, frame, elapsed_sec, speed_kmh, label=""):
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (335, 118), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.58, frame, 0.42, 0, frame)
        cv2.rectangle(frame, (0, 0), (335, 118), (0, 200, 80), 1)

        def txt(t, y, sc=0.52, c=(255, 255, 255), b=1):
            cv2.putText(frame, t, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, c, b, cv2.LINE_AA)

        txt(f"Cloud  : {self.cloud_type}",  22, c=(140, 230, 255), b=2)
        txt(f"Height : {self.height_m:,} m", 43)
        txt(f"Speed  : {speed_kmh:.1f} km/h  ({speed_kmh/3.6:.2f} m/s)",
            64, c=(80, 255, 160))
        txt(f"Dir    : {self.direction}", 85)
        if label:
            txt(label, 110, sc=0.45, c=(255, 215, 60))

        mins = int(elapsed_sec) // 60
        secs = int(elapsed_sec) % 60
        cv2.putText(frame, f"T + {mins:02d}:{secs:02d}",
                    (self.width - 165, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (255, 255, 180), 2, cv2.LINE_AA)

    # ── UPDATE ────────────────────────────────────────────────────
    def _update_clouds(self, dt_sec):
        rng = np.random.default_rng()
        dx = self.direction_vector[0] * self.pixel_speed * dt_sec
        dy = self.direction_vector[1] * self.pixel_speed * dt_sec
        for c in self.clouds:
            c["x"] += dx
            c["y"] += dy
            if c["x"] > self.width + 200:
                c["x"] = -200.0
                c["y"] = float(rng.integers(40, self.height - 130))
            elif c["x"] < -200:
                c["x"] = float(self.width + 200)
                c["y"] = float(rng.integers(40, self.height - 130))
            if c["y"] > self.height - 100:
                c["y"] = float(rng.integers(40, 90))
            elif c["y"] < 20:
                c["y"] = float(rng.integers(self.height - 200, self.height - 130))

    # ── MAKE FRAME ────────────────────────────────────────────────
    def _make_frame(self, elapsed_sec, speed_kmh, label=""):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_sky(frame)
        for lyr in range(3):
            for c in self.clouds:
                if c["layer"] == lyr:
                    self._draw_cloud(frame, c, speed_kmh)
        self._draw_ground(frame)
        self._draw_arrow(frame)
        self._draw_hud(frame, elapsed_sec, speed_kmh, label)
        return frame

    # ── PUBLIC: FRAME SEQUENCE ────────────────────────────────────
    def generate_frame_sequence(self, num_frames=12):
        speed_kmh = self.pixel_speed * 3.6
        frames = []
        for i in range(num_frames):
            self._update_clouds(2.0)
            frames.append(self._make_frame(i * 2.0, speed_kmh).copy())
        return frames

    # ── PUBLIC: SAVE VIDEO ────────────────────────────────────────
    def save_video_with_prediction(self, output_path, prediction_minutes=15):
        speed_kmh = self.pixel_speed * 3.6
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        out       = cv2.VideoWriter(output_path, fourcc, self.fps,
                                    (self.width, self.height))
        dt = 1.0 / self.fps

        # Phase 1 — Live (10s)
        for i in range(10 * self.fps):
            self._update_clouds(dt)
            frame = self._make_frame(i * dt, speed_kmh, "[ Live Observation ]")
            cv2.rectangle(frame, (0, self.height - 5),
                          (self.width, self.height), (0, 210, 80), -1)
            out.write(frame)

        # Phase 2 — +5 min (8s)
        dist5 = speed_kmh * (5 / 60)
        for i in range(8 * self.fps):
            elapsed = 10 + i * dt
            self._update_clouds(dt * 8)
            frame = self._make_frame(elapsed, speed_kmh,
                                     f"[ +5 min Prediction  |  ~{dist5:.2f} km ]")
            cv2.rectangle(frame, (0, self.height - 5),
                          (self.width, self.height), (0, 180, 255), -1)
            out.write(frame)

        # Phase 3 — +15 min (8s)
        dist15 = speed_kmh * (15 / 60)
        for i in range(8 * self.fps):
            elapsed = 18 + i * dt
            self._update_clouds(dt * 20)
            frame = self._make_frame(elapsed, speed_kmh,
                                     f"[ +15 min Prediction  |  ~{dist15:.2f} km ]")
            cv2.rectangle(frame, (0, self.height - 5),
                          (self.width, self.height), (255, 160, 0), -1)
            out.write(frame)

        out.release()
        return output_path