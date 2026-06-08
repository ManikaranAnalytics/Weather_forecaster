import cv2
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os

class CloudMotionVisualizer:
    """Generate cloud motion visualization as video or multi-image sequence"""
    
    def __init__(self, cloud_type="Cumulus", height_m=1500, 
                 direction="West", pixel_speed=8.4):
        self.cloud_type = cloud_type
        self.height_m = height_m
        self.direction = direction
        self.pixel_speed = pixel_speed
        self.frame_width = 800
        self.frame_height = 600
        self.bg_color = (135, 206, 235)  # Sky blue
        
    def get_direction_vector(self):
        """Convert direction string to vector"""
        direction_map = {
            "North": (0, -1),
            "South": (0, 1),
            "East": (1, 0),
            "West": (-1, 0),
            "NE": (0.707, -0.707),
            "NW": (-0.707, -0.707),
            "SE": (0.707, 0.707),
            "SW": (-0.707, 0.707)
        }
        return direction_map.get(self.direction, (-1, 0))
    
    def get_cloud_color(self):
        """Get RGB color based on cloud type"""
        colors = {
            "Cumulus": (255, 255, 255),  # White
            "Altocumulus": (230, 230, 250),  # Light blue
            "Cirrus": (240, 248, 255),  # Alice blue
            "Stratocumulus": (200, 200, 220),  # Light gray
            "Cumulonimbus": (64, 64, 64),  # Dark gray
            "Mixed": (220, 220, 240)  # Very light blue
        }
        return colors.get(self.cloud_type, (255, 255, 255))
    
    def draw_cloud(self, img_array, x, y, scale=1.0):
        """Draw a cloud shape at position (x, y)"""
        pil_image = Image.fromarray(img_array)
        draw = ImageDraw.Draw(pil_image)
        
        color = self.get_cloud_color()
        cloud_width = int(120 * scale)
        cloud_height = int(60 * scale)
        
        # Draw cloud as multiple overlapping circles
        circle_radius = int(30 * scale)
        positions = [
            (x - cloud_width//2, y),
            (x - cloud_width//4, y - circle_radius//2),
            (x, y - circle_radius),
            (x + cloud_width//4, y - circle_radius//2),
            (x + cloud_width//2, y)
        ]
        
        for cx, cy in positions:
            draw.ellipse([cx - circle_radius, cy - circle_radius, 
                         cx + circle_radius, cy + circle_radius], 
                        fill=color, outline=(200, 200, 200))
        
        return np.array(pil_image)
    
    def generate_frame_sequence(self, num_frames=30, time_minutes=5):
        """Generate sequence of frames showing cloud motion"""
        frames = []
        
        # Calculate total motion for the time period
        total_pixels = self.pixel_speed * (time_minutes * 60)
        direction_vector = self.get_direction_vector()
        
        # Divide motion into frame intervals
        pixels_per_frame = total_pixels / num_frames
        
        # Starting cloud position (center)
        start_x = self.frame_width // 2
        start_y = self.frame_height // 3
        
        for frame_idx in range(num_frames):
            # Create blank frame with sky gradient
            frame = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            frame[:, :] = self.bg_color
            
            # Add subtle gradient
            for i in range(self.frame_height):
                intensity = int(self.bg_color[0] - (i * 30 / self.frame_height))
                frame[i, :] = [intensity, intensity + 20, intensity + 50]
            
            # Calculate cloud position for this frame
            motion_x = direction_vector[0] * pixels_per_frame * frame_idx
            motion_y = direction_vector[1] * pixels_per_frame * frame_idx
            
            cloud_x = int(start_x + motion_x)
            cloud_y = int(start_y + motion_y)
            
            # Clamp position to frame boundaries
            cloud_x = max(60, min(cloud_x, self.frame_width - 60))
            cloud_y = max(40, min(cloud_y, self.frame_height - 100))
            
            # Draw cloud
            frame = self.draw_cloud(frame, cloud_x, cloud_y)
            
            # Draw motion trail
            if frame_idx > 0:
                trail_color = (100, 149, 237)  # Cornflower blue
                prev_motion_x = direction_vector[0] * pixels_per_frame * (frame_idx - 1)
                prev_motion_y = direction_vector[1] * pixels_per_frame * (frame_idx - 1)
                prev_x = int(start_x + prev_motion_x)
                prev_y = int(start_y + prev_motion_y)
                
                if frame_idx % 3 == 0:  # Draw dots every 3 frames
                    cv2.circle(frame, (prev_x, prev_y), 3, trail_color, -1)
            
            # Add direction arrow
            self._draw_direction_arrow(frame, direction_vector)
            
            # Add text overlay
            self._add_text_overlay(frame, frame_idx, num_frames, time_minutes)
            
            frames.append(frame)
        
        return frames
    
    def _draw_direction_arrow(self, frame, direction_vector):
        """Draw direction indicator arrow"""
        start_pos = (50, 50)
        arrow_length = 40
        end_x = int(start_pos[0] + direction_vector[0] * arrow_length)
        end_y = int(start_pos[1] + direction_vector[1] * arrow_length)
        end_pos = (end_x, end_y)
        
        cv2.arrowedLine(frame, start_pos, end_pos, (255, 100, 0), 3)
    
    def _add_text_overlay(self, frame, frame_idx, total_frames, time_minutes):
        """Add information text to frame"""
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Calculate elapsed time
        elapsed_seconds = (frame_idx / total_frames) * time_minutes * 60
        elapsed_minutes = int(elapsed_seconds // 60)
        elapsed_secs = int(elapsed_seconds % 60)
        
        # Create text
        text_lines = [
            f"Cloud Type: {self.cloud_type}",
            f"Height: {self.height_m:,}m",
            f"Direction: {self.direction}",
            f"Speed: {self.pixel_speed:.1f} px/s",
            f"Time: {elapsed_minutes:02d}:{elapsed_secs:02d}"
        ]
        
        y_offset = 10
        for line in text_lines:
            draw.text((10, y_offset), line, fill=(0, 0, 0))
            y_offset += 25
        
        return np.array(pil_image)
    
    def save_multi_image(self, output_dir, num_images=9):
        """Save motion sequence as separate PNG files"""
        frames = self.generate_frame_sequence(num_frames=num_images, time_minutes=5)
        
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        
        for idx, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            filename = f"cloud_motion_{idx+1:02d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            image_paths.append(filepath)
        
        return image_paths
    
    def save_video(self, output_path, duration_minutes=5, fps=24):
        """Save motion sequence as MP4 video"""
        num_frames = int(duration_minutes * 60 * fps)
        frames = self.generate_frame_sequence(num_frames=num_frames, 
                                             time_minutes=duration_minutes)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                             (self.frame_width, self.frame_height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return output_path
    
    def save_video_with_prediction(self, output_path, prediction_minutes=15, fps=30):
        """Save video showing 5-min and 15-min predictions"""
        # Generate frames for 15 minutes
        num_frames = int(prediction_minutes * 60 * fps)
        frames = self.generate_frame_sequence(num_frames=num_frames, 
                                             time_minutes=prediction_minutes)
        
        # Add markers at 5 and 15 minute marks
        five_min_frame = int(5 * 60 * fps)
        fifteen_min_frame = int(15 * 60 * fps)
        
        for idx in range(min(five_min_frame, len(frames))):
            if idx == five_min_frame - 1:
                cv2.putText(frames[idx], "5 Minute Prediction", (250, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if fifteen_min_frame < len(frames):
            cv2.putText(frames[fifteen_min_frame - 1], "15 Minute Prediction", 
                       (240, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                             (self.frame_width, self.frame_height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return output_path


def generate_cloud_motion_video(cloud_type="Cumulus", height_m=1500, 
                                direction="West", pixel_speed=8.4, 
                                output_path="cloud_motion.mp4"):
    """Convenience function to generate video"""
    visualizer = CloudMotionVisualizer(cloud_type, height_m, direction, pixel_speed)
    return visualizer.save_video(output_path)


def generate_cloud_motion_images(cloud_type="Cumulus", height_m=1500,
                                 direction="West", pixel_speed=8.4,
                                 output_dir="cloud_frames"):
    """Convenience function to generate image sequence"""
    visualizer = CloudMotionVisualizer(cloud_type, height_m, direction, pixel_speed)
    return visualizer.save_multi_image(output_dir)
