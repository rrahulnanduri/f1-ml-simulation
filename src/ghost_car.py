import arcade
import bisect
import math
from PIL import Image

class GhostCar(arcade.Sprite):
    def __init__(self, trajectory, offset=(0, 0), scale=1.0):
        super().__init__(scale=scale)
        
        # Apply offset to trajectory points
        self.trajectory = [(p[0] + offset[0], p[1] + offset[1], p[2]) for p in trajectory]
        self.times = [p[2] for p in self.trajectory] # Extract times for binary search
        
        # Appearance: Translucent Teal
        # Arcade requires a texture, so we create a simple one
        self.width = 20
        self.height = 10
        img = Image.new('RGBA', (20, 10), (0, 255, 255, 128)) # Teal, 50% opacity
        self.texture = arcade.Texture(img)
        self.alpha = 128
        
        # Initial pos
        if trajectory:
            self.center_x = trajectory[0][0]
            self.center_y = trajectory[0][1]

    def update_position(self, sim_time):
        """
        Updates the ghost car position based on simulation time.
        Interpolates between known telemetry points.
        """
        if not self.trajectory:
            return

        # Handle time before start
        if sim_time <= self.times[0]:
            self.center_x = self.trajectory[0][0]
            self.center_y = self.trajectory[0][1]
            return

        # Handle time after end
        if sim_time >= self.times[-1]:
            self.center_x = self.trajectory[-1][0]
            self.center_y = self.trajectory[-1][1]
            return

        # Find the two points bounding the current time
        idx = bisect.bisect_right(self.times, sim_time)
        # idx is the first index > sim_time, so we want idx-1 and idx
        
        p0 = self.trajectory[max(0, idx - 2)]
        p1 = self.trajectory[idx - 1]
        p2 = self.trajectory[idx]
        p3 = self.trajectory[min(len(self.trajectory) - 1, idx + 1)]
        
        t1, t2 = p1[2], p2[2]
        
        # Interpolation factor (0.0 to 1.0)
        alpha = (sim_time - t1) / (t2 - t1) if t2 > t1 else 0
        
        # Linear Interpolation: Position
        self.center_x = p1[0] + (p2[0] - p1[0]) * alpha
        self.center_y = p1[1] + (p2[1] - p1[1]) * alpha
        
        # Smooth Angle Interpolation
        # Calculate angle of current segment and next segment
        a1 = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        a2 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
        
        # Normalize a2 to be within +/- 180 of a1 to prevent spinning 360 degrees
        while a2 - a1 > 180: a2 -= 360
        while a2 - a1 < -180: a2 += 360
        
        # Morph between a1 (at start of segment) and halfway to a2 (at end)
        # Actually, simpler: Lerp between a1 and a2
        self.angle = a1 + (a2 - a1) * alpha
        
        # DEBUG LOGGING (Every frame - TEMPORARY)
        self._debug_counter = getattr(self, '_debug_counter', 0) + 1
        if self._debug_counter % 60 == 0:  # Log every 1 second of physics time
            print(f"[GHOST] T={sim_time:.3f}s | Idx={idx}/{len(self.trajectory)} | Alpha={alpha:.3f} | t1={t1:.3f} t2={t2:.3f}")

    def draw(self, **kwargs):
        """Draw the ghost car as a blue triangle (same shape as AI cars)."""
        # Triangle dimensions (same as Car class)
        w, h = 10.0, 6.0
        rad = math.radians(self.angle)
        
        # Relative corners for a triangle
        corners = [
            (w/2.0, 0),          # Tip
            (-w/2.0, -h/2.0),    # Back left
            (-w/2.0, h/2.0)      # Back right
        ]
        
        # Rotate and translate
        pts = []
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            pts.append((self.center_x + rx, self.center_y + ry))
        
        # Draw blue triangle
        arcade.draw_polygon_filled(pts, arcade.color.BLUE)
        
        # Cyan indicator at the tip (equivalent to AI car's yellow)
        nox = self.center_x + math.cos(rad) * 4
        noy = self.center_y + math.sin(rad) * 4
        arcade.draw_circle_filled(nox, noy, 1, arcade.color.CYAN)
