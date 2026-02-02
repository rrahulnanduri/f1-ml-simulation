import arcade
import math
from shapely.geometry import Polygon, Point
from PIL import Image
import os

class Car(arcade.Sprite):
    def __init__(self, x=0, y=0, angle=0.0, track_ls=None):
        super().__init__()
        self.center_x = x
        self.center_y = y
        self.angle = angle  # Arcade uses degrees, 0 is right
        self.track_ls = track_ls
        
        # Physics State
        self.speed = 0.0
        
        # Physics Constants (Pixels & Seconds)
        self.MAX_SPEED = 600.0
        self.ACCEL = 400.0
        self.FRICTION = 0.95  # Drag coefficient per second
        self.TURN_RATE = 150.0 # Degrees per second
        
        # Control State
        self.throttle = 0.0 # -1 to 1
        self.steering = 0.0 # -1 to 1
        
        # Arcade 3.0 Sprite REQUIRES a texture. 
        # We create a tiny transparent one since we are drawing custom shapes.
        img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        self.texture = arcade.Texture(img)
        
        # Sensor Configuration
        # WIDENED: For better peripheral vision on corners
        self.sensor_angles = [-110, -55, 0, 55, 110] # Relative to car angle
        self.sensor_range = 100.0  # Reduced for better sensitivity on narrow tracks
        self.sensors = [] # List of (hit_point, distance)
        
        # Neural Network Brain
        self.brain = None # Will be set for autonomous driving
        
        
        # Evolution Stats
        self.fitness = 0.0
        self.is_alive = True
        self.time_alive = 0.0

    def update(self, delta_time: float):
        if not self.is_alive:
            return
            
        self.time_alive += delta_time

        # Kill if stalling (not moving for > 3 seconds)
        # This prevents the simulation from freezing if cars decide to just park.
        if self.time_alive > 3.0 and self.speed < 1.0:
            self.is_alive = False
            # PENALTY: Stalling shouldn't be rewarded
            self.fitness -= 50
            return

        # Use Brain if available
        if self.brain:
            # Prepare normalized sensor inputs (0 to 1)
            # 1.0 = Max distance (safe), 0.0 = Collision (dangerous)
            inputs = []
            for hit_pt, dist in self.sensors:
                inputs.append(dist / self.sensor_range)
            
            # If sensors aren't updated yet (first frame), use safe values
            if not inputs:
                inputs = [1.0] * len(self.sensor_angles)
            
            # ADDED: Speed Input (Normalized 0.0 to 1.0)
            # This allows the car to know how fast it is going
            inputs.append(self.speed / self.MAX_SPEED)
            
            # Feed through NN
            self.steering, self.throttle = self.brain.feed_forward(inputs)
            
            # CONSTRAINT: Disallow reversing (negative throttle) for autonomous cars
            # REMOVED: Simple max(0.2) clamp which disabled breaking
            
            # SMART THROTTLE:
            if self.speed < 20.0:
                # Anti-Stall: If stopped or slow, force acceleration
                self.throttle = max(0.2, self.throttle)
            else:
                # If moving, allow braking (negative throttle)
                # But don't change it, just let it be negative
                pass

        # Store old pos for fitness
        old_x, old_y = self.center_x, self.center_y

        # Apply throttle/brake
        self.speed += self.throttle * self.ACCEL * delta_time
        
        # Apply friction/drag (relative to speed)
        drag = self.speed * (1.0 - self.FRICTION) * delta_time * 10
        if self.speed > 0:
            self.speed = max(0, self.speed - drag)
        else:
            self.speed = min(0, self.speed - drag)
            
        # Hard cap
        max_s = float(self.MAX_SPEED)
        if self.speed > max_s:
            self.speed = max_s
        elif self.speed < 0:
            # PHYSICS CONSTRAINT: No Reverse Gear
            self.speed = 0

        # Steering (only if moving)
        if abs(self.speed) > 1.0:
            self.angle -= self.steering * self.TURN_RATE * delta_time
            
        # Move
        rad = math.radians(self.angle)
        self.center_x += math.cos(rad) * self.speed * delta_time
        self.center_y += math.sin(rad) * self.speed * delta_time

        # Update Fitness (Distance traveled)
        # PERFORMANCE FIX: Reverted Shapely projection (too slow for 20x speed)
        # We simplify fitness as the distance moved in the "right" direction
        dist_moved = math.sqrt((self.center_x - old_x)**2 + (self.center_y - old_y)**2)
        if self.speed > 0:
            self.fitness += dist_moved
        else:
            self.fitness -= dist_moved * 0.5 # Penalty for reversing

    def update_sensors(self, inner_wall_ls, outer_wall_ls, spatial_hash=None):
        """Perform raycasting to update sensor readings.
        
        Args:
            inner_wall_ls: Inner wall LineString (legacy, ignored if spatial_hash provided)
            outer_wall_ls: Outer wall LineString (legacy, ignored if spatial_hash provided)
            spatial_hash: Optional SpatialHash for O(1) lookups (preferred)
        """
        self.sensors = []
        
        # Use optimized spatial hash if available
        if spatial_hash is not None:
            for rel_angle in self.sensor_angles:
                abs_angle_rad = math.radians(self.angle + rel_angle)
                hit_point, dist = spatial_hash.raycast(
                    (self.center_x, self.center_y),
                    abs_angle_rad,
                    self.sensor_range
                )
                self.sensors.append((hit_point, dist))
            return
        
        # Fallback to Shapely (legacy)
        if not inner_wall_ls or not outer_wall_ls:
            return

        from shapely.geometry import LineString, Point
        
        car_pos = (self.center_x, self.center_y)
        
        for rel_angle in self.sensor_angles:
            abs_angle_rad = math.radians(self.angle + rel_angle)
            dest_x = self.center_x + math.cos(abs_angle_rad) * self.sensor_range
            dest_y = self.center_y + math.sin(abs_angle_rad) * self.sensor_range
            ray = LineString([car_pos, (dest_x, dest_y)])
            
            closest_dist = self.sensor_range
            closest_hit = (dest_x, dest_y)
            
            for wall in [inner_wall_ls, outer_wall_ls]:
                intersection = ray.intersection(wall)
                if not intersection.is_empty:
                    if intersection.geom_type == 'Point':
                        hits = [intersection]
                    elif intersection.geom_type == 'MultiPoint':
                        hits = list(intersection.geoms)
                    else:
                        continue
                    
                    for hit in hits:
                        dist = math.sqrt((hit.x - self.center_x)**2 + (hit.y - self.center_y)**2)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_hit = (hit.x, hit.y)
            
            self.sensors.append((closest_hit, closest_dist))

    def get_polygon(self):
        """Returns a shapely Polygon representing the car's current triangle footprint."""
        # Triangle dimensions (facing right)
        w, h = 10.0, 6.0
        rad = math.radians(self.angle)
        
        # Relative corners for a triangle
        # Tip at (w/2, 0), back corners at (-w/2, -h/2) and (-w/2, h/2)
        corners = [
            (w/2.0, 0),
            (-w/2.0, -h/2.0),
            (-w/2.0, h/2.0)
        ]
        
        # Rotate and translate
        pts = []
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            pts.append((self.center_x + rx, self.center_y + ry))
            
        return Polygon(pts)

    def draw(self, **kwargs):
        # Draw car as a red triangle
        poly = self.get_polygon()
        points = list(poly.exterior.coords)
        arcade.draw_polygon_filled(points, arcade.color.RED)
        
        # Tiny yellow indicator at the tip
        rad = math.radians(self.angle)
        nox = self.center_x + math.cos(rad) * 4
        noy = self.center_y + math.sin(rad) * 4
        arcade.draw_circle_filled(nox, noy, 1, arcade.color.YELLOW)
