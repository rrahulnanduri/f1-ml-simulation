"""
Arcade based renderer for the simulation window.
"""
import arcade
import arcade.gui
import numpy as np
import math
from src.car import Car
from src.neural_net import NeuralNetwork
from src.evolution_manager import EvolutionManager
from shapely.geometry import LineString

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1000
SCREEN_TITLE = "F1 ML Simulation - Silverstone"
TRACK_COLOR = arcade.color.ORANGE
WALL_COLOR_OUTER = arcade.color.WHITE
WALL_COLOR_INNER = arcade.color.WHITE
WALL_WIDTH = 2
WALL_WIDTH_INNER = 1
TRACK_WIDTH = 25.0  # Pixels - Reduced width as requested
POPULATION_SIZE = 30  # Reduced from 50 for better performance
SIDEBAR_WIDTH = 300   # Dedicated area for controls

class SimulationWindow(arcade.Window):
    def __init__(self, track_service):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=True)
        
        # Track Data
        self.track_service = track_service
        self.track_points = []
        self.inner_wall = []
        self.outer_wall = []
        self.inner_wall_ls = None
        self.outer_wall_ls = None
        
        # Camera/View
        self.camera = arcade.Camera2D()
        self.gui_camera = arcade.Camera2D() # For HUD
        
        # UI Manager
        self.ui_manager = arcade.gui.UIManager()
        self.ui_manager.enable()
        self.time_scale = 1.0
        
        # State
        self.is_loading = True
        self.cars = []
        # UPDATED: Mutation Rate 0.3 for more aggressive learning
        self.evolution_manager = EvolutionManager(mutation_rate=0.3)
        self.best_fitness = 0.0
        self.fastest_lap = None # None means no full lap yet
        
        # Generation Timer
        self.generation_timer = 0.0
        self.GENERATION_DURATION = 300.0 # Seconds (5 minutes simulated time)
        
        # Physics State
        self.physics_accumulator = 0.0
        self.PHYSICS_STEP = 1.0 / 60.0 # Fixed 60 Hz physics
        self.MAX_FRAME_TIME = 0.5 # Clamp to avoid spiral of death (increased for 20x speed)
        
        # Background
        arcade.set_background_color(arcade.color.BLACK)
        
        # Build UI
        self.setup_ui()

    def on_resize(self, width, height):
        """Handle window resizing."""
        super().on_resize(width, height)
        # Update cameras to match new window size
        self.camera.match_window()
        self.gui_camera.match_window()
        
        # Update HUD Text Positions
        # No manual update needed for Layout Widgets! 
        pass
            
        # Note: Track layout is kept as-is, view just expands/contracts.

    def setup_ui(self):
        from src.track_service import TrackService
        
        # Create a vertical BoxGroup to align controls
        self.v_box = arcade.gui.UIBoxLayout()

        # ===== TRACK/YEAR SELECTION =====
        # Year Label
        year_label = arcade.gui.UILabel(
            text="Year",
            text_color=arcade.color.WHITE,
            width=150,
            height=20,
            font_size=12
        )
        self.v_box.add(year_label)
        
        # Year Dropdown (2018-2024)
        self.available_years = [2024, 2023, 2022, 2021, 2020, 2019, 2018]
        self.selected_year = 2024
        self.year_dropdown = arcade.gui.UIDropdown(
            default=str(self.selected_year),
            options=[str(y) for y in self.available_years],
            width=150,
            height=30
        )
        
        @self.year_dropdown.event("on_change")
        def on_year_change(event):
            self.selected_year = int(self.year_dropdown.value)
            self._update_track_dropdown()
        
        self.v_box.add(self.year_dropdown)
        
        # Track Label
        track_label = arcade.gui.UILabel(
            text="Circuit",
            text_color=arcade.color.WHITE,
            width=150,
            height=20,
            font_size=12
        )
        self.v_box.add(track_label)
        
        # Track Dropdown (populated dynamically)
        self.available_tracks = TrackService.get_available_tracks(self.selected_year)
        track_names = [t[0] for t in self.available_tracks]
        default_track = track_names[0] if track_names else "No Tracks"
        self.track_dropdown = arcade.gui.UIDropdown(
            default=default_track,
            options=track_names,
            width=150,
            height=30
        )
        self.v_box.add(self.track_dropdown)
        
        # Load Button
        load_button = arcade.gui.UIFlatButton(text="Load Track", width=150, height=35)
        
        @load_button.event("on_click")
        def on_load_click(event):
            self._load_selected_track()
        
        self.v_box.add(load_button)
        
        # Spacer
        spacer = arcade.gui.UISpace(width=150, height=20)
        self.v_box.add(spacer)
        
        # ===== SIM SPEED SLIDER =====
        ui_text_label = arcade.gui.UILabel(
            text="Sim Speed",
            text_color=arcade.color.WHITE,
            width=100,
            height=20,
            font_size=12,
            font_name="Arial"
        )
        self.v_box.add(ui_text_label)

        self.ui_slider = arcade.gui.UISlider(value=1.0, min_value=1.0, max_value=20.0, width=150, height=20)
        
        @self.ui_slider.event("on_change")
        def on_change_slider(event):
            self.time_scale = self.ui_slider.value
            
        self.v_box.add(self.ui_slider)
        
        # Spacer
        self.v_box.add(arcade.gui.UISpace(width=150, height=20))
        
        # ===== HUD STATS (Labels in Layout) =====
        # 1. Track Name
        self.lbl_track = arcade.gui.UILabel(text="TRACK: --", width=150, height=20, font_size=12, text_color=arcade.color.ORANGE, align="left")
        self.v_box.add(self.lbl_track)
        
        # 2. Generation
        self.lbl_gen = arcade.gui.UILabel(text="Gen: 0", width=150, height=20, font_size=12, text_color=arcade.color.WHITE, align="left")
        self.v_box.add(self.lbl_gen)
        
        # 3. Fastest Lap
        self.lbl_fastest = arcade.gui.UILabel(text="Fastest: --.--", width=150, height=20, font_size=12, text_color=arcade.color.YELLOW, align="left")
        self.v_box.add(self.lbl_fastest)
        
        # 4. Alive
        self.lbl_alive = arcade.gui.UILabel(text="Alive: 0/0", width=150, height=20, font_size=12, text_color=arcade.color.CYAN, align="left")
        self.v_box.add(self.lbl_alive)
        
        # 5. Speed
        self.lbl_speed = arcade.gui.UILabel(text="Speed: 1.0x", width=150, height=20, font_size=12, text_color=arcade.color.MAGENTA, align="left")
        self.v_box.add(self.lbl_speed)
        
        # 6. Time (Timer)
        self.lbl_time = arcade.gui.UILabel(text="Time: 300.0s", width=150, height=20, font_size=12, text_color=arcade.color.GREEN, align="left")
        self.v_box.add(self.lbl_time)

        # Create a widget to hold the v_box widget, anchored top-right in sidebar area
        layout = arcade.gui.UIAnchorLayout()
        # Anchor to right edge, with slight padding
        layout.add(child=self.v_box, anchor_x="right", anchor_y="top", align_x=-20, align_y=-20)
        
        self.ui_manager.add(layout)
        
        # Remove old text objects (replaced by UILabels)
        if hasattr(self, 'track_label'):
             del self.track_label

    
    def _update_track_dropdown(self):
        """Update track dropdown options when year changes."""
        from src.track_service import TrackService
        self.available_tracks = TrackService.get_available_tracks(self.selected_year)
        track_names = [t[0] for t in self.available_tracks]
        # Note: arcade.gui.UIDropdown doesn't support dynamic option updates easily
        # We rebuild the options list for next load
        print(f"[UI] Year changed to {self.selected_year}. Available tracks: {len(track_names)}")
    
    def _load_selected_track(self):
        """Load the selected track and reinitialize simulation."""
        from src.track_service import TrackService
        
        selected_track_name = self.track_dropdown.value
        print(f"[UI] Loading track: {selected_track_name} ({self.selected_year})")
        
        # Create new TrackService with selected parameters
        self.track_service = TrackService(year=self.selected_year, gp_name=selected_track_name)
        
        # Reset simulation state
        self.cars = []
        self.ghost_car = None
        self.generation_timer = 0.0
        self.best_fitness = 0.0
        self.fastest_lap = None
        self.evolution_manager = EvolutionManager(mutation_rate=0.3)
        
        # Reload track
        self.setup()
        
        # Update HUD
        self.track_label.text = f"TRACK: {selected_track_name.upper()}"

    def setup(self):
        """
        Load the track and prepare the shapes.
        """
        print("Setting up simulation window...")
        # Reset physics accumulator
        self.physics_accumulator = 0.0
        
        # Load Track Telemetry
        raw_coords = self.track_service.get_track_layout()
        
        # Reverted B-Spline Downsampling due to stability issues (cars flying off track)
        # We go back to using raw_coords directly.
        from src.utils import fit_to_screen, generate_track_walls
        
        # Fit track to screen (accounting for sidebar)
        # Use available width: screen width (removed sidebar offset)
        available_width = self.width
        # Reduced padding to Make Map Bigger
        self.track_points, scale, min_pt, offset = fit_to_screen(raw_coords, available_width, self.height, padding=20)
        
        # Procedurally Generate Walls
        self.inner_wall, self.outer_wall = generate_track_walls(self.track_points, track_width=TRACK_WIDTH)
        
        # Create Shapely LineStrings for collisions (legacy fallback)
        if len(self.inner_wall) > 1:
            self.inner_wall_ls = LineString(self.inner_wall + [self.inner_wall[0]])
            self.outer_wall_ls = LineString(self.outer_wall + [self.outer_wall[0]])
        
        # PERFORMANCE: Create Spatial Hash for O(1) raycasting
        from src.spatial_hash import SpatialHash
        self.spatial_hash = SpatialHash(cell_size=50.0)
        self.spatial_hash.add_wall(self.inner_wall + [self.inner_wall[0]])  # Close the loop
        self.spatial_hash.add_wall(self.outer_wall + [self.outer_wall[0]])
        print(f"[PERF] SpatialHash built with {len(self.spatial_hash.grid)} cells")
        
        # Track Centerline for Fitness
        if len(self.track_points) > 1:
            self.track_ls = LineString(self.track_points)
            self.track_length = self.track_ls.length
            print(f"Track Length: {self.track_length:.1f} pixels")
        else:
            self.track_ls = None
            self.track_length = 0.0

        # GHOST CAR: Lewis Hamilton
        print("Fetching Lewis Hamilton's Ghost Data...")
        raw_trajectory = self.track_service.get_fastest_lap_trajectory()
        
        # Transform trajectory to match track alignment
        # Formula: (P - min_pt) * scale + offset
        trajectory = []
        for x, y, t in raw_trajectory:
            tx = (x - min_pt[0]) * scale + offset[0]
            ty = (y - min_pt[1]) * scale + offset[1]
            trajectory.append([tx, ty, t])
            
        # SMOOTH TRAJECTORY: Apply light smoothing (60Hz data is already clean from resampling)
        if len(trajectory) > 10:
            from src.utils import smooth_coords
            coords_only = [[p[0], p[1]] for p in trajectory]
            # Minimal window (3) to avoid introducing lag - 60Hz data is already smooth
            smoothed_pts = smooth_coords(coords_only, window_size=3)
            for i in range(len(trajectory)):
                trajectory[i][0] = smoothed_pts[i][0]
                trajectory[i][1] = smoothed_pts[i][1]

        from src.ghost_car import GhostCar
        self.ghost_car = GhostCar(trajectory)
        print("Ghost Car Initialized.")

        # Initialize Population
        self.spawn_population()

        self.is_loading = False
        print("Setup complete.")

    def spawn_population(self, brains=None):
        """Spawns a new generation of cars."""
        self.generation_timer = 0.0
        # Reset Ghost Car for new race
        if self.ghost_car:
            # We don't need to reset state, update_position handles time
            pass
            
        self.cars = []
        if len(self.track_points) > 1:
            # FIND SAFE SPAWN POINT
            # Search for a point where the car doesn't instantly collide with walls
            start_pos = self.track_points[0]
            start_angle = 0
            
            if self.inner_wall_ls and self.outer_wall_ls:
                found_safe = False
                for i in range(min(50, len(self.track_points)-1)):
                    pos = self.track_points[i]
                    next_p = self.track_points[i+1]
                    angle = math.degrees(math.atan2(next_p[1]-pos[1], next_p[0]-pos[0]))
                    
                    # Test Car Collision
                    test_car = Car(pos[0], pos[1], angle)
                    poly = test_car.get_polygon()
                    if not poly.intersects(self.inner_wall_ls) and not poly.intersects(self.outer_wall_ls):
                        start_pos = pos
                        start_angle = angle
                        found_safe = True
                        print(f"Index {i}: Safe Spawn Found")
                        break
                    else:
                        print(f"Index {i}: Spawn Collision Detected!")
                
                if not found_safe:
                    print("WARNING: Could not find safe spawn point in first 50 points. Using Index 0.")
                    p0 = self.track_points[0]
                    p1 = self.track_points[1]
                    start_pos = p0
                    start_angle = math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0]))
            else:
                 # Fallback if no walls yet
                 next_pos = self.track_points[1]
                 start_angle = math.degrees(math.atan2(next_pos[1] - start_pos[1], next_pos[0] - start_pos[0]))
            
            for i in range(POPULATION_SIZE):
                car = Car(start_pos[0], start_pos[1], start_angle, self.track_ls)
                if brains and i < len(brains):
                    car.brain = brains[i]
                else:
                    # +1 for Speed Input
                    car.brain = NeuralNetwork(input_size=len(car.sensor_angles) + 1)
                self.cars.append(car)
            
            # Store start angle for Wrong-Way detection
            self.start_angle = start_angle

    def on_draw(self):
        """Render the screen."""
        self.clear()
        self.camera.use()
        
        # Draw Sidebar Background (Behind everything else in 2D Camera, actually needs to be Screen Space?)
        # For Simplicity, we draw it in World Space but it covers the right side gap?
        # NO: Sidebar is UI, should be drawn with GUI camera or just as a World overlay.
        # But UI Manager handles buttons. We need a background.
        
        # ACTUALLY: Let's draw it in GUI Camera space (on top of map)
        
        # Draw track walls in Sectors
        # REMOVED: Center line (Track Points)
        
        SECTOR_COLORS = [arcade.color.CYAN, arcade.color.MAGENTA, arcade.color.YELLOW]
        
        for wall_points in [self.outer_wall, self.inner_wall]:
            if len(wall_points) < 3:
                continue
                
            n = len(wall_points)
            chunk_size = n // 3
            
            # Sector 1
            pts1 = wall_points[0 : chunk_size + 1]
            arcade.draw_line_strip(pts1, SECTOR_COLORS[0], WALL_WIDTH)
            
            # Sector 2
            pts2 = wall_points[chunk_size : chunk_size * 2 + 1]
            arcade.draw_line_strip(pts2, SECTOR_COLORS[1], WALL_WIDTH)
            
            # Sector 3
            pts3 = wall_points[chunk_size * 2 :] 
            # Force close the loop visually if needed, but walls usually wrap via generation? 
            # The generation logic didn't explicitly close the loop in the list, 
            # but usually we want to see the gap or the start line handles it.
            # Let's just draw to the end.
            arcade.draw_line_strip(pts3, SECTOR_COLORS[2], WALL_WIDTH)

        # Start Line
        if len(self.outer_wall) > 1 and len(self.inner_wall) > 1:
            def get_third_pt(p1, p2):
                return (p1[0] + (p2[0] - p1[0]) / 3.0, p1[1] + (p2[1] - p1[1]) / 3.0)
            p_third_out = get_third_pt(self.outer_wall[0], self.outer_wall[1])
            p_third_in = get_third_pt(self.inner_wall[0], self.inner_wall[1])
            arcade.draw_line(self.outer_wall[0][0], self.outer_wall[0][1], p_third_out[0], p_third_out[1], arcade.color.RED, WALL_WIDTH + 2)
            arcade.draw_line(self.inner_wall[0][0], self.inner_wall[0][1], p_third_in[0], p_third_in[1], arcade.color.RED, WALL_WIDTH + 2)

        # Draw Ghost Car
        if self.ghost_car:
            self.ghost_car.draw()
            # Label is now rendered inside ghost_car.draw() - REMOVED slow draw_text

        # Draw AI Cars
        for car in self.cars:
            if car.is_alive:
                car.draw()
                # DEBUG: Draw Sensors
                for hit_pt, dist in car.sensors:
                    # Color based on distance (Red = close, Green = safe)
                    color = arcade.color.RED if dist < car.sensor_range * 0.3 else arcade.color.GREEN
                    arcade.draw_line(car.center_x, car.center_y, hit_pt[0], hit_pt[1], color, 1)
        
        # Draw HUD - Optimized using arcade.Text objects
        self.gui_camera.use()
        
        # ===== HUD RENDERING =====
        # Update UI Labels (Values updated every frame)
        self.lbl_track.text = f"TRACK: {self.track_service.gp_name.upper()}"
        
        fastest_val = f"Fastest: {self.fastest_lap:.2f}s" if self.fastest_lap else "Fastest: --.--"
        self.lbl_fastest.text = fastest_val
        
        self.lbl_gen.text = f"Gen: {self.evolution_manager.generation}"
        self.lbl_alive.text = f"Alive: {sum(1 for c in self.cars if c.is_alive)}/{POPULATION_SIZE}"
        self.lbl_speed.text = f"Speed: {self.time_scale:.1f}x"
        
        time_left = max(0, self.GENERATION_DURATION - self.generation_timer)
        self.lbl_time.text = f"Time: {time_left:.1f}s"
        # self.lbl_time.style["text_color"] = ... (Removed to fix crash)

        # Draw UI
        self.ui_manager.draw()

    def update_physics(self, dt):
        """Fixed step physics update."""
        all_dead = True
        lap_finished = False
        
        # Advance Simulation Clock
        self.generation_timer += dt
        
        # Update Ghost Car (Synchronized with physics clock)
        # Update Ghost Car (Synchronized with physics clock)
        if self.ghost_car:
            self.ghost_car.update_position(self.generation_timer)
            # Check if Ghost has finished the lap
            if self.generation_timer >= self.ghost_car.times[-1]:
                lap_finished = True
                print(f"[GHOST] Finished lap in {self.generation_timer:.2f}s")
        
        for car in self.cars:
            if not car.is_alive:
                continue
            
            all_dead = False
            car.update(dt)
            # Remove sensors update_sensors if they are not needed for visuals? 
            # NO! Neural net needs them! We just don't draw them.
            car.update_sensors(self.inner_wall_ls, self.outer_wall_ls, self.spatial_hash)
            
            # Check for Lap Completion
            if self.track_length > 0 and car.fitness > self.track_length:
                lap_finished = True
                
                # Update Fastest Lap
                if self.fastest_lap is None or car.time_alive < self.fastest_lap:
                    self.fastest_lap = car.time_alive
                    
                # Bonus for finishing (Time based: quicker is better)
                # Base 1000 + 10 points per second left on clock
                time_bonus = (self.GENERATION_DURATION - car.time_alive) * 10
                car.fitness += 1000 + max(0, time_bonus)
                
                print(f"Car finished lap in {car.time_alive:.2f}s! Fitness: {car.fitness}")
            
            # Collision Detection
            if self.inner_wall_ls and self.outer_wall_ls:
                car_poly = car.get_polygon()
                if car_poly.intersects(self.inner_wall_ls) or car_poly.intersects(self.outer_wall_ls):
                    car.is_alive = False
                    
                    # IMPACT PENALTY:
                    # If you hit the wall at 600px/s, you lose 300 points.
                    # This makes "fast crash" worse than "slow drive".
                    penalty = (car.speed / car.MAX_SPEED) * 300.0
                    penalty = (car.speed / car.MAX_SPEED) * 300.0
                    car.fitness -= penalty
                    
                    if car.fitness > self.best_fitness:
                        # Only update best fitness if it's actually a good run (net positive)
                        self.best_fitness = car.fitness
            
            # WRONG WAY DETECTION (Behind Start Line)
            # If car is near start and facing backwards (>90 deg from start angle), kill it.
            if hasattr(self, 'start_angle') and self.track_points:
                start_pt = self.track_points[0]
                dist_to_start = math.sqrt((car.center_x - start_pt[0])**2 + (car.center_y - start_pt[1])**2)
                
                if dist_to_start < 300:
                    # Calculate angle difference
                    diff = abs((car.angle - self.start_angle + 180) % 360 - 180)
                    if diff > 100: # Allow slight spins, but not full reverse
                        car.is_alive = False
                        car.fitness -= 500 # Heavy penalty for going wrong way
                        print(f"Car killed for driving backwards at start! (Angle Diff: {diff:.1f})")
                        
        return all_dead, lap_finished

    def on_update(self, delta_time: float):
        """Movement and game logic."""
        if self.is_loading:
            return

        # 1. Apply Time Scale
        # Multiply input time by slider value (e.g. 2.0x)
        dt_scaled = delta_time * self.time_scale

        # 2. Clamp Delta Time (Option 1)
        # Prevent spiral of death if frame time is huge (e.g. startup lag)
        dt = min(dt_scaled, self.MAX_FRAME_TIME)
        
        # 3. Accumulate time
        self.physics_accumulator += dt
        
        # 4. Fixed Step Loop (Option 3)
        all_dead = True
        lap_finished = False
        
        while self.physics_accumulator >= self.PHYSICS_STEP:
            all_dead, lap_finished = self.update_physics(self.PHYSICS_STEP)
            if lap_finished:
                break
            self.physics_accumulator -= self.PHYSICS_STEP

        # UPDATE: Generation Timer ("The Grim Reaper")
        # If the generation lasts too long (e.g. cars driving slowly forever),
        # we kill it. This forces them to race against the clock.
        timeout = self.generation_timer >= self.GENERATION_DURATION
        
        # REMOVED: Ghost Car update from here (moved to update_physics)

        # Next Generation if all cars crashed OR time is up OR lap finished
        if all_dead or timeout or lap_finished:
            reason = "Timeout" if timeout else ("Lap Finished" if lap_finished else "Crash")
            print(f"Gen {self.evolution_manager.generation} ended ({reason}). Evolving...")
            new_brains = self.evolution_manager.evolve(self.cars)
            self.spawn_population(new_brains)

    def on_key_press(self, key, modifiers):
        """Disabled for evolution mode."""
        pass

    def on_key_release(self, key, modifiers):
        """Disabled for evolution mode."""
        pass
