"""
Track Service using FastF1 and MVPAPI (MultiViewer) as suggested by user.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
import fastf1
import fastf1.mvapi.api
import numpy as np
import pandas as pd
import os
import json
from rapidfuzz import fuzz, process

# Additional rotation (degrees) to convert portrait tracks to landscape
# Set to 0 for now - the track should already have correct orientation from MVAPI
LANDSCAPE_ROTATION = 0.0

@dataclass
class CircuitInfo:
    corners: pd.DataFrame
    marshal_lights: pd.DataFrame
    marshal_sectors: pd.DataFrame
    rotation: float

# Mapping from FastF1 GP names to GeoJSON circuit names
CIRCUIT_NAME_MAP = {
    'British Grand Prix': 'Silverstone Circuit',
    'Monaco Grand Prix': 'Circuit de Monaco',
    'Italian Grand Prix': 'Monza Circuit',
    'Belgian Grand Prix': 'Circuit de Spa-Francorchamps',
    'Australian Grand Prix': 'Albert Park Circuit',
    'Bahrain Grand Prix': 'Bahrain International Circuit',
    'Saudi Arabian Grand Prix': 'Jeddah Corniche Circuit',
    'Japanese Grand Prix': 'Suzuka International Racing Course',
    'Chinese Grand Prix': 'Shanghai International Circuit',
    'Miami Grand Prix': 'Miami International Autodrome',
    'Emilia Romagna Grand Prix': 'Imola Circuit',
    'Canadian Grand Prix': 'Circuit Gilles Villeneuve',
    'Spanish Grand Prix': 'Circuit de Barcelona-Catalunya',
    'Austrian Grand Prix': 'Red Bull Ring',
    'Hungarian Grand Prix': 'Hungaroring',
    'Dutch Grand Prix': 'Circuit Zandvoort',
    'Azerbaijan Grand Prix': 'Baku City Circuit',
    'Singapore Grand Prix': 'Marina Bay Street Circuit',
    'United States Grand Prix': 'Circuit of the Americas',
    'Mexico City Grand Prix': 'Autódromo Hermanos Rodríguez',
    'São Paulo Grand Prix': 'Interlagos Circuit',
    'Las Vegas Grand Prix': 'Las Vegas Strip Circuit',
    'Qatar Grand Prix': 'Lusail International Circuit',
    'Abu Dhabi Grand Prix': 'Yas Marina Circuit',
}

class TrackService:
    def __init__(self, year=2024, circuit_key=None, gp_name='Silverstone'):
        self.year = year
        self.circuit_key = circuit_key
        self.gp_name = gp_name
        self.cache_dir = "cache/"
        self.geojson_path = "data/f1-circuits.geojson"
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        fastf1.Cache.enable_cache(self.cache_dir)
    
    @staticmethod
    def get_available_tracks(year: int) -> list:
        """
        Returns a list of (EventName, RoundNumber) tuples for all races in a given year.
        Excludes pre-season testing.
        """
        schedule = fastf1.get_event_schedule(year)
        tracks = []
        for _, row in schedule.iterrows():
            # Skip pre-season testing (RoundNumber 0)
            if row['RoundNumber'] > 0:
                tracks.append((row['EventName'], row['RoundNumber']))
        return tracks

    def load_circuit_info(self) -> Optional[CircuitInfo]:
        """
        Loads circuit info similar to the user-provided snippet.
        """
        # If circuit_key is not provided, we need to find it? 
        # For simplicity, we can fetch a session first to get the circuit key or just try to pass it.
        # Silverstone 2024 circuit key is likely 63 (or we fetch a session to find it).
        
        # Let's fetch a session to ensure we have the right key if not provided
        if not self.circuit_key:
            print(f"Fetching session to determine circuit key for {self.gp_name}...")
            session = fastf1.get_session(self.year, self.gp_name, 'Q')
            # Check fastf1 version capability or just hardcode for valid test
            # The user snippet uses mvapi.get_circuit(year, circuit_key)
            # We can try to get circuit details from session.event
            session.load(telemetry=False, laps=False, weather=False, messages=False)
            self.circuit_key = session.session_info['Meeting']['Circuit']['Key']
            print(f"Found Circuit Key: {self.circuit_key}")

        print(f"Loading MVAPI circuit data for key {self.circuit_key}...")
        data = fastf1.mvapi.api.get_circuit(year=self.year, circuit_key=self.circuit_key)
        
        if not data:
            print("Failed to load circuit info from MVAPI.")
            return None

        # Parse Corners
        rows = []
        for entry in (data.get('corners') or []):
            rows.append((
                float(entry.get('trackPosition', {}).get('x', 0.0)),
                float(entry.get('trackPosition', {}).get('y', 0.0)),
                int(entry.get('number', 0)),
                float(entry.get('angle', 0.0))
            ))
        
        corners_df = pd.DataFrame(rows, columns=['X', 'Y', 'Number', 'Angle'])
        rotation = float(data.get('rotation', 0.0))
        
        return CircuitInfo(corners=corners_df, marshal_lights=pd.DataFrame(), marshal_sectors=pd.DataFrame(), rotation=rotation)

    def _find_circuit_in_geojson(self, geojson_data: dict) -> Optional[dict]:
        """
        Find the matching circuit in GeoJSON using fuzzy matching.
        """
        # First try exact mapping
        mapped_name = CIRCUIT_NAME_MAP.get(self.gp_name)
        
        # Get all circuit names from GeoJSON
        circuit_names = []
        for feature in geojson_data.get('features', []):
            name = feature.get('properties', {}).get('Name', '')
            location = feature.get('properties', {}).get('Location', '')
            circuit_names.append((name, location, feature))
        
        # Try exact match first
        if mapped_name:
            for name, location, feature in circuit_names:
                if name.lower() == mapped_name.lower():
                    print(f"[GEOJSON] Exact match found: {name}")
                    return feature
        
        # Try fuzzy match on GP name
        search_terms = [self.gp_name, self.gp_name.replace(' Grand Prix', '')]
        for search_term in search_terms:
            # Try matching against circuit name
            name_matches = process.extract(search_term, [n[0] for n in circuit_names], scorer=fuzz.partial_ratio, limit=3)
            for match_name, score, _ in name_matches:
                if score > 70:
                    for name, location, feature in circuit_names:
                        if name == match_name:
                            print(f"[GEOJSON] Fuzzy match found: {name} (score: {score})")
                            return feature
            
            # Try matching against location
            location_matches = process.extract(search_term, [n[1] for n in circuit_names], scorer=fuzz.partial_ratio, limit=3)
            for match_loc, score, _ in location_matches:
                if score > 70:
                    for name, location, feature in circuit_names:
                        if location == match_loc:
                            print(f"[GEOJSON] Location match found: {name} @ {location} (score: {score})")
                            return feature
        
        print(f"[GEOJSON] No match found for: {self.gp_name}")
        return None

    def get_fixed_track_layout(self) -> Optional[List[Tuple[float, float]]]:
        """
        Load fixed track layout from GeoJSON file.
        Returns list of (x, y) coordinates in meters, or None if not found.
        """
        import pickle
        from src.utils import convert_lat_lon_to_xy
        
        # Try cache first
        cache_key = f"fixed_track_{self.gp_name.replace(' ', '_')}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            print(f"[CACHE HIT] Loading fixed track layout from {cache_key}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Load GeoJSON
        if not os.path.exists(self.geojson_path):
            print(f"[ERROR] GeoJSON file not found: {self.geojson_path}")
            return None
        
        with open(self.geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Find matching circuit
        circuit_feature = self._find_circuit_in_geojson(geojson_data)
        if not circuit_feature:
            print(f"[WARN] Circuit not found in GeoJSON, falling back to telemetry")
            return None
        
        # Extract coordinates (lon, lat format in GeoJSON)
        coords = circuit_feature.get('geometry', {}).get('coordinates', [])
        if not coords:
            print(f"[ERROR] No coordinates in GeoJSON feature")
            return None
        
        # Convert lat/lon to XY (meters)
        # Use first point as origin
        origin_lon, origin_lat = coords[0][0], coords[0][1]
        
        xy_coords = []
        for lon, lat in coords:
            x, y = convert_lat_lon_to_xy(lat, lon, origin_lat, origin_lon)
            xy_coords.append((x, y))
        
        # Apply same rotation as telemetry uses (from MVAPI)
        circuit_info = self.load_circuit_info()
        rotation_deg = circuit_info.rotation if circuit_info else 0.0
        
        if rotation_deg != 0:
            theta = np.radians(rotation_deg)
            c, s = np.cos(theta), np.sin(theta)
            rotated_coords = []
            for x, y in xy_coords:
                x_rot = x * c - y * s
                y_rot = x * s + y * c
                rotated_coords.append((x_rot, y_rot))
            xy_coords = rotated_coords
            print(f"[GEOJSON] Applied rotation: {rotation_deg}°")
        
        # Close the loop if not already closed
        if xy_coords[0] != xy_coords[-1]:
            xy_coords.append(xy_coords[0])
        
        print(f"[GEOJSON] Loaded {len(xy_coords)} points for {self.gp_name}")
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(xy_coords, f)
        print(f"[CACHE SAVE] Saved fixed track layout to {cache_key}")
        
        return xy_coords

    def get_telemetry_track_layout(self):
        """
        Get track layout from fastest lap telemetry.
        Uses pickle caching for faster subsequent loads.
        """
        import pickle
        
        # Try to load from cache first
        cache_key = f"track_{self.year}_{self.gp_name.replace(' ', '_')}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            print(f"[CACHE HIT] Loading track layout from {cache_key}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"[CACHE MISS] Fetching track layout from FastF1...")
        session = fastf1.get_session(self.year, self.gp_name, 'Q')
        session.load(telemetry=True, laps=True)
        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry()
        
        x = tel['X'].values
        y = tel['Y'].values
        
        # rotation from MVAPI
        circuit_info = self.load_circuit_info()
        rotation_deg = circuit_info.rotation if circuit_info else 0.0
        
        # Add landscape rotation to convert portrait to landscape
        total_rotation = rotation_deg + LANDSCAPE_ROTATION
        
        # Apply rotation
        theta = np.radians(total_rotation)
        c, s = np.cos(theta), np.sin(theta)
        
        # Rotate coordinates
        x_rot = x * c - y * s
        y_rot = x * s + y * c
        
        result = list(zip(x_rot, y_rot))
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"[CACHE SAVE] Saved track layout to {cache_key}")
        
        return result

    def get_track_layout(self):
        """
        Get track layout - tries fixed map first, falls back to telemetry.
        """
        fixed_layout = self.get_fixed_track_layout()
        if fixed_layout:
            return fixed_layout
        return self.get_telemetry_track_layout()


    def get_fastest_lap_trajectory(self):
        """
        Returns a list of (x, y, time_in_seconds) for the ghost car.
        Uses pickle caching for faster subsequent loads.
        """
        import pickle
        
        # Try to load from cache first
        cache_key = f"ghost_{self.year}_{self.gp_name.replace(' ', '_')}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            print(f"[CACHE HIT] Loading ghost trajectory from {cache_key}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"[CACHE MISS] Fetching ghost trajectory from FastF1...")
        session = fastf1.get_session(self.year, self.gp_name, 'Q')
        session.load(telemetry=True, laps=True)
        # Specifically pick driver 44
        ham_laps = session.laps.pick_driver('44')
        if not ham_laps.empty:
            lap = ham_laps.pick_fastest()
        else:
            lap = session.laps.pick_fastest()
        tel = lap.get_telemetry()

        
        # Remove duplicates based on Time to prevent bisect errors
        tel = tel.drop_duplicates(subset=['Time'])
        
        x = tel['X'].values
        y = tel['Y'].values
        # Relative time in seconds
        t_raw = tel['Time'].dt.total_seconds().values
        t = t_raw - t_raw[0] # RELATIVE TO LAP START
        
        # rotation from MVAPI + landscape
        circuit_info = self.load_circuit_info()
        rotation_deg = circuit_info.rotation if circuit_info else 0.0
        total_rotation = rotation_deg + LANDSCAPE_ROTATION
        
        # Apply rotation
        theta = np.radians(total_rotation)
        c, s = np.cos(theta), np.sin(theta)
        
        # Rotate coordinates
        x_rot = x * c - y * s
        y_rot = x * s + y * c
        
        # ===== 60 Hz RESAMPLING (ROOT CAUSE FIX) =====
        # FastF1 telemetry is ~7.7 Hz with gaps up to 700ms.
        # Resample to 60 Hz to match simulation physics rate.
        from scipy.interpolate import interp1d
        
        duration = t[-1] - t[0]
        target_hz = 60
        num_samples = int(duration * target_hz)
        t_resampled = np.linspace(t[0], t[-1], num_samples)
        
        # Create interpolation functions
        interp_x = interp1d(t, x_rot, kind='linear', fill_value='extrapolate')
        interp_y = interp1d(t, y_rot, kind='linear', fill_value='extrapolate')
        
        x_resampled = interp_x(t_resampled)
        y_resampled = interp_y(t_resampled)
        
        # Build resampled trajectory
        trajectory = list(zip(x_resampled, y_resampled, t_resampled))
        
        # DEBUG: Log Telemetry Statistics (After Resampling)
        print(f"[TELEMETRY DEBUG] Original Points: {len(t)} | Resampled Points: {len(trajectory)}")
        print(f"[TELEMETRY DEBUG] Time Range: {t_resampled[0]:.3f}s to {t_resampled[-1]:.3f}s (Duration: {duration:.3f}s)")
        print(f"[TELEMETRY DEBUG] Resampled Rate: {target_hz} Hz")
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(trajectory, f)
        print(f"[CACHE SAVE] Saved ghost trajectory to {cache_key}")
        
        return trajectory
