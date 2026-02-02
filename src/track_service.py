"""
Track Service using FastF1 and MVPAPI (MultiViewer) as suggested by user.
"""
from dataclasses import dataclass
from typing import Optional
import fastf1
import fastf1.mvapi.api
import numpy as np
import pandas as pd
import os
import arcade

@dataclass
class CircuitInfo:
    corners: pd.DataFrame
    marshal_lights: pd.DataFrame
    marshal_sectors: pd.DataFrame
    rotation: float

class TrackService:
    def __init__(self, year=2024, circuit_key=None, gp_name='Silverstone'):
        self.year = year
        self.circuit_key = circuit_key
        self.gp_name = gp_name
        self.cache_dir = "cache/"
        
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

    def get_track_layout(self):
        """
        Fallback to telemetry if needed, or use the MVAPI data to construct spacing.
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
        
        # rotation
        circuit_info = self.load_circuit_info()
        rotation_deg = circuit_info.rotation if circuit_info else 0.0
        
        # Apply rotation
        theta = np.radians(rotation_deg)
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
        
        # rotation
        circuit_info = self.load_circuit_info()
        rotation_deg = circuit_info.rotation if circuit_info else 0.0
        
        # Apply rotation
        theta = np.radians(rotation_deg)
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
