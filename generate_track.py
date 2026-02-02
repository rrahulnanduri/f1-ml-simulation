"""
Generate Track Data JSON
Fetches f1 telemetry and exports it for the React frontend.
"""
import fastf1
import json
import numpy as np
import os

CACHE_DIR = "cache/"

def fetch_and_export():
    # Setup
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    fastf1.Cache.enable_cache(CACHE_DIR)
    
    # Fetch
    print("Fetching Silverstone 2024 Qualifying...")
    session = fastf1.get_session(2024, 'Silverstone', 'Q')
    session.load(telemetry=True, laps=True, weather=False)
    
    print("Processing fastest lap...")
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()
    
    # Extract X, Y, Z (if needed)
    # FastF1 returns 1/10th of a meter. We'll convert to meters for simplicity.
    x = tel['X'].values / 10.0 
    y = tel['Y'].values / 10.0
    
    # Create list of points [x, y]
    coords = np.column_stack((x, y)).tolist()
    
    # Calculate bounds for normalization
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    
    metadata = {
        "circuit": "Silverstone",
        "year": 2024,
        "session": "Qualifying",
        "bounds": {
            "minX": min_x, "maxX": max_x,
            "minY": min_y, "maxY": max_y
        },
        "centerline": coords
    }
    
    # Export
    output_path = "src/constants/track_data.json"
    with open(output_path, 'w') as f:
        json.dump(metadata, f)
        
    print(f"Successfully exported {len(coords)} points to {output_path}")

if __name__ == "__main__":
    fetch_and_export()
