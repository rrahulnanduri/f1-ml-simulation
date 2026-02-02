import arcade
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.track_service import TrackService
from src.renderer import SimulationWindow
from src.utils import fit_to_screen, generate_track_walls, smooth_coords

def main():
    # 1. Fetch Data
    track_svc = TrackService(year=2024, gp_name='Silverstone')
    raw_coords = track_svc.get_track_layout()
    
    # 2. Start Simulation Window
    window = SimulationWindow(track_svc)
    
    # 3. Setup window (loads track and spawns car)
    window.setup()
    
    arcade.run()

if __name__ == "__main__":
    main()
