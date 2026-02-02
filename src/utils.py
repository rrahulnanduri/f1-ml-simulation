"""
Coordinates utility functions for transforming F1 telemetry to screen-space in Arcade.
"""
import numpy as np

def convert_lat_lon_to_xy(lat, lon, origin_lat, origin_lon, globe_radius=6371000):
    """
    Project lat/lon onto a 2D plane (meters) using an equirectangular projection centered on origin.
    Simplification is valid for track-sized areas.
    """
    x = (lon - origin_lon) * (np.pi / 180) * globe_radius * np.cos(origin_lat * np.pi / 180)
    y = (lat - origin_lat) * (np.pi / 180) * globe_radius
    return x, y

def fit_to_screen(coords, screen_width, screen_height, padding=100):
    """
    Scale and center (x, y) coordinates to fit within the screen dimensions.
    Returns: (final_coords, scale, min_pt, offset)
    """
    coords = np.array(coords)
    min_pt = np.min(coords, axis=0)
    max_pt = np.max(coords, axis=0)
    
    range_x = max_pt[0] - min_pt[0]
    range_y = max_pt[1] - min_pt[1]
    
    # Calculate scale factor to fit within padded screen
    scale_x = (screen_width - 2 * padding) / range_x
    scale_y = (screen_height - 2 * padding) / range_y
    scale = min(scale_x, scale_y)
    
    # Apply transformation:
    # 1. Translate to origin (0,0)
    # 2. Scale
    scaled_coords = (coords - min_pt) * scale
    
    # Calculate offset to center the shape
    center_offset_x = (screen_width - (range_x * scale)) / 2
    center_offset_y = (screen_height - (range_y * scale)) / 2
    offset = np.array([center_offset_x, center_offset_y])
    
    final_coords = scaled_coords + offset
    
    return final_coords.tolist(), scale, min_pt.tolist(), offset.tolist()

def resample_spline(coords, num_points=300):
    """
    Resample track coordinates using B-Spline interpolation.
    This dramatically reduces points while preserving smooth curvature.
    Fixes overlapping issues at tight corners (hairpins, chicanes).
    """
    from scipy.interpolate import splprep, splev
    
    coords = np.array(coords)
    
    # Close the loop if not already closed
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    # Fit a periodic B-Spline to the track
    # s=0 means interpolate through all points (no smoothing at fit stage)
    # per=True makes it periodic (closed loop)
    try:
        tck, u = splprep([coords[:, 0], coords[:, 1]], s=len(coords) * 5, per=True, k=3)
    except Exception as e:
        print(f"[WARNING] Spline fitting failed: {e}. Using original coords.")
        return coords.tolist()
    
    # Evaluate spline at evenly spaced points
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    
    resampled = np.column_stack([x_new, y_new])
    print(f"[SPLINE] Downsampled from {len(coords)} to {num_points} points")
    
    return resampled.tolist()

def smooth_coords(coords, window_size=5):
    """
    Apply a simple moving average to smooth the track coordinates.
    """
    coords = np.array(coords)
    smoothed = []
    # Pad array to handle circularity
    padded = np.vstack((coords[-window_size:], coords, coords[:window_size]))
    
    for i in range(len(coords)):
        # Average of window around point
        # taking slices from padded array
        # center index in padded is i + window_size
        start = i + window_size - window_size // 2
        end = start + window_size
        segment = padded[start:end]
        avg = np.mean(segment, axis=0)
        smoothed.append(avg)
        
    return np.array(smoothed)

def generate_track_walls(center_line, track_width=12.0):
    """
    Generate inner and outer wall coordinates from a center line.
    """
    points = np.array(center_line)
    num_points = len(points)
    inner_wall = []
    outer_wall = []
    
    for i in range(num_points):
        p1 = points[i]
        # Use next and prev points to calculate smoothed normal
        p_prev = points[(i - 1) % num_points]
        p_next = points[(i + 1) % num_points]
        
        # Vector from prev to next (tangent approximation)
        dx = p_next[0] - p_prev[0]
        dy = p_next[1] - p_prev[1]
        
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            nx, ny = 0, 0
        else:
            # Perpendicular (Normal)
            nx = -dy / length
            ny = dx / length
            
        # Offset
        inner_wall.append((p1[0] - nx * track_width/2, p1[1] - ny * track_width/2))
        outer_wall.append((p1[0] + nx * track_width/2, p1[1] + ny * track_width/2))
        
    return inner_wall, outer_wall
