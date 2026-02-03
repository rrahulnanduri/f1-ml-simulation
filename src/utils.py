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


def align_datasets(source_coords, target_coords):
    """
    Align source coordinates to target coordinates using uniform scaling.
    
    This function aligns two differently-scaled coordinate systems (e.g., telemetry and GeoJSON)
    by normalizing both to their centers and scaling source to match target's size.
    Uses UNIFORM scaling to preserve aspect ratio and speed consistency.
    
    Args:
        source_coords: List of (x, y) tuples in source coordinate system (e.g., telemetry)
        target_coords: List of (x, y) tuples in target coordinate system (e.g., GeoJSON)
    
    Returns:
        List of (x, y) tuples with source aligned to target space
    """
    source = np.array(source_coords)
    target = np.array(target_coords)
    
    # Get bounds and centers of both datasets
    src_min = np.min(source, axis=0)
    src_max = np.max(source, axis=0)
    src_range = src_max - src_min
    src_center = (src_min + src_max) / 2
    
    tgt_min = np.min(target, axis=0)
    tgt_max = np.max(target, axis=0)
    tgt_range = tgt_max - tgt_min
    tgt_center = (tgt_min + tgt_max) / 2
    
    # Avoid division by zero
    src_range[src_range == 0] = 1
    tgt_range[tgt_range == 0] = 1
    
    # Calculate UNIFORM scale (use the smaller scale to fit within target bounds)
    scale_x = tgt_range[0] / src_range[0]
    scale_y = tgt_range[1] / src_range[1]
    uniform_scale = min(scale_x, scale_y)  # Uniform to preserve aspect ratio
    
    # Center source at origin, scale uniformly, then translate to target center
    aligned = (source - src_center) * uniform_scale + tgt_center
    
    print(f"[ALIGN] Uniform scale: {uniform_scale:.4f} (src: {src_range[0]:.1f}x{src_range[1]:.1f} -> tgt: {tgt_range[0]:.1f}x{tgt_range[1]:.1f})")
    
    return aligned.tolist()

def align_procrustes_data(source, target, search_steps=360):
    """
    Aligns source points to target points using Procrustes Analysis with start-point search.
    Returns aligned source points in target's coordinate space.
    """
    A_orig = np.array(source)
    B_orig = np.array(target)
    
    # 1. Resample to uniform count for correspondence
    def resample(pts, n):
        if len(pts) < 2: return pts
        d = np.cumsum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))
        d = np.insert(d, 0, 0)
        tot = d[-1]
        u_d = np.linspace(0, tot, n+1)[:-1]
        nx = np.interp(u_d, d, pts[:,0])
        ny = np.interp(u_d, d, pts[:,1])
        return np.column_stack((nx, ny))

    N = search_steps
    A = resample(A_orig, N)
    B = resample(B_orig, N)
    
    # Centroids
    cA = np.mean(A, axis=0)
    cB = np.mean(B, axis=0)
    A0 = A - cA
    B0 = B - cB
    
    # Scale
    sA = np.sqrt(np.sum(A0**2))
    sB = np.sqrt(np.sum(B0**2))
    scale = sB / sA if sA > 0 else 1.0
    A1 = A0 * scale
    
    # Find Best Rotation & Shift
    best_err = float('inf')
    best_R = np.eye(2)
    best_shift_idx = 0
    
    for k in range(0, N, 1): # Scan all start shifts
        A_rolled = np.roll(A1, k, axis=0)
        
        # Kabsch Algorithm
        H = A_rolled.T @ B0
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Det check (force right-handed if needed, or allow mirror)
        # Assuming track direction matches, determinant should be +1.
        if np.linalg.det(R) < 0:
            Vt[1,:] *= -1
            R = Vt.T @ U.T
            
        aligned_scan = A_rolled @ R
        err = np.sum((aligned_scan - B0)**2)
        
        if err < best_err:
            best_err = err
            best_R = R
            best_shift_idx = k
            
    print(f"[ALIGN PROCRUSTES] Best Shift: {best_shift_idx}/{N} | Error: {best_err:.2f} | Scale: {scale:.3f}")
    
    # Apply to FULL Original Data
    # 1. Roll Original Data (approximate, based on arc length ratio)
    # Better: Apply the geometric transform (Translate->Scale->Rotate) 
    # AND handle the cyclic shift logically in the renderer/playback.
    # BUT we want the coordinates to match.
    # The 'shift' just matches the SHAPE. The Point Correspondence depends on where they start.
    # If we rotate the shape, the points move to the right place.
    # So we apply R, scale, translation.
    
    final_centered = (A_orig - cA) * scale
    final_rotated = final_centered @ best_R
    final_aligned = final_rotated + cB
    
    return final_aligned.tolist()

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
