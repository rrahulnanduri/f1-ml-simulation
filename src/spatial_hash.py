"""
Spatial Hash Grid for efficient wall segment lookups.
Divides the track into a grid and stores which wall segments intersect each cell.
This reduces raycasting from O(n) to O(1) average case.
"""
import math
from shapely.geometry import LineString, box

class SpatialHash:
    def __init__(self, cell_size=50.0):
        """
        Initialize a spatial hash grid.
        
        Args:
            cell_size: Size of each grid cell in pixels. Smaller = more precise but more memory.
        """
        self.cell_size = cell_size
        self.grid = {}  # Dict of (cell_x, cell_y) -> list of (segment_start, segment_end)
        self.wall_segments = []  # All segments for fallback
    
    def _get_cell(self, x, y):
        """Convert world coordinates to cell coordinates."""
        return (int(x // self.cell_size), int(y // self.cell_size))
    
    def _get_cells_for_segment(self, p1, p2):
        """Get all cells that a line segment passes through (Bresenham-like)."""
        x1, y1 = p1
        x2, y2 = p2
        
        # Get cell bounds
        cell1 = self._get_cell(x1, y1)
        cell2 = self._get_cell(x2, y2)
        
        # Simple: just return all cells in bounding box (good enough for short segments)
        min_cx = min(cell1[0], cell2[0])
        max_cx = max(cell1[0], cell2[0])
        min_cy = min(cell1[1], cell2[1])
        max_cy = max(cell1[1], cell2[1])
        
        cells = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cells.append((cx, cy))
        return cells
    
    def add_wall(self, wall_points):
        """
        Add wall segments to the spatial hash.
        
        Args:
            wall_points: List of (x, y) tuples forming a polyline.
        """
        for i in range(len(wall_points) - 1):
            p1 = wall_points[i]
            p2 = wall_points[i + 1]
            segment = (p1, p2)
            self.wall_segments.append(segment)
            
            # Add to all cells this segment touches
            for cell in self._get_cells_for_segment(p1, p2):
                if cell not in self.grid:
                    self.grid[cell] = []
                self.grid[cell].append(segment)
    
    def get_nearby_segments(self, x, y, radius=None):
        """
        Get wall segments near a point.
        
        Args:
            x, y: Query point
            radius: Optional radius to check multiple cells (defaults to 1 cell)
        
        Returns:
            List of (p1, p2) segment tuples
        """
        cell = self._get_cell(x, y)
        
        # Check current cell and immediate neighbors (9 cells total)
        segments = []
        seen = set()  # Avoid duplicates
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor in self.grid:
                    for seg in self.grid[neighbor]:
                        seg_id = id(seg)
                        if seg_id not in seen:
                            seen.add(seg_id)
                            segments.append(seg)
        
        return segments
    
    def raycast(self, origin, direction_rad, max_dist=250.0):
        """
        Cast a ray and find the closest intersection with walls.
        
        Args:
            origin: (x, y) start point
            direction_rad: Angle in radians
            max_dist: Maximum ray distance
        
        Returns:
            (hit_point, distance) - hit_point is (x, y) or None, distance is float
        """
        ox, oy = origin
        dx = math.cos(direction_rad)
        dy = math.sin(direction_rad)
        end_x = ox + dx * max_dist
        end_y = oy + dy * max_dist
        
        # Get cells along the ray path
        ray_cells = self._get_cells_for_segment(origin, (end_x, end_y))
        
        closest_dist = max_dist
        closest_hit = (end_x, end_y)
        
        # Check segments in ray cells
        checked = set()
        for cell in ray_cells:
            if cell not in self.grid:
                continue
            for seg in self.grid[cell]:
                seg_id = id(seg)
                if seg_id in checked:
                    continue
                checked.add(seg_id)
                
                # Line-line intersection
                hit = self._line_segment_intersection(origin, (end_x, end_y), seg[0], seg[1])
                if hit:
                    dist = math.sqrt((hit[0] - ox)**2 + (hit[1] - oy)**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_hit = hit
        
        return closest_hit, closest_dist
    
    def _line_segment_intersection(self, p1, p2, p3, p4):
        """
        Find intersection point of two line segments.
        Returns (x, y) or None if no intersection.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Parallel
        
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
        
        t = t_num / denom
        u = u_num / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        
        return None
