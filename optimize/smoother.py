import numpy as np 
from typing import List
import copy 
from env.robot import PandaRobot
from scipy.interpolate import splprep, splev

def is_segment_safe(
    robot: PandaRobot,
    q1: np.ndarray,
    q2: np.ndarray,
    obstacle_ids: List[int],
    plane_id:int,
    steps:int = 10
) -> bool:
    """
    drawing a straight line between two joint configurations 
    and checking every few inches to make sure the robot doesn't hit a wall
    """

    for t in np.linspace(0,1,steps):
        q_check = q1 + t*(q2 - q1)
        if robot.is_collision(q_check, obstacle_ids, plane_id):
            return False
    return True

def smooth_path(
    path: List[np.ndarray],
    robot: PandaRobot,
    obstacle_ids: List[int],
    plane_id: int,
    max_iterations: int = 150
) -> List[np.ndarray]:
    """
    Randomly select 2 points and if there is nothing between them , 
    reduce the point in the middle 
    """
    if not path or len(path) < 3:
        return path

    smoothed_path = copy.deepcopy(path)

    for _ in range(max_iterations):
        n = len(smoothed_path)
        if n < 3:
            break

        i = np.random.randint(0, n -2)
        j = np.random.randint(i+2, n)
        q_from = smoothed_path[i]
        q_to = smoothed_path[j]

        is_safe = is_segment_safe(robot, q_from, q_to, obstacle_ids, plane_id)

        if is_safe:
            smoothed_path = smoothed_path[:i +1] + smoothed_path[j:]

    return smoothed_path

def generate_bspline(path: List[np.ndarray], smooth_factor: float = 0.05, num_points: int = 150) -> List[np.ndarray]:
    """
    Generates a continuous B-Spline curve through the given joint space waypoints.
    """
    if len(path) == 2:
        return [path[0] + t * (path[1] - path[0]) for t in np.linspace(0, 1, num_points)]
    elif len(path) < 2:
        return path
    
    path_array = np.array(path).T  # Shape (7, num_waypoints)
    
    # Calculate spline representation
    # k=3 for cubic spline, s controls smoothness 
    tck, _ = splprep(path_array, s=smooth_factor, k=min(3, len(path)-1))
    
    # Generate fine points along spline
    u_fine = np.linspace(0, 1, num_points)
    smooth_path_array = splev(u_fine, tck)
    
    # Transpose back to a list of waypoints 
    result = np.array(smooth_path_array).T.tolist()
    return [np.array(q) for q in result]

def generate_catmull_rom(path: List[np.ndarray], alpha: float = 0.5, pts_per_seg: int = 20) -> List[np.ndarray]:
    """
    Generates a centripetal Catmull-Rom spline that mathematically passes exactly through every waypoint.
    alpha = 0.5  Centripetal 
    alpha = 0.0  Uniform
    alpha = 1.0  Chordal
    """
    if len(path) == 2:
        return [path[0] + t * (path[1] - path[0]) for t in np.linspace(0, 1, pts_per_seg)] + [path[-1]]
    elif len(path) < 2: 
        return path

    # Duplicate start and end points to securely anchor the curve
    points = [path[0]] + path + [path[-1]]
    curved_path = []
    
    for i in range(len(points) - 3):
        p0, p1, p2, p3 = points[i], points[i+1], points[i+2], points[i+3]
        
        def get_t(t: float, pt1: np.ndarray, pt2: np.ndarray) -> float:
            d = np.linalg.norm(pt2 - pt1)
            return t + d**alpha + 1e-6 # small epsilon to avoid div by 0
            
        t0 = 0.0
        t1 = get_t(t0, p0, p1)
        t2 = get_t(t1, p1, p2)
        t3 = get_t(t2, p2, p3)
        
        for t in np.linspace(t1, t2, pts_per_seg, endpoint=False):
            A1 = (t1-t)/(t1-t0)*p0 + (t-t0)/(t1-t0)*p1
            A2 = (t2-t)/(t2-t1)*p1 + (t-t1)/(t2-t1)*p2
            A3 = (t3-t)/(t3-t2)*p2 + (t-t2)/(t3-t2)*p3
            
            B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
            B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3
            
            C = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2
            curved_path.append(C)
            
    curved_path.append(path[-1]) 
    return curved_path
