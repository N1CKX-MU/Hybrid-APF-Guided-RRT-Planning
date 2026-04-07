import numpy as np 
from typing import List
import copy 
from env.robot import PandaRobot

def is_segment_safe(
    robot: PandaRobot,
    q1: np.ndarray,
    q2: np.ndarray,
    obstacle_ids: List[int],
    plane_id:int,
    steps:int = 10
) -> bool:
    """
    Interpolates along a straight line between q1 and q1 in joint space, returns False if any point along 
    line hits the obstacle
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
    Greedy path smoother (Phase B).
    Randomly selects 2 nodes in the path and attempts to connect them directly. 
    If safe, intermediate nodeas are deleted.
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
    
