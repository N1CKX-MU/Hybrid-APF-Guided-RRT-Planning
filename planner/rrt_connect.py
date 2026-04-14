import time
import numpy as np
from typing import List, Optional, Dict, Any

from env.robot import PandaRobot
from planner.rrt import Node, nearest, steer, path_length
from planner.apf import (
    compute_apf_gradient, get_obstacle_positions,
    DEFAULT_K_ATT, DEFAULT_K_REP, DEFAULT_RHO
)
from planner.config import (
    GOAL_BIAS, APF_BLEND, GOAL_THRESHOLD, 
    STEP_SIZE_DEFAULT, STEP_SIZE_MIN, STEP_SIZE_MAX,
    MAX_ITER_DEFAULT
)

def extract_path_connect(nodeA: Node, nodeB: Node, is_treeA_start: bool) -> List[np.ndarray]:
    """
    nodeA and nodeB are the connection points in treeA and treeB.
    is_treeA_start is True if treeA originates from q_start.
    """
    pathA = []
    curr = nodeA
    while curr is not None:
        pathA.append(curr.q.copy())
        curr = curr.parent
        
    pathB = []
    curr = nodeB
    while curr is not None:
        pathB.append(curr.q.copy())
        curr = curr.parent
        
    if is_treeA_start:
        pathA.reverse()
        return pathA + pathB
    else:
        pathB.reverse()
        return pathB + pathA

def apf_rrt_connect(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    robot: PandaRobot,
    obstacle_ids: List[int],
    plane_id: int,
    max_iter: int = MAX_ITER_DEFAULT,
    step_size: float = STEP_SIZE_DEFAULT,
    goal_threshold: float = GOAL_THRESHOLD,
    apf_blend: float = APF_BLEND,
) -> Dict[str, Any]:
    """
    Bi-directional APF-Guided RRT-Connect.
    Grows a tree from Start to Goal, and another from Goal to Start!
    """
    t0 = time.perf_counter()
    treeA: List[Node] = [Node(q_start)]
    treeB: List[Node] = [Node(q_goal)]
    is_treeA_start = True # Tracks which tree is connected to the start
    
    obs_positions = get_obstacle_positions(obstacle_ids)
    
    # Adaptive counters 
    stuck_counterA = 0
    stuck_counterB = 0
    
    for iteration in range(max_iter):
        # 1. Sample Random State 
        q_rand = robot.sample_random_config()
        
        # 2. Extract active parameters 
        current_apf_blend = apf_blend
        if (is_treeA_start and stuck_counterA > 15) or (not is_treeA_start and stuck_counterB > 15):
            current_apf_blend = 1.0 # Pure random escape out of APF local minima
            
        # 3. Extend treeA
        near_nodeA = nearest(treeA, q_rand)
        q_nearA = near_nodeA.q
        
        # APF gradient direction. 
        # If treeA is start tree, it pulls towards q_goal. Else towards q_start
        target_q = q_goal if is_treeA_start else q_start
        grad = compute_apf_gradient(robot, q_nearA, target_q, obs_positions, DEFAULT_K_ATT, DEFAULT_K_REP, DEFAULT_RHO)
        
        rand_dir = q_rand - q_nearA
        rand_norm = np.linalg.norm(rand_dir)
        grad_norm = np.linalg.norm(grad)

        if rand_norm > 1e-9: rand_dir /= rand_norm
        if grad_norm > 1e-9: grad_dir = grad / grad_norm
        else: grad_dir = rand_dir

        direction = current_apf_blend * rand_dir + (1.0 - current_apf_blend) * grad_dir
        dir_norm  = np.linalg.norm(direction)
        if dir_norm > 1e-9: direction /= dir_norm
        
        q_newA = robot.clip_to_limits(q_nearA + step_size * direction)
        
        if robot.is_collision(q_newA, obstacle_ids, plane_id):
            near_nodeA.failures += 1
            if is_treeA_start: stuck_counterA += 1
            else: stuck_counterB += 1
        else:
            if is_treeA_start: stuck_counterA = 0
            else: stuck_counterB = 0
            
            cost = near_nodeA.cost + np.linalg.norm(q_newA - q_nearA)
            new_nodeA = Node(q_newA, parent=near_nodeA, cost=cost)
            treeA.append(new_nodeA)
            
            # 4. Attempt direct connection (Greedy Connect)
            # RRT-Connect typically attempts an iterative stepping direct to the node
            curr_nodeB = nearest(treeB, q_newA, penalty_weight=0.0)
            
            while True:
                q_nearB = curr_nodeB.q
                dist_to_A = np.linalg.norm(q_newA - q_nearB)
                
                if dist_to_A < goal_threshold:
                    # Trees connected!
                    path = extract_path_connect(new_nodeA, curr_nodeB, is_treeA_start)
                    elapsed = time.perf_counter() - t0
                    return {
                        "path": path,
                        "success": True,
                        "node_count": len(treeA) + len(treeB),
                        "iterations": iteration + 1,
                        "time_s": elapsed,
                        "path_length": path_length(path)
                    }
                
                # Steer B towards A structurally
                q_stepB = steer(q_nearB, q_newA, step_size)
                q_stepB = robot.clip_to_limits(q_stepB)
                
                if robot.is_collision(q_stepB, obstacle_ids, plane_id):
                    # Connection blocked
                    curr_nodeB.failures += 1
                    break 
                
                # Valid step, add to tree B and continue marching towards A
                costB = curr_nodeB.cost + np.linalg.norm(q_stepB - q_nearB)
                new_nodeB = Node(q_stepB, parent=curr_nodeB, cost=costB)
                treeB.append(new_nodeB)
                curr_nodeB = new_nodeB
                
        # 5. Swap Trees Roles
        treeA, treeB = treeB, treeA
        is_treeA_start = not is_treeA_start
        
    elapsed = time.perf_counter() - t0
    return {
        "path": None,
        "success": False,
        "node_count": len(treeA) + len(treeB),
        "iterations": max_iter,
        "time_s": elapsed,
        "path_length": 0.0
    }
