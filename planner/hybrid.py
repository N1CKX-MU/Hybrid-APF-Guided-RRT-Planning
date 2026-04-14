import time 
import numpy as np 
from typing import List , Optional , Dict, Any 

from env.robot import PandaRobot
from planner.rrt import Node, nearest, steer, extract_path, path_length
from planner.apf import (
    compute_apf_gradient, get_obstacle_positions,
    DEFAULT_K_ATT, DEFAULT_K_REP, DEFAULT_RHO
)
from planner.config import (
    GOAL_BIAS, APF_BLEND, GOAL_THRESHOLD, 
    STEP_SIZE_DEFAULT, STEP_SIZE_MIN, STEP_SIZE_MAX,
    MAX_ITER_DEFAULT
)


def _make_result(path: Optional[List[np.ndarray]], tree: List[Node], elapsed: float, iterations: int) -> Dict[str,Any]:
    """ Helper to cleanly package the results for our benchmark table later."""
    return {
        "path": path,
        "success": path is not None,
        "node_count": len(tree),
        "iterations": iterations,
        "time_s": elapsed,
        "path_length": path_length(path) if path else None,
    }

def apf_rrt(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    robot: PandaRobot,
    obstacle_ids: List[int],
    plane_id: int,
    max_iter: int = MAX_ITER_DEFAULT,
    step_size: float = STEP_SIZE_DEFAULT,
    goal_threshold: float = GOAL_THRESHOLD,
    goal_bias: float = GOAL_BIAS,
    apf_blend: float = APF_BLEND,
) -> Dict[str, Any]:
        """
    Baseline Hybrid APF-RRT planner (Phase A)
    """
    t0 = time.perf_counter()
    tree: List[Node] = [Node(q_start)]

    obs_positions = get_obstacle_positions(obstacle_ids)

    stuck_counter = 0

    for iteration in range(max_iter):
        # Default parameters
        current_goal_bias = goal_bias
        current_apf_blend = apf_blend

        # Escaping local minima 
        if stuck_counter > 20: 
            # Temporarily turn off APF entirely
            current_goal_bias = 0.0
            current_apf_blend = 1.0
            
        elif iteration > 5000:
            # Lean heavily towards the goal if we are stuck exploring for too long but aren't in a pocket
            current_goal_bias = max(goal_bias, 0.6)
            current_apf_blend = min(apf_blend, 0.3)

        # 1. Goal-Based Sampling
        if np.random.random() < current_goal_bias:
            q_rand = q_goal.copy()
        else: 
            q_rand = robot.sample_random_config()
    
        # 2. Find Nearest Branch 
        near_node = nearest(tree, q_rand)
        q_near    = near_node.q 

        # 3. APF Gradient Check
        # Compute gradient direction to avoid obstacles and progress towards goal
        grad = compute_apf_gradient(
            robot, q_near, q_goal, obs_positions, DEFAULT_K_ATT, DEFAULT_K_REP, DEFAULT_RHO
        )
        # 4. Blend the Directions
        rand_dir = q_rand - q_near
        rand_norm = np.linalg.norm(rand_dir)
        grad_norm = np.linalg.norm(grad)

        # Normalize both vectors so they are a scale of 1.0
        if rand_norm > 1e-9: rand_dir = rand_dir / rand_norm
        if grad_norm > 1e-9: grad_dir = grad / grad_norm
        else: grad_dir = rand_dir

        # Combine them using blend weight 
        direction = current_apf_blend * rand_dir + (1.0 - current_apf_blend) * grad_dir

        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-9: direction = direction / dir_norm

        # 5. Take the Step 
        q_new = robot.clip_to_limits(q_near + step_size * direction)

        # 6. Check for collision
        if robot.is_collision(q_new, obstacle_ids, plane_id):
            near_node.failures += 1
            stuck_counter += 1
            continue

        stuck_counter = 0 # Reset exploration counter
        # 7. Grow the tree
        cost = near_node.cost + np.linalg.norm(q_new - q_near)
        new_node = Node(q_new, parent=near_node, cost=cost)
        tree.append(new_node)

        # 8. Check if we did it 
        if np.linalg.norm(q_new - q_goal) < goal_threshold:
            # Connect the final dot directly to the goal 
            if not robot.is_collision(q_goal, obstacle_ids, plane_id):
                goal_node = Node(q_goal, parent=new_node, cost=cost + np.linalg.norm(q_goal - q_new))
                path = extract_path(goal_node)
            else: 
                path = extract_path(new_node)

            elapsed = time.perf_counter() - t0
            return _make_result(path, tree, elapsed, iteration + 1) 

    # If we run out of iterations, return a failure
    elapsed = time.perf_counter() - t0
    return _make_result(None, tree, elapsed, max_iter)
                

def apf_rrt_enhanced(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    robot: PandaRobot,
    obstacle_ids: List[int],
    plane_id: int,
    max_iter: int = MAX_ITER_DEFAULT,
    step_size_min: float = STEP_SIZE_MIN,
    step_size_max: float = STEP_SIZE_MAX,
    goal_threshold: float = GOAL_THRESHOLD,
    goal_bias: float = GOAL_BIAS,
    apf_blend: float = APF_BLEND,
) -> Dict[str, Any]:
    """
    Enhanced APF-RRT (Phase B).
    Uses Adaptive Step Size based on local obstacle density to sneak through narrow passages.
    """
    t0 = time.perf_counter()
    tree: List[Node] = [Node(q_start)]
    obs_positions = get_obstacle_positions(obstacle_ids)
    
    # How sensitive the robot is to cramped spaces
    lambda_density = 3.0   
    stuck_counter = 0

    for iteration in range(max_iter):
        # Default parameters
        current_goal_bias = goal_bias
        current_apf_blend = apf_blend

        # Escaping local minima (Stuck pocket)
        if stuck_counter > 20: 
            current_goal_bias = 0.0
            current_apf_blend = 1.0
            
        elif iteration > 5000:
            current_goal_bias = max(goal_bias, 0.6)
            current_apf_blend = min(apf_blend, 0.3)

        if np.random.random() < current_goal_bias:
            q_rand = q_goal.copy()
        else:
            q_rand = robot.sample_random_config()

        near_node = nearest(tree, q_rand)
        q_near    = near_node.q

        grad = compute_apf_gradient(robot, q_near, q_goal, obs_positions, DEFAULT_K_ATT, DEFAULT_K_REP, DEFAULT_RHO)

        #  Adaptive Step Size Logic 
        ee_pos = robot.get_end_position(q_near)
        density = sum(
            max(0.0, 1.0 - np.linalg.norm(ee_pos - obs_p) / DEFAULT_RHO)
            for obs_p in obs_positions
        )
        
        # Scale down delta incrementally based on proximity threshold sum
        step_size = step_size_max / (1.0 + lambda_density * density)
        step_size = max(step_size_min, step_size)

        rand_dir  = q_rand - q_near
        rand_norm = np.linalg.norm(rand_dir)
        grad_norm = np.linalg.norm(grad)

        if rand_norm > 1e-9: rand_dir = rand_dir / rand_norm
        if grad_norm > 1e-9: grad_dir = grad / grad_norm
        else: grad_dir = rand_dir

        direction = current_apf_blend * rand_dir + (1.0 - current_apf_blend) * grad_dir
        dir_norm  = np.linalg.norm(direction)
        if dir_norm > 1e-9: direction /= dir_norm

        q_new = robot.clip_to_limits(q_near + step_size * direction)

        if robot.is_collision(q_new, obstacle_ids, plane_id):
            near_node.failures += 1
            stuck_counter += 1
            continue

        stuck_counter = 0 # Reset structural counter
        cost = near_node.cost + np.linalg.norm(q_new - q_near)
        new_node = Node(q_new, parent=near_node, cost=cost)
        tree.append(new_node)

        if np.linalg.norm(q_new - q_goal) < goal_threshold:
            if not robot.is_collision(q_goal, obstacle_ids, plane_id):
                goal_node = Node(q_goal, parent=new_node, cost=new_node.cost + np.linalg.norm(q_goal - q_new))
                tree.append(goal_node)
                path = extract_path(goal_node)
            else:
                path = extract_path(new_node)
            return _make_result(path, tree, time.perf_counter() - t0, iteration + 1)

    return _make_result(None, tree, time.perf_counter() - t0, max_iter)



