import numpy as np 
import pybullet as p 
from typing import List 
from env.robot import PandaRobot

DEFAULT_K_ATT = 1.0   # attractive gain 
DEFAULT_K_REP = 0.8
DEFAULT_RHO = 0.35 # Obstacle influence radius 
DEFAULT_D_MIN = 0.01 # Minimum distance clamp to avoid division by zero 

def attractive_forces(q: np.ndarray, q_goal: np.ndarray , K_att: float = DEFAULT_K_ATT) -> np.ndarray:
    """
    Quadratic attractive potential gradient in joint space
    Simple proportional pull toward the goal configuration.
    """
    # return a 7-element vector pointing directly from q to q_goal, scaled by K_att
    return K_att * (q_goal - q)

def repulsive_force_joint_space(
    robot: PandaRobot,
    q: np.ndarray,
    obstacle_positions: List[np.ndarray],
    K_rep: float = DEFAULT_K_REP,
    rho: float = DEFAULT_RHO,
    d_min: float = DEFAULT_D_MIN
)-> np.ndarray:
    """
    Repulsive force for all obstacles, projected to joint space via J^T.
    """
    # Figure out where the end effector is 
    ee_pos = robot.get_end_position(q) # Shape (3,)

    # Get Jacobian matrix for this exact posture
    J = robot.get_jacobian(q) # Shape (3,7)

    #Calculate the Cartesian 3D push away from all nearbvy obstacles 
    F_rep_task = np.zeros(3)

    for obs_pos in obstacle_positions:
        diff = ee_pos - obs_pos
        d = float(np.linalg.norm(diff))
        d = max(d, d_min)

        #Only Calculate if obstacle is within influence radius
        if d < rho:
            # Calculate the magnitude of the repulsive force
            magnitude = float(K_rep * (1.0/d  - 1.0/rho) * (1.0/d**2))

            # Calculate the direction of the repulsive force
            direction = diff / d

            # Add the repulsive force to the total repulsive force
            # Expanding to strict assignment to avoid pyright's += confusion
            F_rep_task = F_rep_task + (magnitude * direction)

    return J.T @ F_rep_task

def get_obstacle_positions(obstacle_ids: List[int]) -> List[np.ndarray]:
    """
    Get Cartesian positions of obstacles by their PyBullet body IDs.
    """
    positions = []
    for obs_id in obstacle_ids:
        pos, _ = p.getBasePositionAndOrientation(obs_id)
        positions.append(np.array(pos))
    return positions

def compute_apf_gradient(
    robot: PandaRobot,
    q: np.ndarray,
    q_goal: np.ndarray,
    obstacle_positions: List[np.ndarray],
    K_att: float = DEFAULT_K_ATT,
    K_rep: float = DEFAULT_K_REP,
    rho: float = DEFAULT_RHO,
    d_min: float = DEFAULT_D_MIN
) -> np.ndarray:
    """
    Compute total APF gradient as sum of attractive and repulsive fields.
    """
    f_att = attractive_forces(q, q_goal, K_att)
    f_rep = repulsive_force_joint_space(robot, q, obstacle_positions, K_rep, rho, d_min)
    return f_att + f_rep
