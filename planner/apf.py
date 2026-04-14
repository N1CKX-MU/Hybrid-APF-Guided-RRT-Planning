import numpy as np 
import pybullet as p 
from typing import List 
from env.robot import PandaRobot

from planner.config import K_ATT, K_REP, RHO, D_MIN

DEFAULT_K_ATT = K_ATT
DEFAULT_K_REP = K_REP
DEFAULT_RHO = RHO
DEFAULT_D_MIN = D_MIN

# The links you want to protect from obstacles
CONTROL_LINKS = [4, 7, 11]  # elbow, wrist, end-effector

# Optional: weight each link differently
LINK_WEIGHTS = {4: 0.5, 7: 0.8, 11: 1.0}

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
    Calculates forces on multiple links to protect the whole arm.
    """
    robot.set_joint_angles(q)

    total_f_rep_q = np.zeros(robot.NUM_JOINTS)

    for link_idx in CONTROL_LINKS:
        link_pos = robot.get_link_position(link_idx, q)
        J = robot.get_link_jacobian(q, link_idx)
        w = LINK_WEIGHTS.get(link_idx, 1.0)

        F_rep_task = np.zeros(3)
        for obs_pos in obstacle_positions:
            diff = link_pos - obs_pos
            d = max(float(np.linalg.norm(diff)), d_min)

            #Only Calculate if obstacle is within influence radius
            if d < rho:
                # Calculate the magnitude of the repulsive force
                magnitude = K_rep * (1.0/d  - 1.0/rho) * (1.0/d**2)

                # Calculate the direction of the repulsive force
                direction = diff / d

                # Add the repulsive force for this obstacle
                F_rep_task = F_rep_task + (magnitude * direction)

        # Project this link's Cartesian force into joint space
        total_f_rep_q = total_f_rep_q + w * (J.T @ F_rep_task)

    return total_f_rep_q

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
