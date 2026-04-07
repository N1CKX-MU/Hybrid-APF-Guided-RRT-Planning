import numpy as np 
import pybullet as p 
from typing import List, Optional, Tuple

PANDA_NUM_JOINTS = 7
PANDA_EE_LINK = 11

JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])

HOME_CONFIG = np.array([0.0,-0.785,0.0,-2.356,0.0,1.571,0.785])


def set_joint_angles(robot_id: int, q: np.ndarray) -> None:

    for i,angle in enumerate(q):
        p.resetJointState(robot_id,i,angle)

def clip_to_limits(q: np.ndarray) -> np.ndarray:
    return np.clip(q,JOINT_LIMITS[:,0],JOINT_LIMITS[:,1])

def sample_random_config() -> np.ndarray:
    return np.array([
        np.random.uniform(lo,hi) for lo,hi in JOINT_LIMITS
    ])    

def get_end_position(robot_id: int, q: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return end-effector world position.
    If q is provided, temporarily sets joints to q
    """
    if q is not None:
        set_joint_angles(robot_id,q)
    
    state = p.getLinkState(robot_id,PANDA_EE_LINK,computeForwardKinematics=True)
    return np.array(state[4])

def get_end_pose(robot_id: int, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray]:
    """
    Return (position,quaternion) of the end-effector
    """
    if q is not None:
        set_joint_angles(robot_id,q)

    state = p.getLinkState(robot_id,PANDA_EE_LINK,computeForwardKinematics=True)

    return np.array(state[4]), np.array(state[5])

def get_jacobian(robot_id: int, q: np.ndarray) -> np.ndarray:
    """
    Computes the 3x7 translation Jacobian at configuration q.
    Used for APF gradient projection from task space-> Joint space.
    """
    set_joint_angles(robot_id, q)
    q_list = q.tolist()
    zeroes = [0.0] * PANDA_NUM_JOINTS
    
    jac_t , _= p.calculateJacobian(
        robot_id,
        PANDA_EE_LINK,
        localPositions=[0,0,0],
        objPosition=q_list,
        objVelocities=zeroes,
        objAccelerations=zeroes
    )
    return np.array(jac_t)

def is_collision(robot_id: int, q: np.ndarray, obstacle_ids: List[int], plane_id: Optional[int] = None) -> bool:
    """
    Return True if the robot at configuration q collides with any 
    obstacle or with itself. Uses PyBullet's discrete collision detection
    """
    set_joint_angles(robot_id,q)
    p.performCollisionDetection()

    for obs_id in obstacle_ids:
        contacts = p.getContactPoints(robot_id, obs_id)
        if contacts:
            return True
    
    if plane_id is not None:
        contacts = p.getContactPoints(robot_id, plane_id)
        if contacts:
            return True
    
    contacts = p.getContactPoints(robot_id, robot_id)
    if contacts:
        return True

    return False

def collision_free_segment(q_from: np.ndarray, q_to: np.ndarray, robot_id:int,
                           obstacle_ids: List[int], plane_id: Optional[int], n_checks: int = 10,) -> bool:
    
    for t in np.linspace(0,1,n_checks):
        q_interp = (1-t)*q_from + t * q_to
        if is_collision(robot_id,q_interp,obstacle_ids,plane_id):
            return False
    return True 

def solve_ik(robot_id:int , target_pos : np.ndarray, target_orn: Optional[np.ndarray]=None, max_iters:int=100) -> Optional[np.ndarray]:
    
    """
    Uses Pybullet's IK Solver to find joint angles for the given target Pose
    """
    if target_orn is None:
        q = calculateInverseKinematics(robot_id,PANDA_EE_LINK,
                                        targetPosition=target_pos.tolist(),
                                        maxNumIterations=max_iters,)
    else: 
        q = p.calculateInverseKinematics(
            robot_id,PANDA_EE_LINK,
            targetPosition=target_pos.tolist(),
            targetOrientation=target_orn.tolist(),
            maxNumIterations=max_iters,
            lowerLimits=JOINT_LIMITS[:,0].tolist(),
            upperLimits=JOINT_LIMITS[:,1].tolist(),
            jointDamping=[0.01]*PANDA_NUM_JOINTS,
            
        )

    return clip_to_limits(np.array(q[:PANDA_NUM_JOINTS]))