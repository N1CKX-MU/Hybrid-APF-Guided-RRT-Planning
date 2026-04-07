import numpy as np 
import pybullet as p 
from typing import List, Optional, Tuple

class PandaRobot:
    NUM_JOINTS = 7
    EE_LINK = 11

    JOINT_LIMITS = np.array([
        [-2.8973,  2.8973],
        [-1.7628,  1.7628],
        [-2.8973,  2.8973],
        [-3.0718, -0.0698],
        [-2.8973,  2.8973],
        [-0.0175,  3.7525],
        [-2.8973,  2.8973],
    ])

    HOME_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def __init__(self, robot_id: int):
        self.robot_id = robot_id

    def set_joint_angles(self, q: np.ndarray) -> None:
        for i, angle in enumerate(q):
            p.resetJointState(self.robot_id, i, angle)

    def clip_to_limits(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1])

    def sample_random_config(self) -> np.ndarray:
        return np.array([
            np.random.uniform(lo, hi) for lo, hi in self.JOINT_LIMITS
        ])

    def get_end_position(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return end-effector world position.
        If q is provided, temporarily sets joints to q
        """
        if q is not None:
            self.set_joint_angles(q)
        
        state = p.getLinkState(self.robot_id, self.EE_LINK, computeForwardKinematics=True)
        return np.array(state[4])

    def get_end_pose(self, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (position, quaternion) of the end-effector
        """
        if q is not None:
            self.set_joint_angles(q)

        state = p.getLinkState(self.robot_id, self.EE_LINK, computeForwardKinematics=True)

        return np.array(state[4]), np.array(state[5])

    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Computes the 3x7 translation Jacobian at configuration q.
        Used for APF gradient projection from task space-> Joint space.
        """
        self.set_joint_angles(q)
        
        # calculateJacobian requires arrays to match all non-fixed joints
        num_joints_total = p.getNumJoints(self.robot_id)
        mvmnt_joints = [i for i in range(num_joints_total) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        
        q_full = [p.getJointState(self.robot_id, i)[0] for i in mvmnt_joints]
        zeroes = [0.0] * len(mvmnt_joints)
        
        jac_t, _ = p.calculateJacobian(
            self.robot_id,
            self.EE_LINK,
            [0.0, 0.0, 0.0],
            q_full,
            zeroes,
            zeroes
        )
        return np.array(jac_t)[:, :self.NUM_JOINTS]

    def is_collision(self, q: np.ndarray, obstacle_ids: List[int], plane_id: Optional[int] = None) -> bool:
        """
        Return True if the robot at configuration q collides with any 
        obstacle or with itself. Uses PyBullet's discrete collision detection
        """
        self.set_joint_angles(q)
        p.performCollisionDetection()

        for obs_id in obstacle_ids:
            contacts = p.getContactPoints(self.robot_id, obs_id)
            if contacts:
                return True
        
        if plane_id is not None:
            contacts = p.getContactPoints(self.robot_id, plane_id)
            if contacts:
                return True
        
        contacts = p.getContactPoints(self.robot_id, self.robot_id)
        if contacts:
            return True

        return False

    def collision_free_segment(self, q_from: np.ndarray, q_to: np.ndarray,
                               obstacle_ids: List[int], plane_id: Optional[int], n_checks: int = 10) -> bool:
        
        for t in np.linspace(0, 1, n_checks):
            q_interp = (1 - t) * q_from + t * q_to
            if self.is_collision(q_interp, obstacle_ids, plane_id):
                return False
        return True 

    def solve_ik(self, target_pos: np.ndarray, target_orn: Optional[np.ndarray] = None, max_iters: int = 100) -> Optional[np.ndarray]:
        """
        Uses Pybullet's IK Solver to find joint angles for the given target Pose
        """
        if target_orn is None:
            q = p.calculateInverseKinematics(
                self.robot_id,
                self.EE_LINK,
                targetPosition=target_pos.tolist(),
                maxNumIterations=max_iters,
            )
        else: 
            q = p.calculateInverseKinematics(
                self.robot_id,
                self.EE_LINK,
                targetPosition=target_pos.tolist(),
                targetOrientation=target_orn.tolist(),
                maxNumIterations=max_iters,
                lowerLimits=self.JOINT_LIMITS[:, 0].tolist(),
                upperLimits=self.JOINT_LIMITS[:, 1].tolist(),
                jointDamping=[0.01] * self.NUM_JOINTS,
            )

        return self.clip_to_limits(np.array(q[:self.NUM_JOINTS]))