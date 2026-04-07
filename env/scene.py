import numpy as np 
import pybullet as p 
import pybullet_data
from typing import List, Tuple, Optional, Dict, Any
from env.robot import PandaRobot

# ─── Obstacle layout (Mix of spheres and cuboids) ────────
DEFAULT_OBSTACLES = [
    {"type": "sphere", "pos": (0.35, -0.20, 0.55), "radius": 0.07},
    {"type": "box",    "pos": (0.45,  0.15, 0.50), "halfExtents": (0.05, 0.05, 0.08)},
    {"type": "sphere", "pos": (0.30,  0.25, 0.65), "radius": 0.08},
    {"type": "box",    "pos": (0.55, -0.05, 0.60), "halfExtents": (0.06, 0.04, 0.06)},
    {"type": "sphere", "pos": (0.40,  0.00, 0.40), "radius": 0.06},
    {"type": "box",    "pos": (0.25, -0.30, 0.70), "halfExtents": (0.10, 0.02, 0.05)},
    {"type": "sphere", "pos": (0.60,  0.20, 0.45), "radius": 0.07},
    {"type": "box",    "pos": (0.35,  0.35, 0.55), "halfExtents": (0.04, 0.08, 0.04)},
]

# ─── Upgraded Obstacle Layout (Pillars & Canopy) ────────
DEFAULT_OBSTACLES_2 = [
    # --- Ground-level obstacles (Pillars resting on the floor) ---
    # Z-position is exactly equal to the Z-halfExtent so it touches the ground
    {"type": "box",    "pos": (0.4, -0.3, 0.15), "halfExtents": (0.05, 0.15, 0.15)}, 
    {"type": "box",    "pos": (0.4,  0.3, 0.20), "halfExtents": (0.10, 0.05, 0.20)}, 

    # --- The "Narrow Passage" Archway ---
    {"type": "sphere", "pos": (0.4, -0.15, 0.5), "radius": 0.07}, # Left pillar of the arch
    {"type": "sphere", "pos": (0.4,  0.15, 0.5), "radius": 0.07}, # Right pillar of the arch
    {"type": "box",    "pos": (0.4,  0.00, 0.6), "halfExtents": (0.02, 0.2, 0.02)}, # Top bar of the arch

    # --- Flanking Obstacles (Surrounding the arm) ---
    {"type": "sphere", "pos": (0.0,  0.45, 0.3), "radius": 0.1},  # Left side
    {"type": "sphere", "pos": (0.0, -0.45, 0.3), "radius": 0.1},  # Right side
    
    # --- Deep Canopy (Near the goal) ---
    {"type": "sphere", "pos": (0.6,  0.2, 0.45), "radius": 0.06},
    {"type": "box",    "pos": (0.6, -0.1, 0.30), "halfExtents": (0.05, 0.05, 0.05)},
]
class Scene:
    def __init__(self, gui: bool = True):
        """
        Initializes the PyBullet physics simulation and sets up the basic scene.
        """
        self.gui = gui
        mode = p.GUI if gui else p.DIRECT
        self.client_id = p.connect(mode)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        # We step the simulation manually to check for collisions
        p.setRealTimeSimulation(0)
        
        self.plane_id = self._load_plane()
        self.robot = self._load_robot()
        self.obstacle_ids: List[int] = []
        
        # Automatically spawn obstacles when building the scene
        self.spawn_obstacles()

    def _load_plane(self) -> int:
        """ Load the ground plane """
        return p.loadURDF("plane.urdf")

    def _load_robot(self) -> PandaRobot:
        """ Load the panda robot """
        robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0.0], useFixedBase=True)
        return PandaRobot(robot_id)

    def spawn_obstacles(self, obstacle_defs: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """ Spawn obstacles in the scene """
        if obstacle_defs is None:
            obstacle_defs = DEFAULT_OBSTACLES_2

        color = [0.85, 0.25, 0.15, 0.85] # Reddish colour

        for obs in obstacle_defs:
            pos = obs["pos"]
            if obs["type"] == "sphere":
                col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=obs["radius"])
                vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=obs["radius"], rgbaColor=color)
            elif obs["type"] == "box":
                col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obs["halfExtents"])
                vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=obs["halfExtents"], rgbaColor=color)
            else:
                continue

            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=pos
            )
            self.obstacle_ids.append(obs_id)
            
        return self.obstacle_ids

    def draw_path(self, path: List[np.ndarray]) -> None:
        """
        Draw the planned path as green lines in the simulation
        """
        if len(path) < 2:
            return
        
        positions = [self.robot.get_end_position(q) for q in path]

        for i in range(len(positions) - 1):
            p.addUserDebugLine(
                positions[i].tolist(),
                positions[i+1].tolist(),
                lineColorRGB=[0, 1, 0],
                lineWidth=2,
                lifeTime=0,
            )

    def spawn_marker(self, position: np.ndarray, color: List[float]) -> int:
        """Utility to spawn a visual-only sphere that the robot can pass through."""
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=color)

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_id,
            basePosition=position.tolist()
        )

    def disconnect(self) -> None:
        """ Disconnect from the simulation """
        p.disconnect(self.client_id)
