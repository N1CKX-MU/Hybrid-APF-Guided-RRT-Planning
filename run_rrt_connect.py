import time
import numpy as np
import pybullet as p
from env.scene import Scene
from planner.rrt_connect import apf_rrt_connect
from optimize.smoother import smooth_path, generate_catmull_rom

def main():
    print("Initializing Simulation...")
    scene = Scene(gui=True)

    # 1. Define the Start Configuration
    q_start = scene.robot.HOME_CONFIG
    scene.robot.set_joint_angles(q_start)
    
    # 2. Define the Goal 
    target_xyz = np.array([0.35, 0.0, 0.45]) 
    scene.spawn_marker(target_xyz, color=[0.1, 0.9, 0.1, 1.0]) # Green target sphere
    
    print("🎯 Searching for a safe Inverse Kinematics solution...")
    safe_q_goal = None
    
    for i in range(50):
        # Give PyBullet a random starting seed to force a different IK solution
        scene.robot.set_joint_angles(scene.robot.sample_random_config())
        q_guess = scene.robot.solve_ik(target_pos=target_xyz)
        
        if not scene.robot.is_collision(q_guess, scene.obstacle_ids, scene.plane_id):
            safe_q_goal = q_guess
            print(f"Goal posture found on attempt {i+1}.")
            break

    scene.robot.set_joint_angles(q_start)

    if safe_q_goal is None:
        print("Error: Could not find a safe IK solution. Target is blocked.")
        return 
        
    q_goal = safe_q_goal

    if scene.robot.is_collision(q_goal, scene.obstacle_ids, scene.plane_id):
        print("Warning: IK solver produced a state in collision.")
    else:
        print("Goal configuration verified safe.")

    # 3. Run Planner
    print("Running APF-Guided Bi-Directional RRT-Connect Planner...")
    result = apf_rrt_connect(
        q_start=q_start,
        q_goal=q_goal,
        robot=scene.robot,
        obstacle_ids=scene.obstacle_ids,
        plane_id=scene.plane_id,
    )

    # 4. Display Results
    if result["success"]:
        original_path = result["path"]
        print(f"Path found. Length: {len(original_path)} waypoints. Evaluated {result['node_count']} nodes.")
        
        # --- RUN THE SMOOTHER ---
        print("Smoothing path...")
        smoothed_path = smooth_path(
            path=original_path,
            robot=scene.robot,
            obstacle_ids=scene.obstacle_ids,
            plane_id=scene.plane_id,
            max_iterations=150
        )
        print(f"Path compressed from {len(original_path)} to {len(smoothed_path)} waypoints.")
        
        # --- RUN THE SPLINE ENGINE ---
        print("Applying Catmull-Rom spline interpolation...")
        final_path = generate_catmull_rom(smoothed_path, pts_per_seg=20)
        print(f"Final path generated with {len(final_path)} frames.")

        scene.draw_path(final_path)
        
        print("Executing trajectory...")
        
        for q_anim in final_path:
            scene.robot.set_joint_angles(q_anim)
            p.stepSimulation()
            time.sleep(0.02)
            
        print("\nTarget reached. Press Ctrl+C in terminal to exit.")
    else:
        print("Planner failed to find a path within max iterations.")
            
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nShutting down...")
        scene.disconnect()

if __name__ == "__main__":
    main()
