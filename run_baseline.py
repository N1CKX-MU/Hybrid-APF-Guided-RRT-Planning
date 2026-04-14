import time
import numpy as np
import pybullet as p
from env.scene import Scene
from planner.hybrid import apf_rrt,apf_rrt_enhanced
from optimize.smoother import smooth_path, generate_catmull_rom

def main():
    print("Booting Simulation...")
    scene = Scene(gui=True)

    # 1. Define the Start Configuration
    q_start = scene.robot.HOME_CONFIG
    scene.robot.set_joint_angles(q_start)
    
    # 2. Define the Goal (Reaching through the archway!)
    target_xyz = np.array([0.35, 0.0, 0.45]) 
    scene.spawn_marker(target_xyz, color=[0.1, 0.9, 0.1, 1.0]) # Green target sphere
    
    print("Searching for a safe Inverse Kinematics solution...")
    safe_q_goal = None
    
    # Try 50 different times to find a safe posture
    for i in range(50):
        # Give PyBullet a random starting seed to force a different IK solution
        scene.robot.set_joint_angles(scene.robot.sample_random_config())
        
        # Ask IK to solve for the target from this new twisted posture
        q_guess = scene.robot.solve_ik(target_pos=target_xyz)
        
        # Check if the elbow/arm is safely avoiding the walls
        if not scene.robot.is_collision(q_guess, scene.obstacle_ids, scene.plane_id):
            safe_q_goal = q_guess
            print(f"✅ Found safe goal posture on attempt {i+1}!")
            break

    # Reset the robot back to the start!
    scene.robot.set_joint_angles(q_start)

    if safe_q_goal is None:
        print("Error: Could not find a safe IK solution. Target blocked.")
        return # Abort the script
        
    q_goal = safe_q_goal

    if scene.robot.is_collision(q_goal, scene.obstacle_ids, scene.plane_id):
        print("Warning: IK solver output in collision.")
    else:
        print("Goal configuration is safe.")

    # 3. Run the Baseline Planner
    print("Running APF-RRT Baseline Planner...")
    result = apf_rrt(
        q_start=q_start,
        q_goal=q_goal,
        robot=scene.robot,
        obstacle_ids=scene.obstacle_ids,
        plane_id=scene.plane_id,
    )

    # 4. Display the Results
    if result["success"]:
        original_path = result["path"]
        print(f"Path found! Length: {result['path_length']:.2f} rads | Nodes Explored: {result['node_count']}")
        
        print("Smoothing path...")
        smoothed_path = smooth_path(
            path=original_path,
            robot=scene.robot,
            obstacle_ids=scene.obstacle_ids,
            plane_id=scene.plane_id,
            max_iterations=150
        )
        print(f"Path compressed to {len(smoothed_path)} waypoints.")
        
        print("Applying Catmull-Rom spline...")
        final_path = generate_catmull_rom(smoothed_path, pts_per_seg=20)
        print(f"Spline generated with {len(final_path)} frames.")

        scene.draw_path(final_path)
        
        print("Executing path...")
        for q_anim in final_path:
            scene.robot.set_joint_angles(q_anim)
            p.stepSimulation()
            time.sleep(0.05) 
        
        print("\nTarget reached. Press Ctrl+C to exit.")
    else:
        print("Planner failed to find a path within max iterations.")

    # Keep the window open so you can look around
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nShutting down...")
        scene.disconnect()

if __name__ == "__main__":
    main()