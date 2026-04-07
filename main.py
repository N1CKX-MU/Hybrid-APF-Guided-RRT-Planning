import time
import numpy as np
import pybullet as p
from env.scene import Scene
from planner.hybrid import apf_rrt,apf_rrt_enhanced

def main():
    print("🌍 Booting Simulation...")
    scene = Scene(gui=True)

    # 1. Define the Start Configuration
    q_start = scene.robot.HOME_CONFIG
    scene.robot.set_joint_angles(q_start)
    
    # 2. Define the Goal (Reaching through the archway!)
    target_xyz = np.array([0.35, 0.0, 0.45]) 
    scene.spawn_marker(target_xyz, color=[0.1, 0.9, 0.1, 1.0]) # Green target sphere
    
    print("🎯 Searching for a safe Inverse Kinematics solution...")
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
        print("❌ Could not find a safe IK solution. The target is completely blocked!")
        return # Abort the script
        
    q_goal = safe_q_goal

    if scene.robot.is_collision(q_goal, scene.obstacle_ids, scene.plane_id):
        print("🚨 WARNING: The IK solver twisted the arm into a wall! The goal is impossible!")
    else:
        print("✅ Goal configuration is safe.")

    # 3. Run the Brain
    print("🧠 Running APF-RRT Planner... (Give it a few seconds to think)")
    result = apf_rrt_enhanced(
        q_start=q_start,
        q_goal=q_goal,
        robot=scene.robot,
        obstacle_ids=scene.obstacle_ids,
        plane_id=scene.plane_id,
        step_size_max=0.15,
        step_size_min = 0.02,
        apf_blend=0.60,
        goal_bias=0.09,
    )

    # 4. Display the Results
    if result["success"]:
        print(f"✅ Path found! Length: {result['path_length']:.2f} rads | Nodes Explored: {result['node_count']}")
        path = result["path"]
        
        # Draw the green laser path in the air
        scene.draw_path(path)
        
        # Animate the robot actually moving along the waypoints
        print("🤖 Moving robot along path...")
        for q in path:
            scene.robot.set_joint_angles(q)
            p.stepSimulation()
            time.sleep(0.05)  # Slow down the animation so you can watch it dodge
        
        print("\n🎉 Reached Goal! Press Ctrl+C in terminal to exit.")
    else:
        print("❌ Planner failed to find a path within max iterations.")

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