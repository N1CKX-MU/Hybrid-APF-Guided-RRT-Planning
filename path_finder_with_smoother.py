import time
import numpy as np
import pybullet as p
from env.scene import Scene
from planner.hybrid import apf_rrt,apf_rrt_enhanced
from optimize.smoother import smooth_path

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
    )

    # 4. Display the Results
    # 4. Display the Results
    if result["success"]:
        original_path = result["path"]
        print(f"✅ Original Path found! Length: {len(original_path)} waypoints.")
        
        # --- RUN THE SMOOTHER ---
        print("🪚 Smoothing the path...")
        final_path = smooth_path(
            path=original_path,
            robot=scene.robot,
            obstacle_ids=scene.obstacle_ids,
            plane_id=scene.plane_id,
            max_iterations=150
        )
        print(f"📉 Path compressed from {len(original_path)} down to {len(final_path)} waypoints!")
        
        # Draw the final optimized laser path
        scene.draw_path(final_path)
        
        # Animate the robot moving along the optimized waypoints
        print("🤖 Moving robot along smoothed path...")
        # Animate the robot moving smoothly between the sparse waypoints
        print("🤖 Moving robot along smoothed path...")
        
        # Loop through each segment of the path
        for i in range(len(final_path) - 1):
            q_current = final_path[i]
            q_next    = final_path[i + 1]
            
            # Create 40 "in-between" frames for this segment
            frames = 40
            for t in np.linspace(0, 1, frames):
                # Calculate the exact middle position for this frame
                q_anim = q_current + t * (q_next - q_current)
                
                scene.robot.set_joint_angles(q_anim)
                p.stepSimulation()
                time.sleep(0.01)  # 10ms delay per frame (100 FPS)
            
        print("\n🎉 Reached Goal! Press Ctrl+C in terminal to exit.")
            
        print("\n🎉 Reached Goal! Press Ctrl+C in terminal to exit.")
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