import time 
import pybullet as p 
from env.scene import Scene


def test_environment(): 
    # Instantiate your new Scene class!
    scene = Scene(gui=True)
    
    print("✅ Physics Engine Connected")
    print("✅ Ground Plane Loaded")
    print(f"✅ Robot Loaded (ID: {scene.robot.robot_id})")
    print(f"✅ Spawned {len(scene.obstacle_ids)} Obstacles (Spheres & Cuboids)")
    
    print("\nThe simulation is live! Click and drag in the window to rotate the camera.")
    print("Press Ctrl+C in this terminal to close it.")
    
    try:
        # Keep the window open so you can inspect your world
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nShutting down simulation...")
        scene.disconnect()

if __name__ == "__main__":
    test_environment()