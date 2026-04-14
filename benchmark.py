import numpy as np
import time
from env.scene import Scene
from planner.hybrid import apf_rrt, apf_rrt_enhanced
from planner.config import MAX_ITER_BENCHMARK

def run_benchmark(num_trials: int = 10):
    print("🌍 Booting Headless Simulation (No GUI)...")
    # Setting gui=False makes PyBullet run mathematically in the background at lightning speed
    scene = Scene(gui=False)
    
    q_start = scene.robot.HOME_CONFIG
    target_xyz = np.array([0.35, 0.0, 0.45])
    
    print("🎯 Pre-calculating safe goal configuration...")
    safe_q_goal = None
    for _ in range(50):
        scene.robot.set_joint_angles(scene.robot.sample_random_config())
        q_guess = scene.robot.solve_ik(target_pos=target_xyz)
        if not scene.robot.is_collision(q_guess, scene.obstacle_ids, scene.plane_id):
            safe_q_goal = q_guess
            break
            
    scene.robot.set_joint_angles(q_start)
    
    if safe_q_goal is None:
        print("❌ Could not find a safe IK solution. Aborting benchmark.")
        scene.disconnect()
        return
        
    q_goal = safe_q_goal

    # ─── Tracking Dictionaries ───
    metrics = {
        "Baseline": {"successes": 0, "times": [], "nodes": [], "lengths": []},
        "Enhanced": {"successes": 0, "times": [], "nodes": [], "lengths": []}
    }

    print(f"\n🚀 Running {num_trials} trials for each algorithm...\n")

    for i in range(num_trials):
        print(f"Running Trial {i+1}/{num_trials}...")
        
        # --- 1. Run Baseline ---
        res_base = apf_rrt(
            q_start=q_start, q_goal=q_goal, robot=scene.robot,
            obstacle_ids=scene.obstacle_ids, plane_id=scene.plane_id,
            max_iter=MAX_ITER_BENCHMARK
        )
        if res_base["success"]:
            metrics["Baseline"]["successes"] += 1
            metrics["Baseline"]["times"].append(res_base["time_s"])
            metrics["Baseline"]["nodes"].append(res_base["node_count"])
            metrics["Baseline"]["lengths"].append(res_base["path_length"])

        # --- 2. Run Enhanced ---
        res_enh = apf_rrt_enhanced(
            q_start=q_start, q_goal=q_goal, robot=scene.robot,
            obstacle_ids=scene.obstacle_ids, plane_id=scene.plane_id,
            max_iter=MAX_ITER_BENCHMARK
        )
        if res_enh["success"]:
            metrics["Enhanced"]["successes"] += 1
            metrics["Enhanced"]["times"].append(res_enh["time_s"])
            metrics["Enhanced"]["nodes"].append(res_enh["node_count"])
            metrics["Enhanced"]["lengths"].append(res_enh["path_length"])

    scene.disconnect()

    # ─── Print the Report ───
    print("\n" + "="*60)
    print(f"{'APF-RRT PERFORMANCE BENCHMARK':^60}")
    print(f"{'(Averaged over '+str(num_trials)+' trials)':^60}")
    print("="*60)
    print(f"{'Metric':<20} | {'Phase A (Baseline)':<15} | {'Phase B (Enhanced)':<15}")
    print("-" * 60)

    for algo in ["Baseline", "Enhanced"]:
        m = metrics[algo]
        if m["successes"] == 0:
            m["times"] = [0]
            m["nodes"] = [0]
            m["lengths"] = [0]

    # Calculate averages safely
    b_succ = (metrics["Baseline"]["successes"] / num_trials) * 100
    e_succ = (metrics["Enhanced"]["successes"] / num_trials) * 100
    
    b_time = np.mean(metrics["Baseline"]["times"]) if metrics["Baseline"]["times"] else 0
    e_time = np.mean(metrics["Enhanced"]["times"]) if metrics["Enhanced"]["times"] else 0
    
    b_nodes = np.mean(metrics["Baseline"]["nodes"]) if metrics["Baseline"]["nodes"] else 0
    e_nodes = np.mean(metrics["Enhanced"]["nodes"]) if metrics["Enhanced"]["nodes"] else 0
    
    b_len = np.mean(metrics["Baseline"]["lengths"]) if metrics["Baseline"]["lengths"] else 0
    e_len = np.mean(metrics["Enhanced"]["lengths"]) if metrics["Enhanced"]["lengths"] else 0

    print(f"{'Success Rate':<20} | {b_succ:>14.1f}% | {e_succ:>14.1f}%")
    print(f"{'Avg Compute Time':<20} | {b_time:>12.2f} s | {e_time:>12.2f} s")
    print(f"{'Avg Nodes Explored':<20} | {b_nodes:>14.0f} | {e_nodes:>14.0f}")
    print(f"{'Avg Path Length':<20} | {b_len:>12.2f} rad | {e_len:>12.2f} rad")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_benchmark(num_trials=50)