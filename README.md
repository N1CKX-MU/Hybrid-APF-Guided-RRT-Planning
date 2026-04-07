# 🤖 Hybrid APF-Guided RRT Path Planning

A custom Python and PyBullet implementation of an advanced robotic path planner for a 7-DOF Franka Panda manipulator. This project combines **Rapidly-exploring Random Trees (RRT)** with **Artificial Potential Fields (APF)** to navigate complex 3D environments, complete with a custom solution for the "Narrow Passage Problem" and trajectory smoothing.

## ✨ Key Features

* **Custom PyBullet Environment:** Object-oriented simulation featuring a 7-DOF Franka Panda arm and highly customizable obstacle layouts (spheres, boxes, archways).
* **Multi-Start Inverse Kinematics (IK):** A robust IK wrapper that utilizes stochastic joint-space seeding to ensure the final goal configuration doesn't collide with obstacles.
* **Hybrid APF-RRT Planner:** Blends the rapid, global exploration of RRT with the local obstacle-repelling gradients of APF.
* **Adaptive Step-Sizing (Narrow Passage Solver):** dynamically scales the algorithm's step size based on local obstacle density. It takes massive leaps in open space and automatically switches to high-resolution "tip-toes" to thread the arm through narrow gaps.
* **Greedy Path Smoother:** A post-processing optimizer that utilizes ray-casting collision checks to delete redundant waypoints, compressing erratic RRT paths into fluid, direct trajectories.
* **Trajectory Interpolation:** Generates high-FPS linear joint-space interpolations for smooth, industrial-grade robot animation.
* **Headless Benchmarking:** Automated testing suite to statistically compare the baseline algorithm against the enhanced adaptive algorithm.

## 📂 Project Structure

```text
├── env/
│   ├── robot.py        # Object-Oriented wrapper for the Panda robot (Jacobians, IK, collisions)
│   └── scene.py        # PyBullet environment setup and obstacle generation
├── planner/
│   ├── rrt.py          # Core RRT data structures (Node, steer, nearest)
│   ├── apf.py          # Gradient math (Attractive/Repulsive forces, Jacobians)
│   └── hybrid.py       # The Hybrid algorithms (Baseline & Enhanced APF-RRT)
├── optimize/
│   └── smoother.py     # Greedy path smoothing and segment safety checks
├── run_raw.py          # Executes the unoptimized path planner
├── run_smoothed.py     # Executes the path planner with the post-processing smoother
└── benchmark.py        # Headless benchmarking suite for statistical analysis
```

## 🚀 Installation & Setup

**Clone the repository:**
```bash
git clone https://github.com/N1CKX-MU/Hybrid-APF-Guided-RRT-Planning.git
cd Hybrid-APF-RRT-Planner
```

**Install dependencies:**
This project relies on standard scientific computing and simulation libraries.
```bash
pip install numpy pybullet
```

## 🎮 Usage

### 1. Watch the Robot Plan and Move
To run the simulation with the graphical interface (GUI) and watch the robot navigate the narrow archway:
```bash
python path_finder_with_smoothing.py
```
*(Note: You can easily swap between `apf_rrt` and `apf_rrt_enhanced` in this file to see the difference in behavior).*

### 2. Run the Benchmark Suite
To run statistical comparisons of the algorithms without rendering the GUI (Headless mode):
```bash
python benchmark.py
```

## 📊 Performance Benchmarks

The integration of Adaptive Step Sizing (Phase B) dramatically out-performs the baseline APF-RRT algorithm, particularly in environments with narrow passages.

Averaged over 10 trials in headless simulation:

| Metric | Phase A (Baseline) | Phase B (Enhanced Adaptive) |
| :--- | :--- | :--- |
| **Success Rate** | 100.0% | 100.0% |
| **Avg Compute Time** | 9.34 s | 5.49 s |
| **Avg Nodes Explored** | 3500 | 2648 |

> **Note:** The Enhanced planner uses 25% less memory and solves complex narrow-passage mazes nearly twice as fast by intelligently scaling its exploration bounds.

## 🧠 The Math Behind the Code

- **The Jacobian:** The APF repulsive forces are calculated in task space (Cartesian coordinates) and translated into joint space (Radians) using the robot's dynamically calculated 3x7 Jacobian matrix.
- **Local Minima Avoidance:** By blending the APF gradient with random RRT sampling (75% Random, 25% APF), the robot naturally escapes the "invisible bowls" created by overlapping repulsive forcefields.
- **Obstacle Density Metric:** The Enhanced algorithm calculates the distance from the end-effector to all obstacles. If the sum of these distances breaches a critical threshold, the algorithm dynamically reduces its step size to navigate safely.