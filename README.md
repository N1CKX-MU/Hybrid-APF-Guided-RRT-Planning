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
│   ├── hybrid.py       # The Hybrid algorithms (Baseline & Enhanced APF-RRT)
│   └── config.py       # Centralized configuration (APF gains, RRT biases, step sizes)
├── optimize/
│   └── smoother.py     # Greedy path smoothing and segment safety checks
├── main.py             # Highly polished interactive CLI entry point for the suite
├── run_baseline.py     # Execution script for Baseline APF-RRT
├── run_enhanced.py     # Execution script for Enhanced APF-RRT
├── run_rrt_connect.py  # Execution script for Bi-Directional RRT-Connect
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

### 1. Interactive CLI (Recommended)
The project features a fully polished, interactive text-menu to execute all available motion planners.
```bash
python main.py
```
*This handles everything automatically, allowing you to choose between Baseline, Enhanced, or RRT-Connect algorithms and run them effortlessly.*

### 2. Direct Execution
To run the planners individually and bypass the CLI menu:
```bash
python run_rrt_connect.py
python run_enhanced.py
python run_baseline.py
```

### 3. Run the Benchmark Suite
To run statistical comparisons of the algorithms without rendering the GUI (Headless mode):
```bash
python benchmark.py
```

## 📊 Performance Benchmarks

The integration of Bi-Directional RRT-Connect (Phase C) dramatically out-performs the baseline APF-RRT algorithms, particularly in heavily constrained environments with narrow passages.

Averaged over 20 trials in headless simulation:

| Metric | Phase A (Baseline) | Phase B (Enhanced Adaptive) | Phase C (Bi-directional Connect) |
| :--- | :--- | :--- | :--- |
| **Success Rate** | 85.0% | 100.0% | 100.0% |
| **Avg Compute Time** | 0.99 s | 0.42 s | 0.23 s |
| **Avg Nodes Explored** | 521 | 574 | 368 |
| **Avg Path Length** | 3.65 rad | 3.60 rad | 6.53 rad |

> **Note:** The Bi-directional RRT-Connect planner completely dominates the environment, cutting computation time in half compared to the Enhanced approach. Its organic tree-connection structure trivially solves extremely complex narrow-passage mazes that cause the baseline algorithm to occasionally trap out entirely (85% success rate).

## 🧠 The Math Behind the Code

- **The Jacobian:** The APF repulsive forces are calculated in task space (Cartesian coordinates) and translated into joint space (Radians) using the robot's dynamically calculated 3x7 Jacobian matrix.
- **Local Minima Avoidance:** By blending the APF gradient with random RRT sampling (75% Random, 25% APF), the robot naturally escapes the "invisible bowls" created by overlapping repulsive forcefields.
- **Obstacle Density Metric:** The Enhanced algorithm calculates the distance from the end-effector to all obstacles. If the sum of these distances breaches a critical threshold, the algorithm dynamically reduces its step size to navigate safely.