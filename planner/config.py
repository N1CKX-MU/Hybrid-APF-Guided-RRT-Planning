"""
Standard Configuration for Hybrid APF-RRT Planner.
Centralizing these parameters ensures consistency across main.py, benchmark.py, and other scripts.
"""

# ─── CORE PLANNER PARAMETERS ───
# How much to bias the random sampling toward the goal configuration
# Higher = More aggressive pull toward goal, Lower = More exploration
GOAL_BIAS = 0.09

# Blending weight between random direction and APF gradient
# Higher = More random/RRT behavior, Lower = More APF/gradient descent behavior
APF_BLEND = 0.60

# Distance threshold to consider the goal reached (in radians)
GOAL_THRESHOLD = 0.15

# ─── ADAPTIVE STEP SIZE (ENHANCED ONLY) ───
# The maximum step size allowed in open spaces
STEP_SIZE_MAX = 0.15

# The minimum step size allowed in crowded spaces
STEP_SIZE_MIN = 0.02

# Standard step size for the baseline (non-enhanced) planner
STEP_SIZE_DEFAULT = 0.08

# ─── ITERATION LIMITS ───
# Default limit for standard runs
MAX_ITER_DEFAULT = 8000

# Higher limit for benchmarks or complex scenes
MAX_ITER_BENCHMARK = 15000

# ─── ARTIFICIAL POTENTIAL FIELD (APF) CONSTANTS ───
K_ATT = 1.0     # Attractive gain 
K_REP = 0.8     # Repulsive gain
RHO = 0.35      # Obstacle influence radius
D_MIN = 0.01    # Min distance clamp to avoid division by zero
