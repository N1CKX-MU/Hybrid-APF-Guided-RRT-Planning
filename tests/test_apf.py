import numpy as np
from planner.apf import attractive_forces, repulsive_force_joint_space

def test_attractive_force_joint_space():
    q = np.array([0.0, 0.0])
    q_goal = np.array([1.0, 0.0])
    
    # Force should point directly to goal
    f_att = attractive_forces(q, q_goal, K_att=2.0)
    
    expected = np.array([2.0, 0.0])
    np.testing.assert_array_almost_equal(f_att, expected)

def test_attractive_force_at_goal():
    q = np.array([1.0, 1.0])
    q_goal = np.array([1.0, 1.0])
    
    f_att = attractive_forces(q, q_goal, K_att=5.0)
    
    expected = np.array([0.0, 0.0])
    np.testing.assert_array_almost_equal(f_att, expected)
