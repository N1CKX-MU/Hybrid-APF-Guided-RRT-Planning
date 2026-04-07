import numpy as np
from planner.rrt import Node, nearest, steer, extract_path, path_length

def test_steer_takes_partial_step():
    """
    If the target is FARTHER than the step_size, 
    it should take exactly one step_size toward it.
    """
    q_from = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_to   = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step_size = 0.1
    
    q_new = steer(q_from, q_to, step_size)
    
    expected = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(q_new, expected)


def test_steer_snaps_to_target():
    """
    If the target is CLOSER than the step_size, 
    it should just return the exact target coordinate.
    """
    q_from = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_to   = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step_size = 0.1
    
    q_new = steer(q_from, q_to, step_size)
    
    np.testing.assert_array_almost_equal(q_new, q_to)


def test_nearest_node_in_tree():
    """
    Given a random coordinate, the nearest() function must return 
    the Node in the tree with the shortest Euclidean distance.
    """
    node1 = Node(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    node2 = Node(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    node3 = Node(np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0]))
    
    tree = [node1, node2, node3]
    
    # This guess is closest to node2
    q_guess = np.array([1.1, 1.1, 1.0, 0.0, 0.0, 0.0, 0.0])
    
    closest = nearest(tree, q_guess)
    
    # Assert that the object returned is exactly node2
    assert closest is node2


def test_path_length_calculation():
    """
    Verifies the path_length function accurately sums the Euclidean 
    distances between consecutive waypoints.
    """
    # Moving 1 unit on the X-axis, then 2 units on the Y-axis. Total should be 3.0.
    path = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.0])
    ]
    
    length = path_length(path)
    assert np.isclose(length, 3.0)


def test_extract_path():
    """
    Verifies that the path is correctly extracted by following 
    parent pointers backward from the goal, and that the final 
    list is reversed (Start -> Goal).
    """
    n_start = Node(np.array([0.0]))
    n_mid   = Node(np.array([1.0]), parent=n_start)
    n_goal  = Node(np.array([2.0]), parent=n_mid)
    
    path = extract_path(n_goal)
    
    assert len(path) == 3
    # Verify the order is Start -> Mid -> Goal
    np.testing.assert_array_almost_equal(path[0], n_start.q)
    np.testing.assert_array_almost_equal(path[1], n_mid.q)
    np.testing.assert_array_almost_equal(path[2], n_goal.q)