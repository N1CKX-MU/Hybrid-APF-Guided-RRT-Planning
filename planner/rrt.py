import numpy as np 
from typing import List, Optional

class Node:
    """
    A Single node in the Rapidly-exploring Random Tree
    """
    __slots__ = ("q","parent", "cost")

    def __init__(self, q: np.ndarray, parent: Optional["Node"] = None, cost: float = 0.0):
        self.q = np.asarray(q,dtype=float)
        self.parent = parent
        self.cost = float(cost)
    def __repr__(self) -> str:
        return f"Node(q={np.round(self.q,3)},cost={self.cost:.4f})"


def nearest(tree: List[Node], q_rand: np.ndarray) -> Node:
    """
    Finds the closest existing node in the tree to a randomly guessed coordinate
    """
    
    #Calc Euclidean distance between our random guess and every node in the tree
    dists = np.array([np.linalg.norm(node.q - q_rand) for node in tree])

    # return the node with the abs min distance
    return tree[int(np.argmin(dists))]

def steer(q_from : np.ndarray, q_to: np.ndarray, step_size: float = 0.10) -> np.ndarray:
    """
    Moves from q_from to q_to by at most step size in JOINT SPACE,
    Return the new configuration.
    """
    delta = q_to - q_from
    dist = np.linalg.norm(delta)

    if dist <1e-9:
        return q_from.copy()

    if dist<= step_size:
        return q_to.copy()
    
    q_new = q_from + step_size * (delta / dist)
    return q_new

def extract_path(goal_node: Node) -> List[np.ndarray]:
    """
    Backtracks from the goal node to the root node by following parent pointers,
    Returns the final path as a list of joint configs [Start , ... , End]
    """

    path = []
    current_node = goal_node

    while current_node is not None:
        path.append(current_node.q.copy())
        current_node = current_node.parent
    
    return list(reversed(path))


def path_length(path: List[np.ndarray]) -> float:
    """
    Computes the total euclidean path length in joint space
    Sum of ||q[i+1] - q[i] || for all consecutive waypoints
    """
    
    if not path or len(path) < 2:
        return 0.0
    
    return sum(
        np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)
    )
    

        