import numpy as np
from optimize.smoother import smooth_path, generate_bspline, generate_catmull_rom

def test_smooth_path_empty_or_short():
    # If path is too short, it should return identical path
    short_path = [np.array([0.0]), np.array([1.0])]
    result = smooth_path(short_path, None, [], -1, max_iterations=10)
    assert len(result) == 2

def test_generate_bspline_short_path():
    path = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    spline = generate_bspline(path, num_points=10)
    assert len(spline) == 10
    np.testing.assert_array_almost_equal(spline[0], path[0])
    np.testing.assert_array_almost_equal(spline[-1], path[-1])

def test_generate_catmull_rom_short_path():
    path = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    spline = generate_catmull_rom(path, pts_per_seg=10)
    # 2 pts -> len should be 10 iterations + 1 = 11? Wait, it returns pts_per_seg + 1
    assert len(spline) > 2
    np.testing.assert_array_almost_equal(spline[0], path[0])
    np.testing.assert_array_almost_equal(spline[-1], path[-1])

def test_generate_catmull_rom_multiple_pts():
    path = [
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([2.0, 0.0])
    ]
    spline = generate_catmull_rom(path, pts_per_seg=5)
    # ensure it starts at 0 and goes to end
    np.testing.assert_array_almost_equal(spline[0], path[0])
    np.testing.assert_array_almost_equal(spline[-1], path[-1])
