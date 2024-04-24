import numpy as np
from utils import local_to_global

def test_local_to_global():
    # Test case 1
    base_x = 0
    base_y = 0
    base_yaw = 0
    local_x = 1
    local_y = 1
    local_yaw = np.pi/2
    base_v = 1
    local_vx = 0
    local_vy = 0
    expected_result = (1, 1, np.pi/2, 1, 0)
    assert local_to_global(base_x, base_y, base_yaw, base_v, local_x, local_y, local_yaw, local_vx, local_vy) == expected_result

    # Test case 2
    base_x = 10
    base_y = 5
    base_yaw = np.pi/4
    local_x = 2
    local_y = -3
    local_yaw = np.pi/6
    base_v = 2
    local_vx = 1
    local_vy = -1
    expected_result = (10 + 2 * np.cos(np.pi/4) - (-3) * np.sin(np.pi/4),
                       5 + 2 * np.sin(np.pi/4) + (-3) * np.cos(np.pi/4),
                       np.pi/4 + np.pi/6,
                       2 * np.cos(np.pi/4) + 1 * np.cos(np.pi/4) - (-1) * np.sin(np.pi/4),
                       2 * np.sin(np.pi/4) + 1 * np.sin(np.pi/4) + (-1) * np.cos(np.pi/4))
    assert local_to_global(base_x, base_y, base_yaw, base_v, local_x, local_y, local_yaw, local_vx, local_vy) == expected_result

    # Test case 3
    base_x = -5
    base_y = 3
    base_yaw = -np.pi/3
    local_x = -1
    local_y = 0
    local_yaw = -np.pi/4
    base_v = 0.5
    local_vx = -0.5
    local_vy = 0.5
    expected_result = (-5 + (-1) * np.cos(-np.pi/3) - 0 * np.sin(-np.pi/3),
                       3 + (-1) * np.sin(-np.pi/3) + 0 * np.cos(-np.pi/3),
                       -np.pi/3 + (-np.pi/4),
                       0.5 * np.cos(-np.pi/3) + (-0.5) * np.cos(-np.pi/3) - 0.5 * np.sin(-np.pi/3),
                       0.5 * np.sin(-np.pi/3) + (-0.5) * np.sin(-np.pi/3) + 0.5 * np.cos(-np.pi/3))
    assert local_to_global(base_x, base_y, base_yaw, base_v, local_x, local_y, local_yaw, local_vx, local_vy) == expected_result

    print("All test cases passed!")

test_local_to_global()