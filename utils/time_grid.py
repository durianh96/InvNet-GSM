import numpy as np


def find_closest_time_grid_index(grid_points, target_t):
    closest_index = np.argmin(np.abs(np.array(grid_points) - target_t))
    return closest_index


def find_closest_time_grid_value(grid_points, target_t):
    closest_index = find_closest_time_grid_index(grid_points, target_t)
    closest_value = grid_points[closest_index]
    return closest_value


def find_right_time_grid_index(grid_points, target_t):
    if target_t < grid_points[-1]:
        right_index = min([i for i, t in enumerate(grid_points) if t - target_t >= 0])
    else:
        right_index = len(grid_points) - 1
    return right_index


def find_right_time_grid_value(grid_points, target_t):
    right_index = find_right_time_grid_index(grid_points, target_t)
    right_value = grid_points[right_index]
    return right_value
