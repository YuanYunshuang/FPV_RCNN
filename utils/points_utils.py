import numpy as np


def rotate_points_along_z(points, angle):
    """
    Args:
        points: Numpy (B, N, 3 + C)
        angle: (B), angle along z-axis in degree

    """
    angle = angle / 180 * np.pi
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array([
        [cosa,  sina, 0],
        [-sina, cosa, 0],
        [0, 0, 1],
    ], dtype=np.float)
    #print('noise tf:')
    #print(rot_matrix)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, 3:]), axis=-1)
    return points_rot


def add_gps_noise_bev(points, noise):
    """
        Args:
        points: Numpy (N, 3 + C)
        noise: (6), [std_x, std_y, std_z, std_roll, std_pitch, std_yaw]
    """
    rot = np.random.normal(0, noise[-1], 1)
    points_out = rotate_points_along_z(points, rot)
    shift = np.array([np.random.normal(0, noise[0], 1), np.random.normal(0, noise[1], 1)])
    #print('shift:', shift)
    points_out[:, :2] = points_out[:, :2] + shift.reshape((1, 2))

    return points_out, [*shift, rot]