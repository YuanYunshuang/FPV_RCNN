import numpy as np


def get_tf_matrix(vector, inv=False):
    x = vector[0]
    y = vector[1]
    angle = vector[2]
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array([
        [cosa,  -sina, 0],
        [sina, cosa, 0],
        [0, 0, 1]
    ]).astype(np.float)
    shift_matrix = np.array([
        [1,  0, x],
        [0, 1, y],
        [0, 0, 1]
    ]).astype(np.float)
    if inv:
        mat = rot_matrix @ shift_matrix
    else:
        mat = shift_matrix @ rot_matrix
    return mat, rot_matrix, shift_matrix

def global_rotation(gt_boxes, points, angle):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        angle: angle in rad
    Returns:
    """
    points = rotate_points_along_z(points, angle)
    gt_boxes[:, 0:3] = rotate_points_along_z(gt_boxes[:, 0:3],angle)
    gt_boxes[:, 6] += angle
    return gt_boxes, points


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (N, 3 + C)
        angle: float, angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array([
        [cosa,  sina, 0],
        [-sina, cosa, 0],
        [0, 0, 1]
    ]).astype(np.float)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, 3:]), axis=-1)
    return points_rot


def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)


def rot2eul(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = np.pi/2.0
        psi = np.arctan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -np.pi/2.0
        psi = np.atan2(-R[0,1],-R[0,2])
    else:
        theta = -np.arcsin(R[2,0])
        cos_theta = np.cos(theta)
        psi = np.arctan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = np.arctan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return [psi, theta, phi]
