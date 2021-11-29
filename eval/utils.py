import numpy as np
import math

z_normed_corners_default = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]]) / 2

def center_to_corner_BEV(boxBEV):
    #WZN boxBEV to corners in BEV
    #boxBEV is [xcam, zcam, l, w, ry]
    x0 = np.array(boxBEV).reshape(5)
    x0[4] *= -1
    ry = x0[4]
    rotmat = np.array([[math.cos(ry), -math.sin(ry)], [math.sin(ry), math.cos(ry)]])
    corners = np.matmul(z_normed_corners_default * x0[2:4], rotmat.transpose()) + x0[0:2]
    return corners

def create_BEV_box_sample_grid(sample_grid):
    x = np.arange(-0.5 + sample_grid / 2, 0.5, sample_grid)
    y = x
    zx, zy = np.meshgrid(x, y)
    z_normed = np.concatenate((zx.reshape((-1, 1)), zy.reshape((-1, 1))), axis=1)
    return z_normed