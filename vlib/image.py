import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.common_utils import mask_points_by_range
from utils.box_utils import mask_boxes_outside_range_numpy
from ops.iou3d_nms.iou3d_nms_utils import decode_boxes


def draw_box_plt(boxes_dec, ax, color=None, linewidth_scale=1.0):
    """
    draw boxes in a given plt ax
    :param boxes_dec: (N, 5) or (N, 7) in metric
    :param ax:
    :return: ax with drawn boxes
    """
    if not len(boxes_dec)>0:
        return ax
    boxes_np= boxes_dec
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = boxes_np.cpu().detach().numpy()
    if boxes_np.shape[-1]>5:
        boxes_np = boxes_np[:, [0, 1, 3, 4, 6]]
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes_np[:, 4:5]
    # bl, fl, fr, br
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1]
              - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1]
              - y[:, None]) * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)
    for corner in corners:
        ax.plot(corner[[0,1,2,3,0], 0], corner[[0,1,2,3,0], 1], color=color, linewidth=0.5*linewidth_scale)
        # draw front line (
        ax.plot(corner[[2, 3], 0], corner[[2, 3], 1], color=color, linewidth=2*linewidth_scale)
    return ax


def draw_boxes_img(boxes, image, color=(255, 255, 255)):
    """
    draw boxes on an image
    :param boxes: (N, 5) [x, y, dx, dy, heading] in pixels
    :param image:
    :return: image with boxes drawn on it
    """
    boxes_np, image_np = boxes, image
    if not isinstance(boxes, np.ndarray):
        boxes_np = boxes.cpu().detach().numpy()
    if not isinstance(image, np.ndarray):
        image_np = image
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    for i in range(len(x)):
        # OpenCV draws with raw as x-axis, col as y-axis
        image_np = cv2.circle(image_np, (y[i], x[i]), radius=4, color=color, thickness=-1)

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes_np[:, 4:]
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1] - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1] - y[:, None]) * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_y.astype(np.int32), new_x.astype(np.int32)], axis=2)
    for corner in corners:
        for i in range(4):
            image_np = cv2.line(image_np, tuple(corner[i]), tuple(corner[(i+1) % 4]), color, 1)
    return image_np


def draw_gt_boxes(boxes, image, x_min, y_min, resolution, color=(255, 255, 255)):
    """
    draw boxes and box centers on grid map
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading] in meters
    :param image:
    :param x_min: lidar sensor range x min
    :param y_min: lidar sensor range y min
    :param resolution: resolution of projecting box points to the 2d image

    :return: images drawn with boxes and centers of the boxes
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    dx = boxes[:, 3]
    dy = boxes[:, 4]

    centers = [
        np.floor((x - x_min) / resolution).astype(np.int32),
        np.floor((y - y_min) / resolution).astype(np.int32)
    ]
    for i in range(len(centers[0])):
        # OpenCV draws with raw as x-axis, col as y-axis
        image = cv2.circle(image, (centers[1][i], centers[0][i]), radius=4, color=color, thickness=-1)

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes[:, 6:7]
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1] - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1] - y[:, None]) * (np.cos(theta)) + y[:, None]
    new_x = np.floor((new_x - x_min) / resolution).astype(np.int32)
    new_y = np.floor((new_y - y_min) / resolution).astype(np.int32)
    corners = np.stack([new_y, new_x], axis=2)
    for corner in corners:
        for i in range(4):
            image = cv2.line(image, tuple(corner[i]), tuple(corner[(i+1) % 4]), color, 1)

    return image


def detection_map(points, map_size, min_x, min_y, resolution):
    """
    Project points to 2d images based on the number of detection in each grid cell
    :param points: List(Nx3)
    :param map_size: Tuple(2) 2d grid map size
    :param x_min: lidar sensor range x min
    :param y_min: lidar sensor range y min
    :param resolution: resolution of projecting box points to the 2d image

    :return: 2d grid map of detections
    """
    map_detection = np.zeros(map_size, dtype=np.int32)
    for p in points:
        row = min(int(np.floor((p[0] - min_x) / resolution)), map_size[0]-1)
        col = min(int(np.floor((p[1] - min_y) / resolution)), map_size[1]-1)
        if p[3] > 0:
            map_detection[row, col] += 1  # some points have 0 intensity
    return map_detection


def mask_points_and_boxes_outside_range(points, gt_boxes, pc_range):
    mask = mask_points_by_range(points, pc_range)
    points = points[mask]
    mask = mask_boxes_outside_range_numpy(
        gt_boxes, pc_range, min_num_corners=1
    )
    gt_boxes = gt_boxes[mask]
    return points, gt_boxes





    