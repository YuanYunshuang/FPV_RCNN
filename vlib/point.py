''' Visualization code for point clouds and 3D bounding boxes with mayavi.
Modified by Charles R. Qi
Date: September 2017
Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
'''

import numpy as np
from ops.iou3d_nms.iou3d_nms_utils import centroid_to_corners
import matplotlib.pyplot as plt
import wandb
from vlib.image import draw_box_plt
import torch

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


def draw_points_boxes_plt(pc_range, points=None, boxes_pred=None, boxes_gt=None, wandb_name=None,
                          points_c='k.', bbox_gt_c='green', bbox_pred_c='red', return_ax=False, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(10, 10)).add_subplot(1, 1, 1)
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(pc_range[0], pc_range[3]),
               ylim=(pc_range[1], pc_range[4]))
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], points_c, markersize=0.3)
    if (boxes_gt is not None) and len(boxes_gt)>0:
        ax = draw_box_plt(boxes_gt, ax, color=bbox_gt_c)
    if (boxes_pred is not None) and len(boxes_pred)>0:
        ax = draw_box_plt(boxes_pred, ax, color=bbox_pred_c)
    plt.xlabel('x')
    plt.ylabel('y')

    # plt.savefig('/media/hdd/ophelia/tmp/tmp.png')
    if wandb_name is not None:
        wandb.log({wandb_name: wandb.Image(plt)})
    if return_ax:
        return ax
    plt.close()


def draw_points_boxes_wandb(points=None, boxes_pred=None, boxes_gt=None):
    boxes = []
    vectors = []
    boxes_w, vectors_w = get_wanndb_boxes3d(boxes_pred, [255, 0, 0])
    if len(boxes_w)>0:
        boxes.extend(boxes_w)
        vectors.extend(vectors_w)

    boxes_w, vectors_w = get_wanndb_boxes3d(boxes_gt, [0, 255, 0])
    if len(boxes_w)>0:
        boxes.extend(boxes_w)
        vectors.extend(vectors_w)

    wandb.log(
        {
            "point_scene": wandb.Object3D(
                {
                    "type": "lidar/beta",
                    "points": points,
                    "boxes": np.array(boxes),
                    "vectors": np.array(vectors)
                }
            )
        }
    )


def get_wanndb_boxes3d(boxes, color, labels=None):
    boxes_wandb = []
    vectors_wandb = []
    if (boxes is not None) and len(boxes)>0:
        boxes_corners = centroid_to_corners(boxes)
        if isinstance(boxes_corners, torch.Tensor):
            boxes_corners = boxes_corners.cpu().numpy()
        for corners in boxes_corners:
            boxes_wandb.append({
                "corners": corners.tolist(),
                "color": color,
            })
            corners_front = corners[[0,3,4,7]]
            center = corners.mean(axis=0)
            center_front = corners_front.mean(axis=0)
            vec_end = center + (center_front - center) * 2
            vectors_wandb.append({
                "start": center.tolist(),
                "end": vec_end.tolist(),
                "color": color
            })
    return boxes_wandb, vectors_wandb



