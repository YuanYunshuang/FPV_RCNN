from vlib.point import draw_points_boxes_wandb, draw_points_boxes_plt, draw_box_plt
import matplotlib.pyplot as plt


def draw_points_boxes_bev_3d(points, pred_boxes, gt_boxes, pc_range):
    """
    visualize the result of the first batch, input shoulb be data from only one batch
    """
    draw_points_boxes_wandb(points, boxes_pred=pred_boxes, boxes_gt=gt_boxes)
    draw_points_boxes_plt(pc_range, points[:, :2], boxes_pred=pred_boxes, boxes_gt=gt_boxes)


def draw_points_boxes_bev(points, pred_boxes, gt_boxes, pc_range):
    """
    visualize the result of the first batch, input shoulb be data from only one batch
    """
    draw_points_boxes_plt(pc_range, points[:, :2], boxes_pred=pred_boxes, boxes_gt=gt_boxes)


def draw_points_boxes_3d(points, pred_boxes, gt_boxes, pc_range):
    """
    visualize the result of the first batch, input shoulb be data from only one batch
    """
    draw_points_boxes_wandb(points, boxes_pred=pred_boxes, boxes_gt=gt_boxes)


def draw_points_boxes_plt_2d(pc_range, points=None, boxes_pred=None, boxes_gt=None):
    ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(pc_range[0], pc_range[3]),
           ylim=(pc_range[1], pc_range[4]))
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], 'y.', markersize=0.3)
    if (boxes_gt is not None) and len(boxes_gt)>0:
        ax = draw_box_plt(boxes_gt, ax, color='green')
    if (boxes_pred is not None) and len(boxes_pred)>0:
        ax = draw_box_plt(boxes_pred, ax, color='red')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig('/media/hdd/ophelia/tmp/tmp.png')
    plt.close()