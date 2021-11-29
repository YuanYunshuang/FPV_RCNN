from abc import ABCMeta, abstractmethod, abstractproperty
from utils import box_np_ops, box_torch_ops
import numpy as np
import torch
from ops.roiaware_pool3d import roiaware_pool3d_utils
import cv2 as cv



class BoxCoder(object):
    """Abstract base class for box coder."""

    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass


class GroundBox3dCoder(BoxCoder):
    def __init__(self, linear_dim=False, angle_vec_encode=False, angle_residual_encode=False,
                 n_dim=7, norm_velo=False, **kwargs):
        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = angle_vec_encode
        self.residual_encode = angle_residual_encode
        self.norm_velo = norm_velo
        self.n_dim = n_dim

    @property
    def code_size(self):
        # return 8 if self.vec_encode else 7
        # return 10 if self.vec_encode else 9
        return self.n_dim + 1 if self.vec_encode else self.n_dim

    def _encode(self, boxes, anchors):
        return box_np_ops.second_box_encode(
            boxes,
            anchors,
            encode_angle_to_vector=self.vec_encode,
            encode_angle_with_residual=self.residual_encode,
            smooth_dim=self.linear_dim,
            norm_velo=self.norm_velo,
        )

    def _decode(self, encodings, anchors):
        return box_np_ops.second_box_decode(
            encodings,
            anchors,
            encode_angle_to_vector=self.vec_encode,
            encode_angle_with_residual=self.residual_encode,
            smooth_dim=self.linear_dim,
            norm_velo=self.norm_velo,
        )


class GroundBox3dCoderTorch(GroundBox3dCoder):
    # This func inherit funcs like _encode in GroundBox3dCoder.
    def encode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_encode(boxes, anchors,
                                               self.vec_encode,
                                               self.residual_encode,
                                               self.linear_dim)

    def decode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_decode(boxes, anchors,
                                               self.vec_encode,
                                               self.residual_encode,
                                               self.linear_dim)


class GroundBoxBevGridCoder(BoxCoder):
    def __init__(self, linear_dim=False, encode_angle_vector=False, n_dim=7, norm_velo=False, **kwargs):
        super().__init__()
        self.box_means = np.array(kwargs['box_means'])
        self.box_stds = np.array(kwargs['box_stds'])
        self.linear_dim = linear_dim
        self.vec_encode = encode_angle_vector
        self.norm_velo = norm_velo
        self.n_dim = n_dim

    @property
    def code_size(self):
        # return 8 if self.vec_encode else 7
        # return 10 if self.vec_encode else 9
        return self.n_dim + 1 if self.vec_encode else self.n_dim

    def encode(self, boxes_in, cfg):
        boxes = boxes_in.copy()
        boxes_ = boxes_in.copy()
        boxes_[:, 2] = 0.0
        boxes_[:, 5] = 1.0

        resolution = cfg.voxel_size[0] * cfg.label_downsample
        x = np.arange(cfg.pc_range[0], cfg.pc_range[3], resolution, dtype=np.float32)
        y = np.arange(cfg.pc_range[1], cfg.pc_range[4], resolution, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        points = np.zeros((xx.size, 3), dtype=np.float32)
        points[:, 0] = xx
        points[:, 1] = yy

        point_indices = np.where(roiaware_pool3d_utils.points_in_boxes_cpu(points, boxes_[:, :7]) > 0)
        box_idxs_of_pts = - np.ones(xx.size, dtype=np.int)
        box_idxs_of_pts[point_indices[1]] = point_indices[0]
        label_size = (cfg.grid_size[0] // cfg.label_downsample,
                      cfg.grid_size[1] // cfg.label_downsample)
        label = np.zeros(label_size, dtype=np.float32)
        x_inds = np.clip(((xx - cfg.pc_range[0]) / resolution).astype(np.int), 0, label_size[0] - 1)
        y_inds = np.clip(((yy - cfg.pc_range[1]) / resolution).astype(np.int), 0, label_size[1] - 1)
        mask = (box_idxs_of_pts >= 0)
        selected_pts_xy = points[mask][:, :2]
        box_xy_of_selected_pts = boxes[box_idxs_of_pts[mask]][:, :2]
        #centerness = np.exp(-1 * np.linalg.norm((selected_pts_xy - box_xy_of_selected_pts), axis=1))
        #centerness = 2 * centerness / (1 + centerness) # [0, 1]
        label[x_inds[mask], y_inds[mask]] = 1#centerness
        if cfg.BOX_CODER['n_dim'] == 7:
            reg = np.zeros(label_size + (8,), dtype=np.float32)
            reg[x_inds[mask], y_inds[mask], 0] = boxes[box_idxs_of_pts[mask], 0] - xx[mask]
            reg[x_inds[mask], y_inds[mask], 1] = boxes[box_idxs_of_pts[mask], 1] - yy[mask]
            reg[x_inds[mask], y_inds[mask], 2] = boxes[box_idxs_of_pts[mask], 2]
            reg[x_inds[mask], y_inds[mask], 3] = boxes[box_idxs_of_pts[mask], 3]
            reg[x_inds[mask], y_inds[mask], 4] = boxes[box_idxs_of_pts[mask], 4]
            reg[x_inds[mask], y_inds[mask], 5] = boxes[box_idxs_of_pts[mask], 5]
            reg[x_inds[mask], y_inds[mask], 6] = np.sin(boxes[box_idxs_of_pts[mask], 6])
            reg[x_inds[mask], y_inds[mask], 7] = np.cos(boxes[box_idxs_of_pts[mask], 6])
        elif cfg.BOX_CODER['n_dim'] == 5:
            reg = np.zeros(label_size + (6,), dtype=np.float32)
            reg[x_inds[mask], y_inds[mask], 0] = boxes[box_idxs_of_pts[mask], 0] - xx[mask]
            reg[x_inds[mask], y_inds[mask], 1] = boxes[box_idxs_of_pts[mask], 1] - yy[mask]
            reg[x_inds[mask], y_inds[mask], 2] = np.log(boxes[box_idxs_of_pts[mask], 3])
            reg[x_inds[mask], y_inds[mask], 3] = np.log(boxes[box_idxs_of_pts[mask], 4])
            reg[x_inds[mask], y_inds[mask], 4] = np.sin(boxes[box_idxs_of_pts[mask], 6])
            reg[x_inds[mask], y_inds[mask], 5] = np.cos(boxes[box_idxs_of_pts[mask], 6])
            reg[x_inds[mask], y_inds[mask]] = (reg[x_inds[mask], y_inds[mask]] - self.box_means) / self.box_stds
        else:
            raise ValueError('box encode dim should be 5 or 7.')

        return label, reg

    def decode_torch(self, cls_preds, reg_preds, score_thr, cfg):
        batch_size = cls_preds.shape[0]
        scores = torch.sigmoid(cls_preds)
        kernel = np.ones((3, 3), np.uint8)
        means = torch.tensor(self.box_means, dtype=reg_preds.dtype, device=reg_preds.device)
        stds = torch.tensor(self.box_stds, dtype=reg_preds.dtype, device=reg_preds.device)
        reg_preds = reg_preds * stds[None, None, None, :] + means[None, None, None, :]
        resolution = cfg.voxel_size[0] * cfg.label_downsample

        batch_boxes = []
        batch_scores = []
        for b in range(batch_size):
            image = cv.UMat((scores[b].squeeze().cpu().numpy() > score_thr).astype(np.float32))
            scores_mask = cv.UMat.get(cv.erode(image, kernel, iterations=1))
            scores_mask = torch.tensor(scores_mask, device=scores.device)

            cur_xs, cur_ys = torch.where(scores_mask)
            boxes = torch.zeros([len(cur_xs), 7], dtype=reg_preds.dtype, device=reg_preds.device)
            if len(cur_xs)<=0:
                batch_boxes.append(boxes)
                batch_scores.append(torch.zeros([0, 1], dtype=reg_preds.dtype, device=reg_preds.device))
                continue
            boxes_reg = reg_preds[b, cur_xs, cur_ys, :]
            xs = (cur_xs * resolution + cfg.pc_range[0])
            ys = (cur_ys * resolution + cfg.pc_range[1])
            if cfg.BOX_CODER['n_dim'] == 5:
                boxes[:, 0] = boxes_reg[:, 0] + xs
                boxes[:, 1] = boxes_reg[:, 1] + ys
                boxes[:, 3] = boxes_reg[:, 2].exp()
                boxes[:, 4] = boxes_reg[:, 3].exp()
                boxes[:, 5] = 1.0
                boxes[:, 6] = torch.atan2(boxes_reg[:, 4], boxes_reg[:, 5])
            else:
                raise NotImplementedError

            batch_boxes.append(boxes)
            batch_scores.append(scores[b, cur_xs, cur_ys, :].reshape(-1, 1))

        return batch_scores, batch_boxes







