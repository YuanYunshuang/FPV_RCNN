from functools import partial
from utils.common_utils import *

import numpy as np


class Augmentor:
    def __init__(self, dcfg):

        self.data_augmentor_queue = []

        for cur_augmentor_name, cur_cfg in dcfg.AUGMENTOR.items():
            cur_augmentor = getattr(self, cur_augmentor_name)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(self, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = self.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = self.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict

    @staticmethod
    def random_flip_along_x(gt_boxes, points):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C)
        Returns:
        """
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            gt_boxes[:, 6] = -gt_boxes[:, 6]
            points[:, 1] = -points[:, 1]

            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 8] = -gt_boxes[:, 8]

        return gt_boxes, points

    @staticmethod
    def random_flip_along_y(gt_boxes, points):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C)
        Returns:
        """
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
            points[:, 0] = -points[:, 0]

            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7] = -gt_boxes[:, 7]

        return gt_boxes, points

    @staticmethod
    def global_rotation(gt_boxes, points, rot_range):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C),
            rot_range: [min, max] in °
        Returns:
        """
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1]) / 180 * np.pi
        points = rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        gt_boxes[:, 0:3] = rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
        gt_boxes[:, 6] += noise_rotation
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7:9] = rotate_points_along_z(
                np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                np.array([noise_rotation])
            )[0][:, 0:2]

        return gt_boxes, points

    @staticmethod
    def global_scaling(gt_boxes, points, scale_range):
        """
        Args:
            gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
            points: (M, 3 + C),
            scale_range: [min, max]
        Returns:
        """
        if scale_range[1] - scale_range[0] < 1e-3:
            return gt_boxes, points
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        points[:, :3] *= noise_scale
        gt_boxes[:, :6] *= noise_scale
        return gt_boxes, points