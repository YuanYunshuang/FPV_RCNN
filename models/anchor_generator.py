import numpy as np
from utils import box_np_ops


class AnchorGeneratorRange:
    def __init__(self, cfg):
        self._sizes = cfg['sizes']
        self._anchor_ranges = cfg['anchor_ranges']
        self._rotations = cfg['rotations']
        self._dtype = np.float32
        self._class_name = cfg['class_name']
        self._match_threshold = cfg['match_threshold']
        self._unmatch_threshold = cfg['unmatch_threshold']

    @property
    def class_name(self):
        return self._class_name

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    @property
    def ndim(self):
        # return 7 + len(self._custom_values)
        return self._anchors.shape[-1]

    def generate(self, feature_map_size):
        self._anchors = box_np_ops.create_anchors_3d_range(
            feature_map_size,
            anchor_range=self._anchor_ranges,
            sizes=self._sizes,
            rotations=self._rotations,
            dtype=self._dtype,
        )
        return self._anchors