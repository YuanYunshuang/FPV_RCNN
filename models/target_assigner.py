from collections import OrderedDict
import numpy as np
from utils import box_np_ops


class TargetAssigner:
    def __init__(
            self,
            box_coder,
            anchor_generator,
            positive_fraction=None,
            sample_size=512,
    ):
        self._box_coder = box_coder
        self._anchor_generator = anchor_generator
        self._positive_fraction = positive_fraction   # None
        self._sample_size = sample_size

    @property
    def box_coder(self):
        return self._box_coder

    @property
    def classes(self):
        return [a.class_name for a in self._anchor_generator]

    def assign(self, anchors_dict, gt_boxes, anchors_mask=None,
               gt_classes=None, gt_names=None, enable_similar_type=False):
        '''
            anchors_dict: {
                            'Car': {
                                      'anchors': (1, 200, 176, 2, 7),
                                      'matched_thresholds': (70400,),
                                      'unmatched_thresholds': (70400,),
                                    }
                           }
        '''

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, -1]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, -1]]

            return region_similarity_func(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        targets_list = []
        anchor_loc_idx = 0
        for class_name, anchor_dict in anchors_dict.items():

            mask = np.array([c == class_name for c in gt_names], dtype=np.bool_)  # to obtain specific class, like car
            if enable_similar_type:
                mask = np.ones(gt_names.shape, dtype=np.bool_)
                # to avoid occurence of 2,3.. in labels in create_target_np.
                gt_classes = np.ones(gt_names.shape, dtype=np.int32)

            feature_map_size = anchor_dict["anchors"].shape[:3]  # (1, 200, 176)
            num_loc = anchor_dict["anchors"].shape[-2]           # 2

            if anchors_mask is not None: # False
                anchors_mask = anchors_mask.reshape(*feature_map_size, -1)
                anchors_mask_class = anchors_mask[..., anchor_loc_idx : anchor_loc_idx + num_loc].reshape(-1)
                prune_anchor_fn = lambda _: np.where(anchors_mask_class)[0]
            else:
                prune_anchor_fn = None

            targets = create_target_np(
                anchor_dict["anchors"].reshape(-1, self.box_coder.n_dim),   # (1, 200, 176, 2, 7) -> (70400, 7)
                gt_boxes[mask],
                similarity_fn,
                box_encoding_fn,
                prune_anchor_fn=prune_anchor_fn,            # None
                gt_classes=gt_classes[mask],
                matched_threshold=anchor_dict["matched_thresholds"],
                unmatched_threshold=anchor_dict["unmatched_thresholds"],
                positive_fraction=self._positive_fraction,  # 8
                rpn_batch_size=self._sample_size,           # 512
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size,     # 7
            )
            anchor_loc_idx += num_loc
            targets_list.append(targets)

        targets_dict = {
            "labels": [t["labels"] for t in targets_list],
            "bbox_targets": [t["bbox_targets"] for t in targets_list],
            "bbox_outside_weights": [t["bbox_outside_weights"] for t in targets_list],
            "positive_gt_id": [t["positive_gt_id"] for t in targets_list],
        }
        targets_dict["bbox_targets"] = np.concatenate \
            ([ v.reshape(*feature_map_size, -1, self.box_coder.code_size) for v in targets_dict["bbox_targets"]], axis=-2, )
        targets_dict["bbox_targets"] = targets_dict["bbox_targets"].reshape(-1, self.box_coder.code_size)
        targets_dict["labels"] = np.concatenate([v.reshape(*feature_map_size, -1) for v in targets_dict["labels"]], axis=-1)
        targets_dict["bbox_outside_weights"] = np.concatenate \
            ([v.reshape(*feature_map_size, -1) for v in targets_dict["bbox_outside_weights"]], axis=-1 ,)
        targets_dict["labels"] = targets_dict["labels"].reshape(-1)
        targets_dict["bbox_outside_weights"] = targets_dict["bbox_outside_weights"].reshape(-1)

        return targets_dict

    def generate_anchors_dict(self, feature_map_size):
        '''
           This function can generate anchors based on feature_map_size with anchor_generator.generate ( actually
           box_np_ops.create_anchors_3d_range ).
        '''
        anchors_dict = {self._anchor_generator.class_name: {} }
        anchors_dict = OrderedDict(anchors_dict)
        anchors = self._anchor_generator.generate(feature_map_size)
        anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])
        num_anchors = np.prod(anchors.shape[:-1])
        match_thr = np.full(num_anchors, self._anchor_generator.match_threshold, anchors.dtype)
        unmatch_thr = np.full(num_anchors, self._anchor_generator.unmatch_threshold, anchors.dtype)
        class_name = self._anchor_generator.class_name
        anchors_dict[class_name]["anchors"] = anchors
        anchors_dict[class_name]["matched_thresholds"] = match_thr
        anchors_dict[class_name]["unmatched_thresholds"] = unmatch_thr
        return anchors_dict


    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in [self._anchor_generator]:
            num += a_generator.num_anchors_per_localization
        return num


def create_target_np(
    all_anchors,      # (70400, 7)
    gt_boxes,         # (M, 7)
    similarity_fn,
    box_encoding_fn,
    prune_anchor_fn=None,   # None
    gt_classes=None,        # (M,)
    matched_threshold=0.6,
    unmatched_threshold=0.45,
    bbox_inside_weight=None,
    positive_fraction=None, # None
    rpn_batch_size=300,     # 512
    norm_by_num_examples=False,
    box_code_size=7,
):
    """

    """
    num_anchors = all_anchors.shape[0]   # 70400
    anchors = all_anchors                # [70400, 7]
    num_inside = num_anchors             # 70400

    if gt_classes is None:
        gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)

    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_anchors,), dtype=np.int32)
    gt_ids = np.empty((num_anchors,), dtype=np.int32)
    labels.fill(-1)           # (70400,)
    gt_ids.fill(-1)           # (70400,)

    if len(gt_boxes) > 0:
        # anchors: [N, 7], gt_boxes: [M, 7]
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)      # (N, M), nearest bev iou

        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)    # (N,), get the index of the gt_box with the largest iou for each anchor, (iou=0 -> index:0)
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_anchors), anchor_to_gt_argmax]  # (N,), get the largest iou with one gt_box for each anchor
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)    # (M,), get the index of the anchor with the largest iou for each gt_box.
        gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])]  # (M,), get the largest iou with one anchor for each gt_box

        # must remove gt which doesn't match any anchor. but it should be impossible in voxelization scene.
        empty_gt_mask = gt_to_anchor_max == 0
        gt_to_anchor_max[empty_gt_mask] = -1

        # todo: Important
        # (M+m, ) Find all anchors that have the same max iou with the same gt_box, for each of gt_boxes.
        # which means sometimes one gt box may have the same iou (also maximum among all anchors) with multiple anchors.
        pos_inds_force = np.where(anchor_by_gt_overlap == gt_to_anchor_max)[0]  # (M+m,), index of positive anchors

        # fg label: for each gt, use anchors of highest overlap with it as positive targets.
        # usually one anchor can have only one gt_box with large iou. While one gt_box can have multiple anchors with large iou.
        # While I still feel there exist a little possibility that one gt box has max iou with one anchor, and this anchor has larger iou
        # with another gt_box, but this gt_box has max iou with another anchor, so the orginal gt box may not have corresponind target.
        gt_inds_force = anchor_to_gt_argmax[pos_inds_force]   # (M+m,) indices of targeted gt_boxes for the M+m positive anchors
        labels[pos_inds_force] = gt_classes[gt_inds_force]    # these anchors are labeled with `1` as positive
        gt_ids[pos_inds_force] = gt_inds_force                # these anchors are saved with corresponding gt_box indices as targets

        # fg label: above threshold IOU. Notice we use anchor_to_gt_max
        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds]
        gt_ids[pos_inds] = gt_inds

        # bg label: below threshold IOU. Notice we use anchor_to_gt_max
        bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    else:
        bg_inds = np.arange(num_anchors) # all set as background

    fg_inds = np.where(labels > 0)[0]           # indices of positive anchors
    fg_max_overlap = None
    if len(gt_boxes) > 0:
        fg_max_overlap = anchor_to_gt_max[fg_inds]  # array of max iou of one positive anchor with all gt boxes
    gt_pos_inds = gt_ids[fg_inds]

    if len(gt_boxes) == 0:
        labels[:] = 0
    else:
        labels[bg_inds] = 0
        # some gt box may have iou with anchors less than threshold and be taken as bg_inds, so needed to label re-assignment.
        # notice fg_inds and gt_ids keep unchanged for positive targets with assignment of negative labels.
        labels[pos_inds_force] = gt_classes[gt_inds_force]

    bbox_targets = np.zeros((num_anchors, box_code_size), dtype=all_anchors.dtype)   # (70400, 7)
    if len(gt_boxes) > 0:
        # see box_np_ops.second_box_encode
        bbox_targets[fg_inds, :] = box_encoding_fn(gt_boxes[anchor_to_gt_argmax[fg_inds], :],
                                                   anchors[fg_inds, :])  # targets: (num_pos_anchor, 7)
        #bbox_targets[fg_inds, :] = box_encoding_fn(gt_boxes[gt_pos_inds, :], anchors[fg_inds, :])

    bbox_outside_weights = np.zeros((num_inside,), dtype=all_anchors.dtype)

    if norm_by_num_examples:  # False
        num_examples = np.sum(labels >= 0)  # neg + pos
        num_examples = np.maximum(1.0, num_examples)
        bbox_outside_weights[labels > 0] = 1.0 / num_examples
    else:
        bbox_outside_weights[labels > 0] = 1.0

    ret = { "labels": labels,                             # [70400,]
            "bbox_targets": bbox_targets,                 # [num_pos_anchors, 7]
            "bbox_outside_weights": bbox_outside_weights, # [70400,]
            "assigned_anchors_overlap": fg_max_overlap,   # [num_pos_anchors,], max iou with each anchor among all gt boxes
            "positive_gt_id": gt_pos_inds,                # [num_pos_anchors,], index of targeted gt boxes for positive anchors
            "assigned_anchors_inds": fg_inds,}            # [num_pos_anchors, ], indices of positive anchors

    return ret


def region_similarity_func(boxes1, boxes2):
    '''nearest_iou_similarity'''
    boxes1_bv = box_np_ops.rbbox2d_to_near_bbox(boxes1)
    boxes2_bv = box_np_ops.rbbox2d_to_near_bbox(boxes2)
    ret = box_np_ops.iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
    return ret
