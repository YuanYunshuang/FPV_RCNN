from models import anchor_generator, box_coders
from models.target_assigner import TargetAssigner
import numpy as np
from utils import box_np_ops


class AssignTarget(object):
    '''
        This func was modified for processing only one class
    '''

    def __init__(self,pc_range, **kwargs):
        # get target assigner & box_coder configs.
        self.training = True if kwargs["mode"]=="train" else False

        assigner_cfg = kwargs["cfg"]
        anchor_cfg = assigner_cfg['anchor_generator']
        anchor_cfg['anchor_ranges'] = [*pc_range[:2], -1, *pc_range[3:5], -1]
        box_coder_cfg = assigner_cfg['box_coder'].copy()  # "ground_box3d_coder"

        anchor_gen = getattr(anchor_generator, anchor_cfg['type'])(anchor_cfg)

        self.target_class_names = anchor_gen.class_name
        self.target_class_ids = [1]  # for car id
        self.enable_similar_type = assigner_cfg.get("enable_similar_type", False)
        if self.enable_similar_type:
            self.target_class_ids = [1, 2]  # for car id  # todo: addition of similar type

        # get target_assigner
        positive_fraction = assigner_cfg['sample_positive_fraction']  # None

        self.target_assigner = TargetAssigner(
            box_coder=getattr(box_coders, box_coder_cfg['type'])(**box_coder_cfg),  # "ground_box3d_coder"
            anchor_generator=anchor_gen, # nearest iou
            positive_fraction=positive_fraction,  # None
            sample_size=assigner_cfg['sample_size'],  # 512
        )

        self.out_size_factor = assigner_cfg['out_size_factor']  # 8
        self.anchor_dict = self.target_assigner.generate_anchors_dict(assigner_cfg['feature_map_size'])
        pass

    def __call__(self, data_dict):

        # get anchors: [x, y, z, w(x-axis), l(y-axis), h, ry].  [(70400, 7),]
        data_dict["anchors"] = self.anchor_dict[self.target_class_names]["anchors"].reshape([-1, 7])
        data_dict["batch_types"]["anchors"] = 'gpu_float32'

        # get gt labels of targeted classes; limit ry range in [-pi, pi].
        if self.training:
            gt_mask = np.zeros(data_dict["gt_classes"].shape, dtype=np.bool)
            for target_class_id in self.target_class_ids:
                gt_mask = np.logical_or(gt_mask, data_dict["gt_classes"] == target_class_id)

            gt_boxes = data_dict["gt_boxes"][gt_mask]
            gt_boxes[:, -1] = box_np_ops.limit_period(gt_boxes[:, -1], offset=0.5,
                                                      period=np.pi * 2)  # limit ry to [-pi, pi]

            data_dict["gt_boxes"] = gt_boxes
            data_dict["gt_classes"] = data_dict["gt_classes"][gt_mask]
            data_dict["gt_names"] = data_dict["gt_names"][gt_mask]

        # get anchor classification labels and localization regression labels
            targets_dict = self.target_assigner.assign(
                self.anchor_dict,
                data_dict["gt_boxes"],  # (x, y, z, w, l, h, r)
                anchors_mask=None,
                gt_classes=data_dict["gt_classes"],
                gt_names=data_dict["gt_names"],
                enable_similar_type=self.enable_similar_type,
            )

            data_dict.update({
                "labels": targets_dict["labels"],
                "reg_targets": targets_dict["bbox_targets"],
                "reg_weights": targets_dict["bbox_outside_weights"],
                "positive_gt_id": targets_dict["positive_gt_id"],
            })

            data_dict['batch_types'].update({
                "labels": 'gpu_int32',
                "reg_targets": 'gpu_float32',
                "reg_weights": 'gpu_float32',
                "positive_gt_id": 'cpu_none'
            })

        return data_dict