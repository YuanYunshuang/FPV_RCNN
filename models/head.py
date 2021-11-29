import torch
from torch import nn
import numpy as np
from losses import build_loss
from models.utils import kaiming_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
from ops.iou3d_nms.iou3d_nms_utils import aligned_boxes_iou3d_gpu
from models import box_coders
from enum import Enum
import wandb


class MultiGroupHead(nn.Module):
    def __init__(self, cfg, pc_range):
        super(MultiGroupHead, self).__init__()

        assert cfg['with_cls'] or cfg['with_reg']

        num_classes = cfg['num_class']   # 1
        num_dirs = cfg['num_dirs']
        self.class_names = cfg['class_names']   # ['Car']
        self.num_anchor_per_loc = num_dirs * num_classes# 2

        box_coder_cfg =  cfg['box_coder'].copy()
        self.box_coder = getattr(box_coders,box_coder_cfg['type'])(**box_coder_cfg)
        box_code_size = self.box_coder.code_size * num_classes    # [8]*1

        self.with_cls = cfg['with_cls']                                 # True
        self.with_reg = cfg['with_reg']                                 # True
        self.in_channels = cfg['in_channels']                           # 128
        self.num_classes = num_classes                           # 1
        self.reg_class_agnostic = cfg['reg_class_agnostic']             # False
        self.encode_rad_error_by_sin = cfg['encode_rad_error_by_sin']   # True
        self.encode_background_as_zeros = cfg['encode_background_as_zeros']  # True
        self.use_sigmoid_score = cfg['use_sigmoid_score']               # True
        self.pred_var = cfg['pred_var']
        self.box_n_dim = self.box_coder.n_dim                # 7

        self.loss_cls = build_loss(cfg['loss_cls'])
        self.loss_reg = build_loss(cfg['loss_bbox'])
        self.loss_iou_pred = build_loss(cfg['loss_iou'])

        if cfg['loss_aux'] is not None:   # True
            self.loss_aux = build_loss(cfg['loss_aux'])

        self.loss_norm = cfg['loss_norm']

        self.use_direction_classifier = cfg['use_dir_classifier']  # WeightedSoftmaxClassificationLoss
        if self.use_direction_classifier:
            self.direction_offset = cfg['direction_offset']          # 0

        self.bev_only = True if cfg['mode'] == "bev" else False  # mode='3d' -> False

        # get output_size by calculating num_cls&num_pred(loc)&num_dir
        # 1, 2, 7
        if self.encode_background_as_zeros: # actually discard this category
            num_cls = self.num_anchor_per_loc * num_classes         # 2
        else:
            num_cls = self.num_anchor_per_loc * (num_classes + 1)  # 2

        if self.bev_only:
            num_pred = self.num_anchor_per_loc * (box_code_size - 2)
        else:
            num_pred = self.num_anchor_per_loc * box_code_size    # 14

        if self.use_direction_classifier:
            num_dir = self.num_anchor_per_loc * 2             # 4, 2 * softmax(2,)
        else:
            num_dir = None


        # it seems can add multiple head here.
        self.tasks = Head(self.in_channels, num_pred, num_cls, num_iou=self.num_anchor_per_loc,
                          use_dir=self.use_direction_classifier, pred_var=self.pred_var,
                          num_dir=num_dir if self.use_direction_classifier else None, header=False)

        self.nms_type = cfg['nms']['name']
        if self.nms_type=='normal':
            from ops.iou3d_nms.iou3d_nms_utils import nms_gpu
            self.nms = nms_gpu
        elif self.nms_type=='iou_weighted':
            from utils.box_torch_ops import rotate_weighted_nms
            self.nms = rotate_weighted_nms

        post_center_range = pc_range.copy()
        post_center_range[2] = -5
        post_center_range[-1] = 5
        self.post_center_range = torch.tensor(pc_range, dtype=torch.float).cuda()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        return self.tasks(x)

    def prepare_loss_weights(self, labels, loss_norm=None, dtype=torch.float32,):
        '''
            get weight of each anchor in each sample; all weights in each sample sum as 1.
        '''
        loss_norm_type = getattr(LossNormType, loss_norm["type"])   # norm_by_num_positives
        pos_cls_weight = loss_norm["pos_cls_weight"]                # 1.0
        neg_cls_weight = loss_norm["neg_cls_weight"]                # 1.0

        cared = labels >= 0                                         # [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(dtype) * neg_cls_weight
        positive_cls_weights = positives.type(dtype) * pos_cls_weight
        cls_weights = negative_cls_weights + positive_cls_weights
        reg_weights = positives.type(dtype)

        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.type(dtype).sum(1, keepdim=True)
            num_examples = torch.clamp(num_examples, min=1.0)
            cls_weights /= num_examples
            bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(bbox_normalizer, min=1.0)

        elif loss_norm_type == LossNormType.NormByNumPositives:           # True
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)   # [batch_size, 1]
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)           # [N, num_anchors], average in each sample;
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)           # todo: interesting, how about the negatives samples

        elif loss_norm_type == LossNormType.NormByNumPosNeg:
            pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
            normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
            cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
            cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = torch.clamp(normalizer, min=1.0)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer

        elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        else:
            raise ValueError(f"unknown loss norm type. available: {list(LossNormType)}")

        return cls_weights, reg_weights, cared

    def loss(self, batch_dict, **kwargs):
        batch_anchors = batch_dict["anchors"]   # (batch_size, 27600, 7)
        preds_dict = batch_dict['preds_dict']
        batch_size_device = batch_anchors.shape[0]  # (batch_size,)

        # get predictions.
        box_preds = preds_dict["box_preds"]        # [batch_size, 100, 138, 14]，
        cls_preds = preds_dict["cls_preds"]        # [batch_size, 100, 138, 2]，

        # get targets and weights.
        labels = batch_dict["labels"]          # cls_labels: [batch_size, 27600], elem in [-1, 0, 1].
        reg_targets = batch_dict["reg_targets"] # reg_labels: [batch_size, 27600, 7].
        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels,
                                          loss_norm=self.loss_norm, dtype=torch.float32,) # all: [batch_size, 27600]
        cls_targets = labels * cared.type_as(labels)   # filter -1 in labels.
        cls_targets = cls_targets.unsqueeze(-1)        # [batch_size, 27600, 1].

        # get localization and classification loss.
        batch_size = int(box_preds.shape[0])
        box_preds = box_preds.view(batch_size, -1, self.box_coder.code_size)   # [batch_size, 100, 138, 14] -> [batch_size, 27600, 7].
        cls_preds = cls_preds.view(batch_size, -1, self.num_classes)  # [batch_size, 27600] -> [batch_size, 27600, 1].
        # tmp = torch.sigmoid(cls_preds)[0].cpu().data.numpy()
        # wandb.log({'cls_preds': wandb.Histogram(tmp[tmp>0.1])})
        # wandb.log({'num_score_over_0.3': (tmp>0.3).sum()})
        # wandb.log({'num_pos_target': (cls_targets[0].cpu().data.numpy()>0).sum()})

        if self.encode_rad_error_by_sin:  # True
            # todo: Notice, box_preds.ry has changed to box_preds.sinacosb.
            # sin(a - b) = sinacosb-cosasinb, a: pred, b: gt;
            # box_preds: ry_a -> sinacosb; reg_targets: ry_b -> cosasinb.
            encoded_box_preds, encoded_reg_targets = add_sin_difference(box_preds, reg_targets)
        else:
            encoded_box_preds, encoded_reg_targets = box_preds, reg_targets

        loc_loss = self.loss_reg(encoded_box_preds, encoded_reg_targets, weights=reg_weights)  # [N, 27600, 7], WeightedSmoothL1Loss, averaged in sample.
        cls_loss = self.loss_cls(cls_preds, cls_targets, weights=cls_weights)  # [N, 27600, 1], SigmoidFocalLoss, averaged in sample.
        var_weight_loss = 0
        if self.pred_var:
            log_var_box_pred = preds_dict['var_box_preds'].view(loc_loss.shape)
            log_var_box_pred = torch.clamp(log_var_box_pred, min=-7, max=7)
            var_box_pred = torch.exp(-log_var_box_pred)
            loc_loss = loc_loss * var_box_pred + log_var_box_pred * reg_weights[:, :, None]
            loc_loss[reg_weights==0] = 0
            var_weight_loss = (self.tasks.conv_var.weight.abs().mean()
                               + self.tasks.conv_var.bias.abs().mean()) * 50000
            # probabilistic_loss_weight = min(1.0, self.current_step/self.annealing_step)
            # probabilistic_loss_weight = (100**probabilistic_loss_weight-1.0)/(100.0-1.0)
            # loss_box_reg = (1.0 - probabilistic_loss_weight)*standard_regression_loss + \
            #                probabilistic_loss_weight*loss_box_reg
        loc_loss_reduced = self.loss_reg._loss_weight * loc_loss.sum() / batch_size_device   # 2.0, averaged on batch_size
        cls_loss_reduced = self.loss_cls._loss_weight * cls_loss.sum() / batch_size_device   # 1.0, average on batch_size
        loss = loc_loss_reduced + cls_loss_reduced + var_weight_loss

        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)  # for analysis, average on batch
        cls_pos_loss /= self.loss_norm["pos_cls_weight"]
        cls_neg_loss /= self.loss_norm["neg_cls_weight"]

        if self.use_direction_classifier:   # True
            dir_targets = get_direction_target(batch_dict["anchors"], reg_targets, dir_offset=self.direction_offset,)  # [8, 70400, 2]
            dir_logits = preds_dict["dir_cls_preds"].view(batch_size_device, -1, 2)     # [8, 27600, 2], WeightedSoftmaxClassificationLoss.
            weights = (labels > 0).type_as(dir_logits)                                  # [8, 27600], only for positive anchors.
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)              # [8, 27600], averaged in sample.
            dir_loss = self.loss_aux(dir_logits, dir_targets, weights=weights)          # [8, 27600].
            dir_loss = dir_loss.sum() / batch_size_device                               # averaged in batch.
            loss += dir_loss * self.loss_aux._loss_weight                               # *0.2

        # for analysis.
        loc_loss_elem = [loc_loss[:, :, i].sum() / batch_size_device for i in range(loc_loss.shape[-1])]

        # for iou prediction
        iou_preds = preds_dict["iou_preds"]
        pos_pred_mask = reg_weights > 0
        iou_pos_preds = iou_preds.view(batch_size, -1, 1)[pos_pred_mask]
        qboxes = self.box_coder.decode_torch(box_preds[pos_pred_mask], batch_dict["anchors"][pos_pred_mask])
        gboxes = self.box_coder.decode_torch(reg_targets[pos_pred_mask], batch_dict["anchors"][pos_pred_mask])
        iou_weights = reg_weights[pos_pred_mask]
        iou_pos_targets = aligned_boxes_iou3d_gpu(qboxes, gboxes).detach()
        iou_pos_targets = 2 * iou_pos_targets - 1
        iou_pred_loss = self.loss_iou_pred(iou_pos_preds, iou_pos_targets, iou_weights)
        iou_pred_loss = iou_pred_loss.sum() / batch_size
        loss += iou_pred_loss


        ret = {
            "loss": loss,
            "cls_pos": cls_pos_loss.detach().cpu(),
            "cls_neg": cls_neg_loss.detach().cpu(),
            "dir_red.": dir_loss.detach().cpu() if self.use_direction_classifier else None,
            "cls_red.": cls_loss_reduced.detach().cpu().mean(),
            "loc_red.": loc_loss_reduced.detach().cpu().mean(),
            # "var_weight": var_weight_loss,
            # "loc_loss_elem": [elem.detach().cpu() for elem in loc_loss_elem],
            "iou_pred": iou_pred_loss.detach().cpu(),
            # "num_pos": (labels > 0)[0].sum(),
            # "num_neg": (labels == 0)[0].sum(),
        }

        return ret

    def post_processing(self, batch_data, test_cfg):
        preds_dict = batch_data['preds_dict']
        anchors = batch_data['anchors']
        batch_gt_boxes = batch_data['gt_boxes']
        batch_size = anchors.shape[0]
        anchors_flattened = anchors.view(batch_size, -1, self.box_n_dim)
        batch_cls_preds = preds_dict["cls_preds"].view(batch_size, -1, self.num_classes)  # [8, 70400, 1]
        batch_box_preds = preds_dict["box_preds"].view(batch_size, -1, self.box_coder.code_size)  # [batch_size, 70400, 7]
        batch_iou_preds = preds_dict["iou_preds"].view(batch_size, -1, 1)
        if self.use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"].view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        batch_reg_preds = self.box_coder.decode_torch(batch_box_preds[:, :, :self.box_coder.code_size],
                                                      anchors_flattened)

        detections = self.get_task_detections(test_cfg,
                                              batch_cls_preds, batch_reg_preds,
                                              batch_dir_preds, batch_iou_preds,
                                              batch_gt_boxes,
                                              anchors)

        return detections

    def get_task_detections(self, test_cfg, batch_cls_preds, batch_reg_preds, batch_dir_preds=None,
                            batch_iou_preds=None, batch_coop_boxes=None, batch_anchors=None):
        predictions_dicts = []
        post_center_range = self.post_center_range
        anchors = batch_anchors[0] # Anchors in all batches are the same
        if batch_coop_boxes is None:
            batch_coop_boxes = [None] * len(batch_anchors)
        batch_zip = zip(batch_reg_preds, batch_cls_preds, batch_dir_preds, batch_iou_preds, batch_coop_boxes)
        for box_preds, cls_preds, dir_preds, iou_preds, coop_boxes in batch_zip:
            # get reg and cls predictions
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()

            # get dir labels
            dir_labels = None
            if self.use_direction_classifier:
                dir_labels = torch.max(dir_preds, dim=-1)[1]

            # get scores from cls_preds
            total_scores = torch.sigmoid(cls_preds)       # [N_anchors, 1]
            top_scores = total_scores.squeeze(-1)         # [N_anchors]
            top_labels = torch.zeros([top_scores.shape[0]], dtype=torch.long).cuda()

            # SCORE_THRESHOLD: remove boxes with score lower than 0.3.
            if test_cfg['score_threshold'] > 0.0:
                thresh = test_cfg['score_threshold']
                top_scores_keep = top_scores >= thresh
                if top_scores_keep.sum() == 0:
                    print("No score over threshold.")

            # NMS: obtain remained box_preds & dir_labels & cls_labels after score threshold.
            top_scores = top_scores.masked_select(top_scores_keep)
            if top_scores.shape[0] != 0:
                # todo: confidence function.
                iou_preds = (iou_preds.squeeze() + 1) * 0.5
                top_scores *= torch.pow(iou_preds.masked_select(top_scores_keep), 4)
                if test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if self.use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                # ADD additional bboxes of cooperative vehicles to the predictions since they are known
                if coop_boxes is not None:
                    coop_boxes = coop_boxes.reshape(-1, 7)
                    box_preds = torch.cat([box_preds, coop_boxes], dim=0)
                    top_scores = torch.cat([top_scores, torch.tensor([1.0] * len(coop_boxes),
                                           device=top_scores.device)], dim=0)
                    top_labels = torch.cat([top_labels, torch.tensor([0] * len(coop_boxes),
                                           device=top_labels.device)], dim=0)
                    if self.use_direction_classifier:
                        dir_labels = torch.cat([dir_labels, torch.tensor([0] * len(coop_boxes),
                                             device=dir_labels.device)],dim=0)

                # REMOVE overlap boxes by bev rotate-nms.
                if self.nms_type=='normal':
                    box_inds = self.nms(box_preds, top_scores, test_cfg['nms_iou_threshold'],
                                        pre_max_size=test_cfg['nms_pre_max_size'])
                    box_preds = box_preds[box_inds[0]]
                    scores = top_scores[box_inds[0]]
                    label_preds = top_labels[box_inds[0]]
                    if self.use_direction_classifier:
                        dir_labels = dir_labels[box_inds[0]]
                elif self.nms_type=='iou_weighted':
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]
                    iou_preds_for_nms = torch.cat([iou_preds[top_scores_keep], torch.tensor([1.0],
                                                device=top_scores.device)], dim=0)
                    anchors_for_nms = anchors[top_scores_keep]
                    box_preds, dir_labels, label_preds, scores = self.nms(box_preds,boxes_for_nms,
                                                                      dir_labels, top_labels, top_scores,
                                                                      iou_preds_for_nms,
                                                                      anchors_for_nms,
                                                                      cnt_threshold = test_cfg['cnt_threshold'],
                                                                      pre_max_size=test_cfg['nms_pre_max_size'],
                                                                      post_max_size=test_cfg['nms_post_max_size'],
                                                                      iou_threshold=test_cfg['nms_iou_threshold'])


            else:
                box_preds = torch.zeros([0, 7], dtype=float)

            # POST-PROCESSING of predictions.
            if box_preds.shape[0] != 0:
                # move pred boxes direction by pi, eg. pred_ry < 0 while pred_dir_label > 0.
                if self.use_direction_classifier:
                    top_labels = ((box_preds[..., -1] - self.direction_offset) > 0) ^ (dir_labels.byte() == 1)
                    box_preds[..., -1] += torch.where(top_labels, torch.tensor(np.pi).type_as(box_preds),
                                                      torch.tensor(0.0).type_as(
                                                          box_preds))  # useful for dir accuracy, but has no impact on localization

                # remove pred boxes out of POST_VALID_RANGE
                mask = torch.norm(box_preds[:, :3], dim=1) < post_center_range[3]
                # mask = (box_preds[:, :3] >= post_center_range[:3]).all(1)
                # mask &= (box_preds[:, :3] <= post_center_range[3:]).all(1)
                predictions_dict = {"box_lidar": box_preds[mask],
                                    "scores": scores[mask],
                                    "label_preds": label_preds[mask]}
            else:
                # TODO: what can zero_pred can be used for? and how to eval zero results?
                dtype = batch_reg_preds.dtype
                device = batch_reg_preds.device
                predictions_dict = {
                    "box_lidar": torch.zeros([0, self.box_n_dim], dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros([0], dtype=top_labels.dtype, device=device),
                }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts


class Head(nn.Module):
    def __init__(self, num_input, num_pred, num_cls, num_iou=2, use_dir=False, pred_var=True, num_dir=0,
                 header=True, name="", focal_loss_init=False):
        super(Head, self).__init__()
        self.use_dir = use_dir
        self.pred_var = pred_var

        self.conv_box = nn.Conv2d(num_input, num_pred, 1)  # 128 -> 14
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)   # 128 -> 2
        self.conv_iou = nn.Conv2d(in_channels=num_input, out_channels=num_iou, kernel_size=1,
                                  stride=1, padding=0, bias=False)

        if self.use_dir:
            self.conv_dir = nn.Conv2d(num_input, num_dir, 1)  # 128 -> 4
        if self.pred_var:
            self.conv_var = nn.Conv2d(num_input, num_pred, 1)
            torch.nn.init.normal_(self.conv_var.weight, mean=0, std=0.0001)
            torch.nn.init.constant_(self.conv_var.bias, 0)

    def forward(self, x):                                              # x.shape=[8, 128, 200, 176]
        box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()      # box_preds.shape=[8, 200, 176, 14]
        cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()      # cls_preds.shape=[8, 200, 176, 2]
        ret_dict = {"box_preds": box_preds, "cls_preds": cls_preds}
        if self.use_dir:
            dir_preds = self.conv_dir(x).permute(0, 2, 3, 1).contiguous()  # dir_preds.shape=[8, 200, 176, 4]
            ret_dict["dir_cls_preds"] = dir_preds
        else:
            ret_dict["dir_cls_preds"] = torch.zeros((len(box_preds), 1, 2))
        if self.pred_var:
            var_preds = self.conv_var(x).permute(0, 2, 3, 1).contiguous()
            ret_dict["var_box_preds"] = var_preds
        else:
            ret_dict["var_box_preds"] = torch.zeros(len(box_preds))

        ret_dict["iou_preds"] = self.conv_iou(x).permute(0, 2, 3, 1).contiguous()

        return ret_dict


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])   # ry -> sin(pred_ry)*cos(gt_ry)
    rad_gt_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])     # ry -> cos(pred_ry)*sin(gt_ry)
    res_boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    res_boxes2 = torch.cat([boxes2[..., :-1], rad_gt_encoding], dim=-1)
    return res_boxes1, res_boxes2


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [Nde'f, num_anchors, num_class], it has been averaged on each sample
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size  # averaged on batch
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    '''
        Tensor: [4, 70400], elem is in range(depth), like 0 or 1 for depth=2, which means index of label 1 in last \
                dimension of tensor_onehot;
        tensor_onehot.scatter_(dim, index_matrix, value): dim mean target dim of tensor_onehot to pad value, index_matrix
                has the shape of tensor_onehot.shape[:-1], denoting index of label 1 in the target dim.
    '''
    tensor_onehot = torch.zeros(*list(tensor.shape), depth, dtype=dtype, device=tensor.device) # [4, 70400, 2]
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                        # [4, 70400, 2]
    return tensor_onehot


def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0.0):
    '''
        Paras:
            anchors: [batch_size, w*h*num_anchor_per_pos, anchor_dim], [4, 70400, 7];
            reg_targets: same shape as anchors, [4, 70400, 7] here;
        return:
            dir_clas_targets: [batch_size, w*h*num_anchor_per_pos, 2]
    '''
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., -1] + anchors[..., -1]           # original ry, [4, 70400]
    dir_cls_targets = ((rot_gt - dir_offset) > 0).long()       # [4, 70400], elem: 0 or 1, todo: physical scene
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets