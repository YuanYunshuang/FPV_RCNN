
import torch
from enum import Enum
from losses.focal_loss import SigmoidFocalLoss
from losses.smooth_l1_loss import WeightedSmoothL1Loss
from losses.cross_entropy import WeightedSigmoidClassificationLoss, \
                                 WeightedSoftmaxClassificationLoss, \
                                 WeightedSigmoidBinaryCELoss

__all__={
    'SigmoidFocalLoss',
    'WeightedSmoothL1Loss',
    'WeightedSigmoidClassificationLoss',
    'WeightedSoftmaxClassificationLoss',
    'WeightedSigmoidBinaryCELoss'
}
loss_dict = {
    'SigmoidFocalLoss': SigmoidFocalLoss,
    'WeightedSmoothL1Loss': WeightedSmoothL1Loss,
    'WeightedSigmoidClassificationLoss': WeightedSigmoidClassificationLoss,
    'WeightedSoftmaxClassificationLoss': WeightedSoftmaxClassificationLoss,
    'WeightedSigmoidBinaryCELoss': WeightedSigmoidBinaryCELoss
}

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"


def build_loss(cfg):
    loss_cfg = cfg.copy()
    loss_module = loss_dict[loss_cfg.pop('type')]

    return loss_module(**loss_cfg)


def prepare_loss_weights(labels, loss_norm=None, dtype=torch.float32,):
    '''
        get weight of each anchor in each sample; all weights in each sample sum as 1.
    '''
    loss_norm_type = getattr(LossNormType, loss_norm["type"])   # norm_by_num_positives
    pos_cls_weight = loss_norm["pos_cls_weight"]
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


