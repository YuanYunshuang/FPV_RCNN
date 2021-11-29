import torch
from torch import nn


class WeightedSmoothL1Loss(nn.Module):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as 0.5*x^2 if |x|<1 and |x|-0.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, reduction="none", code_weights=None, codewise=True, loss_weight=1.0,):
        super(WeightedSmoothL1Loss, self).__init__()

        # if code_weights is not None:
        #     self._code_weights = torch.tensor(code_weights, dtype=torch.float32)
        # else:
        #     self._code_weights = None

        self._sigma = sigma               # 3
        self._code_weights = None if code_weights is None else torch.tensor(code_weights)
        self._codewise = codewise         # True
        self._reduction = reduction       # mean
        self._loss_weight = loss_weight   # 2.0 here

    def forward(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.
            Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors, code_size] representing the (encoded) predicted locations of objects.
            target_tensor: A float tensor of shape [batch_size, num_anchors, code_size] representing the regression targets
            weights: a float tensor of shape [batch_size, num_anchors]

            Returns:
            loss: a float tensor of shape [batch_size, num_anchors] tensor representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:   # False: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], xyzhwlr are equal
            diff = self._code_weights.view(1, 1, -1).to(diff.device) * diff

        # this sml1: 0.5*(3x)^2 if |x|<1/3^2 otherwise |x|-0.5/3^2
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(abs_diff)   # compare elements in abs_diff with 1/9, less -> 1.0, otherwise -> 0.0

        # todo???: why 1/9
        # if abs_diff_lt_1 = 1 (abs_diff < 1/9), loss = 0.5 * 9 * (abs_diff)^2, when abs_diff=1/9, loss=0.5/9
        # else if abs_diff_lt=0, (abs_diff > 1/9), loss = abs_diff - (0.5/9), when abs_diff=1/9, loss=0.5/9
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) + \
               (abs_diff - 0.5 / (self._sigma ** 2)) * (1.0 - abs_diff_lt_1)

        if self._codewise:    # True
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1) # pos_anchors multiply the weight: 1/num_pos_anchor in each sample
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)  #  * weights
            if weights is not None:
                anchorwise_smooth_l1norm *= weights
        if self._reduction=='mean':
            anchorwise_smooth_l1norm = anchorwise_smooth_l1norm.sum() / weights.sum() / anchorwise_smooth_l1norm.shape[-1]
        return anchorwise_smooth_l1norm


class SmoothL1Loss(nn.Module):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as 0.5*x^2 if |x|<1 and |x|-0.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, reduction="mean", loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()

        # if code_weights is not None:
        #     self._code_weights = torch.tensor(code_weights, dtype=torch.float32)
        # else:
        #     self._code_weights = None

        self._sigma = sigma               # 3
        self._reduction = reduction       # mean
        self._loss_weight = loss_weight   # 10.0 here

    def forward(self, source, target, mask):
        """Compute loss function.
            Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors, code_size] representing the (encoded) predicted locations of objects.
            target_tensor: A float tensor of shape [batch_size, num_anchors, code_size] representing the regression targets
            weights: a float tensor of shape [batch_size, num_anchors]

            Returns:
            loss: a float tensor of shape [batch_size, num_anchors] tensor representing the value of the loss function.
        """
        d = source.shape[1]
        mask = mask.view(-1, 2).contiguous().any(dim=1)
        prediction_tensor = source.view(1, d, -1)[:, :, mask]
        target_tensor = target.view(1, d, -1)[:, :, mask]
        diff = prediction_tensor - target_tensor

        # this sml1: 0.5*(3x)^2 if |x|<1/3^2 otherwise |x|-0.5/3^2
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(abs_diff)   # compare elements in abs_diff with 1/9, less -> 1.0, otherwise -> 0.0

        # todo???: why 1/9
        # if abs_diff_lt_1 = 1 (abs_diff < 1/9), loss = 0.5 * 9 * (abs_diff)^2, when abs_diff=1/9, loss=0.5/9
        # else if abs_diff_lt=0, (abs_diff > 1/9), loss = abs_diff - (0.5/9), when abs_diff=1/9, loss=0.5/9
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) + \
               (abs_diff - 0.5 / (self._sigma ** 2)) * (1.0 - abs_diff_lt_1)

        smooth_l1norm = loss.mean() * self._loss_weight

        return smooth_l1norm