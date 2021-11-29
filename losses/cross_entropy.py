from torch import nn
from losses.loss_base import Loss
from losses.utils import *
import torch.nn.functional as F


class WeightedSigmoidClassificationLoss(nn.Module):
    """Sigmoid cross entropy classification loss function."""

    def __init__(self, logit_scale=1.0, loss_weight=1.0, name=""):
        """Constructor.

        Args:
        logit_scale: When this value is high, the prediction is "diffused" and
                    when this value is low, the prediction is made peakier.
                    (default 1.0)

        """
        super(WeightedSigmoidClassificationLoss, self).__init__()
        self.name = name
        self._loss_weight = loss_weight   # 0.2
        self._logit_scale = logit_scale   # default: 1.0

    def forward(self, prediction_tensor, target_tensor, weights, class_indices=None):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]
        class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        weights = weights.unsqueeze(-1)
        if class_indices is not None:
            weights *= (
                indices_to_dense_vector(class_indices, prediction_tensor.shape[2])
                .view(1, 1, -1)
                .type_as(prediction_tensor)
            )
        per_entry_cross_ent = sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor
        )
        return per_entry_cross_ent * weights


class WeightedSigmoidBinaryCELoss(nn.Module):
    """Sigmoid cross entropy classification loss function."""

    def __init__(self, logit_scale=1.0, loss_weight=1.0, name=""):
        """Constructor.

        Args:
        logit_scale: When this value is high, the prediction is "diffused" and
                    when this value is low, the prediction is made peakier.
                    (default 1.0)

        """
        super(WeightedSigmoidBinaryCELoss, self).__init__()
        self.name = name
        self._loss_weight = loss_weight   # 0.2
        self._logit_scale = logit_scale   # default: 1.0

    def forward(self, prediction_tensor, target_tensor, weights=None, class_indices=None):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]
        class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        if weights is not None:
            weights = weights.unsqueeze(-1)
        if class_indices is not None:
            weights *= (
                indices_to_dense_vector(class_indices, prediction_tensor.shape[2])
                .view(1, 1, -1)
                .type_as(prediction_tensor)
            )
        per_entry_cross_ent = F.binary_cross_entropy_with_logits(prediction_tensor, target_tensor, weights)
        return per_entry_cross_ent


class WeightedSoftmaxClassificationLoss(nn.Module):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0, loss_weight=1.0, name=""):
        """Constructor.

        Args:
        logit_scale: When this value is high, the prediction is "diffused" and
                    when this value is low, the prediction is made peakier.
                    (default 1.0)

        """
        super(WeightedSoftmaxClassificationLoss, self).__init__()
        self.name = name
        self._loss_weight = loss_weight   # 0.2
        self._logit_scale = logit_scale   # default: 1.0

    def forward(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors, num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors, num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
            loss: a float tensor of shape [batch_size, num_anchors] representing the value of the loss function.
        """
        num_classes = prediction_tensor.shape[-1]     # 2
        prediction_tensor = torch.div(prediction_tensor, self._logit_scale)
        per_row_cross_ent = softmax_cross_entropy_with_logits(labels=target_tensor.view(-1, num_classes), logits=prediction_tensor.view(-1, num_classes),)  # mix all voxels in a batch

        return per_row_cross_ent.view(weights.shape) * weights