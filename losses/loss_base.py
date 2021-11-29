from abc import ABCMeta, abstractmethod
import torch

class Loss(object):
    """Abstract base class for loss functions."""

    __metaclass__ = ABCMeta

    def __call__(
        self,
        prediction_tensor,
        target_tensor,
        ignore_nan_targets=False,
        scope=None,
        **params
    ):
        """Call the loss function.

        Args:
        prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
            representing predicted quantities.
        target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
            regression or classification targets.
        ignore_nan_targets: whether to ignore nan targets in the loss computation.
            E.g. can be used if the target tensor is missing groundtruth data that
            shouldn't be factored into the loss.
        scope: Op scope name. Defaults to 'Loss' if None.
        **params: Additional keyword arguments for specific implementations of
                the Loss.

        Returns:
        loss: a tensor representing the value of the loss function.
        """

        return self._compute_loss(prediction_tensor, target_tensor, **params)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overridden by implementations.

        Args:
        prediction_tensor: a tensor representing predicted quantities
        target_tensor: a tensor representing regression or classification targets
        **params: Additional keyword arguments for specific implementations of
                the Loss.

        Returns:
        loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
            anchor
        """
        pass