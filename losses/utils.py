import torch
import numpy as np

def indices_to_dense_vector(
    indices, size, indices_value=1.0, default_value=0, dtype=np.float32
):
    """Creates dense vector with indices set to specific value and rest to zeros.

    This function exists because it is unclear if it is safe to use
        tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

    Args:
        indices: 1d Tensor with integer indices which are to be set to
            indices_values.
        size: scalar with size (integer) of output Tensor.
        indices_value: values of elements specified by indices in the output vector
        default_value: values of other elements in the output vector.
        dtype: data type.

    Returns:
        dense 1D Tensor of shape [size] with indices set to indices_values and the
            rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense


def sigmoid_cross_entropy_with_logits(logits, labels):
    '''
       logits: y'', labels: y
       sigmoid: y' = 1 / (1 + e^(-y''))
       cross_entropy:
                   -y*logy' - (1-y)*log(1-y') = -y*log(1 / (1 + e^(-y''))) - (1-y)*log(1 - 1 / (1 + e^(-y'')))
                                              = y*log(1 + e^(-y'')) - (1-y)*log(e^(-y'')/1+e^(-y''))
                                              = y*log(1 + e^(-y'')) - (1-y)*(-y''-log(1+e^(-y'')))
                                              = y'' - y*y'' + log(1 + e^(-y''))
            to avoid overflow of e^(-y'') when y'' < 0, we can get =>
                        = max(y'', 0) - y''*y + log(1 + e^(-abs(y'')))  # this code
                            | y'' - y*y'' + log(1 + e^(-y'')) if y'' > 0,
                        =
                            | -y*y'' + log(1 + e^(y'')) = -y*y'' + log(1 + e^(-y'')) + log(e^y'')
    '''
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss


def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))             # [0, 1, 2]
    transpose_param = [0] + [param[-1]] + param[1:-1]  # [0, 2, 1]
    logits = logits.permute(*transpose_param)          # [N, ..., C] -> [N, C, ...]: [8, 70400, 2] -> [8, 2, 70400]
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])    # logits: [8, 2, 70400]; labels: [8, 70400, 2]; max: [8, 70400]  -> loss: [8, 70400]
    return loss