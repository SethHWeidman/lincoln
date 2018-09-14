import torch
from torch import Tensor

from typing import Tuple


def to_2d(a: Tensor,
          type: str="col") -> Tensor:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.dim() == 1, \
    "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(tensor_size(a), 1)
    elif type == "row":
        return a.reshape(1, tensor_size(a))


def tensor_size(tensor: Tensor) -> int:
    '''
    Returns the number of elements in a 1D Tensor
    '''
    assert tensor.dim() == 1, \
    "Input tensors must be 1 dimensional"

    return list(tensor.size())[0]


def permute_data(X, y):
    perm = torch.randperm(X.shape[0])
    return X[perm], y[perm]


Batch = Tuple[Tensor, Tensor]

def generate_batch(X: Tensor,
                   y: Tensor,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:

    assert (X.dim() == 2) and (y.dim() == 2), \
    "X and Y must be 2 dimensional"

    if start+batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start:start+batch_size], y[start:start+batch_size]

    return X_batch, y_batch


def generate_batches(X: Tensor,
                     y: Tensor,
                     size: int = 32) -> Batch:

    if X.shape[0] != y.shape[0]:
        raise ValueError('feature and label arrays must have the same first dimension')

    N = X.shape[0]

    for ii in range(0, N, size):
        X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

        yield X_batch, y_batch


def assert_same_shape(output: Tensor,
                      output_grad: Tensor):
    assert output.shape == output_grad.shape, \
    '''
    Two tensors should have the same shape; instead, first Tensor's shape is {0}
    and second Tensor's shape is {1}.
    '''.format(tuple(output_grad.shape), tuple(output.shape))
    return None


def assert_dim(t: Tensor,
               dim: Tensor):
    assert len(t.shape) == dim, \
    '''
    Tensor expected to have dimension {0}, instead has dimension {1}
    '''.format(dim, len(t.shape))
    return None
