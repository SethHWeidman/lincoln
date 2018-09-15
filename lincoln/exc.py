from torch import Tensor

""" Module to hold exceptions, errors, etc. """

class MatchError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

class DimensionError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

class BackwardError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

def assert_same_shape(output: Tensor, 
                      output_grad: Tensor):
    assert output.shape == output_grad.shape, \
    '''
    Two tensors should have the same shape; instead, first Tensor's shape is {0}
    and second Tensor's shape is {1}.
    '''.format(tuple(output_grad.shape), tuple(output.shape))
    return None