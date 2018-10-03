import torch
from torch import Tensor

from ..utils import assert_same_shape, assert_dim

from .base import Operation


class LogSigmoid(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> Tensor:
        self.output = self.input - torch.log(torch.exp(self.input) + 1)
        return self.output

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return (1 - torch.exp(self.output))*output_grad

    def __repr__(self):
        return "LogSigmoid"


class Sigmoid(Operation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        super().__init__()


    def _output(self) -> Tensor:
        return 1.0/(1.0+torch.exp(-1.0 * self.input))


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        # Lines specific to this class
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad

    def __repr__(self):
        return "Sigmoid"


class ReLU(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> Tensor:
        self.output = torch.clamp(self.input, 0, 1e5)
        return self.output

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        relu_backward = (self.output > 0).type(self.output.dtype)
        return relu_backward * output_grad

    def __repr__(self):
        return "ReLU"
