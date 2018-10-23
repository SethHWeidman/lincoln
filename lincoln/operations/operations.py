import torch
from torch import Tensor

from ..utils import assert_same_shape, assert_dim
from .base import Operation, ParamOperation


class WeightMultiply(ParamOperation):

    def __init__(self, W: Tensor):
        super().__init__(W)

    def _output(self) -> Tensor:
        return torch.mm(self.input, self.param)

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return torch.mm(output_grad, self.param.transpose(0, 1))

    def _param_grad(self, output_grad: Tensor) -> Tensor:
        return torch.mm(self.input.transpose(0, 1), output_grad)


class BiasAdd(ParamOperation):

    def __init__(self,
                 B: Tensor):
        super().__init__(B)


    def _output(self) -> Tensor:
        return torch.add(self.input, self.param)


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return torch.ones_like(self.input) * output_grad


    def _param_grad(self, output_grad: Tensor) -> Tensor:
        param_grad = torch.ones_like(self.param) * output_grad
        return torch.sum(param_grad, dim=0).reshape(1, param_grad.shape[1])
