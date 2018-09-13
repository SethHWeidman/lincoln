import torch
from torch import Tensor

from .utils import assert_same_shape

class Operation(object):

    def __init__(self):
        pass


    def forward(self, input_: Tensor):
        self.input_ = input_

        self.output = self._output()

        return self.output


    def backward(self, output_grad: Tensor) -> Tensor:

        assert_same_shape(self.output, output_grad)

        self._compute_grads(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad


    def _compute_grads(self, output_grad: Tensor) -> Tensor:
        self.input_grad = self._input_grad(output_grad)


    def _output(self) -> Tensor:
        raise NotImplementedError()


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()


class ParamOperation(Operation):

    def __init__(self, param: Tensor) -> Tensor:
        super().__init__()
        self.param = param


    def _compute_grads(self, output_grad: Tensor) -> Tensor:
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)


    def _param_grad(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()


class WeightMultiply(ParamOperation):

    def __init__(self, W: Tensor):
        super().__init__(W)


    def _output(self) -> Tensor:
        return torch.mm(self.input_, self.param)


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return torch.mm(output_grad, self.param.transpose(0, 1))


    def _param_grad(self, output_grad: Tensor) -> Tensor:
        return torch.mm(self.input_.transpose(0, 1), output_grad)


class BiasAdd(ParamOperation):

    def __init__(self,
                 B: Tensor):
        super().__init__(B)


    def _output(self) -> Tensor:
        return torch.add(self.input_, self.param)


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return torch.ones_like(self.input_) * output_grad


    def _param_grad(self, output_grad: Tensor) -> Tensor:
        param_grad = torch.ones_like(self.param) * output_grad
        return torch.sum(param_grad, dim=0).reshape(1, param_grad.shape[1])
