import torch
from torch import Tensor

from ..utils import assert_same_shape, assert_dim


class Operation(object):

    def __init__(self):
        pass


    def forward(self, input: Tensor):

        self.input = input

        self.output = self._output()

        return self.output


    def backward(self, output_grad: Tensor) -> Tensor:

        assert_same_shape(self.output, output_grad)

        self._compute_grads(output_grad)

        assert_same_shape(self.input, self.input_grad)
        return self.input_grad


    def _compute_grads(self, output_grad: Tensor) -> Tensor:

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input, self.input_grad)
        return self.input_grad

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


class PyTorchOperation(ParamOperation):

    def __init__(self, param: Tensor) -> Tensor:
        super().__init__(param)
        self.op = nn.Linear(param.shape[0],
                            param.shape[0])


    def _output(self) -> Tensor:

        self.input_with_grad = self.input.detach()
        self.input_with_grad.requires_grad = True

        return self.op(self.input_with_grad)


    def _input_grad(self, output_grad: Tensor) -> Tensor:

        self.output.backward(gradient=output_grad)

        return self.input_with_grad.grad

    def _param_grad(self, output_grad: Tensor) -> Tensor:

        return self.op.weight.grad
