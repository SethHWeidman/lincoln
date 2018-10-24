from .base import Operation, ParamOperation
from .utils import assert_same_shapes
from torch import Tensor
from typing import Dict, List, Tuple


class OperationBlock(object):

    def __init__(self) -> None:
        self.params: Dict[Tensor] = {}
        self.param_grads: Tuple[Tensor] = []
        self.ops: Dict[Operation] = {}
        self.first: bool = True

    def _setup_block(self) -> Tuple[Tensor]:
        pass

    def forward(self, *inputs) -> Tuple[Tensor]:

        if self.first:
            self._setup_block()
            self.first = False

        self.inputs = inputs

        self.inputs_with_grad = self._inputs_autograd()
        self.params_with_grad = self._params_autograd()
        self._gradify_operations()

        self.outputs = self._outputs()

        return self.outputs

    def _inputs_autograd(self) -> Tuple[Tensor]:
        inputs_with_grad = tuple(inp.detach() for inp in self.inputs)
        for inp in inputs_with_grad:
            inp.requires_grad = True
        return inputs_with_grad

    def _params_autograd(self) -> Tuple[Tensor]:
        params_with_grad = tuple(param.detach()
                                 for param in self.params.values())
        for param in params_with_grad:
            param.requires_grad = True
        return params_with_grad

    def _gradify_operations(self) -> Tuple[Tensor]:
        for op, tensor in zip([op for op in self.ops.values()
                               if issubclass(op.__class__, ParamOperation)],
                              self.params_with_grad):
            setattr(op, "param", tensor)

    def backward(self, *output_grads) -> Tuple[Tensor]:

        assert_same_shapes(self.outputs, output_grads)

        self.input_grads = self._input_grads(output_grads)

        if self.params:
            self.param_grads = self._param_grads()

        assert_same_shapes(self.inputs, self.input_grads)
        return self.input_grads

    def _input_grads(self, output_grads: Tuple[Tensor]) -> Tuple[Tensor]:

        if len(output_grads) == 1:
            self.outputs.backward(output_grads)
        else:
            for out, grad in zip(self.outputs, output_grads):
                out.backward(gradient=grad, retain_graph=True)

        input_grads = tuple()
        for inp in self.inputs_with_grad:
            input_grads = input_grads + (inp.grad,)

        return input_grads

    def _param_grads(self) -> List[Tensor]:
        return tuple(param.grad for param in self.params_with_grad)

    def _params(self) -> None:
        return tuple(param.data for param in self.params_with_grad)

    def _outputs(self) -> Tuple[Tensor]:
        raise NotImplementedError()
