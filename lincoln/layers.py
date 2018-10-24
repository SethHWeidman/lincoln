from typing import List, Tuple, Dict

import torch
from torch import Tensor

from .operations.activations import Sigmoid
from .operations.base import Operation, ParamOperation
from .operations.dense import WeightMultiply, BiasAdd
from .operations.conv import Conv2D_Op, Conv2D_Op_cy, Conv2D_Op_Pyt
from .operations.reshape import Flatten
from .utils import assert_same_shapes


class Layer(object):

    def __init__(self,
                 neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: List[Tensor] = []
        self.param_grads: List[Tensor] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:
        pass

    def forward(self, input_: Tensor) -> Tensor:
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: Tensor) -> Tensor:

        assert_same_shapes(self.output, output_grad)

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        assert_same_shapes(self.input_, input_grad)

        self._param_grads()

        return input_grad

    def _param_grads(self) -> Tensor:

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> Tensor:

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    '''
    Once we define all the Operations and the outline of a layer, all that remains to implement here
    is the _setup_layer function!
    '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid(),
                 conv_in: bool = False) -> None:
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: Tensor) -> None:

        # weights
        self.params.append(torch.empty(input_.shape[1], self.neurons).uniform_(-1, 1))

        # bias
        self.params.append(torch.empty(1, self.neurons).uniform_(-1, 1))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1])] + [self.activation]

        return None


class Conv2D(Layer):
    '''
    Once we define all the Operations and the outline of a layer,
    all that remains to implement here is the _setup_layer function!
    '''
    def __init__(self,
                 out_channels: int,
                 param_size: int,
                 activation: Operation = Sigmoid(),
                 cython: bool = False,
                 pytorch: bool = False,
                 flatten: bool = False) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.cython = cython
        self.pytorch = pytorch
        self.flatten = flatten
        self.neurons = out_channels

    def _setup_layer(self, input_: Tensor) -> Tensor:

        conv_param = torch.empty(self.neurons,
                                 input_.shape[1],
                                 self.param_size,
                                 self.param_size).uniform_(-1, 1)
        self.params.append(conv_param)

        self.operations = []

        if self.pytorch:
            self.operations.append(Conv2D_Op_Pyt(self.params[0]))
        elif self.cython:
            self.operations.append(Conv2D_Op_cy(self.params[0]))
        else:
            self.operations.append(Conv2D_Op(self.params[0]))

        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(Flatten())

        return None


# LSTMLayer class - series of operations
class LSTMLayer(object):

    def __init__(self,
                 max_len: int,
                 vocab_size: int,
                 hidden_size: int = 100):
        super().__init__()
        self.nodes = [LSTMNode(hidden_size, vocab_size) for _ in range(max_len)]
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.first: bool = True
        self.start_H: Tensor = None
        self.start_C: Tensor = None
        self.params: Dict[Tensor] = {}


    def _init_params(self, input_: Tensor) -> Tensor:
        '''
        First dimension of input_ will be batch size
        '''
        self.start_H = torch.zeros(input_.shape[0], self.hidden_size)
        self.start_C = torch.zeros(input_.shape[0], self.hidden_size)
        self.params['Wf'] = torch.rand(self.hidden_size + self.vocab_size,
                                       self.hidden_size)
        self.params['Bf'] = torch.rand(1, self.hidden_size)

        self.params['Wi'] = torch.rand(self.hidden_size + self.vocab_size,
                                       self.hidden_size)
        self.params['Bi'] = torch.rand(1, self.hidden_size)

        self.params['Wc'] = torch.rand(self.hidden_size + self.vocab_size,
                                       self.hidden_size)
        self.params['Bc'] = torch.rand(1, self.hidden_size)

        self.params['Wo'] = torch.rand(self.hidden_size + self.vocab_size,
                                       self.hidden_size)
        self.params['Bo'] = torch.rand(1, self.hidden_size)

        self.params['Wv'] = torch.rand(self.hidden_size,
                                       self.vocab_size)
        self.params['Bv'] = torch.rand(1, self.vocab_size)

        for param in self.params.values():
            param.requires_grad = True


    def _zero_param_grads(self) -> None:
        for param in self.params.values():
            if param.grad is not None:
                param.grad.data.zero_()


    def _params(self) -> Tuple[Tensor]:
        return tuple(self.params.values())


    def _param_grads(self) -> Tuple[Tensor]:
        return tuple(param.grad for param in self.params.values())


    def forward(self, input_: Tensor) -> Tensor:
        if self.first:
            self._init_params(input_)
            self.first = False

        # shape: batch size by sequence length by vocab_size
        self.input_ = input_

        H_in = torch.clone(self.start_H)
        C_in = torch.clone(self.start_C)

        self.output = torch.zeros_like(self.input_)

        seq_len = self.input_.shape[1]

        for i in range(seq_len):

            # pass info forward through the nodes
            elem_out, H_in, C_in = self.nodes[i].forward(self.params, self.input_[:, i, :],
                                                         H_in, C_in)

            self.output[:, i, :] = elem_out

        self.start_H = H_in
        self.start_C = C_in

        return self.output


    def backward(self, output_grad: Tensor) -> Tensor:

#         self._zero_param_grads()

        dH_next = torch.zeros_like(self.start_H)
        dC_next = torch.zeros_like(self.start_C)

        self.input_grad = torch.zeros_like(self.input_)

        for i in reversed(range(self.input_.shape[1])):

            # pass info forward through the nodes
            grad_out, dH_next, dC_next = \
                self.nodes[i].backward(self.params, output_grad[:, i, :],
                                       dH_next, dC_next)

            self.input_grad[:, i, :] = grad_out

        return self.input_grad
