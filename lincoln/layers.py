import typing
from typing import List, Tuple, Dict

import torch
from torch import Tensor

from .operations.activations import Sigmoid
from .operations.base import Operation, ParamOperation
from .operations.operations import WeightMultiply, BiasAdd
from .operations.conv import Conv2D_Op, Conv2D_Op_cy, Conv2D_Op_Pyt
from .operations.reshape import Flatten
from .utils import assert_same_shape, assert_dim



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

        assert_same_shape(self.output, output_grad)

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        assert_same_shape(self.input_, input_grad)

        self._param_grads()

        return input_grad

    def _param_grads(self) -> Tensor:

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)


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
    Once we define all the Operations and the outline of a layer, all that remains to implement here
    is the _setup_layer function!
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
