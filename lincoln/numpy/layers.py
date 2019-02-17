from typing import List

import torch
from torch import Tensor

import numpy as np

from .activations import Sigmoid, Linear
from .base import Operation, ParamOperation
from .dense import WeightMultiply, BiasAdd
from .dropout import Dropout
# from .conv import Conv2D_Op, Conv2D_Op_cy, Conv2D_Op_Pyt
# from .operations.reshape import Flatten
from ..np_utils import assert_same_shape


class Layer(object):

    def __init__(self,
                 neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, input_: np.ndarray) -> None:
        pass

    def forward(self, input_: np.ndarray,
                inference=False) -> np.ndarray:

        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        assert_same_shape(self.output, output_grad)

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        assert_same_shape(self.input_, input_grad)

        self._param_grads()

        return input_grad

    def _param_grads(self) -> None:

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:

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
                 activation: Operation = Linear(),
                 conv_in: bool = False,
                 dropout: float = 1.0,
                 weight_init: str = "standard") -> None:
        super().__init__(neurons)
        self.activation = activation
        self.conv_in = conv_in
        self.dropout = dropout
        self.weight_init = weight_init

    def _setup_layer(self, input_: np.ndarray) -> None:
        np.random.seed(self.seed)
        num_in = input_.shape[1]
        # if self.weight_init not in ["xavier", "he"]:

        if self.weight_init == "glorot":
            scale = 2/(num_in + self.neurons)
        else:
            scale = 1.0

        # weights
        self.params = []
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(num_in, self.neurons)))

        # bias
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(1, self.neurons)))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

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
            self.operations.append(Conv2D_Op_Pyt(self.params_dict[0]))
        elif self.cython:
            self.operations.append(Conv2D_Op_cy(self.params_dict[0]))
        else:
            self.operations.append(Conv2D_Op(self.params_dict[0]))

        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(Flatten())

        return None


class BatchNorm(Layer):

    def __init__(self) -> None:
        pass

    def _setup_layer(self, input_: np.ndarray) -> None:
        obs = input_[0]

        self.aggregates = (0,
                           np.zeros_like(obs),
                           np.zeros_like(obs))

        self.params: List[float] = []
        self.params.append(0.)
        self.params.append(1.)

    def _update_stats(self, new_input: np.ndarray):

        (count, mean, M2) = self.aggregates
        count += 1
        delta = new_input - mean
        mean += delta / count
        delta2 = new_input - mean
        M2 += delta * delta2

        self.aggregates = (count, mean, M2)


    def forward(self, input_: np.ndarray,
                inference=False) -> np.ndarray:

        self.input_ = input_
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        if not inference:
            for obs in input_:
                self._update_stats(obs)

            self.mean = input_.mean(axis=0)
            self.var = input_.var(axis=0)
        else:
            self.mean, self.var, samp_var = finalize(self.aggregates)

        self.output = (input_ - self.mean) / (self.var + 1e-8)

        self.output *= self.params[0] # gamma
        self.output += self.params[0] # beta

        return self.output

    def backward(self,
                 output_grad: Tensor) -> Tensor:

        assert_same_shape(self.output, output_grad)

        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        dbeta = np.sum(output_grad, axis=0)
        dgamma = np.sum((self.input_ - mu) * \
                        np.sqrt((self.var + 1e-8)) * output_grad, axis=0)

        self.param_grads = [dbeta, dgamma]

        input_grad = (self.params[1] * np.sqrt(self.var + 1e-8) / N) * \
                     (N * output_grad - np.sum(output_grad, axis=0) - \
                      (self.input_ - self.mean) * (self.var + 1e-8)**(-1.0) * \
                      np.sum(output_grad * (input_ - self.mean), axis=0))

        assert_same_shape(self.input_, input_grad)

        return input_grad
