import numpy as np

from typing import Dict

from lincoln.autograd.tensor import Tensor
from lincoln.autograd.param import Parameter
from lincoln.autograd.activations import sigmoid


class Layer(object):

    def __init__(self,
                 neurons: int,
                 activation: sigmoid) -> None:
        self.num_hidden = neurons
        self.activation = activation
        self.first = True
        self.params: Dict[['str'], Tensor] = {}

    def _init_params(self, input_: Tensor) -> None:
        np.random.seed(self.seed)
        self.params['W'] = Parameter(input_.shape[1], self.num_hidden)
        self.params['B'] = Parameter(self.num_hidden)

    def forward(self, input_: Tensor) -> Tensor:
        if self.first:
            self._init_params(input_)
            self.first = False

        output = input_ @ self.params['W'] + self.params['B']

        return self.activation(output)

    def _params(self) -> Tensor:

        return list(self.params.values())
