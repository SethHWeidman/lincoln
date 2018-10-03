from typing import Tuple

from torch import Tensor
from .layers import Layer


class Optimizer(object):
    def __init__(self,
                 net: NeuralNetwork):
        self.net = net
        self.params = net._params()


    def step(self) -> None:
        self.param_grads = self._param_grads()
        for i, (param, param_grad) in zip(self.params,
                                          self.param_grads):
            self._update_rule(param, param_grad)


    def _update_rule(self, *args: Tuple[Tensor]) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self,
                 net: NeuralNetwork,
                 lr: float = 0.003) -> None:
        super().__init__(net)
        self.lr = lr


    def _update_rule(self, *args: Tuple[Tensor]) -> None:
        param.sub_(self.lr*grad)


class SGDMomentum(Optimizer):
    def __init__(self,
                 net: NeuralNetwork,
                 lr: float = 0.003,
                 momentum: float = 0.9) -> None:
        super().__init__(net)
        self.lr = lr
        self.velocities = [torch.zeros_like(param) for param in self.params]



    def step(self, net: NeuralNetwork) -> None:

        for i, (param, param_grad, velocity) in zip(self.params,
                                                    self.param_grads,
                                                    self.velocities):
            self._update_rule(param, param_grad, velocity)


    def _update_rule(self, *args: Tuple[Tensor]) -> None:

            # Update velocity
            velocity.mul_(self.momentum).add_(self.lr * param_grad)

            # Use this to update parameters
            param.sub_(velocity)
