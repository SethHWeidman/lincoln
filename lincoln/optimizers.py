import typing
from .layers import Layer


class SGD:
    def __init__(self, network: typing.Type[Layer], lr: float = 0.003):
        self.network = network
        self.lr = lr
        
    def step(self):
        for param, grad in zip(self.network.parameters(), self.network.grads()):
            param.sub_(self.lr*grad)