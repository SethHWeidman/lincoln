import typing
from .layers import Layer, NeuralNetwork

class SGD:
    def __init__(self,  
                 lr: float = 0.003):
        self.lr = lr
        
    def step(self, network: NeuralNetwork):
        for layer in network.layers:
            for param, grad in zip(layer.params, layer.param_grads):
                param.add_(self.lr*grad)