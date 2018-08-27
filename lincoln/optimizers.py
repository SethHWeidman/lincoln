import typing
from torch import Tensor
from .layers import Layer


class Optimizer(object):
    def __init__(self):
        pass
        
    def step(self, layer: Layer) -> None:
        assert len(layer.params) == len(layer.param_grads)
        for param, grad in zip(layer.params, layer.param_grads):
            self._update_rule(param, grad)
            
    def _update_rule(self, param: Tensor, grad: Tensor):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self,  
                 lr: float = 0.003) -> None:
        super().__init__()
        self.lr = lr
        
    def _update_rule(self, param: Tensor, grad: Tensor) -> None:
        param.sub_(self.lr*grad)