import torch
from torch import Tensor

from .utils import assert_same_shape

class Operation(object):

    def __init__(self):
        pass
    

    def forward(self, 
                input_: Tensor):
        self.input_ = input_
        
        self.output = self._compute_output()

        return self.output
    

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        assert_same_shape(self.output, output_grad)       
        
        input_grad = self._compute_grads(output_grad)
               
        assert_same_shape(self.input_, input_grad)
        return input_grad
    

    def backward(self, output_grad: Tensor) -> Tensor:
        return self._input_grad(output_grad)

    def _compute_output(self, input_: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def _compute_grads(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()
        

class ParamOperation(Operation):

    def __init__(self, param: Tensor) -> Tensor:
        super().__init__()
        self.param = param
        
    def backward(self, output_grad: Tensor) -> Tensor:
        
        self.param_grad = self._param_grad(output_grad)
        
        assert_same_shape(self.param, self.param_grad)
        
        return self._input_grad(output_grad)
        
    def _param_grad(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()


class WeightMultiply(ParamOperation):

    def __init__(self, 
                 W: Tensor):
        super().__init__(W)
    
    def _compute_output(self):
        return torch.mm(self.input_, self.param)
    
    def _compute_grads(self, output_grad):
        return torch.mm(output_grad, self.param.transpose(0, 1))
    
    def _param_grad(self, 
                    output_grad: Tensor):
        
        return torch.mm(self.input_.transpose(0, 1), output_grad)
    

class BiasAdd(ParamOperation):

    def __init__(self, 
                 B: Tensor):
        super().__init__(B)

    def _compute_output(self):
        return torch.add(self.input_, self.param)      

    def _compute_grads(self, output_grad):
        return torch.ones_like(self.input_) * output_grad   

    def _param_grad(self, 
                    output_grad: Tensor):
        param_grad = torch.ones_like(self.param) * output_grad       
        return torch.sum(param_grad, dim=0).reshape(1, param_grad.shape[1])