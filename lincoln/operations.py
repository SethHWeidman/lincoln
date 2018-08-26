import torch
from torch import Tensor

from .utils import assert_same_shape

class Operation(object):

    def __init__(self):
        raise NotImplementedError()
    
    def forward(self, input_: Tensor) -> Tensor:
        raise NotImplementedError()

    def backward(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()
        

class ParamOperation(Operation):

    def __init__(self, param: Tensor) -> Tensor:
        super().__init__()
        self.param = param
        
    def _param_grad(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()


class WeightMultiply(ParamOperation):

    def __init__(self, 
                 W: Tensor, 
                 param_name: str='W'):
        self.param = W
        self.param_name = param_name
    
    def forward(self, 
                input_: Tensor):
        self.input_ = input_

        # Lines specific to this class
        assert self.input_.shape[1] == self.param.shape[0], \
        "Mismatch of shapes in WeightMultiply operation"
        self.output = torch.mm(input_, self.param)

        return self.output

    def backward(self, 
                 output_grad: Tensor):
        assert_same_shape(self.output, output_grad)

        # Lines specific to this class        
        input_grad = torch.mm(output_grad, self.param.transpose(0, 1))
        
        self.param_grad = self._param_grad(output_grad)
        
        assert_same_shape(self.input_, input_grad)
        return input_grad
    
    def _param_grad(self, 
                    output_grad: Tensor):

        # Lines specific to this class
        param_grad = torch.mm(self.input_.transpose(0, 1), output_grad)
        
        assert_same_shape(self.param, param_grad)
        return param_grad
    

class BiasAdd(ParamOperation):

    def __init__(self, 
                 B: Tensor,
                 param_name: str='B'):
        self.param = B
        self.param_name = param_name
    
    def forward(self, 
                input_: Tensor):
        self.input_ = input_
        
        # Lines specific to this class        
        assert self.input_.shape[1] == self.param.shape[1], \
        "Mismatch of shapes in BiasAdd operation"
        self.output = torch.add(self.input_, self.param)
        
        return self.output

    def backward(self, 
                 output_grad: Tensor):
        assert_same_shape(self.output, output_grad)
        
        # Lines specific to this class 
        input_grad = torch.ones_like(self.input_) * output_grad
        
        self.param_grad = self._param_grad(output_grad)
        
        assert_same_shape(self.input_, input_grad)
        return input_grad
    
    def _param_grad(self, 
                   output_grad: Tensor):
 
        # Lines specific to this class
        param_grad = torch.ones_like(self.param) * output_grad       
        param_grad = torch.sum(param_grad, dim=0).reshape(1, param_grad.shape[1])
        
        assert_same_shape(self.param, param_grad)
        return param_grad