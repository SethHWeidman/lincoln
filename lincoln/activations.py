import torch
from torch import Tensor

from .operations import Operation
from .utils import assert_same_shape


class Activation(Operation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        pass


class Sigmoid(Activation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, 
                input_: Tensor) -> Tensor:
        
        self.input_ = input_

        # Lines specific to this class
        self.output = 1.0/(1.0+torch.exp(-1.0 * input_))
        
        return self.output

    def backward(self, 
                 output_grad: Tensor) -> Tensor:

        assert_same_shape(self.output, output_grad)           
        
        # Lines specific to this class
        sigmoid_backward = self.output*(1.0-self.output)
        input_grad = sigmoid_backward * output_grad
        
        assert_same_shape(self.input_, input_grad)
        
        return input_grad


class LinearAct(Activation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, 
                input_: Tensor) -> Tensor:
        
        self.input_ = input_
        
        # Lines specific to this class        
        self.output = input_
        
        return self.output

    def backward(self, 
                 output_grad: Tensor) -> Tensor:

        assert_same_shape(self.output, output_grad)           

        # Lines specific to this class
        input_grad = output_grad
        
        assert_same_shape(self.input_, input_grad)
        
        return input_grad
    

class LogSigmoid(Activation):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_: Tensor) -> Tensor:
       
        self.input_ = input_

        # Lines specific to this class
        self.output = input_ - torch.log(torch.exp(input_) + 1)

        return self.output
    
    def backward(self, output_grad: Tensor) -> Tensor:
        
        if not hasattr(self, 'output'):
            message = "The forward method must be run before the backward method"
            raise lnc.exc.BackwardError(message)  
        assert_same_shape(self.output, output_grad) 

        # Lines specific to this class
        input_grad = (1 - torch.exp(self.output))*output_grad
        
        assert_same_shape(self.input_, input_grad)        
        return input_grad