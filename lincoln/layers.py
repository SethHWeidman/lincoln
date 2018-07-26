import typing
from typing import List, Tuple

import torch
from torch import Tensor

from lincoln.exc import MatchError, DimensionError


class Layer(object):
    '''
    Defining basic functions that all classes inheriting from Layer must implement.
    '''

    def __init__(self):
        pass

    def forward(self, input):
        raise NotImplementedError()

    def backward(self, output_grad):
        raise NotImplementedError()
        
    def parameters(self):
        yield from ()
    
    def grads(self):
        yield from ()
        
    def __call__(self, input):
        return self.forward(input)


class Sequential(Layer):
    
    def __init__(self, *layers: typing.Type[Layer]):
        super().__init__()
        self.layers = tuple(layers)
              
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def backward(self, grad: Tensor = None) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters()
        
    def grads(self):
        for layer in self.layers:
            yield from layer.grads()
    
    def __iter__(self):
        return iter(self.layers)
    
    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(layer_strs) + ")"


class Linear(Layer):

    def __init__(self, size: int) -> None:

        super().__init__()
        self.size = size
        self.first = True
        self.parameters_ = {'W': None, 'B': None}
        self.grads_ = {'W': None, 'B': None}
    
    def forward(self, input: Tensor) -> Tensor:
        """ Takes a tensor and performs a linear (affine) transformation """
        
        if input.dim() != 2:
            raise DimensionError(f"Tensor should have dimension 2, instead it has dimension {input.dim()}")

        self.last_input = input
        
        # Sets up the weights on the first iteration. Doing this so the
        # input size isn't defined until we pass in our first tensor
        if self.first:
            n_input = input.size()[1]
            
            # Intialize a 2D tensor for the weights
            self.W = torch.randn((n_input, self.size))*0.01
            # Register the weight parameter
            self.parameters_.update({'W': self.W})
            
            # Intialize the bias terms (one for each output value)
            self.B = torch.randn((1, self.size))*0.01
            # Register the bias parameter
            self.parameters_.update({'B': self.B})
            
            self.first = False
        
        # The linear transformation here
        self.output = torch.mm(self.last_input, self.W) + self.B
        
        return self.output

    def backward(self, in_grad: Tensor) -> Tensor:
        """ Takes a gradient from another operation, then calculates the gradients
            for this layer's parameters, and returns the gradient for this layer to pass
            backwards in the network
        """
        
        # Key assertion
        if self.output.shape != in_grad.shape:
            message = (f"Two tensors should have the same shape; instead, first Tensor's shape "
                       f"is {in_grad.shape} and second Tensor's shape is {self.output.shape}.")
            raise MatchError(message)
        
        # Number of examples
        n = in_grad.shape[0]
        
        # Parameter gradients
        x = self.last_input
        dW = torch.mm(x.t(), in_grad)   # dL/dW
        dB = torch.sum(in_grad, dim=0).view(*self.B.shape)   # dL/dB
        
        # Register parameter gradients
        self.grads_.update({'W': dW})
        self.grads_.update({'B': dB})
        
        # This layer's gradient which we'll pass on to previous layers, for dL/dx
        backward_grad = torch.mm(in_grad, self.W.t())
        
        # Key assertion
        if self.last_input.shape != backward_grad.shape:
            message = (f"Two tensors should have the same shape; instead, first Tensor's shape "
                       f"is {self.last_input.shape} and second Tensor's shape is {backward_grad.shape}.")
            raise MatchError(message)

        return backward_grad
    
    def parameters(self):
        for tensor in self.parameters_.values():
            yield tensor
    
    def grads(self):
        for param in self.parameters_:
            yield self.grads_[param]
    
    def __repr__(self):
        return f"Linear({self.size})"


class Sigmoid(Layer):
    '''
    Sigmoid activation function
    '''
    def __init__(self):
        super().__init__()
        

    def forward(self, input: Tensor) -> Tensor:
        
        self.last_input = input
        
        self.output = 1.0/(1.0+torch.exp(-1.0 * input))
        return self.output


    def backward(self, in_grad: Tensor) -> Tensor:

        # Key assertion
        if self.output.shape != in_grad.shape:
            message = (f"Two tensors should have the same shape; instead, first Tensor's shape "
                       f"is {in_grad.shape} and second Tensor's shape is {self.output.shape}.")
            raise MatchError(message)           
        
        sigmoid_backward = self.output*(1.0-self.output)
        backward_grad = sigmoid_backward * in_grad
        
        # Key assertion
        if self.last_input.shape != backward_grad.shape:
            message = (f"Two tensors should have the same shape; instead, first Tensor's shape "
                       f"is {self.last_input.shape} and second Tensor's shape is {backward_grad.shape}.")
            raise MatchError(message)
        
        return backward_grad
    

    def __repr__(self):
        return f"Sigmoid"


class Dense(Sequential):


    def __init__(self, size: int, activation: typing.Any = 'sigmoid') -> None:
        
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        else:
            self.activation = activation

        super().__init__(Linear(size), self.activation)
