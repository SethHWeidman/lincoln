import typing
from typing import List, Tuple, Dict

import torch
from torch import Tensor

from .operations import Operation, ParamOperation, WeightMultiply, BiasAdd
from .activations import Activation, LinearAct
from .exc import MatchError, DimensionError
from .utils import assert_same_shape


class Layer(object):

    def __init__(self, 
                 neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: List[Tensor] = []
        self.param_grads: List[Tensor] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:
        pass
        
    def forward(self, input_: Tensor) -> Tensor:
        if self.first:
            self._setup_layer(input_.shape[1])
            self.first = False            
        self.input_ = input_
        
        for operation in self.operations:

            input_ = operation.forward(input_)
            
        self.output = input_

        return self.output

    def backward(self, output_grad: Tensor) -> Tensor:
        
        assert_same_shape(self.output, output_grad)
        
        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)
            
        input_grad = output_grad
        
        assert_same_shape(self.input_, input_grad)        
        
        self._param_grads()
        
        return input_grad
      
    def _param_grads(self) -> Tensor:

        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)      

class Dense(Layer):
    '''
    Once we define all the Operations and the basi.
    '''
    def __init__(self, 
                 neurons: int, 
                 activation: Activation = LinearAct) -> None:
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, num_in: int) -> None:
        # weights
        self.params.append(torch.empty(num_in, self.neurons).uniform_(-1, 1))
        
        # bias
        self.params.append(torch.empty(1, self.neurons).uniform_(-1, 1))
        
        self.operations = [WeightMultiply(self.params[0]), 
                           BiasAdd(self.params[1])] + [self.activation]

        
class LayerBlock(object):
    '''
    We will ultimately want another level on top of operations and Layers, for example when we get to ResNets.
    For now, I'm calling that a "LayerBlock" and defining a "NeuralNetwork" to be identical to it. 
    '''
    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers
              
    def forward(self, X_batch: Tensor) -> Tensor:
        
        X_out = X_batch
        for layer in self.layers:
            X_out = layer.forward(X_out)

        return X_out
        
    def backward(self, loss_grad: Tensor) -> Tensor:
    
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        return grad
    
    def params(self):
        for layer in self.layers:
            yield from layer.params()
        
    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads()
    
    def __iter__(self):
        return iter(self.layers)
    
    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(layer_strs) + ")"
    

from .loss import Loss, MeanSquaredError
class NeuralNetwork(LayerBlock):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self, layers: List[Layer], 
                 learning_rate: float = 0.01, 
                 loss: Loss = MeanSquaredError):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.loss = loss
        
    def forward_loss(self, 
                     X_batch: Tensor, 
                     y_batch: Tensor) -> float:
        
        prediction = self.forward(X_batch)
        return self.loss.loss_gradient(prediction, y_batch)
        
    def train_batch(self, 
                    X_batch: Tensor,
                    y_batch: Tensor) -> float:
        
        loss = self.forward_loss(X_batch, y_batch)
        
        self.backward(self.loss.loss_grad)
        
        self.update_params()
         
        return loss

    def update_params(self) -> None:

        for layer in self.layers:
            
            for param, grad in zip(layer.params, layer.param_grads):
                param.sub_(self.learning_rate * grad)
  