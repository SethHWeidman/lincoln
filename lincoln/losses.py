import typing

import torch
from torch import Tensor

from .layers import Layer
from .exc import MatchError


class Loss:
    """ Base class for losses """
    def __init__(self, network: typing.Type[Layer]):
        self.network = network
    
    def forward(self, input: Tensor, targets: Tensor) -> float:
        raise NotImplementedError()

    def gradient(self) -> Tensor:
        """ Gradient of the loss with respect to it's inputs """
        raise NotImplementedError()

    def backward(self) -> Tensor:
        backward_grad = self.gradient()
        # Calculate gradients for the network
        grad = self.network.backward(backward_grad)
        return grad
        
    def __call__(self, input: Tensor, targets: Tensor) -> float:
        return self.forward(input, targets)


class LogLoss(Loss):
    """ Log loss error specifically for logistic regression, requires a sequence of layers as input """
    def __init__(self, network: typing.Type[Layer], eta=1e-9):
        super().__init__(network)
        
        # Small parameter to avoid explosions when our probabilities get near 0 or 1
        # A better way to do this is use log probabilities everywhere
        self.eta = eta
        
    def forward(self, features: Tensor, labels: Tensor) -> float:
        
        self.last_input = p = self.network(features)
        self.labels = y = labels
        
        loss = torch.sum(-y*torch.log(p + self.eta) - (1-y)*torch.log(1 - p + self.eta))
        return loss.item()
    
    def gradient(self) -> Tensor:
        """ Gradient of the loss with respect to it's inputs """
        y, p = self.labels, self.last_input
        N = y.shape[0]
        
        backward_grad = torch.sum(-y/(p + self.eta) + (1-y)/(1 - p  + self.eta), dim=1).view(N, -1)
        
        return backward_grad

    def __repr__(self):
        return f"LogLoss"
    
    
class LogSigmoidLoss(Loss):
    def __init__(self, network):
        super().__init__(network)
        
    def forward(self, input: Tensor, labels: Tensor) -> float:
        
        # Here we're assuming z is a log-probability
        self.last_input = z = self.network(input)
        self.labels = y = labels
        
        loss = torch.sum(-y*z - (1-y)*torch.log(1-torch.exp(z)))
        return loss.item()
    
    def gradient(self) -> Tensor:
        y, z = self.labels, self.last_input
        n = y.shape[0]
        exp_z = torch.exp(z)
        
        grad = torch.sum(-y + (1-y)*exp_z/(1 - exp_z), dim=1).view(n, -1)
        
        return grad