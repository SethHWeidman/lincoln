import typing

import torch
from torch import Tensor

from .layers import Layer
from .exc import MatchError
from .utils import assert_same_shape


class Loss(object):

    def __init__(self) -> None:
        pass
    
    
    def loss_gradient(self, 
                      prediction: Tensor, 
                      target: Tensor) -> float:
        raise NotImplementedError()


class LogLoss(Loss):
    """ Log loss error specifically for logistic regression, requires a sequence of layers as input """
    def __init__(self, eta=1e-9):
        super().__init__()
        
        # Small parameter to avoid explosions when our probabilities get near 0 or 1
        # A better way to do this is use log probabilities everywhere
        self.eta = eta
        
    def loss_gradient(self, prediction: Tensor, 
                    target: Tensor) -> float:
        
        loss = torch.sum(-target*torch.log(prediction + self.eta) - \
                         (1-target)*torch.log(1 - prediction + self.eta))
        
        N = target.shape[0]
        self.loss_grad = torch.sum(-target/(prediction + self.eta) + \
                                   (1-target)/(1 - prediction  + self.eta), dim=1).view(N, -1)
        
        assert_same_shape(prediction, self.loss_grad)
        
        return loss.item()

    def __repr__(self):
        return f"LogLoss"
 
    
class LogSigmoidLoss(Loss):
    def __init__(self):
        super().__init__()
        
    def loss_gradient(self, predictions: Tensor, 
                target: Tensor) -> float:
        
        
        loss = torch.sum(-target*predictions - (1-target)*torch.log(1-torch.exp(predictions)))

        exp_z = torch.exp(predictions)
        N = target.shape[0]        
        self.loss_grad = torch.sum(-target + (1-target)*exp_z/(1 - exp_z), dim=1).view(N, -1)
        
        assert_same_shape(prediction, self.loss_grad)
        
        return loss.item()
    

class MeanSquaredError(Loss):

    def __init__(self) -> None:
        pass
    
    
    def loss_gradient(self, 
                      prediction: Tensor, 
                      target: Tensor) -> Tensor:
        loss = torch.sum(torch.pow(prediction - target, 2))
        
        loss_grad = -2.0 * torch.add(target, -1.0 * prediction)
        
        self.loss_grad = loss_grad
        
        assert_same_shape(prediction, loss_grad)
        
        return loss 