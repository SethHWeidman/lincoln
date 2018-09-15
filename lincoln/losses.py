import typing

import torch
from torch import Tensor

from .exc import MatchError, DimensionError
from .utils import assert_same_shape, assert_dim


class Loss:

    def __init__(self):
        pass

    def forward(self, prediction: Tensor, target: Tensor) -> float:

        if len(prediction.shape) != 2:
            raise DimensionError("Prediction tensor must have two dimensions.")

        self.prediction = prediction
        self.target = target.view(-1, 1)

        self.output = self._output()

        return self.output

    def backward(self) -> Tensor:

        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> Tensor:
        raise NotImplementedError()

    def _input_grad(self) -> Tensor:
        raise NotImplementedError()


class LogLoss(Loss):
    """ Log loss error specifically for logistic regression, requires a sequence of layers as input """
    def __init__(self, eta=1e-6):
        super().__init__()

        # Small parameter to avoid explosions when our probabilities get near 0 or 1
        # A better way to do this is use log probabilities everywhere
        self.eta = eta

    def _output(self) -> float:
        prediction, target = self.prediction, self.target
        loss = torch.sum(-target*torch.log(prediction + self.eta) - \
                         (1-target)*torch.log(1 - prediction + self.eta))

        return loss.item()

    def _input_grad(self)-> Tensor:

        prediction, target = self.prediction, self.target
        N = target.shape[0]
        loss_grad = torch.sum(-target/(prediction + self.eta) + \
                              (1-target)/(1 - prediction  + self.eta), dim=1).view(N, -1)

        assert_same_shape(prediction, loss_grad)

        return loss_grad


    def _input_grad(self)-> Tensor:

        prediction, target = self.prediction, self.target
        N = target.shape[0]
        loss_grad = torch.sum(-target/(prediction + self.eta) + \
                              (1-target)/(1 - prediction  + self.eta), dim=1).view(N, -1)

        assert_same_shape(prediction, loss_grad)

        return loss_grad

    def __repr__(self):
        return f"LogLoss"


class LogSigmoidLoss(Loss):
    def __init__(self, eta=1e-6):
        super().__init__()

        # Small parameter to avoid explosions when our probabilities get near 0 or 1
        # A better way to do this is use log probabilities everywhere
        self.eta = eta

    def _output(self) -> float:
        prediction, target = self.prediction, self.target
        loss = torch.sum(-target*prediction - (1-target)*torch.log(1-torch.exp(prediction) + self.eta))

        return loss.item()


    def _input_grad(self) -> Tensor:

        prediction, target = self.prediction, self.target

        exp_z = torch.exp(prediction)
        N = target.shape[0]
        self.loss_grad = torch.sum(-target + (1-target)*exp_z/(1 - exp_z + self.eta), dim=1).view(N, -1)

        assert_same_shape(prediction, self.loss_grad)

        return self.loss_grad

    def _input_grad(self) -> Tensor:

        prediction, target = self.prediction, self.target

        exp_z = torch.exp(prediction)
        N = target.shape[0]
        self.loss_grad = torch.sum(-target + (1-target)*exp_z/(1 - exp_z + self.eta), dim=1).view(N, -1)

        assert_same_shape(prediction, self.loss_grad)

        return self.loss_grad


class MeanSquaredError(Loss):

    def __init__(self) -> None:
        super().__init__()


    def _output(self) -> float:
        prediction, target = self.prediction, self.target
        loss = torch.sum(torch.pow(prediction - target, 2))
        return loss.item()


    def _input_grad(self) -> Tensor:
        prediction, target = self.prediction, self.target

        loss_grad = -2.0 * torch.add(target, -1.0 * prediction)

        assert_same_shape(prediction, loss_grad)

    def _input_grad(self) -> Tensor:
        prediction, target = self.prediction, self.target

        loss_grad = -2.0 * torch.add(target, -1.0 * prediction)

        assert_same_shape(prediction, loss_grad)

        return loss_grad


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()


    def _output(self) -> float:

        ps = self.prediction

    def _output(self) -> float:

        ps = self.prediction
        ys = self.target
        loss = torch.sum(-torch.log(torch.gather(ps, 1, ys)))
        return loss.item()

    def _input_grad(self) -> Tensor:
        ps = self.prediction
        ys = self.target

        # Create a mask for our correct labels, with 1s for the true labels, 0 elsewhere
        mask = torch.zeros_like(ps)
        mask.scatter_(1, ys, 1)

        # Picking out particular elements denoted by the correct labels
        grads = mask * -1/ps

        # Create a mask for our correct labels, with 1s for the true labels, 0 elsewhere
        mask = torch.zeros_like(ps)
        mask.scatter_(1, ys, 1)

        # Picking out particular elements denoted by the correct labels
        grads = mask * -1/ps

        return grads

class NLLLoss(Loss):
    def __init__(self):
        super().__init__()

    def _output(self) -> float:
        logps, target = self.prediction, self.target

        zeros = torch.zeros_like(logps)
        mask = zeros.scatter(1, target, 1)
        L = mask * -logps

        loss = L.sum().item()
        return loss

    def _input_grad(self) -> Tensor:

        zeros = torch.zeros_like(self.prediction)
        backward_grad = zeros.scatter(1, self.target, -1)

        return backward_grad

    def __repr__(self):
        return "NLLLoss"


    def _output(self) -> float:
        logps, target = self.prediction, self.target

        zeros = torch.zeros_like(logps)
        mask = zeros.scatter(1, target, 1)
        L = mask * -logps

        loss = L.sum().item()
        return loss

    def _input_grad(self) -> Tensor:

        zeros = torch.zeros_like(self.prediction)
        backward_grad = zeros.scatter(1, self.target, -1)

        return backward_grad

    def __repr__(self):
        return "NLLLoss"
