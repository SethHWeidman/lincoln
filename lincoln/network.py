import torch
from torch import Tensor
from typing import List, Callable

from .optimizers import Optimizer, SGD
from .layers import Layer
from .losses import Loss, MeanSquaredError
from .utils import generate_batches, to_2d, permute_data


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


class NeuralNetwork(LayerBlock):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self, layers: List[Layer],
                 loss: Loss = MeanSquaredError,
                 optimizer: Optimizer = SGD(),
                 batch_gen: Callable = generate_batches):
        super().__init__(layers)
        self.loss = loss
        self.optimizer = optimizer
        self.batch_gen = batch_gen


    def train_batch(self,
                    X_batch: Tensor,
                    y_batch: Tensor) -> float:

        prediction = self.forward(X_batch)

        loss = self.loss.forward(prediction, y_batch)
        loss_grad = self.loss.backward()

        self.backward(loss_grad)

        self.update_params()

        return loss

    def update_params(self) -> None:

        for layer in self.layers:
            self.optimizer.step(layer)


    def fit(self, X_train: Tensor, y_train: Tensor,
            X_test: Tensor, y_test: Tensor,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            single_output: bool = False)-> None:

        if single_output:
            y_train, y_test = to_2d(y_train, "col"), to_2d(y_test, "col")

        torch.manual_seed(seed)

        for e in range(epochs):
            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.batch_gen(X_train, y_train, size=batch_size)
            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                print(self.train_batch(X_batch, y_batch))

            if (e+1) % eval_every == 0:
                test_preds = self.forward(X_test)
                # import pdb; pdb.set_trace()
                loss = self.loss.forward(test_preds, y_test)
                print(f"Validation loss after {e+1} epochs is {loss:.3f}")
