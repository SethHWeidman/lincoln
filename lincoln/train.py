
from typing import Callable

import torch
from torch import Tensor

from .network import NeuralNetwork
from .optimizers import Optimizer
from .utils import generate_batches, to_2d, permute_data


class Trainer(object):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer,
                 batch_gen: Callable = generate_batches) -> None:
        self.net = net
        self.optim = optim
        setattr(self.optim, 'net', self.net)
        self.batch_gen = batch_gen

    def update_params(self) -> None:

        self.optim.step()

    def fit(self, X_train: Tensor, y_train: Tensor,
            X_test: Tensor, y_test: Tensor,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            single_output: bool = False,
            restart: bool = True)-> None:

        if restart:
            self.optim.first = True

        if single_output:
            y_train, y_test = to_2d(y_train, "col"), to_2d(y_test, "col")

        torch.manual_seed(seed)

        for e in range(epochs):
            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.batch_gen(X_train, y_train, size=batch_size)
            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)
                self.update_params()

            if (e+1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)
                print(f"Validation loss after {e+1} epochs is {loss:.3f}")
