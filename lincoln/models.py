import typing
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from .layers import Layer
from .losses import Loss, LogLoss
from .optimizers import SGD
from .metrics import accuracy


def generate_batches(features: np.ndarray, 
                     labels: np.ndarray,
                     size: int = 32,
                     shuffle: bool = True) -> Tuple[Tensor, Tensor]:
    
    if features.shape[0] != labels.shape[0]:
        raise ValueError('feature and label arrays must have the same first dimension')
    
    n = features.shape[0]
    
    if shuffle:
        idx = np.arange(n)
        shuffled = np.random.shuffle(idx)
        features = features[shuffled].reshape((n, -1)) 
        labels = labels[shuffled].reshape((n, 1))
    
    for ii in range(0, n, size):
        out_features = torch.from_numpy(features[ii:ii+size, :]).type(torch.FloatTensor)
        out_labels = torch.from_numpy(labels[ii:ii+size, :]).type(torch.FloatTensor)
        yield out_features, out_labels

class Logistic:
    def __init__(self, network: typing.Type[Layer], 
                       loss: Loss=LogLoss, 
                       optimizer: typing.Any=SGD, 
                       metric: typing.Callable=accuracy, 
                       batch_gen: typing.Callable=generate_batches):
        self.network = network
        if loss is not LogLoss:
            self.loss = loss
        else:
            self.loss = LogLoss(network)
        
        if optimizer is not SGD:
            self.optim = optimizer
        else:
            self.optim = optimizer(network)
        
        self.metric = metric
        self.batch_gen = batch_gen
        
    def fit(self, features: np.ndarray = None, labels: np.ndarray = None, 
                  epochs: int=500, print_every: int=100, 
                  batch_size: int=32, log_ps=False)-> None:
        steps = 0
        for e in range(epochs):
            running_loss = 0
            if features is not None:
                batch_generator = self.batch_gen(features, labels, size=batch_size)
            else:
                batch_generator = self.batch_gen
            for ii, (x, y) in enumerate(batch_generator):
                steps += 1
                running_loss += self.loss(x, y)
                self.loss.backward()
                self.optim.step()
            
                if steps % print_every == 0:
                    ps = self.network(torch.from_numpy(features).type(torch.FloatTensor))
                    if log_ps is True:
                        ps = torch.exp(ps)
                    predictions = np.round(ps.numpy())
                    acc = accuracy(predictions, labels)         
                    print(f"Epoch {e+1}.. Train loss: {running_loss/print_every:.4f}.. ", f"Accuracy: {acc*100:.3f}%")
                    running_loss = 0
