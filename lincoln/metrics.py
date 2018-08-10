import numpy as np

def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    accuracy = np.mean(predictions.squeeze() == labels.squeeze())
    return accuracy