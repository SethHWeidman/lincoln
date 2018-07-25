import numpy as np


def standardize(arr: np.ndarray) -> np.ndarray:
    
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    
    return (arr - means) / stds