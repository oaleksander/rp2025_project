import numpy as np

def rmse(x1: np.array, x2: np.array) -> float:
    return np.sqrt(np.mean(np.power(x1 - x2, 2)))