from project_repo.src.estimators.derivative_estimator import DerivativeEstimator
import numpy as np

class FilteredDerivativeEstimator(DerivativeEstimator):
    def __init__(self, ksize):
        self.ksize = ksize

    def estimate(self, t: np.array, x: np.array) -> np.array:
        kernel = np.ones(self.ksize) / self.ksize
        return np.convolve(super().estimate(t, x), kernel, 'same')