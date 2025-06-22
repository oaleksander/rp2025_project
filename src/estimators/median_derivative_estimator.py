from project_repo.src.estimators.derivative_estimator import DerivativeEstimator
import numpy as np

class MedianDerivativeEstimator(DerivativeEstimator):
    def __init__(self, ksize):
        self.window_size = ksize

    def estimate(self, t: np.array, x: np.array) -> np.array:
        est = np.pad(super().estimate(t, x), (round(self.window_size / 2), ), 'edge')
        idx = np.arange(self.window_size) + np.arange(len(est) - self.window_size + 1)[:, None]
        b = [row for row in est[idx]]
        res = np.array([np.median(c) for c in b])
        return res