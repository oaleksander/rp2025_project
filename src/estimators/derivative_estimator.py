from project_repo.src.estimators.estimator import Estimator
import numpy as np

class DerivativeEstimator(Estimator):
    def estimate(self, t: np.array, x: np.array) -> np.array:
        dx = np.concat(([0], x[1:] - x[0:-1]))
        dt = np.concat(([np.inf], t[1:] - t[0:-1]))
        return dx / dt