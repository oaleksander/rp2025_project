from project_repo.src.control.state_space import StateSpace
from project_repo.src.estimators.estimator import Estimator
import numpy as np

class TrackingLoopEstimator(Estimator):
    def __init__(self, pos_gain: float, vel_gain: float):
        self.pos_gain = pos_gain
        self.vel_gain = vel_gain

    def estimate(self, t: np.array, x: np.array) -> np.array:
        ss = StateSpace(A=np.array([[0, 1],
                                    [0, 0]]),
                        B=np.array([[self.pos_gain], [self.vel_gain]]),
                        C=np.array(np.eye(2)))
        state, result = np.zeros((2, 1)), np.empty(np.shape(x))
        dt = np.concat(([0], t[1:] - t[0:-1]))

        for i in range(len(result)):
            state = ss.step(state, x[i] - ss.out(state)[0], float(dt[i]))
            result[i] = float(ss.out(state)[1][0])
        return result