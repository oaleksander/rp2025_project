from project_repo.src.control.state_space import StateSpace
from project_repo.src.estimators.estimator import Estimator
import numpy as np


class UnknownInputKalmanEstimator(Estimator):
    def __init__(self, Q: np.array, R: np.array, H: np.array, sys: StateSpace):
        self.Q = Q
        self.R = R
        self.H = H
        A = np.block([
            [sys.A, sys.B],
            [np.zeros((sys.B.shape[1], len(sys.A))), np.zeros((sys.B.shape[1], sys.B.shape[1]))]
        ])
        C = np.block([sys.C, np.zeros((1, sys.B.shape[1]))])
        self.system = StateSpace(A, None, C, sys.D)


    def estimate(self, t, x) -> np.array:

        P = np.eye(len(self.system.A))

        state, result = np.zeros((self.system.A.shape[0], 1)), np.empty(np.shape(x))
        dt = np.concat(([0], t[1:] - t[0:-1]))

        for i in range(len(result)):
            state = self.system.step(state, None, float(dt[i]))
            F, _ = self.system.to_discrete(float(dt[i]))
            P = F @ P @ np.transpose(F) + self.Q
            y = x[i] - self.system.out(state)
            S = self.system.C @ P @ np.transpose(self.system.C) + self.R
            K = P @ np.transpose(self.system.C) @ np.linalg.inv(S)
            state = state + K @ y
            P = (np.eye(len(P)) - K @ self.system.C) @ P
            result[i] = float((self.H @ state)[0])

        return result

