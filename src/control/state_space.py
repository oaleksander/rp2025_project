import numpy as np


def to_mat(m):
    if np.isscalar(m) or (np.ndim(m) != 2 and len(m) == 1):
        return np.array([m]).reshape((1, 1))
    else:
        return m

def expm_approx(m):
    return (np.eye(len(m)) + m / 2) @ np.linalg.inv(np.eye(len(m)) - m / 2)

class StateSpace:
    def __init__(self, A: np.array, B: np.array = None, C: np.array = None, D: np.array = None):
        self.A = to_mat(A)
        if B is None:
            self.B = np.zeros((len(self.A), 1))
        else:
            self.B = to_mat(B)
        self.C = to_mat(C)
        if C is None:
            self.C = np.zeros((1, len(self.A)))
        else:
            self.C = to_mat(C)
        if D is None:
            self.D = to_mat(0)
        else:
            self.D = to_mat(D)

    def to_discrete(self, Ts: float) -> tuple[np.array, np.array]:
        # e^([A, B; 0, 0] * Ts) = ([Ad, Bd; 0, I])
        M = expm_approx(np.block([
            [self.A, self.B],
            [np.zeros((self.B.shape[1], len(self.A))), np.zeros((self.B.shape[1], self.B.shape[1]))]
        ]) * Ts)

        Ad = M[0:len(self.A), 0:len(self.A)]
        Bd = M[0:len(self.A), len(self.A):]

        return Ad, Bd

    def step(self, x: np.array, u: np.array = None, Ts: float = None):
        x = to_mat(x)

        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        else:
            u = to_mat(u)

        if Ts is None:
            return self.A @ x + self.B @ u

        Ad, Bd = self.to_discrete(Ts)

        return (Ad @ x + # Ad @ x
                Bd @ u) # Bd @ u

    def out(self, x: np.array, u: np.array = None):
        if u is None:
            u = np.zeros((self.D.shape[1], 1))
        else:
            u = to_mat(u)
        return self.C @ x + self.D @ u
