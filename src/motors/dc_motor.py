from ..control.state_space import StateSpace
import numpy as np

class DcMotor(StateSpace):
    def __init__(self, J, b, Kt, Ke, R, L):
        self.J = J
        self.b = b
        self.Kt = Kt
        self.Ke = Ke
        self.R = R
        super().__init__(
            A=np.array([[0, 1, 0],
                        [0, -b / J, Kt / J],
                        [0, -Ke / L, -R / L]]),
            B=np.array([[0], [0], [1 / L]]),
            C=np.array([[1, 0, 0]]),
            D=np.array([[0]])
        )

    @staticmethod
    def pos(x) -> float:
        return float(x[0][0])

    @staticmethod
    def vel(x) -> float:
        return float(x[1][0])

    @staticmethod
    def current(x) -> float:
        return float(x[2][0])
