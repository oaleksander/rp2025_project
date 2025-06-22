import unittest
from ..dc_motor import DcMotor
import numpy as np

class DcMotorTestCase(unittest.TestCase):
    """
    Тест двигателя постоянного тока
    """

    def test(self):
        motor = DcMotor(J=2.7e-5, b=1e-5, Ke=0.02, Kt=0.01, R=1.5, L=1e-5)
        Ts = 0.001
        Tstop = 5
        U = 12
        state = np.array([[0], [0], [0]])

        rng = range(int(Tstop / Ts))
        I = []
        for i in rng:
            state = motor.step(state, U, Ts)
            I.append(motor.current(state))

        self.assertAlmostEqual(U * (motor.Kt / (motor.b * motor.R + motor.Ke * motor.Kt)), motor.vel(state), 1)
        self.assertAlmostEqual(U / (motor.R + motor.Ke * motor.Kt / motor.b), motor.current(state), 2)