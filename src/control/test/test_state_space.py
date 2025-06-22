import unittest
from ..state_space import StateSpace
import numpy as np


class StateSpaceTestCase(unittest.TestCase):
    """
    Тест пространства состояний
    """
    def test(self):
        ss = StateSpace(A=np.eye(2), B=np.ones((2, 1)), C=np.eye(2), D=np.ones((2, 1)))
        self.assertTrue(np.all(np.ones((2, 1)) == ss.step(x=np.ones((2, 1)))))
        self.assertTrue(np.all(np.array([[3], [4]]) == ss.step(x=np.array([[3], [4]]))))
        self.assertTrue(np.all(np.ones((2, 1)) == ss.step(np.ones((2, 1)))))
        self.assertTrue(np.all(np.array([[2], [2]]) == ss.step(x=np.array([[0], [0]]), u=2)))
        self.assertTrue(np.all(np.array([[1], [2]]) == ss.out(x=np.array([[1], [2]]))))
        self.assertTrue(np.all(np.array([[2], [2]]) == ss.out(x=np.array([[0], [0]]), u=2)))
