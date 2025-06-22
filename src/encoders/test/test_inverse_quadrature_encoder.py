import unittest
from ..inverse_quadrature_encoder import InverseQuadratureEncoder

class InverseQuadratureEncoderTestCase(unittest.TestCase):
    """
    Тест обратного квадратурного энкодера
    """
    def test(self):
        iqe = InverseQuadratureEncoder()
        self.assertEqual((1, 0), iqe.get(-3))
        self.assertEqual((1, 1), iqe.get(-2))
        self.assertEqual((0, 1), iqe.get(-1))
        self.assertEqual((0, 0), iqe.get(0))
        self.assertEqual((1, 0), iqe.get(1))
        self.assertEqual((1, 1), iqe.get(2))
        self.assertEqual((0, 1), iqe.get(3))
        self.assertEqual((0, 0), iqe.get(4))
        self.assertEqual((1, 0), iqe.get(5))
        self.assertEqual((1, 1), iqe.get(6))

