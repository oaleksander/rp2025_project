import unittest
from ..quadrature_encoder import QuadratureEncoder

class QuadratureEncoderTestCase(unittest.TestCase):
    """
    Тест квадратурного энкодера
    """
    def test(self):
        encoder = QuadratureEncoder(0)
        self.assertEqual(0, encoder.ticks)
        encoder.update(1, 0)
        self.assertEqual(1, encoder.ticks)
        encoder.update(1, 1)
        self.assertEqual(2, encoder.ticks)
        encoder.update(0, 1)
        self.assertEqual(3, encoder.ticks)
        encoder.update(0, 0)
        self.assertEqual(4, encoder.ticks)
        encoder.update(0, 1)
        self.assertEqual(3, encoder.ticks)
        encoder.update(1, 1)
        self.assertEqual(2, encoder.ticks)
        encoder.update(1, 0)
        self.assertEqual(1, encoder.ticks)
        encoder.update(0, 0)
        self.assertEqual(0, encoder.ticks)


if __name__ == '__main__':
    unittest.main()
