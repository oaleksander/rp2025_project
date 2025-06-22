import numpy as np

class Estimator:
    """
    Интерфейс алгоритма оценки скорости
    """

    def estimate(self, t: np.array, x: np.array) -> np.array:
        """
        Оценка скорости

        :param t: Вектор значений времени
        :param x: Вектор значений положения
        :return: Вектор значений скорости
        """
        pass