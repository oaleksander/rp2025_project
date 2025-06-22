from project_repo.src.estimators.estimator import Estimator
import numpy as np

class DerivativeEstimator(Estimator):
    """
    Реализация оценки скорости на основании численного дифференцирования

    Самый наивный алгоритм
    """

    def estimate(self, t: np.array, x: np.array) -> np.array:
        """
        Оценка скорости

        :param t: Вектор значений времени
        :param x: Вектор значений положения
        :return: Вектор значений скорости
        """
        dx = np.concat(([0], x[1:] - x[0:-1]))
        dt = np.concat(([np.inf], t[1:] - t[0:-1]))
        return dx / dt