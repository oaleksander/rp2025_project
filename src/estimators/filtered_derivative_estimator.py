from project_repo.src.estimators.derivative_estimator import DerivativeEstimator
import numpy as np

class FilteredDerivativeEstimator(DerivativeEstimator):
    """
    Реализация оценки скорости при помощи фильтрованной производной

    Реализует фильтр низких частот с произвольным размером ядра
    """

    def __init__(self, ksize):
        """
        Инициализация фильтра

        :param ksize: Размер фильтрующего ядра: целое число - чем больше, тем сильнее фильтрация
        """
        self.ksize = ksize

    def estimate(self, t: np.array, x: np.array) -> np.array:
        """
        Оценка скорости

        :param t: Вектор значений времени
        :param x: Вектор значений положения
        :return: Вектор значений скорости
        """

        # Используем алгоритм дифференцирования, а зачем сворачиваем с соответствующим ядром
        # Например, для ksize = 3 получится kernel = [1/3 1/3 1/3]
        kernel = np.ones(self.ksize) / self.ksize
        return np.convolve(super().estimate(t, x), kernel, 'same')