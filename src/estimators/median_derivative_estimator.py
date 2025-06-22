from project_repo.src.estimators.derivative_estimator import DerivativeEstimator
import numpy as np

class MedianDerivativeEstimator(DerivativeEstimator):
    """
    Реализация оценки скорости при помощи фильтрованной производной

    Реализует бегущую медиану
    """

    def __init__(self, ksize):
        """
        Инициализация произнодной с плавающей медианой

        :param ksize: Размер выборки доля медианы
        """
        self.window_size = ksize

    def estimate(self, t: np.array, x: np.array) -> np.array:
        """
        Оценка скорости

        :param t: Вектор значений времени
        :param x: Вектор значений положения
        :return: Вектор значений скорости
        """

        # Увеличиваем размерность исходного массива
        est = np.pad(super().estimate(t, x), (round(self.window_size / 2), ), 'edge')

        # Получаем индексы для расчета медианы
        idx = np.arange(self.window_size) + np.arange(len(est) - self.window_size + 1)[:, None]
        b = [row for row in est[idx]]

        # Непосредственно расчет
        res = np.array([np.median(c) for c in b])
        return res