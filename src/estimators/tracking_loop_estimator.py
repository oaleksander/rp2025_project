from project_repo.src.control.state_space import StateSpace
from project_repo.src.estimators.estimator import Estimator
import numpy as np

class TrackingLoopEstimator(Estimator):
    """
    Реализация оценки скорости при помощи следящего наблюдателя
    """

    def __init__(self, pos_gain: float, vel_gain: float):
        """
        Инициализация следящего наблюдателя

        :param pos_gain: Коэффициент коррекции положения
        :param vel_gain: Коэффициент коррекции скорости
        """

        self.pos_gain = pos_gain
        self.vel_gain = vel_gain

    def estimate(self, t: np.array, x: np.array) -> np.array:
        """
        Оценка скорости

        :param t: Вектор значений времени
        :param x: Вектор значений положения
        :return: Вектор значений скорости
        """

        # Создаем модель тела, двигающегося с постоянной скоростью
        # Вектор состояния - x = [положение, скорость]
        # Оценка состояния - dx/dt = Ax + B(y - z)
        # Где A - Переходная матрица для экстраполяции
        #     y - Измеренное положение
        #     z = Cx - Полученное положение
        ss = StateSpace(A=np.array([[0, 1],
                                    [0, 0]]),
                        B=np.array([[self.pos_gain], [self.vel_gain]]),
                        C=np.array(np.eye(2)))
        state, result = np.zeros((2, 1)), np.empty(np.shape(x))
        dt = np.concat(([0], t[1:] - t[0:-1]))

        for i in range(len(result)):
            state = ss.step(state, x[i] - ss.out(state)[0], float(dt[i]))
            result[i] = float(ss.out(state)[1][0])
        return result