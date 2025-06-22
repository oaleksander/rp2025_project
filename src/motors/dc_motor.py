import numpy as np
from control.state_space import StateSpace

class DcMotor(StateSpace):
    """
    Класс для моделирования работы двигателя постоянного тока

    Реализован поверх State-space
    """

    def __init__(self, J, b, Kt, Ke, R, L):
        """
        Инициализация двигателя постоянного тока

        :param J: Момент инерции [Кг * м^2]
        :param b: Коэффициент вязкого трения [Нм/рад/с]
        :param Kt: Постоянная силы [Нм/А]
        :param Ke: Постоянная обратной ЭДС [В/рад/с]
        :param R: Сопротивление катушек [Ом]
        :param L: Индуктивность катушек [Гн]
        """

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
        """
        Получить положение мотора

        :param x: Вектор состояния
        :return: Положение [рад]
        """
        return float(x[0][0])

    @staticmethod
    def vel(x) -> float:
        """
        Получить скорость мотора

        :param x: Вектор состояния
        :return: Скорость [рад/с]
        """
        return float(x[1][0])

    @staticmethod
    def current(x) -> float:
        """
        Получить силу тока мотора

        :param x: Вектор состояния
        :return: Сила тока [А]
        """
        return float(x[2][0])
