import numpy as np


def to_mat(m):
    """
    Преобразование поля в матрицу

    :param m: Матрица или скаляр
    :return: Исходная матрица, если m - матрица либо матрица 1x1, если m - скаляр
    """

    if np.isscalar(m) or (np.ndim(m) != 2 and len(m) == 1):
        return np.array([m]).reshape((1, 1))
    else:
        return m

def expm_approx(m):
    """
    Аппроксимация матричной экспоненты

    :param m: Матрица
    :return: Экспонента e^m
    """
    return (np.eye(len(m)) + m / 2) @ np.linalg.inv(np.eye(len(m)) - m / 2)

class StateSpace:
    """
    Класс, представляющий линейную стационарную систему в пространстве состояний (State-Space)
    Поддерживает произвольное количество входов, состояний и выходов

    Позволяет моделировать состояние системы в дискретном времени
    """

    def __init__(self, A: np.array, B: np.array = None, C: np.array = None, D: np.array = None):
        """
        Инициализация системы

        :param A: Обязательная квадратная переходная матрица: количество строк/столбцов = количество состояний
        :param B: Матрица входов: количество строк = количество состояний, количество столбцов = количество входов,
        :param C: Матрица выходов: количество строк = количество выходов, количество столбцов = количество состояний
        :param D: "Сквозная" матрица: количество строк = количество выходов, количество столбцов = количество входов

        Соответствует дифференциальному уравнению

        dx/dt = Ax + Bu;
        y = Cx + Du,

        где x - вектор состояния, u - входной сигнал, y - выходной сигнал
        """
        self.A = to_mat(A)
        if B is None:
            self.B = np.zeros((len(self.A), 1))
        else:
            self.B = to_mat(B)
        self.C = to_mat(C)
        if C is None:
            self.C = np.zeros((1, len(self.A)))
        else:
            self.C = to_mat(C)
        if D is None:
            self.D = to_mat(0)
        else:
            self.D = to_mat(D)

    def to_discrete(self, Ts: float) -> tuple[np.array, np.array]:
        """
        Переход из непрерывного представления в дискретное

        :param Ts: Период дискретизации [c]
        :return: Матрицы Ad и Bd, соответствующие разностному уравнению
        x(t + Ts) = Ad * x(t) + Bd * u(t)
        """

        # e^([A, B; 0, 0] * Ts) = ([Ad, Bd; 0, I])
        M = expm_approx(np.block([
            [self.A, self.B],
            [np.zeros((self.B.shape[1], len(self.A))), np.zeros((self.B.shape[1], self.B.shape[1]))]
        ]) * Ts)

        Ad = M[0:len(self.A), 0:len(self.A)]
        Bd = M[0:len(self.A), len(self.A):]

        return Ad, Bd

    def step(self, x: np.array, u: np.array = None, Ts: float = None):
        """
        Расчет очередного шага системы

        :param x: Исходный вектор состояния x(t)
        :param u: Управляющее воздействие u(t)
        :param Ts: Период дискретизации [c]
        :return: Если Ts != 0: новый вектор состояния x(t + Ts);
                 Если Ts == 0: дифференциал вектора состояния dx(t)
        """

        x = to_mat(x)

        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        else:
            u = to_mat(u)

        if Ts is None:
            return self.A @ x + self.B @ u

        Ad, Bd = self.to_discrete(Ts)

        return (Ad @ x + # Ad @ x
                Bd @ u) # Bd @ u

    def out(self, x: np.array, u: np.array = None):
        """
        Получить вектор измеряемого выхода

        :param x: Вектор состояний x(t)
        :param u: Вектор входов u(t)
        :return: Выход y(t)
        """

        if u is None:
            u = np.zeros((self.D.shape[1], 1))
        else:
            u = to_mat(u)
        return self.C @ x + self.D @ u
