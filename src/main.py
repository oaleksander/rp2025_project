import matplotlib.pyplot as plt
import numpy as np

from project_repo.src.estimators.derivative_estimator import DerivativeEstimator
from project_repo.src.estimators.filtered_derivative_estimator import FilteredDerivativeEstimator
from project_repo.src.estimators.kalman_estimator import KalmanEstimator
from project_repo.src.estimators.median_derivative_estimator import MedianDerivativeEstimator
from project_repo.src.estimators.tracking_loop_estimator import TrackingLoopEstimator
from project_repo.src.estimators.unknown_input_kalman_estimator import UnknownInputKalmanEstimator
from project_repo.src.motors.dc_motor import DcMotor
from project_repo.src.statistics.statistics import rmse

def test_voltage(T):
    """
    Генерация функции входного сигнала
    
    :param T: Время [с] 
    :return: Входное напряжение [B]
    """
    
    u = []
    for t in T:
        if t <= 1:
            u.append(0)
        elif t <= 3:
            u.append(12 * (t - 1) / 2)
        elif t <= 7:
            u.append(12)
        elif t <= 9:
            u.append(12 - 12 * (t - 7) / 2)
        else:
            u.append(0)
    return u


def prepare_data(motor, Ts):
    """
    Подготовка данных для эксперимента
    
    :param motor: Математическая модель двигателя 
    :param Ts: Шаг дискретизации [c]
    :return: t, vel, t_meas, pos_meas, u:
    Реальное время, реальная скорость, измеряемое время, измеряемое положение, управляющее воздействие
    """
    
    # Генерируем время
    t = np.arange(start=0, stop=10, step=Ts)
    # Немного смещаем отсчеты, что бы фактический шаг дискретизации был непостоянным
    t = t + np.random.rand(len(t)) * (Ts / 2) - (Ts / 4)

    # Моделируем значения скорости и положения {x, dx}(t)
    dt = np.concat(([0], t[1:] - t[0:-1]))
    state = np.zeros((3, 1))
    u = test_voltage(t)
    pos, vel = np.zeros(len(dt)), np.zeros(len(dt))
    for i in range(len(dt)):
        state = motor.step(state, u[i], float(dt[i]))
        pos[i] = motor.pos(state)
        vel[i] = motor.vel(state)

    # Уменьшаем точность положения и времени для фильтра
    pos_meas = np.round(pos)
    t_meas = np.round(t + Ts, 3)

    return t, vel, t_meas, pos_meas, u


def display_results(t, vel, estimations, Ts):
    """
    Вывод результатов оценки
    
    :param t: Реальное время
    :param vel: Реальная скорость
    :param estimations: Набор из оценок скорости
    :param Ts: Шаг дискретизации
    """
    
    print('Ts = ', Ts)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(7, 5))

    for estimation in estimations:
        name, data = estimation
        ax1.plot(t, data, label=name)
    ax1.plot(t, vel, label='Original')

    for estimation in estimations:
        name, data = estimation
        ax2.plot(t, vel - data, label=name)
        print('RMS(', name, ') = ', rmse(vel, data))

    fig.suptitle('Ts = ' + str(Ts))
    ax1.grid(True)
    ax1.set_ylabel('Velocity (rad/s)')
    ax1.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_ylabel('Error (rad/s)')
    ax2.set_xlabel('Time (s)')
    print()


def main():
    """
    Основная функция, осуществляющая моделирование работы двигателя и
    тестирование алгоритмов расчета скорости
    """

    # Зафиксируем seed, что бы значения воспроизводились
    np.random.seed(42)

    # Инициализация модели самого двигателя и алгоритмов оценки
    motor = DcMotor(J=2.7e-5, b=1e-5, Ke=0.02, Kt=0.01, R=1.5, L=1e-5)
    derivative = DerivativeEstimator()
    filtered_derivative = FilteredDerivativeEstimator(5)
    median_derivative = MedianDerivativeEstimator(5)
    tracking_loop = TrackingLoopEstimator(2.4, 150)
    Q = np.diag([0, 5, 25])
    Q_ext = np.diag([0, 5, 25, 0.5])
    R = np.diag([1])
    no_input_kalman = UnknownInputKalmanEstimator(Q_ext, R, np.array([0, 1, 0, 0]), motor)
    kalman = KalmanEstimator(Q, R, np.array([0, 1, 0]), motor)

    # Симуляция алгоритмов с различными шагами дискретизации
    times = [0.0033, 0.01, 0.03]
    for Ts in times:
        # Подготовка значений
        t, vel, t_meas, pos_meas, u = prepare_data(motor, Ts)

        # Оценка
        estimations = [
            ('Derivative', derivative.estimate(t_meas, pos_meas)),
            ('Filtered derivative', filtered_derivative.estimate(t_meas, pos_meas)),
            ('Median derivative', median_derivative.estimate(t_meas, pos_meas)),
            ('Tracking loop', tracking_loop.estimate(t_meas, pos_meas)),
            ('Unknown input kalman', no_input_kalman.estimate(t_meas, pos_meas)),
            ('Kalman', kalman.estimate(t_meas, pos_meas, u)),
        ]

        # Вывод графиков и другой информации
        display_results(t, vel, estimations, Ts)

    plt.show()


if __name__ == '__main__':
    main()
