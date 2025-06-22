import matplotlib.pyplot as plt
import numpy as np

from project_repo.src.estimators.derivative_estimator import DerivativeEstimator
from project_repo.src.estimators.filtered_derivative_estimator import FilteredDerivativeEstimator
from project_repo.src.estimators.kalman_estimator import KalmanEstimator
from project_repo.src.estimators.tracking_loop_estimator import TrackingLoopEstimator
from project_repo.src.estimators.unknown_input_kalman_estimator import UnknownInputKalmanEstimator
from project_repo.src.motors.dc_motor import DcMotor
from project_repo.src.statistics.statistics import rmse

def test_voltage(T):
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
    t = np.arange(start=0, stop=10, step=Ts)
    t = t + np.random.rand(len(t)) * (Ts / 2) - (Ts / 4)

    dt = np.concat(([0], t[1:] - t[0:-1]))
    state = np.zeros((3, 1))
    u = test_voltage(t)
    x, dx = np.zeros(len(dt)), np.zeros(len(dt))
    for i in range(len(dt)):
        state = motor.step(state, u[i], float(dt[i]))
        x[i] = motor.pos(state)
        dx[i] = motor.vel(state)

    x_meas = np.round(x)
    t_meas = np.round(t + Ts, 3)

    return t, dx, t_meas, x_meas, u


def display_results(t, dx, estimations, Ts):
    print('Ts = ', Ts)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(7, 5))

    for estimation in estimations:
        name, data = estimation
        ax1.plot(t, data, label=name)
    ax1.plot(t, dx, label='Original')

    for estimation in estimations:
        name, data = estimation
        ax2.plot(t, dx - data, label=name)
        print('RMS(', name, ') = ', rmse(dx, data))

    fig.suptitle('Ts = ' + str(Ts))
    ax1.grid(True)
    ax1.set_ylabel('Velocity (rad/s)')
    ax1.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_ylabel('Error (rad/s)')
    ax2.set_xlabel('Time (s)')
    print()


def main():
    np.random.seed(42)

    motor = DcMotor(J=2.7e-5, b=1e-5, Ke=0.02, Kt=0.01, R=1.5, L=1e-5)
    derivative = DerivativeEstimator()
    filtered_derivative = FilteredDerivativeEstimator(5)
    tracking_loop = TrackingLoopEstimator(2.4, 150)

    Q = np.diag([0, 5, 25])
    Q_ext = np.diag([0, 5, 25, 0.5])
    R = np.diag([1])
    no_input_kalman = UnknownInputKalmanEstimator(Q_ext, R, np.array([0, 1, 0, 0]), motor)
    kalman = KalmanEstimator(Q, R, np.array([0, 1, 0]), motor)

    times = [0.0033, 0.01, 0.03]
    for Ts in times:
        t, dx, t_meas, x_meas, u = prepare_data(motor, Ts)

        estimations = [
            ('Derivative', derivative.estimate(t_meas, x_meas)),
            ('Filtered derivative', filtered_derivative.estimate(t_meas, x_meas)),
            ('Tracking loop', tracking_loop.estimate(t_meas, x_meas)),
            ('Unknown input kalman', no_input_kalman.estimate(t_meas, x_meas)),
            ('Kalman', kalman.estimate(t_meas, x_meas, u)),
        ]

        display_results(t, dx, estimations, Ts)

    plt.show()


if __name__ == '__main__':
    main()
