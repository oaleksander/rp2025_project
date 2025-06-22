# `estimators` - Модуль с алгоритмами оценки скорости

* [estimator](estimator.py) - Общий интерфейс

* [derivative_estimator](derivative_estimator.py) - Производная
* [filtered_derivative_estimator](filtered_derivative_estimator.py) - Производная c низкочастотным фильтром
* [median_derivative_estimator](median_derivative_estimator.py) - Производная с бегущей медианой
* [tracking_loop_estimator](tracking_loop_estimator.py) - Следящий регулятор (наблюдатель)
* [kalman_estimator](kalman_estimator.py) - Фильтр Калмана
* [unknown_input_kalman_estimator](unknown_input_kalman_estimator.py) - Фильтр Калмана расширенем порядка

| Алгоритм             | Требуется модель объекта? | Требуется входной сигнал? |
|----------------------|---------------------------|---------------------------|
| derivative           | -                         | -                         |
| filtered_derivative  | -                         | -                         |
| median_derivative    | -                         | -                         |
| tracking_loop        | -                         | -                         |
| kalman               | +                         | +                         |
| unknown_input_kalman | +                         | -                         |

[↩️ Вверх](..)