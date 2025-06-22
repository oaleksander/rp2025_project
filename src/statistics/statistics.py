import numpy as np

def rmse(x1: np.array, x2: np.array) -> float:
    """
    Расчет среднеквадратичного отклонения двух векторов

    :param x1: Первый вектор
    :param x2: Второй вектор
    :return: СКО (скаляр)
    """
    return np.sqrt(np.mean(np.power(x1 - x2, 2)))