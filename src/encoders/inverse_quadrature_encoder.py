class InverseQuadratureEncoder:
    """
    Класс для имитации работы квадратурного энкодера
    """

    _matrix = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def get(self, ticks) -> tuple:
        """
        Рассчитать сигналы энкодера

        :param ticks: Количество "тиков"
        :return: Сигналы A и B соответственно
        """

        return self._matrix[round(ticks % len(self._matrix))]
