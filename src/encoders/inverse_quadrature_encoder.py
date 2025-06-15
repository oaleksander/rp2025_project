class InverseQuadratureEncoder:
    _matrix = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def get(self, ticks) -> tuple:
        return self._matrix[round(ticks % len(self._matrix))]
