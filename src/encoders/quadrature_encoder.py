class QuadratureEncoder:
    ticks: int = 0
    _matrix = [0, -1, 1, 2, 1, 0, 2, -1, -1, 2, 0, 1, 2, 1, -1, 0]
    _ab = 0b00

    def __init__(self, init: int = 0) -> None:
        self.ticks = init

    def update(self, a: bool, b: bool) -> None:
        new_ab = (int(a) << 1) + int(b)
        self.ticks += self._matrix[(self._ab << 2) + new_ab]
        self._ab = new_ab
