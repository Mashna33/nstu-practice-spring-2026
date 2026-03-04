import numpy as np


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Мелиди Мирон Евстафьевич, ПМ-33"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 1"

    @staticmethod
    def sum(x: int, y: int) -> int:
        return x + y

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        A_arr = np.asarray(A)
        b_arr = np.asarray(b)

        if b_arr.ndim == 2 and b_arr.shape[1] == 1:
            b_arr = b_arr.reshape(-1)

        x = np.linalg.solve(
            A_arr.astype(np.float64, copy=False),
            b_arr.astype(np.float64, copy=False),
        )
        return x
