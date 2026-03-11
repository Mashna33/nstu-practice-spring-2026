import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        # В тестах bias ожидают как np.array(0)
        self.bias = np.array(0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(np.mean((y - pred) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        # R^2 = 1 - MSE / Var(y)
        pred = self.predict(x)
        mse = np.mean((y - pred) ** 2)
        var_y = np.var(y)
        if var_y == 0:
            return 1.0
        return float(1.0 - mse / var_y)

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x)
        # как в тесте:
        # dw = -2 * x.T @ (y - pred) / n
        # db = -2 * mean(y - pred)
        n = x.shape[0]
        dw = (-2.0 / n) * (x.T @ (y - pred))
        db = -2.0 * np.mean(y - pred)
        return dw, np.array(db)


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return self._sigmoid(z)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        # Binary cross-entropy (mean)
        p = self.predict(x)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        # Accuracy
        p = self.predict(x)
        y_pred = (p >= 0.5).astype(y.dtype)
        return float(np.mean(y_pred == y))

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Для BCE:
        # dw = x.T @ (p - y) / n
        # db = mean(p - y)
        p = self.predict(x)
        n = x.shape[0]
        diff = p - y
        dw = (x.T @ diff) / n
        db = np.mean(diff)
        return dw, np.array(db)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Мелиди Мирон Евстафьевич, ПМ-33"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 2"

    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> LinearRegression:
        return LinearRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> LogisticRegression:
        return LogisticRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def fit(model: LinearRegression | LogisticRegression, x: np.ndarray, y: np.ndarray, lr: float, n_iter: int) -> None:
        # Обычный градиентный спуск
        for _ in range(n_iter):
            dw, db = model.grad(x, y)
            model.weights = model.weights - lr * dw
            model.bias = model.bias - lr * db
