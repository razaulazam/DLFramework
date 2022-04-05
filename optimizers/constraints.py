import numpy as np

from abc import ABCMeta, abstractmethod

# -----------------------------------------------------------------

class Constraints(metaclass=ABCMeta):
    """Abstract Constraints Class"""

    @abstractmethod
    def calculate(self):
        raise NotImplementedError

    @abstractmethod
    def norm(self):
        raise NotImplementedError

# -----------------------------------------------------------------

class L2_Regularizer(Constraints):
    """L2 regularization"""

    def __init__(self, alpha: float, weights: np.ndarray):
        self.alpha = alpha
        self.weights = weights

    def calculate(self):
        return self.alpha * self.weights

    def norm(self):
        return self.alpha * np.sqrt(np.sum(np.power(self.weights, 2)))

    @property
    def weights(self):
        return self.weights

# -----------------------------------------------------------------

class L1_Regularizer(Constraints):
    """L1 regularization"""

    def __init__(self, alpha: float, weights: np.ndarray):
        self.alpha = alpha
        self.weights = weights

    def calculate(self):
        return self.alpha * np.sign(self.weights)

    def norm(self):
        return self.alpha * np.sum(np.abs(self.weights))

    @property
    def weights(self):
        return self.weights

# -----------------------------------------------------------------
