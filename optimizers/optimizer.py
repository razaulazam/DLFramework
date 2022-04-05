import math
import numpy as np

from abc import ABCMeta, abstractmethod

# -----------------------------------------------------------------

class Optimizer(metaclass=ABCMeta):
    """Abstract Optimizer Class"""

    @abstractmethod
    def calculate_update(self):
        raise NotImplementedError

# -----------------------------------------------------------------

class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def __init__(self, learning_rate: float):
        self.global_learning_rate = learning_rate

    def calculate_update(self, individual_delta: float, weight_tensor: np.ndarray, gradient_tensor: np.ndarray):
        """Updates the weights with SGD algorithm"""
        
        weight_new_tensor = weight_tensor - (self.global_learning_rate * individual_delta * gradient_tensor)
        return weight_new_tensor

# -----------------------------------------------------------------

class SgdWithMomentum(Optimizer):
    """Stochastic Gradient Descent with Momentum"""

    def __init__(self, learning_rate: float, momentum: float):
        self.global_learning_rate = learning_rate
        self.momentum = momentum
        self.flag = False
        self.velocity = None

    def calculate_update(self, individual_delta: float, weight_tensor: np.ndarray, gradient_tensor: np.ndarray):
        """Update the weights with the momentum algorithm"""

        if not self.flag:
            self.velocity = (-1 * self.global_learning_rate * individual_delta) * gradient_tensor
            weight_new_tensor = weight_tensor + self.velocity
            self.flag = True
        else:
            self.velocity = self.momentum * self.velocity - (self.global_learning_rate * individual_delta *
                                                             gradient_tensor)
            weight_new_tensor = weight_tensor + self.velocity

        return weight_new_tensor

# -----------------------------------------------------------------

class Adam(Optimizer):
    """Adam optimizer"""

    def __init__(self, learning_rate: float, momentum: float, phi: float):
        self.global_learning_rate = learning_rate
        self.momentum = momentum
        self.phi = phi
        self.flag = False
        self.exponent = 1
        self.velocity = []
        self.r_velocity = []

    def calculate_update(self, individual_delta: float, weight_tensor: np.ndarray, gradient_tensor: np.ndarray):
        """Updates the weights by computing the gradients and applying the descent algorithm"""

        if not self.flag:
            self.velocity.append((1 - self.momentum) * gradient_tensor)
            intermediate_gradient = np.multiply(gradient_tensor, gradient_tensor)
            self.r_velocity.append((1 - self.phi) * intermediate_gradient)
            velocity_hat = self.velocity[0]/(1 - math.pow(self.momentum, self.exponent))
            r_hat = self.r_velocity[0]/(1 - math.pow(self.phi, self.exponent))
            weight_new_tensor = weight_tensor - (individual_delta * self.global_learning_rate *
                                                 (velocity_hat + 0.000000001) / (np.sqrt(np.abs(r_hat)) + 0.000000001))
            self.flag = True
            self.exponent = self.exponent + 1
        else:
            self.velocity[0] = self.velocity[0] * self.momentum + ((1 - self.momentum) * gradient_tensor)
            intermediate_gradient = np.multiply(gradient_tensor, gradient_tensor)
            self.r_velocity[0] = self.r_velocity[0] * self.phi + ((1 - self.phi) * intermediate_gradient)
            velocity_hat = self.velocity[0] / (1 - math.pow(self.momentum, self.exponent))
            r_hat = self.r_velocity[0] / (1 - math.pow(self.phi, self.exponent))
            weight_new_tensor = weight_tensor - (individual_delta * self.global_learning_rate *
                                                 (velocity_hat + 0.000000001) / (np.sqrt(np.abs(r_hat)) + 0.000000001))
            self.exponent = self.exponent + 1

        return weight_new_tensor

# -----------------------------------------------------------------
