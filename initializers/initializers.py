import numpy as np
import math

from abc import ABCMeta, abstractmethod
from typing import Tuple

# -----------------------------------------------------------------

class Initializers(metaclass=ABCMeta):
    """Abstract Initializers Class"""
    
    @abstractmethod
    def initialize(self):
        raise NotImplementedError

# -----------------------------------------------------------------

class Constant(Initializers):
    
    def __init__(self, val: int):
        """Constant initialization"""

        self.value = val
        self.weights = []
        self.fan_in = 0
        self.fan_out = 0

    def initialize(self, weights_shape: Tuple[int, int], fan_in: int, fan_out: int):
        """Main method for initialization"""

        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weights = np.zeros(weights_shape, dtype=float)
        self.weights[:, :] = self.value
        return self.weights

# -----------------------------------------------------------------

class UniformRandom(Initializers):
    
    def __init__(self):
        """Uniform random initialization"""

        self.weights = []
        self.fan_in = 0
        self.fan_out = 0

    def initialize(self, weights_shape: Tuple[int, int], fan_in: int, fan_out: int):
        """Main method for initialization"""

        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weights = np.random.random(weights_shape)
        return self.weights

# -----------------------------------------------------------------

class Xavier(Initializers):
    def __init__(self):
        """Xavier initialization"""

        self.mean = 0
        self.sigma = 0
        self.fan_in = 0
        self.fan_out = 0
        self.weights = []

    def initialize(self, weights_shape: Tuple[int, int], fan_in: int, fan_out: int):
        """Main method for initialization"""

        self.fan_in = fan_in
        self.fan_out = fan_out
        num = 2/(fan_out+fan_in)
        self.sigma = math.sqrt(num)
        self.weights = np.random.normal(self.mean, self.sigma, weights_shape)
        return self.weights

# -----------------------------------------------------------------

class He(Initializers):
    def __init__(self):
        """He initialization"""

        self.mean = 0
        self.sigma = 0
        self.fan_in = 0
        self.fan_out = 0
        self.weights = []

    def initialize(self, weights_shape: Tuple[int, int], fan_in: int, fan_out: int):
        """Main method for initialization"""

        self.fan_in = fan_in
        self.fan_out = fan_out
        num = 2/fan_in
        self.sigma = math.sqrt(num)
        self.weights = np.random.normal(self.mean, self.sigma, weights_shape)
        return self.weights

# -----------------------------------------------------------------
