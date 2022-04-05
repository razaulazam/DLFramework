import numpy as np
from layers.base import BaseClass, Phase

# -----------------------------------------------------------------

class Dropout(BaseClass):
    def __init__(self, probability: float):
        """Implementation of the Dropout layer"""

        super().__init__()
        self.probability = probability
        self.phase = Phase.train
        self.drop = 0
        self.new_input_tensor = 0

    def forward(self, input_tensor: np.ndarray):
        """Does the forward pass computations"""

        if self.phase == Phase.train:
            self.drop = np.random.binomial(1, self.probability, size=input_tensor.shape)
            new_input = np.multiply(input_tensor, self.drop)
            new_input /= self.probability
            self.new_input_tensor = new_input
        else:
            self.new_input_tensor = input_tensor

        return self.new_input_tensor

    def backward(self, error_tensor: np.ndarray):
        """Does the backward pass computations"""

        error = np.multiply(self.drop, error_tensor)
        return error

# -----------------------------------------------------------------
