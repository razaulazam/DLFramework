import numpy as np

# -----------------------------------------------------------------

class Sigmoid:
    def __init__(self):
        """Implementation of the sigmoid non-linearity"""

        self.activations = None

    def forward(self, input_tensor: np.ndarray):
        """Implements the forward method"""

        self.activations = 1/(1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor: np.ndarray):
        """Implements the backward method"""

        der = self.activations * (1 - self.activations)
        return error_tensor * der

# -----------------------------------------------------------------
