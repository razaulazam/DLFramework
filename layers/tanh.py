import numpy as np

# -----------------------------------------------------------------

class TanH:
    def __init__(self):
        """Implementation of the TanH non-linearity"""

        self.activations = None

    def forward(self, input_tensor: np.ndarray):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor: np.ndarray):
        der = 1 - self.activations**2
        return error_tensor * der

# -----------------------------------------------------------------
