import numpy as np

# -----------------------------------------------------------------

class ReLU:

    def __init__(self):
        """Implementation of RELU non-linearity"""
        
        self.input = None

    def forward(self, input_tensor: np.ndarray):
        """Performs the forward pass computations"""

        self.input = input_tensor.copy()
        output_tensor = np.maximum(0, input_tensor)
        return output_tensor

    def backward(self, error_tensor: np.ndarray):
        """Performs the backward pass computations"""
        batch_size = self.input.shape[0]
        input_size = self.input.shape[1]
        new_error_tensor = np.zeros((batch_size, input_size), dtype=float)
        for i in range(batch_size):
            for j in range(input_size):
                if self.input[i, j] > 0:
                    new_error_tensor[i, j] = error_tensor[i, j]

        return new_error_tensor

# -----------------------------------------------------------------
