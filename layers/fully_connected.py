import numpy as np

# -----------------------------------------------------------------

class FullyConnected:

    def __init__(self, input_size: int, output_size: int):
        """Fully connected layer implementation"""
        
        total_size = input_size + 1
        self.weights = np.random.random([total_size, output_size])
        self.delta = 1 
        self.input = None
        self.gradient = None

    def forward(self, input_tensor: np.ndarray):
        """Performs the forward pass computations"""

        batch_size = input_tensor.shape[0]
        bias = np.ones((batch_size, 1), dtype=int)

        input_tensor = np.hstack((input_tensor, bias))

        self.input = input_tensor.copy()
        input_weight_transpose = np.transpose(self.weights)
        input_tensor_transpose = np.transpose(input_tensor)

        output_tensor = np.dot(input_weight_transpose, input_tensor_transpose)

        return np.transpose(output_tensor)

    def backward(self, error_tensor: np.ndarray):
        """Performs the backward pass computations"""

        input_weight_transpose = np.transpose(self.weights)

        new_tensor = np.dot(error_tensor, input_weight_transpose)
        num_cols = new_tensor.shape[1] - 1
        new_error_tensor = np.delete(new_tensor, num_cols, 1)
        input_transpose = np.transpose(self.input)

        self.gradient = np.dot(input_transpose, error_tensor)
        self.weights -= self.delta*self.gradient

        return new_error_tensor

    @property
    def gradient_weights(self):
        """Returns the gradient which can be used to verify the computations using e.g. numerical gradient computation."""
        
        return self.gradient

# -----------------------------------------------------------------
