import numpy as np

# -----------------------------------------------------------------

class SoftMax:

    def __init__(self):
        """Implementation of the activation layer"""
        
        self.y_hat = None

    def forward(self, input_tensor: np.ndarray, label_tensor: np.ndarray):
        """Performs the forward pass computations"""
        
        maximums = np.amax(input_tensor, axis=1)

        for x in range(input_tensor.shape[0]):
            input_tensor[x, :] = input_tensor[x, :] - maximums[x]

        input_tensor = np.exp(input_tensor)
        sum_input = np.sum(input_tensor, axis=1)

        for x in range(input_tensor.shape[0]):
            input_tensor[x, :] = input_tensor[x, :] / sum_input[x]

        self.y_hat = input_tensor.copy()
        
        loss = 0

        for x in range(label_tensor.shape[0]):
            for y in range(label_tensor.shape[1]):
                if label_tensor[x, y] == 1:
                    loss = loss + (np.log(input_tensor[x, y]) * -1)

        return loss

    def predict(self, input_tensor: np.ndarray):
        """Performs the prediction"""

        maximums = np.amax(input_tensor, axis=1)

        for x in range(input_tensor.shape[0]):
            input_tensor[x, :] = input_tensor[x, :] - maximums[x]

        input_tensor = np.exp(input_tensor)
        sum_input = np.sum(input_tensor, axis=1)

        for x in range(input_tensor.shape[0]):
            input_tensor[x, :] = input_tensor[x, :] / sum_input[x]

        return input_tensor

    def backward(self, label_tensor: np.ndarray):
        """Performs the backward pass computations"""
        
        batch_size = label_tensor.shape[0]
        categories = label_tensor.shape[1]
        error_tensor = np.zeros((batch_size, categories), dtype=float)

        for i in range(batch_size):
            for j in range(categories):
                if label_tensor[i, j] == 1:
                    error_tensor[i, j] = self.y_hat[i, j] - 1
                else:
                    error_tensor[i, j] = self.y_hat[i, j]

        return error_tensor

# -----------------------------------------------------------------
