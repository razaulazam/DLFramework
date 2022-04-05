import numpy as np

from fully_connected import FullyConnected
from tanh import TanH
from optimizers.optimizer import Optimizer
from initializers.initializers import Initializers

# -----------------------------------------------------------------

class RNN:
  
    def __init__(self, input_size: int, hidden_size: int, output_size: int, bptt_length: int):
        """Implementation of the standard recurrent unit"""

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bpt_length = bptt_length
        self.sub_seq = False

        self.input_tensor = None
        self.fc_activation = FullyConnected((self.hidden_size + self.input_size),
                                                           self.hidden_size)
        self.fc_output = FullyConnected(self.hidden_size, self.output_size)

        self.hidden_states = None
        self.starting_state = np.zeros((1, self.hidden_size), dtype=float)

        self.delta = 1

        self.gradients_activation = None
        self.gradients_output = None

        self.optimizer = None
        self.flag_opt = 0

    def toggle_memory(self):
        """Toggles the memory"""

        self.sub_seq = not self.sub_seq

    def forward(self, input_tensor: np.ndarray):
        """Does the forward pass computations"""

        batch_size = input_tensor.shape[0]
        total_hidden = batch_size + 1
        stacked_size = self.input_size + self.hidden_size

        self.input_tensor = input_tensor
        output_tensor = np.zeros((batch_size, self.output_size), dtype=float)
        self.hidden_states = np.zeros((total_hidden, self.hidden_size), dtype=float)

        activation = TanH.TanH()

        if self.sub_seq is True:
            self.hidden_states[0, :] = self.starting_state[:]

        for i in range(batch_size):
            combined_input = np.hstack((self.hidden_states[i, :], input_tensor[i, :]))
            intermediate = self.fc_activation.forward(combined_input.reshape(1, stacked_size))
            self.hidden_states[(i + 1), :] = activation.forward(intermediate)
            output_tensor[i, :] = self.fc_output.forward(self.hidden_states[(i + 1), :].reshape(1, self.hidden_size))

        self.starting_state[:] = self.hidden_states[batch_size, :]

        return output_tensor

    def backward(self, error_tensor: np.ndarray):
        """Does the backward pass computations"""

        batch_size = self.input_tensor.shape[0]
        output_dim = self.input_tensor.shape[1]
        err_prev = np.zeros((batch_size, output_dim), dtype=float)
        stacked_size = self.hidden_size + self.input_size
        bpt_len = 0
        flag = 0

        self.gradients_output = np.zeros((self.hidden_size + 1, self.output_size), dtype=float)
        self.gradients_activation = np.zeros((stacked_size + 1, self.hidden_size), dtype=float)
        bias = np.ones((1, 1), dtype=int)

        activations_grad = TanH.TanH()
        err_activation_output = np.zeros((1, self.hidden_size), dtype=float)
        err_intermediate = np.zeros((1, self.hidden_size), dtype=float)
        intermediate_tensor = np.zeros((1, stacked_size), dtype=float)

        for i in reversed(range(batch_size)):
            self.fc_output.input = np.hstack((self.hidden_states[(i + 1), :].reshape(1, self.hidden_size), bias))
            err_activation_output[:] = self.fc_output.backward(error_tensor[i, :].reshape(1, self.output_size))

            if flag == 1:
                err_activation_output = err_activation_output + err_intermediate

            activations_grad.activations = self.hidden_states[(i + 1), :]
            gradient_tan = activations_grad.backward(err_activation_output)

            input_stack = np.hstack((self.hidden_states[i, :], self.input_tensor[i, :]))

            self.fc_activation.input = np.hstack((input_stack.reshape(1, stacked_size), bias))
            intermediate_tensor[:] = self.fc_activation.backward(gradient_tan.reshape(1, self.hidden_size))

            err_intermediate[:] = intermediate_tensor[:, 0:self.hidden_size]
            err_prev[i, :] = intermediate_tensor[:, self.hidden_size:stacked_size]
            flag = 1
            bpt_len += 1

            if bpt_len <= self.bpt_length:
                self.gradients_output += self.fc_output.get_gradient_weights()
                self.gradients_activation += self.fc_activation.get_gradient_weights()

        if self.flag_opt == 1:
            self.fc_output.weights = self.optimizer.calculate_update(self.delta, self.fc_output.weights,
                                                                     self.gradients_output)

            self.fc_activation.weights = self.optimizer.calculate_update(self.delta, self.fc_activation.weights,
                                                                         self.gradients_activation)

        return err_prev

    def set_optimizer(self, optimizer: Optimizer):
        """Sets the optimizer"""

        self.optimizer = optimizer
        self.flag_opt = 1

    def get_gradient_weights(self):
        """Get the gradients of the weights"""

        return self.gradients_activation

    def get_weights(self):
        """Get the weights"""

        return self.fc_activation.weights

    def set_weights(self, weights: np.ndarray):
        """Sets the weights"""

        self.fc_activation.weights = weights

    def initialize(self, weights_initializer: Initializers, bias_initializer: Initializers):
        """Sets the weights and biases"""

        self.fc_activation.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        """Calculates the regularization loss"""

        weights = self.get_weights()
        return self.optimizer.regularizer.norm(weights)

# -----------------------------------------------------------------
