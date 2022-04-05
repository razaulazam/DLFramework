import tanh
import numpy as np
import fully_connected as fc
import sigmoid as S

from optimizers.optimizer import Optimizer
from initializers.initializers import Initializers

# -----------------------------------------------------------------

class LSTM:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, bptt_length: int):
        """Implementation of the LSTM layer"""

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bptt_length = bptt_length
        self.sequence_flag = True

        self.optimizer = None
        self.delta = 1

        self.fc_layer1 = fc.FullyConnected(self.input_size + self.hidden_size, 4 * self.hidden_size)
        self.fc_layer3 = fc.FullyConnected(self.hidden_size, self.output_size)
        self.first_hidden = np.zeros(self.hidden_size)
        self.first_cells = np.zeros(self.hidden_size)
        self.record_gate = None
        self.input = None
        self.time_dim = None
        self.hiddens = None
        self.cell_states = None
        self._gradient_weights = None
        self._gradient_weights_output = None
        self.flag_opt = 0

    def toggle_memory(self):
        """Toggles the memory"""

        self.sequence_flag = not self.sequence_flag

    def forward(self, input_tensor: np.ndarray):
        """Does the forward pass computations"""

        self.input = input_tensor
        self.time_dim = input_tensor.shape[0]
        self.hiddens = np.zeros((self.time_dim + 1, self.hidden_size), dtype=float)
        self.cell_states = np.zeros((self.time_dim + 1, self.hidden_size), dtype=float)
        stack_size = self.hidden_size + self.input_size
        self.record_gate = np.zeros((self.time_dim, 4 * self.hidden_size), dtype=float)
        output_tensor = np.zeros((self.time_dim, self.output_size), dtype=float)

        if self.sequence_flag is False:
            self.hiddens[0, :] = self.first_hidden
            self.cell_states[0, :] = self.first_cells

        for dim in range(self.time_dim):
            input_hat = np.concatenate((self.hiddens[dim, :], input_tensor[dim, :]), axis=0)
            new_hidden = self.fc_layer1.forward(input_hat.reshape(stack_size, 1).T)

            new1, new2, new3, new4 = np.hsplit(new_hidden,
                                               [self.hidden_size, 2 * self.hidden_size, 3 * self.hidden_size])

            sigmoid_layer = S.Sigmoid()
            inputGate = sigmoid_layer.forward(new2)
            forgetGate = sigmoid_layer.forward(new1)
            outputGate = sigmoid_layer.forward(new4)

            tan_layer = tanh.TanH()
            c_hat = tan_layer.forward(new3)

            self.record_gate[dim, :] = np.hstack(
                (forgetGate, inputGate, c_hat, outputGate))

            self.cell_states[dim + 1, :] = (inputGate * c_hat) + (forgetGate * self.cell_states[dim, :])
            self.hiddens[dim + 1, :] = outputGate * tan_layer.forward(self.cell_states[dim + 1, :])
            output_tensor[dim, :] = self.fc_layer3.forward(self.hiddens[(dim + 1), :].reshape(1, self.hidden_size))

        self.first_hidden = self.hiddens[self.time_dim, :]
        self.first_cells = self.cell_states[self.time_dim, :]

        return output_tensor

    def backward(self, error_tensor: np.ndarray):
        """Does the backward pass computations with truncated back-propagation"""

        self._gradient_weights_output = np.zeros((self.hidden_size + 1, self.output_size), dtype=float)
        stacked_size = self.hidden_size + self.input_size
        self._gradient_weights = np.zeros((stacked_size + 1, 4 * self.hidden_size), dtype=float)
        err_intermediate = np.zeros((1, self.hidden_size), dtype=float)
        grad_prev_state = np.zeros((1, self.hidden_size), dtype=float)
        backprop_error_tensor = np.zeros((self.time_dim, self.input_size), dtype=float)
        counter_ = 0
        bias = np.ones((1, 1), dtype=int)
        tan_layer = tanh.TanH()
        flag = 0

        for dim in reversed(range(self.time_dim)):
            self.fc_layer3.input = np.hstack((self.hiddens[(dim + 1), :].reshape(1, self.hidden_size), bias))
            err_activation_output = self.fc_layer3.backward(error_tensor[dim, :].reshape(1, self.output_size))

            if flag == 1:
                err_activation_output = err_activation_output + err_intermediate

            forgetGate, inputGate, c_hat, outputGate = np.hsplit(self.record_gate[dim, :],
                                                                 [self.hidden_size, 2 * self.hidden_size,
                                                                  3 * self.hidden_size])
            forgetGate = forgetGate.reshape(1, self.hidden_size)
            inputGate = inputGate.reshape(1, self.hidden_size)
            c_hat = c_hat.reshape(1, self.hidden_size)
            outputGate = outputGate.reshape(1, self.hidden_size)

            temp = tan_layer.forward(self.cell_states[dim + 1, :])
            tan_sq = (1 - (temp ** 2))

            gradient_state = outputGate * err_activation_output * tan_sq.reshape(1, self.hidden_size)
            gradient_cells = gradient_state + grad_prev_state

            gradient_forget_gate = self.cell_states[dim, :].reshape(1, self.hidden_size) * gradient_cells
            gradient_forget_gate = forgetGate * (1 - forgetGate) * gradient_forget_gate

            gradient_input = c_hat * gradient_cells
            gradient_input = inputGate * (1 - inputGate) * gradient_input

            c_square = 1 - (np.square(c_hat))
            gradient_c_hat = inputGate * gradient_cells
            gradient_c_hat = c_square * gradient_c_hat

            gradient_output = tan_layer.forward(self.cell_states[dim + 1, :]).reshape(1,
                                                                                      self.hidden_size) * err_activation_output
            gradient_output = outputGate * (1 - outputGate) * gradient_output

            grad_prev_state = forgetGate * gradient_cells

            stacked_gates_results = np.hstack((gradient_forget_gate, gradient_input, gradient_c_hat, gradient_output))

            input_stack = np.hstack((self.hiddens[dim, :], self.input[dim, :]))
            self.fc_layer1.input = np.hstack((input_stack.reshape(1, stacked_size), bias))

            temp = self.fc_layer1.backward(stacked_gates_results)
            err_intermediate = temp[:, :self.hidden_size].reshape(1, self.hidden_size)
            grad_x = temp[:, self.hidden_size:]
            
            flag = 1
            backprop_error_tensor[dim, :] = grad_x.reshape(self.input_size)
            counter_ += 1

            if counter_ <= self.bptt_length:
                self._gradient_weights_output += self.fc_layer3.get_gradient_weights()
                self._gradient_weights += self.fc_layer1.get_gradient_weights()

        if self.flag_opt == 1:
            self.fc_layer3.weights = self.optimizer.calculate_update(self.delta, self.fc_layer3.weights,
                                                                     self._gradient_weights_output)
            self.fc_layer1.weights = self.optimizer.calculate_update(self.delta, self.fc_layer1.weights,
                                                                     self._gradient_weights)

        return backprop_error_tensor

    def get_gradient_weights(self):
        """Returns the gradients of the weights"""

        return self._gradient_weights

    def get_weights(self):
        """Returns the weights"""

        return self.fc_layer1.get_weights()

    def set_weights(self, weights: np.ndarray):
        """Sets the weights"""

        self.fc_layer1.set_weights(weights)

    def initialize(self, weights_initializer: Initializers, bias_initializer: Initializers):
        """Initializes the weights and biases"""

        self.fc_layer1.initialize(weights_initializer, bias_initializer)
        self.fc_layer3.initialize(weights_initializer, bias_initializer)

    def set_optimizer(self, optimizer: Optimizer):
        """Sets the optimizer"""

        self.optimizer = optimizer
        self.flag_opt = 1

    def calculate_regularization_loss(self):
        """Calculates loss term due to L2 regularization"""

        weights = self.get_weights()
        return self.optimizer.regularizer.norm(weights)

# -----------------------------------------------------------------
