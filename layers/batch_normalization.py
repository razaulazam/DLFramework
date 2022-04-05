import copy
import numpy as np

from layers.base import BaseClass, Phase
from optimizers.optimizer import Optimizer

# -----------------------------------------------------------------

class BatchNormalization(BaseClass):
    def __init__(self, channels:int = 0):
        """

        Implementation of Batch Normalization for avoiding problems due to
        internal covariate shifts
        
        """

        super().__init__()
        self.weights = None
        self.bias = None
        self.channels = channels
        self.alpha = 0.8
        self.running_mean = None
        self.running_variance = None
        self.phase = Phase.train.value
        self.epsilon = 1e-10
        self.weight_optimizer = None
        self.bias_optimizer = None
        self.delta = 1
        self.input_tensor_store = None
        self.input_tensor_scaled = None
        self.normalized_tensor = None
        self.mean = None
        self.var = None
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input_tensor: np.ndarray):
        """Does the forward pass computations that involve scaling the values of every channel in the activation feature map"""
        
        if self.phase == Phase.test.value:
            if self.channels != 0:
                B, H, M, N = input_tensor.shape

                input_tensor_reshaped = input_tensor.reshape(B, H, M * N)
                input_tensor_reshaped = np.transpose(input_tensor_reshaped, (0, 2, 1))
                input_tensor_reshaped = input_tensor_reshaped.reshape(B * M * N, H)
                self.input_tensor_store = input_tensor_reshaped

                if self.weights is None:
                    self.weights = np.ones(input_tensor_reshaped.shape[1])
                if self.bias is None:
                    self.bias = np.zeros(input_tensor_reshaped.shape[1])

                self.input_tensor_scaled = (input_tensor_reshaped - self.running_mean) / np.sqrt(
                    self.running_variance + 1e-8)
                temp = np.multiply(self.input_tensor_scaled, self.weights)
                res = temp + self.bias

                res = res.reshape(B, M * N, H)
                res = np.transpose(res, (0, 2, 1))
                self.normalized_tensor = res.reshape(B, H, M, N)
            else:
                if (self.weights or self.bias) is None:
                    self.weights = np.ones(input_tensor.shape[1])
                    self.bias = np.zeros(input_tensor.shape[1])

                self.input_tensor_store = input_tensor
                self.input_tensor_scaled = (input_tensor - self.running_mean) / np.sqrt(self.running_variance +
                                                                                        self.epsilon)
                temp = np.multiply(self.weights, self.input_tensor_scaled)
                self.normalized_tensor = temp + self.bias

            return self.normalized_tensor

        elif self.phase == Phase.train: 

            if self.channels != 0:

                B, H, M, N = input_tensor.shape

                input_tensor_reshaped = input_tensor.reshape(B, H, M * N)
                input_tensor_reshaped = np.transpose(input_tensor_reshaped, (0, 2, 1))
                input_tensor_reshaped = input_tensor_reshaped.reshape(B * M * N, H)
                self.input_tensor_store = input_tensor_reshaped

                self.mean = np.mean(input_tensor_reshaped, axis=0)
                self.var = np.var(input_tensor_reshaped, axis=0)

                if self.running_mean is None:
                    self.running_mean = self.mean
                else:
                    self.running_mean = self.running_mean * (1 - self.alpha) + (self.alpha * self.mean)

                if self.running_variance is None:
                    self.running_variance = self.var
                else:
                    self.running_variance = self.running_variance * (1 - self.alpha) + (self.alpha * self.var)

                if (self.weights or self.bias) is None:
                    self.weights = np.ones(input_tensor.shape[1])
                    self.bias = np.zeros(input_tensor.shape[1])

                self.input_tensor_scaled = (input_tensor_reshaped - self.mean) / np.sqrt(self.var + self.epsilon)
                temp = np.multiply(self.input_tensor_scaled, self.weights)
                res = temp + self.bias

                res = res.reshape(B, M * N, H)
                res = np.transpose(res, (0, 2, 1))
                self.normalized_tensor = res.reshape(B, H, M, N)

            else:
                self.input_tensor_store = input_tensor
                self.mean = np.mean(input_tensor, axis=0)
                self.var = np.var(input_tensor, axis=0)

                if (self.weights or self.bias) is None:
                    self.weights = np.ones(input_tensor.shape[1])
                    self.bias = np.zeros(input_tensor.shape[1])

                if self.running_mean is None:
                    self.running_mean = self.mean
                else:
                    self.running_mean = self.running_mean * (1 - self.alpha) + (self.alpha * self.mean)

                if self.running_variance is None:
                    self.running_variance = self.var
                else:
                    self.running_variance = self.running_variance * (1 - self.alpha) + (self.alpha * self.var)

                self.input_tensor_scaled = (input_tensor - self.mean) / np.sqrt(self.var + self.epsilon)
                temp = np.multiply(self.weights, self.input_tensor_scaled)
                self.normalized_tensor = temp + self.bias

            return self.normalized_tensor

    def backward(self, error_tensor: np.ndarray):
        """Does the backward pass computations"""

        if self.channels != 0:
            B, H, M, N = error_tensor.shape

            error_tensor = error_tensor.reshape(B, H, M * N)
            error_tensor = np.transpose(error_tensor, (0, 2, 1))
            error_tensor = error_tensor.reshape(B * M * N, H)

            no_batch = error_tensor.shape[0]

            self.gradient_weights = np.sum((error_tensor * self.input_tensor_scaled), axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)

            gradient_input_temp = error_tensor * self.weights

            sub_mean = self.input_tensor_store - self.mean
            sum_input_temp = np.sum((gradient_input_temp * sub_mean), axis=0)
            power_var = np.power((self.var + self.epsilon), -1.5)

            gradient_variance = sum_input_temp * (-0.5) * power_var

            gradient_mult = gradient_input_temp * (-1 / np.sqrt(self.var + self.epsilon))
            gradient_mean = np.sum(gradient_mult, axis=0)

            temp1 = gradient_input_temp * (1 / np.sqrt(self.var + self.epsilon))
            temp2 = gradient_variance * ((2 * (self.input_tensor_store - self.running_mean)) / no_batch)
            temp3 = gradient_mean / no_batch

            gradient_input = temp1 + temp2 + temp3

            if self.weight_optimizer is not None:
                self.weights = self.weight_optimizer.calculate_update(self.delta, self.weights, self.gradient_weights)

            if self.bias_optimizer is not None:
                self.bias = self.bias_optimizer.calculate_update(self.delta, self.bias, self.gradient_bias)

            gradient_input = gradient_input.reshape(B, M * N, H)
            gradient_input = np.transpose(gradient_input, (0, 2, 1))
            gradient_input = gradient_input.reshape(B, H, M, N)

        else:
            no_batch = error_tensor.shape[0]
            self.gradient_weights = np.sum((error_tensor * self.input_tensor_scaled), axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)

            gradient_input_temp = error_tensor * self.weights

            sub_mean = self.input_tensor_store - self.mean
            sum_input_temp = np.sum((gradient_input_temp * sub_mean), axis=0)
            power_var = np.power((self.var + self.epsilon), -1.5)

            gradient_variance = sum_input_temp * (-0.5) * power_var

            gradient_mult = gradient_input_temp * (-1 / np.sqrt(self.var + self.epsilon))
            gradient_mean = np.sum(gradient_mult, axis=0)

            temp_one = gradient_input_temp * (1 / np.sqrt(self.var + self.epsilon))
            temp_two = gradient_variance * ((2 * (self.input_tensor_store - self.running_mean)) / no_batch)
            temp_three = gradient_mean / no_batch

            gradient_input = temp_one + temp_two + temp_three

            if self.weight_optimizer is not None:
                self.weights = self.weight_optimizer.calculate_update(self.delta, self.weights, self.gradient_weights)

            if self.bias_optimizer is not None:
                self.bias = self.bias_optimizer.calculate_update(self.delta, self.bias, self.gradient_bias)

        return gradient_input

    @property
    def gradient_weights(self):
        return self.gradient_weights

    @property
    def gradient_bias(self):
        return self.gradient_bias

    @property
    def weights(self):
        return self.weights

    @weights.setter
    def weights(self, weight: np.ndarray):
        self.weights = weight

    def set_optimizer(self, optimizer: Optimizer):
        self.weight_optimizer = optimizer
        self.bias_optimizer = copy.deepcopy(optimizer)

# -----------------------------------------------------------------
