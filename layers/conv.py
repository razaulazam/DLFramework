import math
import scipy
import copy
import numpy as np

from typing import Tuple
from scipy import signal
from optimizers.optimizer import Optimizer
from initializers.initializers import Initializers

# -----------------------------------------------------------------

class Conv:
    def __init__(self, stride_shape: Tuple[int, int], convolution_shape: Tuple[int, int], num_kernel: int, learning_rate: float):
        """Implementation of the Convolutional layer used in a Convolutional Neural Network"""

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernel = num_kernel
        self.learning_rate = learning_rate
        self.delta = 1
        self.weight_optimizer = None
        self.bias_optimizer = None
        self.flag_set = True
        self.input_cache = None
        self.bias_gradient = None
        self.weight_gradient = None
        if len(self.stride_shape) == 2:
            self.weights = np.random.random((self.num_kernel, self.convolution_shape[0],
                                            self.convolution_shape[1], self.convolution_shape[2]))

        if len(self.stride_shape) == 1:
            self.weights = np.random.random((self.num_kernel, self.convolution_shape[0],
                                            self.convolution_shape[1]))

        self.bias = np.random.random((num_kernel, 1))

    def forward(self, input_tensor: np.ndarray):
        """Forward method called in the forward pass"""

        stride_dim = len(self.stride_shape)

        if stride_dim == 2:
            batch_size = input_tensor.shape[0]
            channels = input_tensor.shape[1]
            input_tensor_height = input_tensor.shape[2]
            input_tensor_width = input_tensor.shape[3]

            dim_x = math.ceil(input_tensor_height/self.stride_shape[0])
            dim_y = math.ceil(input_tensor_width/self.stride_shape[1])

            pad_x = max((((dim_x - 1) * self.stride_shape[0]) + self.convolution_shape[1] - input_tensor_height), 0)
            pad_y = max((((dim_y - 1) * self.stride_shape[1]) + self.convolution_shape[2] - input_tensor_width), 0)

            pad_top = pad_x // 2
            pad_bottom = pad_x - pad_top

            pad_left = pad_y // 2
            pad_right = pad_y - pad_left

            input_tensor_padded = np.zeros((batch_size, channels, (input_tensor_height + pad_x), (input_tensor_width +
                                                                                                  pad_y)), dtype=float)
            if pad_bottom == 0 and pad_right != 0:
                input_tensor_padded[:, :, pad_top:input_tensor_height, pad_left:-pad_right] = input_tensor

            elif pad_right == 0 and pad_bottom != 0:
                input_tensor_padded[:, :, pad_top:-pad_bottom, pad_left:input_tensor_width] = input_tensor

            elif pad_right == 0 and pad_bottom == 0:
                input_tensor_padded[:, :, pad_top:input_tensor_height, pad_left:input_tensor_width] = input_tensor

            else:
                input_tensor_padded[:, :, pad_top:-pad_bottom, pad_left:-pad_right] = input_tensor

            self.input_cache = input_tensor
            convolve_result = np.zeros((batch_size, self.num_kernel, dim_x, dim_y))

            for num_images in range(batch_size):
                for num_kernel in range(self.num_kernel):
                    result = np.zeros((dim_x, dim_y), dtype=float)
                    for num_channel in range(channels):
                        result = result + scipy.signal.correlate2d(input_tensor_padded[num_images, num_channel, :, :],
                                                                   self.weights[num_kernel, num_channel, :, :],
                                                                   mode='valid')[::self.stride_shape[0], ::self.stride_shape[1]]
                    convolve_result[num_images, num_kernel, :, :] = result + self.bias[num_kernel]

        if stride_dim == 1: 
            batch_size = input_tensor.shape[0]
            channels = input_tensor.shape[1]
            input_tensor_height = input_tensor.shape[2]

            dim_x = math.ceil(input_tensor_height / self.stride_shape[0])

            pad_x = max((((dim_x - 1) * self.stride_shape[0]) + self.convolution_shape[1] - input_tensor_height), 0)
            pad_top = pad_x // 2
            pad_bottom = pad_x - pad_top

            input_tensor_padded = np.zeros((batch_size, channels, (input_tensor_height + pad_x)), dtype=float)

            if pad_bottom == 0:
                input_tensor_padded[:, :, pad_top:input_tensor_height] = input_tensor
            else:
                input_tensor_padded[:, :, pad_top:-pad_bottom] = input_tensor

            self.input_cache = input_tensor
            convolve_result = np.zeros((batch_size, self.num_kernel, dim_x))

            for num_images in range(batch_size):
                for num_kernel in range(self.num_kernel):
                    result = np.zeros(dim_x, dtype=float)
                    for num_channel in range(channels):
                        result = result + scipy.signal.correlate(input_tensor_padded[num_images, num_channel, :],
                                                                 self.weights[num_kernel, num_channel, :],
                                                                 mode='valid')[::self.stride_shape[0]]

                    convolve_result[num_images, num_kernel, :] = result + self.bias[num_kernel]

        return convolve_result

    def initialize(self, weights_initializer: Initializers, bias_initializer: Initializers):
        """Initialize the weights and bias"""

        if len(self.stride_shape) == 2:
            fan_in = np.prod(self.convolution_shape)
            fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernel
            weights_shape = (self.num_kernel, *self.convolution_shape)

            weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
            self.weights = weights

        if len(self.stride_shape) == 1:
            fan_in = np.prod(self.convolution_shape)
            fan_out = self.convolution_shape[1] * self.num_kernel
            weights_shape = (self.num_kernel, *self.convolution_shape)

            weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
            self.weights = weights

        bias_shape = (self.num_kernel, 1)
        self.bias = bias_initializer.initialize(bias_shape, self.num_kernel, 1)

    def set_optimizer(self, optimizer: Optimizer):
        """Set the optimizer"""

        self.weight_optimizer = optimizer
        self.bias_optimizer = copy.deepcopy(optimizer)
        self.flag_set = not self.flag_set

    def get_gradient_bias(self):
        """Returns the gradients of the loss with respect to the bias vector"""
        
        return self.bias_gradient

    def backward(self, error_tensor: np.ndarray):
        """For computing the backward pass computations"""
        
        stride_dim = len(self.stride_shape)

        if stride_dim == 2:
            bias_gradient = np.sum(error_tensor, axis=(0, 2, 3))
            self.bias_gradient = bias_gradient.reshape(self.num_kernel, -1)

            batch_size = error_tensor.shape[0]
            channels = error_tensor.shape[1]

            channels_original = self.input_cache.shape[1]
            height_original = self.input_cache.shape[2]
            width_original = self.input_cache.shape[3]

            weight_shape = self.weights.shape
            weights = copy.deepcopy(self.weights)
            weights_new = np.zeros((weight_shape[1], weight_shape[0], weight_shape[2], weight_shape[3]))

            for i in range(channels_original):
                for j in range(self.num_kernel):
                    weights_new[i, j, :, :] = weights[j, i, :, :]

            weights_new = weights_new[:, ::-1, :, :]
            new_error_tensor = np.zeros((batch_size, channels_original, height_original, width_original), dtype=float)
            input_error_tensor = np.zeros((batch_size, channels, height_original, width_original), dtype=float)
            input_error_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor

            pad_input = max(((height_original - 1) + self.convolution_shape[1] - height_original), 0)
            pad_input_width = max(((width_original - 1) + self.convolution_shape[2] - width_original), 0)
            pad_top = pad_input // 2
            pad_bottom = pad_input - pad_top
            pad_left = pad_input_width // 2
            pad_right = pad_input_width - pad_left

            input_error_tensor_padded = np.zeros((batch_size, channels, (input_error_tensor.shape[2] + pad_input),
                                              (input_error_tensor.shape[3] + pad_input_width)), dtype=float)

            if pad_bottom == 0 and pad_right != 0:
                input_error_tensor_padded[:, :, pad_top:height_original, pad_left:-pad_right] = input_error_tensor

            elif pad_right == 0 and pad_bottom != 0:
                input_error_tensor_padded[:, :, pad_top:-pad_bottom, pad_left:width_original] = input_error_tensor

            elif pad_right == 0 and pad_bottom == 0:
                input_error_tensor_padded[:, :, pad_top:height_original, pad_left:width_original] = input_error_tensor

            else:
                input_error_tensor_padded[:, :, pad_top:-pad_bottom, pad_left:-pad_right] = input_error_tensor

            for num_batch in range(batch_size):
                for num_kernel in range(channels_original):
                    new_error_tensor[num_batch, num_kernel, :, :] = scipy.signal.convolve(weights_new[num_kernel, :, :, :],
                                                                                          input_error_tensor_padded[num_batch, :, :, :],
                                                                                          mode='valid')

            self.weight_gradient = np.zeros((self.num_kernel, self.convolution_shape[0], self.convolution_shape[1],
                                        self.convolution_shape[2]), dtype=float)
            input_tensor_required = np.zeros((batch_size, channels_original, height_original + pad_input,
                                          width_original + pad_input_width), dtype=float)

            if pad_bottom == 0 and pad_right != 0:
                input_tensor_required[:, :, pad_top:height_original, pad_left:-pad_right] = self.input_cache

            elif pad_right == 0 and pad_bottom != 0:
                input_tensor_required[:, :, pad_top:-pad_bottom, pad_left:width_original] = self.input_cache

            elif pad_right == 0 and pad_bottom == 0:
                input_tensor_required[:, :, pad_top:height_original, pad_left:width_original] = self.input_cache

            else:
                input_tensor_required[:, :, pad_top:-pad_bottom, pad_left:-pad_right] = self.input_cache

            for kernels in range(self.num_kernel):
                for num_channel in range(channels_original):
                    self.weight_gradient[kernels, num_channel, :, :] = scipy.signal.correlate(
                        input_tensor_required[:, num_channel, :, :],
                        input_error_tensor[:, kernels, :, :],
                        mode='valid')

            if self.flag_set is False:
                self.weights = self.weight_optimizer.calculate_update(self.delta, self.weights, self.weight_gradient)
                self.bias = self.bias_optimizer.calculate_update(self.delta, self.bias, self.bias_gradient)

        if stride_dim == 1:
            batch_size = error_tensor.shape[0]
            channels = error_tensor.shape[1]

            channels_original = self.input_cache.shape[1]
            height_original = self.input_cache.shape[2]

            new_error_tensor = np.zeros((batch_size, channels_original, height_original), dtype=float)
            input_error_tensor = np.zeros((batch_size, channels, height_original), dtype=float)
            input_error_tensor[:, :, ::self.stride_shape[0]] = error_tensor

            pad_input = max(((height_original - 1) + self.convolution_shape[1] - height_original), 0)

            pad_top = pad_input // 2
            pad_bottom = pad_input - pad_top

            input_error_tensor_padded = np.zeros((batch_size, channels, (input_error_tensor.shape[2] + pad_input)),
                                                 dtype=float)

            if pad_bottom == 0:
                input_error_tensor_padded[:, :, pad_top:height_original] = input_error_tensor

            else:
                input_error_tensor_padded[:, :, pad_top:-pad_bottom] = input_error_tensor

            weights = self.weights.copy()
            weights = weights.reshape(weights.shape[1], weights.shape[0], weights.shape[2])
            for num_batch in range(batch_size):
                for num_kernel in range(channels_original):
                    new_error_tensor[num_batch, num_kernel, :] = scipy.signal.convolve(
                        input_error_tensor_padded[num_batch, :, :],
                        weights[num_kernel, :, :],
                        mode='valid')

        return new_error_tensor

    def get_gradient_weights(self):
        """Get the gradients of the weights"""

        return self.weight_gradient

# -----------------------------------------------------------------
