import numpy as np


def he_init(x, input_size):
    return x * np.sqrt(2. / input_size)

class Linear:
    def __init__(self, input_size : int, output_size : int):
        self.weights = he_init(np.random.randn(input_size, output_size))
        self.biases = np.zeros((1, output_size))

        self.input_data = None

        self.d_weights = None
        self.d_biases = None

    def forward(self, input_data):
        self.input_data = input_data

        return np.dot(input_data, self.weights) + self.biases

    def backward(self, d_output):
        self.d_weights = np.dot(self.input_data.T, d_output)
        self.d_biases = np.sum(d_output, axis=0, keepdims=True)

        d_input = np.dot(d_output, self.weights.T)

        return d_input
