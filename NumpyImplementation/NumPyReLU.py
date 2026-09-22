import numpy as np


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        self.input = input_data

        return np.maximum(0, input_data)

    def backward(self, d_output):
        d_input = d_output.copy()

        d_input[self.input <= 0] = 0

        return d_input