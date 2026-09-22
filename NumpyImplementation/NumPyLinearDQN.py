from .NumPyReLU import ReLU
from .NumPyLinearLayer import Linear


class LinearDQN:
    def __init__(self,
                 input_size : int,
                 output_size : int,
                 hidden_size : int,
                 hidden_layers : int
                 ):
        self.layers : list = []

        if hidden_layers > 0:
            self.layers.append(Linear(input_size, hidden_size))
            self.layers.append(ReLU())

            for layer in range(hidden_layers - 1):
                if hidden_layers - 1 - layer > 0:
                    self.layers.append(Linear(hidden_size, hidden_size))

            self.layers.append(Linear(hidden_size, output_size))
        else:
            self.layers.append(Linear(input_size, output_size))

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)

        return input_data

    def backward(self, d_output):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)

        return d_output
