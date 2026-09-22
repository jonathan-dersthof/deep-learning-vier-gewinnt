import numpy as np


class MSELoss:
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean(np.square(predictions - targets))

    def backward(self):
        n = self.predictions.shape[0]
        return (2.0 / n) * (self.predictions - self.targets)
