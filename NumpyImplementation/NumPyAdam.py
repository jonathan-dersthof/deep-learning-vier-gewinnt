import numpy as np

""" Mithilfe von KI implementiert """
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}

        self.t = 0

    def step(self, model):
        self.t += 1

        for layer in model.layers:
            if hasattr(layer, 'weights'):
                layer_id = id(layer)

                if layer_id not in self.m:
                    self.m[layer_id] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}
                    self.v[layer_id] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}

                self.m[layer_id]['w'] = self.beta1 * self.m[layer_id]['w'] + (1 - self.beta1) * layer.d_weights
                self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * layer.d_biases

                self.v[layer_id]['w'] = self.beta2 * self.v[layer_id]['w'] + (1 - self.beta2) * (layer.d_weights ** 2)
                self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * (layer.d_biases ** 2)

                m_hat_w = self.m[layer_id]['w'] / (1 - self.beta1 ** self.t)
                m_hat_b = self.m[layer_id]['b'] / (1 - self.beta1 ** self.t)

                v_hat_w = self.v[layer_id]['w'] / (1 - self.beta2 ** self.t)
                v_hat_b = self.v[layer_id]['b'] / (1 - self.beta2 ** self.t)

                layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                layer.biases -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
