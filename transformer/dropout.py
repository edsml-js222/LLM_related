import numpy as np

class Dropout:
    def __init__(self, p=0.1):
        self.p = p
    def forward(self, inputs, train=True):
        if train:
            mask = np.random.binomial(1, 1-self.p, size=inputs.shape)
            outputs = inputs * mask / (1 - self.p)
        else:
            outputs = inputs
        return outputs
