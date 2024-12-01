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
        return outputs, mask

if __name__ == "__main__":
    test_dropout = Dropout(p=0.2)
    inputs = np.random.randint(1,10, (2,3))
    outputs, mask = test_dropout.forward(inputs, train=True)
    print(f"mask: {mask}")
    print(f"input: {inputs}")
    print(f"outputs: {outputs}")