"""
title : main.py
create : @tarickali 23/11/19
update : @tarickali 23/11/19
"""

import numpy as np
import matplotlib.pyplot as plt

from core.engine import initialize, forward, backward, update, train
from core.activations import sigmoid
from core.losses import binary_crossentropy
from core.metrics import accuracy


def main():
    architecture = [
        {"input_dim": 2, "output_dim": 3, "activation": "sigmoid"},
        {"input_dim": 3, "output_dim": 1, "activation": "linear"},
    ]

    network = initialize(architecture, 0)

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1, 1, 0]]).reshape(-1, 1)

    # o, cache = forward(network, x)
    # cost, grad = binary_crossentropy(y, o)
    # gradients = backward(network, grad, cache)
    # update(network, gradients, 0.001)

    history = train(network, x, y, alpha=0.03, epochs=500)

    losses = [epoch["loss"] for epoch in history]
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()
