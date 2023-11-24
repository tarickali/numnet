"""
title : xor.py
create : @tarickali 23/11/20
update : @tarickali 23/11/24
"""

import numpy as np
import matplotlib.pyplot as plt

from core.types import Network, History
from core.engine import initialize, forward, backward, update
from core.activations import sigmoid
from core.losses import binary_crossentropy
from core.metrics import accuracy

__all__ = ["xor_driver"]


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate data for the XOR problem.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] : x @ (4, 2), y @ (4, 1)

    """

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1, 1, 0]]).reshape(-1, 1)

    return x, y


def train(
    network: Network,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    epochs: int = 300,
) -> History:
    """Training function used for the XOR problem.

    Parameters
    ----------
    network : Network
    X : np.ndarray @ (4, 2)
    y : np.ndarray @ (4, 1)
    alpha : float = 0.1
    epochs : int = 300

    Returns
    -------
    History

    """

    history = []
    for e in range(epochs):
        o, cache = forward(network, X)
        loss, grad = binary_crossentropy(y, o)
        gradients = backward(network, grad, cache)
        network = update(network, gradients, alpha)

        p = np.round(sigmoid(o))
        acc = accuracy(y, p)
        history.append({"epoch": e, "loss": loss, "acc": acc})
        print(history[-1])

    return history


def xor_driver():
    x, y = generate_data()

    architecture = [
        {"input_dim": 2, "output_dim": 4, "weight_init": 1.0, "activation": "sigmoid"},
        {"input_dim": 4, "output_dim": 4, "weight_init": 1.0, "activation": "sigmoid"},
        {"input_dim": 4, "output_dim": 1, "weight_init": 1.0, "activation": "linear"},
    ]

    network = initialize(architecture, seed=0)

    history = train(network, x, y, alpha=0.5, epochs=1000)

    losses = [epoch["loss"] for epoch in history]
    plt.plot(losses)
    plt.show()
