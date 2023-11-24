"""
title : mnist.py
create : @tarickali 23/11/22
update : @tarickali 23/11/22
"""

import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from core.types import Network, History
from core.engine import initialize, forward, backward, update
from core.activations import softmax
from core.losses import categorical_crossentropy
from core.metrics import accuracy

__all__ = ["mnist_driver"]


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate data for the MNIST problem.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] : X @ (n, 28, 28), y @ (n, 10)

    """

    train_df = pd.read_csv("data/mnist/train.csv")

    pixels = train_df.drop("label", axis=1).to_numpy()
    labels = train_df["label"].to_numpy()

    X = pixels.reshape((-1, 784)) / 255.0
    y = one_hot(labels)

    return X, y


def one_hot(x: np.ndarray, k: int = 10) -> np.ndarray:
    """ """

    n = x.shape[0]
    o = np.zeros((n, k))
    o[np.arange(n), x] = 1
    return o


def get_batches(
    X: np.ndarray, y: np.ndarray, m: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """ """

    n = X.shape[0]
    batches = []
    for i in range(n // m):
        a, b = i * m, (i + 1) * m
        batches.append((X[a:b], y[a:b]))
    if b != n:
        batches.append((X[b:], y[b:]))
    return batches


def train(
    network: Network,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    m: int = 32,
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

    n, _ = X.shape

    history = []
    for e in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for Xb, yb in get_batches(X, y, m):
            o, cache = forward(network, Xb)
            loss, grad = categorical_crossentropy(yb, o)
            gradients = backward(network, grad, cache)
            network = update(network, gradients, alpha)

            p = one_hot(np.argmax(softmax(o), axis=1), k=10)
            acc = accuracy(yb, p)

            epoch_loss += loss * m
            epoch_acc += acc * m
        history.append({"epoch": e, "loss": epoch_loss / n, "acc": epoch_acc / n})
        print(history[-1])

    return history


def mnist_driver():
    X, y = generate_data()
    n = X.shape[0]
    X, y = X[: int(0.8 * n)], y[: int(0.8 * n)]

    architecture = [
        {"input_dim": 784, "output_dim": 512, "activation": "relu"},
        {"input_dim": 512, "output_dim": 512, "activation": "relu"},
        {"input_dim": 512, "output_dim": 512, "activation": "relu"},
        {"input_dim": 512, "output_dim": 10, "activation": "linear"},
    ]

    network = initialize(architecture, weight_init=0.01, seed=0)

    start = time.time()
    history = train(network, X, y, alpha=0.01, epochs=5)
    end = time.time()
    print(end - start)

    losses = [epoch["loss"] for epoch in history]
    plt.plot(losses)
    plt.show()
