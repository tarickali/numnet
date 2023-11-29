"""
title : sinusoidal.py
create : @tarickali 23/11/24
update : @tarickali 23/11/28
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from core.types import Network, History
from core.engine import initialize, forward, backward, update
from core.losses import mse

__all__ = ["regression_driver"]


def train(
    network: Network,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    epochs: int = 300,
) -> History:
    """Training function used for the regression problem.

    Parameters
    ----------
    network : Network
    X : np.ndarray @ (m, n_feats)
    y : np.ndarray @ (m, 1)
    alpha : float = 0.1
    epochs : int = 300

    Returns
    -------
    History

    """

    history = []
    for e in range(epochs):
        o, cache = forward(network, X)
        loss, grad = mse(y, o)
        gradients = backward(network, grad, cache)
        update(network, gradients, alpha, inplace=True)

        history.append({"epoch": e, "loss": loss})
        print(history[-1])

    return history


def regression_driver():
    # Create regression data
    X, y = make_regression(n_samples=200, n_features=10)
    y = y.reshape(-1, 1)

    # Normalize data
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Network and training hyperparameters
    EPOCHS = 300
    ALPHA = 0.1
    architecture = [
        {"input_dim": 10, "output_dim": 16, "weight_init": 0.05, "activation": "relu"},
        {"input_dim": 16, "output_dim": 16, "weight_init": 0.05, "activation": "relu"},
        {"input_dim": 16, "output_dim": 1, "weight_init": 0.05, "activation": "linear"},
    ]

    # Initialize network
    network = initialize(architecture, seed=0)

    # Train network
    history = train(network, X_train, y_train, alpha=ALPHA, epochs=EPOCHS)

    # Plot loss curve
    losses = [epoch["loss"] for epoch in history]
    plt.plot(losses)
    plt.show()

    # Compute trained network's loss on training data
    y_train_pred, _ = forward(network, X_train)
    train_loss, _ = mse(y_train, y_train_pred)
    print(f"Train loss: {train_loss}")

    # Compute trained network's loss on test data
    y_test_pred, _ = forward(network, X_test)
    test_loss, _ = mse(y_test, y_test_pred)
    print(f"Test loss: {test_loss}")
