"""
title : mnist.py
create : @tarickali 23/11/22
update : @tarickali 23/11/28
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

    # Read in MNIST training dataframe
    train_df = pd.read_csv("data/mnist/train.csv")

    # Get pixel and label information from dataframe
    pixels = train_df.drop("label", axis=1).to_numpy()
    labels = train_df["label"].to_numpy()

    # Scale pixel data to be between 0.0 - 1.0
    X = pixels.reshape((-1, 784)) / 255.0

    # Create one hot vector of labels
    y = one_hot(labels)

    return X, y


def one_hot(x: np.ndarray, k: int = 10) -> np.ndarray:
    """Create a one-hot array of input array with k classes.

    Parameters
    ----------
    x : np.ndarray @ (n, 1)
    k : int = 10
        Number of classes in the one-hot array.

    Returns
    -------
    np.ndarray @ (n, k)

    """

    n = x.shape[0]
    o = np.zeros((n, k))
    o[np.arange(n), x] = 1
    return o


def get_batches(
    X: np.ndarray, y: np.ndarray, m: int = 32
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create batches of (X, y) data.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    m : int = 32

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]

    """

    n = X.shape[0]
    batches = []

    # Loop for creating batches of size m
    for i in range(n // m):
        a, b = i * m, (i + 1) * m
        batches.append((X[a:b], y[a:b]))

    # Create an extra match of size < m for leftover data
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
    """Training function used for the MNIST problem.

    Parameters
    ----------
    network : Network
    X : np.ndarray @ (n, 784)
    y : np.ndarray @ (n, 1)
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
    # Get MNIST data
    X, y = generate_data()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Network and training hyperparameters
    EPOCHS = 5
    ALPHA = 0.01
    architecture = [
        {
            "input_dim": 784,
            "output_dim": 512,
            "weight_init": 0.01,
            "activation": "relu",
        },
        {
            "input_dim": 512,
            "output_dim": 512,
            "weight_init": 0.01,
            "activation": "relu",
        },
        {
            "input_dim": 512,
            "output_dim": 512,
            "weight_init": 0.01,
            "activation": "relu",
        },
        {
            "input_dim": 512,
            "output_dim": 10,
            "weight_init": 0.01,
            "activation": "linear",
        },
    ]

    # Initialize network
    network = initialize(architecture, seed=0)

    # Train network
    start = time.time()
    history = train(network, X_train, y_train, alpha=ALPHA, epochs=EPOCHS)
    end = time.time()
    print(f"Training time: {end - start}")

    # Plot loss and accuracy curves
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
    losses = [epoch["loss"] for epoch in history]
    accs = [epoch["acc"] for epoch in history]
    axes[0].plot(losses)
    axes[0].set(title="Loss Curve", xlabel="Epochs", ylabel="Loss")
    axes[1].plot(accs)
    axes[1].set(title="Accuracy Curve", xlabel="Epochs", ylabel="Accuracy")
    plt.tight_layout()
    plt.show()

    # Compute trained network's accuracy on training data
    y_train_pred, _ = forward(network, X_train)
    y_train_pred = one_hot(np.argmax(softmax(y_train_pred), axis=1), k=10)
    train_acc = accuracy(y_train, y_train_pred)
    print(f"Train accuracy: {train_acc}")

    # Compute trained network's accuracy on test data
    y_test_pred, _ = forward(network, X_test)
    y_test_pred = one_hot(np.argmax(softmax(y_test_pred), axis=1), k=10)
    test_acc = accuracy(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc}")
