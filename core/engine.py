"""
title : engine.py
create : @tarickali 23/11/19
update : @tarickali 23/11/19
"""

import numpy as np

from core.types import Architecture, Network, Cache, Gradients, History
from core.activations import FUNCTIONS, GRADIENTS
from core.losses import binary_crossentropy, categorical_crossentropy
from core.metrics import accuracy


def initialize(architecture: Architecture, seed: int = None) -> Network:
    """Initialize the parameters of a feedforward neural network.

    Parameters
    ----------
    architecture : Architecture
        The description of the neural network architecture.
    seed : int = None
        The seed to use for rng.

    Returns
    -------
    network : Network
        The parameters and info for each layer of the neural network.

    """

    rng = np.random.RandomState(seed)
    network = []

    for layer in architecture:
        input_dim = layer["input_dim"]
        output_dim = layer["output_dim"]
        activation = layer["activation"]
        network.append(
            {
                "W": rng.normal(loc=0.0, scale=1.0, size=(input_dim, output_dim)),
                "b": np.zeros(shape=(output_dim, 1)),
                "activation": activation,
            }
        )

    return network


def forward(network: Network, x: np.ndarray) -> tuple[np.ndarray, Cache]:
    """Performs forward propogation with network on input array x.

    Parameters
    ----------
    network : Network
    x : np.ndarray @ (m, n_in)

    Returns
    -------
    y : np.ndarray @ (m, n_out)
    cache : Cache

    """

    a = x
    cache = []
    for layer in network:
        W, b, activation = layer["W"], layer["b"], layer["activation"]
        z = W.T @ a + b
        cache.append({"a_in": a, "z_out": z})
        a = FUNCTIONS[activation](z)
    return a, cache


def backward(network: Network, grad: np.ndarray, cache: Cache) -> Gradients:
    """

    Parameters
    ----------
    network : Network
    grad : np.ndarray @ (m, n_out)
    cache : Cache

    Returns
    -------
    gradients : Gradients

    """

    gradients = []
    delta = grad
    for layer, memory in reversed(zip(network, cache)):
        W, b, activation = layer["W"], layer["b"], layer["activation"]
        a_in, z_out = memory["a_in"], memory["z_out"]
        delta = W.T @ delta * GRADIENTS[activation](z_out)
        gradients.append({"W": a_in @ delta.T, "b": delta})


def update(network: Network, gradients: Gradients, alpha: float) -> None:
    """

    Parameters
    ----------
    network : Network
    gradients : Gradients

    """

    for layer, grad in zip(network, gradients):
        layer["W"] -= alpha * grad["W"]
        layer["b"] -= alpha * grad["b"]


def train(
    network: Network,
    X: np.ndarray,
    y: np.ndarray,
    m: int = None,
    alpha: float = 0.001,
    epochs: int = 100,
) -> History:
    """ """

    n = len(X)

    history = []
    for e in epochs:
        o, cache = forward(network, X)
        cost, grad = binary_crossentropy(y, o)
        gradients = backward(network, grad, cache)
        update(network, gradients, alpha)

        acc = accuracy(y, o)
        history.append({"epoch": e, "cost": cost, "acc": acc})

    return history
