"""
title : engine.py
create : @tarickali 23/11/19
update : @tarickali 23/11/19
"""

import numpy as np

from core.types import Architecture, Network, Cache, Gradients, History
from core.activations import FUNCTIONS, GRADIENTS, sigmoid
from core.losses import binary_crossentropy
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
                "b": np.zeros(shape=(output_dim,)),
                "activation": activation,
            }
        )

    return network


def forward(network: Network, x: np.ndarray) -> tuple[np.ndarray, Cache]:
    """Performs forward propogation on a network on an input array x.

    Parameters
    ----------
    network : Network
        The network to perform the forward pas on.
    x : np.ndarray @ (m, n_in)
        The input array to be passed through the network.

    Returns
    -------
    y : np.ndarray @ (m, n_out)
        The output array from the forward pass.
    cache : Cache
        The cache consisting of (z, a) for each layer of the forward pass.

    """

    z = a = x
    cache = []
    for layer in network:
        W, b, activation = layer["W"], layer["b"], layer["activation"]
        cache.append({"z": z, "a": a})
        z = a @ W + b
        a = FUNCTIONS[activation](z)
    return a, cache


def backward(network: Network, grad: np.ndarray, cache: Cache) -> Gradients:
    """Performs backward propogation on a network using a top gradient.

    Parameters
    ----------
    network : Network
        The network to perform the backward pass on.
    grad : np.ndarray @ (m, n_out)
        The top gradient of the loss function with the output layer.
    cache : Cache
        The cache returned from the forward pass.

    Returns
    -------
    gradients : Gradients
        The gradients for each network layer's parameters.

    """

    gradients = []
    delta = grad
    for layer, memory in reversed(list(zip(network, cache))):
        W, activation = layer["W"], layer["activation"]
        z, a = memory["z"], memory["a"]
        m, _ = a.shape
        gradients.append({"W": a.T @ delta / m, "b": np.mean(delta, axis=0)})
        delta = delta @ W.T * GRADIENTS[activation](z)
    return list(reversed(gradients))


def update(network: Network, gradients: Gradients, alpha: float) -> None:
    """Update the parameters of network with gradients.

    Parameters
    ----------
    network : Network
        The neural network to be updated.
    gradients : Gradients
        The gradients of the network parameters.
    alpha : float
        The learning rate to update the network parameters.

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
    for e in range(epochs):
        o, cache = forward(network, X)
        loss, grad = binary_crossentropy(y, o)
        gradients = backward(network, grad, cache)
        update(network, gradients, alpha)

        acc = accuracy(y, np.round(sigmoid(o)))
        history.append({"epoch": e, "loss": loss, "acc": acc})
        print(history[-1])

    return history
