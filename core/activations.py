"""
title : activations.py
create : @tarickali 23/11/19
update : @tarickali 23/11/22
"""

import numpy as np


def linear(z: np.ndarray) -> np.ndarray:
    """Computes the linear activation function.

    Parameters
    ----------
    z : np.ndarray @ (m, k)

    Returns
    -------
    np.ndarray @ (m, k)

    """

    return z


def relu(z: np.ndarray) -> np.ndarray:
    """Computes the relu activation function.

    Parameters
    ----------
    z : np.ndarray @ (m, k)

    Returns
    -------
    np.ndarray @ (m, k)

    """

    return np.maximum(z, 0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Computes the sigmoid activation function.

    Parameters
    ----------
    z : np.ndarray @ (m, k)

    Returns
    -------
    np.ndarray @ (m, k)

    """

    return 1 / (1 + np.exp(-z))


def softmax(z: np.ndarray) -> np.ndarray:
    """Computes the softmax activation function.

    Parameters
    ----------
    z : np.ndarray @ (m, k)

    Returns
    -------
    np.ndarray @ (m, k)

    """

    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def linear_prime(z: np.ndarray) -> np.ndarray:
    """Computes the linear activation gradient function.

    Parameters
    ----------
    z : np.ndarray @ (m, k)

    Returns
    -------
    np.ndarray @ (m, k)

    """

    return np.ones(z.shape)


def relu_prime(z: np.ndarray) -> np.ndarray:
    """Computes the relu activation gradient function.

    Parameters
    ----------
    z : np.ndarray @ (m, k)

    Returns
    -------
    np.ndarray @ (m, k)

    """

    return (z > 0).astype(z.dtype)


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Computes the sigmoid activation gradient function.

    Parameters
    ----------
    z : np.ndarray @ (m, k)

    Returns
    -------
    np.ndarray @ (m, k)

    """

    s = sigmoid(z)
    return s * (1 - s)


def softmax_prime(z: np.ndarray) -> np.ndarray:
    """Computes the softmax activation gradient function.

    NOTE: Returns the all ones matrix with shape of input z,
    since the categorical cross-entropy loss function L computes
    the appropriate gradient of L with respect to z.

    NOTE: The true gradient of softmax with respect to z is a Jacobian,
    and the code is given below:
    s = softmax(z)
    jacob = np.diag(s.flatten()) - np.outer(s, s)

    NOTE: It is important to note that this choice limits the use of
    the softmax activation to only the last layer of a neural network.

    Parameters
    ----------
    z : np.ndarray @ (m, k)

    Returns
    -------
    np.ndarray @ (m, k)

    """

    return np.ones(z.shape)


FUNCTIONS = {"linear": linear, "relu": relu, "sigmoid": sigmoid, "softmax": softmax}

GRADIENTS = {
    "linear": linear_prime,
    "relu": relu_prime,
    "sigmoid": sigmoid_prime,
    "softmax": softmax_prime,
}
