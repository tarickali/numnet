"""
title : losses.py
create : @tarickali 23/11/19
update : @tarickali 23/11/22
"""

import numpy as np

from core.activations import sigmoid, softmax


def mae(y: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the mean absolute error between true and pred arrays.

    Parameters
    ----------
    y : np.ndarray @ (m, 1)
        The target ground truth labels.
    o : np.ndarray @ (m, 1)
        The network output predictions.

    Returns
    -------
    np.ndarray, np.ndarray : loss @ (), grad @ (m, 1)

    """

    assert y.shape == o.shape

    m, _ = y.shape

    loss = np.mean(np.abs(y - o))
    grad = ...

    assert loss.shape == ()
    assert grad.shape == (m, 1)

    return loss, grad


def mse(y: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the mean squared error between true and pred arrays.

    Parameters
    ----------
    y : np.ndarray @ (m, 1)
        The target ground truth labels.
    o : np.ndarray @ (m, 1)
        The network output predictions.

    Returns
    -------
    np.ndarray, np.ndarray : loss @ (), grad @ (m, 1)

    """

    assert y.shape == o.shape

    m, _ = y.shape

    loss = np.mean((y - o) ** 2) / 2
    grad = o - y

    assert loss.shape == ()
    assert grad.shape == (m, 1)

    return loss, grad


def binary_crossentropy(
    y: np.ndarray, o: np.ndarray, logits: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the binary crossentropy between true and pred arrays.

    Parameters
    ----------
    y : np.ndarray @ (m, 1)
        The target ground truth labels.
    o : np.ndarray @ (m, 1)
        The network output predictions.
    logits : bool = True
        Indicates whether the values of o are activated.

    Returns
    -------
    np.ndarray, np.ndarray : loss @ (), grad @ (m, 1)

    """

    assert y.shape == o.shape

    m, _ = y.shape

    if logits == True:
        a = sigmoid(o)
        loss = np.mean(-y * np.log(a) - (1 - y) * np.log(1 - a))
        grad = a - y
    else:
        loss = np.mean(-y * np.log(o) - (1 - y) * np.log(1 - o))
        # grad = o - y
        grad = (o - y) / (o * (1 - o))

    assert loss.shape == ()
    assert grad.shape == (m, 1)

    return loss, grad


def categorical_crossentropy(
    y: np.ndarray, o: np.ndarray, logits: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the categorical crossentropy between true and pred arrays.

    Parameters
    ----------
    y : np.ndarray @ (m, n_out)
        The target ground truth labels.
    o : np.ndarray @ (m, n_out)
        The network output predictions.
    logits : bool = True
        Indicates whether the values of o are activated.

    Returns
    -------
    np.ndarray, np.ndarray : loss @ (), grad @ (m, n_out)


    """

    assert y.shape == o.shape

    m, k = y.shape

    if logits == True:
        a = softmax(o)
        loss = -np.mean(y * np.log(a))
        grad = a - y
    else:
        loss = -np.mean(y * np.log(o))
        grad = o - y

    assert loss.shape == ()
    assert grad.shape == (m, k)

    return loss, grad
