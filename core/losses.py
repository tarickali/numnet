"""
title : losses.py
create : @tarickali 23/11/19
update : @tarickali 23/11/19
"""

import numpy as np

from core.activations import sigmoid, softmax


def mae(y: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    y : np.ndarray @ (m, 1)
        The target ground truth labels.
    o : np.ndarray @ (m, 1)
        The network output predictions.

    Returns
    -------
    np.ndarray, np.ndarray : cost @ (1, 1), grad @ (m, 1)

    """

    assert y.shape == o.shape

    m, _ = y.shape

    cost = np.sum(np.abs(y - o)) / m
    grad = ...

    assert cost.shape == (1, 1)
    assert grad.shape == (m, 1)

    return cost, grad


def mse(y: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    y : np.ndarray @ (m, 1)
        The target ground truth labels.
    o : np.ndarray @ (m, 1)
        The network output predictions.

    Returns
    -------
    np.ndarray, np.ndarray : cost @ (1, 1), grad @ (m, 1)

    """

    assert y.shape == o.shape

    m, _ = y.shape

    cost = np.sum((y - o) ** 2) / m
    grad = ...

    assert cost.shape == (1, 1)
    assert grad.shape == (m, 1)

    return cost, grad


def binary_crossentropy(
    y: np.ndarray, o: np.ndarray, logits: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    y : np.ndarray
        The target ground truth labels.
    o : np.ndarray
        The network output predictions.
    logits : bool = True
        Indicates whether the values of o are activated.

    Returns
    -------
    np.ndarray, np.ndarray : cost, grad

    """

    assert y.shape == o.shape

    m, _ = y.shape

    if logits == True:
        a = sigmoid(o)
        cost = y * np.log(a) + (1 - y) * np.log(1 - a)
        grad = ...
    else:
        cost = ...
        grad = ...

    assert cost.shape == (1, 1)
    assert grad.shape == (m, 1)

    return cost, grad


def categorical_crossentropy(
    y: np.ndarray, o: np.ndarray, logits: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    y : np.ndarray
        The target ground truth labels.
    o : np.ndarray
        The network output predictions.
    logits : bool = True
        Indicates whether the values of o are activated.

    Returns
    -------
    np.ndarray, np.ndarray : cost, grad


    """

    assert y.shape == o.shape

    m, _ = y.shape

    if logits == True:
        a = sigmoid(o)
        cost = y * np.log(a) + (1 - y) * np.log(1 - a)
        grad = ...
    else:
        cost = ...
        grad = ...

    assert cost.shape == (1, 1)
    assert grad.shape == (m, 1)

    return cost, grad
