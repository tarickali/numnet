"""
title : activations.py
create : @tarickali 23/11/19
update : @tarickali 23/11/19
"""

import numpy as np


def linear(z: np.ndarray) -> np.ndarray:
    return z


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / sum(np.exp(z))


def linear_prime(z: np.ndarray) -> np.ndarray:
    return np.ones(z.shape)


def relu_prime(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)


def softmax_prime(z: np.ndarray, grad: np.ndarray) -> np.ndarray:
    return np.diagflat(z) - np.dot(z, z.T)


FUNCTIONS = {"linear": linear, "relu": relu, "sigmoid": sigmoid, "softmax": softmax}

GRADIENTS = {
    "linear": linear_prime,
    "relu": relu_prime,
    "sigmoid": sigmoid_prime,
    "softmax": softmax_prime,
}
