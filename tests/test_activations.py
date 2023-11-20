"""
title : test_activations.py
create : @tarickali 23/11/19
update : @tarickali 23/11/19
"""

import numpy as np
from core.activations import *


def test_linear():
    z = np.array([[1, 2, 3]])
    a = linear(z)
    assert np.all(a == np.array([[1, 2, 3]]))


def test_sigmoid():
    ERROR = 1e-5
    z = np.array([[1, 2, 3]])
    a = sigmoid(z)
    assert np.all((a - np.array([[0.7310585786, 0.880797078, 0.9525741268]])) < ERROR)


def test_relu():
    z = np.array([[-1, 0, 1, 2]])
    a = relu(z)
    assert np.all(a == np.array([[0, 0, 1, 2]]))
