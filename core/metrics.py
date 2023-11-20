"""
title : metrics.py
create : @tarickali 23/11/19
update : @tarickali 23/11/19
"""

import numpy as np


def accuracy(y: np.ndarray, o: np.ndarray) -> np.float32:
    """ """

    return np.mean(y == o)
