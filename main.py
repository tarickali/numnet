"""
title : main.py
create : @tarickali 23/11/19
update : @tarickali 23/11/19
"""

import numpy as np

from core.engine import initialize, forward, backward, update


def main():
    architecture = [
        {"input_dim": 2, "output_dim": 10, "activation": "linear"},
        {"input_dim": 10, "output_dim": 3, "activation": "linear"},
    ]

    network = initialize(architecture, 0)

    x = np.array([1, 2]).reshape(-1, 1)
    y, cache = forward(network, x)

    print(y)
    print(cache)


if __name__ == "__main__":
    main()
