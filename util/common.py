import os

import numpy as np


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def fill_batch(x, y, configs_, curves_, rs):
    for i in range(x.shape[0]):
        index = rs.randint(0, configs_.shape[0])
        x[i] = configs_[index]
        y[i] = curves_[index, -1]
    return x, y


def normalized(x, mean=None, std=None):
    if mean is None or std is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    return (x - mean) / std, mean, std


def loss(y_hat, y):
    """
    Mean squared error.
    y_hat : predicted values
    y : true values
    """
    return np.mean(np.power(y_hat - y, 2))
