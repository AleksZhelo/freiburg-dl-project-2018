import os

import numpy as np


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def date2str(date):
    return date.strftime('%Y-%m-%d_%H_%M_%S')


def fill_batch(x, y, configs_, curves_, rs):
    for i in range(x.shape[0]):
        index = rs.randint(0, configs_.shape[0])
        x[i] = configs_[index]
        y[i] = curves_[index, -1]
    return x, y


def fill_lstm_batch(x, y, n_input, configs_, curves_, rs):
    for i in range(x.shape[0]):
        index = rs.randint(0, configs_.shape[0])
        x[i, 0, :5] = configs_[index]
        x[i, 0, 5] = 0
        y[i, 0] = curves_[index, 0]
        for t in range(1, n_input):
            x[i, t, :5] = 0
            x[i, t, 5] = curves_[index, t - 1]
            y[i, t] = curves_[index, t]
    return x, y


def fill_pred_lstm_batch(x, n_test, configs_, curves_):
    for i in range(x.shape[0]):
        x[i, 0, :5] = configs_[i]
        x[i, 0, 5] = 0
        for t in range(1, n_test):
            x[i, t, :5] = 0
            x[i, t, 5] = curves_[i, t - 1]
    return x


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
