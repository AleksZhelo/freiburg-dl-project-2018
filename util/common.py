import os

import numpy as np
import pandas as pd


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


def print_pd_frame_from_multi_input_performances(performances, estimators):
    data = {
        '5 points': np.mean(performances, axis=1)[:, 0],
        '10 points': np.mean(performances, axis=1)[:, 1],
        '20 points': np.mean(performances, axis=1)[:, 2],
        '30 points': np.mean(performances, axis=1)[:, 3]
    }
    frame = pd.DataFrame(data, index=estimators)
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    frame = frame[sorted(frame.columns.tolist(), key=lambda x: int(x.split(' ')[0]))]
    print(frame)


def get_pd_frame_task2(losses, configs, estimators):
    data = {
        'loss': losses,
        'params': configs
    }
    frame = pd.DataFrame(data, index=estimators)
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)

    return frame
