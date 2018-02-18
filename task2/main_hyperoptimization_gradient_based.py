from __future__ import print_function

import gc
import json
import os
from datetime import datetime

import numpy as np
import scipy.optimize
import tensorflow as tf

from models.mlp_decov import MLP_DeCov
from models.mlp_l1 import MLP_L1
from task2.run_model import run_model
from util.common import ensure_dir
from util.loader import load_data_as_numpy


def evaluate_model(x, *args):  # TODO: correct usage of *args?
    with tf.Session() as session:
        # dct = {'learning_rate': x[0], 'reg_weight': x[1], 'drop_rate': x[2]}
        dct = {'learning_rate': x[0], 'reg_weight': x[1]}
        if decay_lr:
            dct['exponential_decay'] = True
            dct['decay_rate'] = x[2]
            dct['decay_steps'] = configs.shape[0] / batch_size

        cv_loss = run_model(session, configs, learning_curves, None,
                            model, normalize, train_epochs, batch_size, eval_every, dct)
        results.append((cv_loss, dct))
    tf.reset_default_graph()
    gc.collect()  # TODO: still leaks memory, but less?
    return cv_loss


if __name__ == '__main__':
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    res_dir = os.path.join(os.path.dirname(__file__), 'optimization_results')
    ensure_dir(log_dir)
    ensure_dir(res_dir)

    configs, learning_curves = load_data_as_numpy()

    batch_size = 12
    train_epochs = 100  # was 300
    patience = 40
    eval_every = 4
    normalize = True
    decay_lr = False
    run_time = 3600

    model = MLP_DeCov
    rs = np.random.RandomState(1)
    results = []

    start = datetime.now()

    scipy.optimize.minimize(
        # evaluate_model, x0=np.array([0.0009130585273711927, 0.00024719478144616294, 0.10526187]),
        evaluate_model, x0=np.array([0.000075, 0.5]),
        # does not seem to work with the decay
        # evaluate_model, x0=np.array([0.0004807115923157915, 0.004885703203107214, 0.75]),
        method='TNC', bounds=((0, 1), (0, 1)),
        # method='L-BFGS-B', bounds=((0, 1), (0, 1), (0.25, 1.0)),
        # method='TNC', bounds=((0.0, 1.0), (0.0, 1.0), (0.25, 1.0)),
        # options={'disp': True, 'maxfun': 3000, 'eps': 1e-05}
        # options={'disp': True, 'maxfun': 3000, 'eps': 1e-06}
        options={'disp': True, 'eps': 0.5 * 1e-04}
    )

    # TODO: write after every iteration
    with open(os.path.join(res_dir, '{0}_{1}'.format(model.__name__, datetime.now())), 'w') as f:
        json.dump(results, f)
