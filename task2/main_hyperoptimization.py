from __future__ import print_function

import gc
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from models.mlp_decov import MLP_DeCov
from models.mlp_exp_decay import MLP_EXP_DECAY
from models.mlp_exp_decay_sgd import MLP_EXP_DECAY_SGD
from models.mlp_l1_exp_decay import MLP_L1_EXP_DECAY
from models.mlp_l1_sgd import MLP_L1_SGD
from task2.run_model import run_model
from util.common import ensure_dir, date2str
from util.loader import load_data_as_numpy


def evaluate_model_random_search():
    with tf.Session() as session:
        params = model.sample_params(rs)
        if decay_lr:
            model.append_decay_params(params, rs, configs.shape[0] / batch_size)
        cv_loss = run_model(session, configs, learning_curves, None,
                            model, normalize, train_epochs, batch_size, eval_every, params)
        results.append((cv_loss, params))
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
    train_epochs = 300  # was 300
    patience = 40
    eval_every = 4
    normalize = True
    decay_lr = True
    run_time = 4 * 3600

    # model = MLP_L1_SGD
    # model = MLP_EXP_DECAY
    model = MLP_L1_EXP_DECAY
    # model = MLP_EXP_DECAY_SGD
    rs = np.random.RandomState()
    results = []

    start = datetime.now()

    while (datetime.now() - start).total_seconds() < run_time:
        evaluate_model_random_search()

    # TODO: write after every iteration
    with open(os.path.join(res_dir, '{0}_{1}_{2}_epochs'.format(
            model.__name__, date2str(datetime.now()), train_epochs)), 'w') as f:
        json.dump(results, f)
