from __future__ import print_function

import gc
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from hyperband.hyperband import Hyperband
from models.mlp import MLP
from models.mlp_decov import MLP_DeCov
from models.mlp_l1 import MLP_L1
from models.mlp_l1_elu import MLP_L1_ELU
from models.mlp_l1_exp_decay import MLP_L1_EXP_DECAY
from models.mlp_l1_sgd import MLP_L1_SGD
from models.mlp_l2 import MLP_L2
from models.mlp_l2_elu import MLP_L2_ELU
from task2.run_model import run_model
from util.common import ensure_dir, date2str
from util.loader import load_data_as_numpy


def evaluate_model(params, epochs):
    with tf.Session() as session:
        cv_loss = run_model(session, configs, learning_curves, None,
                            model, normalize, epochs, batch_size, eval_every, params)
    tf.reset_default_graph()
    gc.collect()  # TODO: still leaks memory, but less?
    return cv_loss


def gen_sample_params(model, decay_lr, rs):
    def do_sample():
        params = model.sample_params(rs)
        if batch_size is None:
            params['batch_size'] = rs.randint(1, 64)
        if decay_lr:
            b_size = batch_size if batch_size is not None else params['batch_size']
            model.append_decay_params(params, rs, configs.shape[0] / b_size)
        return params

    return do_sample


if __name__ == '__main__':
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    res_dir = os.path.join(os.path.dirname(__file__), 'optimization_results')
    ensure_dir(log_dir)
    ensure_dir(res_dir)

    configs, learning_curves = load_data_as_numpy()

    batch_size = 12  # if None also sampled randomly
    max_epochs = 300
    eval_every = 4
    normalize = True
    decay_lr = True
    run_time = 4 * 3600

    model = MLP_L1_EXP_DECAY
    rs = np.random.RandomState()
    hyperband = Hyperband(gen_sample_params(model, decay_lr, rs),
                          evaluate_model,
                          max_epochs=max_epochs, reduction_factor=3, min_r=5)

    results = []
    start = datetime.now()

    while (datetime.now() - start).total_seconds() < run_time:
        results.extend(hyperband.run())

    # TODO: write after every iteration
    with open(os.path.join(res_dir, '{0}_hyperband_min_r_5_{1}'.format(
            model.__name__, date2str(datetime.now()))), 'w') as f:
        json.dump(results, f)
