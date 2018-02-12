from __future__ import print_function

import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from models.mlp_l1 import MLP_L1
from task2.run_model import run_model
from util.common import ensure_dir
from util.loader import load_data_as_numpy


def evaluate_model_random_search():
    graph = tf.Graph()  # TODO: leaks memory, rewrite to avoid
    session = tf.Session(graph=graph)
    with graph.as_default():
        params = model.sample_params(rs)
        cv_loss = run_model(session, configs, learning_curves, None,
                            model, normalize, train_epochs, batch_size, eval_every, params)
        results.append((cv_loss, params))
    session.close()
    tf.reset_default_graph()
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
    run_time = 3600

    model = MLP_L1
    rs = np.random.RandomState(1)
    results = []

    start = datetime.now()

    while (datetime.now() - start).total_seconds() < run_time:
        evaluate_model_random_search()

    # TODO: write after every iteration
    with open(os.path.join(res_dir, '{0}_{1}'.format(model.__name__, datetime.now())), 'w') as f:
        json.dump(results, f)
