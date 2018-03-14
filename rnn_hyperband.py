from __future__ import print_function

import functools
import gc
import json
import os
import platform
from datetime import datetime

import multiprocessing
import numpy as np
import tensorflow as tf

from hyperband.hyperband import Hyperband
from models.lstm_tf_decov import LSTM_TF_DeCov
from models.lstm_tf_l2 import LSTM_TF_L2
from run_rnn_model import run_rnn_model
from util.common import ensure_dir, date2str
from util.loader import load_data_as_numpy


def evaluate_model(params, epochs, model_desc=None):
    with tf.Session() as session:
        cv_test, cv_valid, extras = \
            run_rnn_model(session, configs, learning_curves, None, save_dir,
                          model, n_input_train, n_input_test, normalize,
                          epochs, batch_size, eval_every, params,
                          early_stopping=early_stopping, patience=patience, model_desc=model_desc,
                          n_folds=3, tf_seed=1123, numpy_seed=1123, verbose=False)
    tf.reset_default_graph()
    gc.collect()  # TODO: still leaks memory, but less?

    extras['cv_valid'] = list(cv_valid)
    extras['cv_test'] = list(cv_test)
    extras['stopped_early'] = bool(extras['stopped_early'])
    extras['num_epochs'] = list(extras['num_epochs'])

    if early_stopping:
        return cv_test.mean(), extras
    else:
        return cv_test.mean()


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
    ensure_dir(log_dir)
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    ensure_dir(save_dir)
    res_dir = os.path.join(os.path.dirname(__file__), 'optimization_results')
    ensure_dir(res_dir)

    configs, learning_curves = load_data_as_numpy()

    batch_size = 32
    n_input_train = None  # if None - randomized between 5 and 20
    n_input_test = None  # 30  # if None - average of results for [5, 10, 20, 30] is used
    train_epochs = 1500
    eval_every = 1
    normalize = True
    decay_lr = False
    early_stopping = True
    patience = 250
    run_time = 1 * 3600

    model = LSTM_TF_DeCov
    # model = LSTM_TF_Dropout
    # model = LSTM_TF_L2

    def worker(process_num, managed_results):
        rs = np.random.RandomState()
        hyperband = Hyperband(
            gen_sample_params(model, decay_lr, rs),
            functools.partial(
                evaluate_model,
                model_desc='{0}_{1}_process{2}'.format(
                    model.__name__, platform.node(), process_num
                )
            ),
            max_epochs=train_epochs, reduction_factor=5, min_r=5
        )
        managed_results.extend(hyperband.run(early_stopping=early_stopping))


    manager = multiprocessing.Manager()
    results = manager.list()
    start = datetime.now()

    while (datetime.now() - start).total_seconds() < run_time:
        jobs = []
        for i in range(3):
            p = multiprocessing.Process(target=worker, args=(i, results))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        jobs[:] = []

    results_non_managed = [x for x in results]
    results_non_managed.append(
        dict(
            batch_size=batch_size,
            n_input_train=n_input_train,
            n_input_test=n_input_test,
            train_epochs=train_epochs,
            eval_every=eval_every,
            normalize=normalize,
            decay_lr=decay_lr,
            early_stopping=early_stopping,
            patience=patience,
            run_time=run_time
        )
    )

    # TODO: write after every iteration
    with open(os.path.join(res_dir, '{0}_hyperband_{1}'.format(
            model.__name__, date2str(datetime.now()))), 'w') as f:
        json.dump(results_non_managed, f)
