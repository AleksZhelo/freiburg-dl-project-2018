from __future__ import print_function

import functools
import gc
import json
import os
import hashlib

import numpy as np
import tensorflow as tf

from models.rnn.lstm_tf import LSTM_TF
from models.rnn.lstm_tf_decov import LSTM_TF_DeCov
from models.rnn.lstm_tf_decov_mlp_init import LSTM_TF_DeCov_MLP_init
from models.rnn.lstm_tf_decov_mlp_init_both import LSTM_TF_DeCov_MLP_init_both
from models.rnn.lstm_tf_decov_mlp_init_cell import LSTM_TF_DeCov_MLP_init_cell
from models.rnn.lstm_tf_dropout import LSTM_TF_Dropout
from models.rnn.lstm_tf_l1 import LSTM_TF_L1
from models.rnn.lstm_tf_l2 import LSTM_TF_L2
from models.rnn.lstm_tf_mlp_init import LSTM_TF_MLP_init
from run_rnn_model import run_rnn_model, loss_and_predictions_for_several_n_input
from util.common import ensure_dir, parse_task_args, model_from_name
from util.loader import load_data_as_numpy


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def post_fold_func(predictions_test,
                   initial_state, phase, rnn, session, test_configs, test_curves,
                   results_list=None):
    _, extrapolations = loss_and_predictions_for_several_n_input(
        initial_state, 40, 100,
        phase, rnn, session,
        test_configs, test_curves
    )
    results_list.append((test_curves, predictions_test, extrapolations))


if __name__ == '__main__':
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    ensure_dir(log_dir)
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints_test')
    ensure_dir(save_dir)

    configs, learning_curves = load_data_as_numpy()

    models = [LSTM_TF, LSTM_TF_DeCov, LSTM_TF_DeCov_MLP_init, LSTM_TF_DeCov_MLP_init_both,
              LSTM_TF_DeCov_MLP_init_cell, LSTM_TF_Dropout, LSTM_TF_L1, LSTM_TF_L2,
              LSTM_TF_MLP_init]

    args = parse_task_args()
    tasks_file = args.tasks_file

    with open(tasks_file, 'r') as f:
        tasks = json.load(f)

    results = []
    for task in tasks:
        for n_input_train in [5, 10, 20, None]:
            model = model_from_name(models, task['name'])
            batch_size = task['settings']['batch_size']
            n_input_test = None
            train_epochs = task['settings']['train_epochs']
            eval_every = task['settings']['eval_every']
            normalize = task['settings']['normalize']
            early_stopping = task['settings']['early_stopping']
            patience = task['settings']['patience']
            params = task['params']
            model_desc = '{0}_{1}_{2}'.format(task['name'],
                                              hashlib.sha256(task['model_desc'].encode('utf-8')).hexdigest(),
                                              str(n_input_train))
            desc_short = '{0}_{1}'.format(task['name'], str(n_input_train))

            each_fold_results = []

            with tf.Session() as session:
                cv_test, cv_valid, extras = \
                    run_rnn_model(session, configs, learning_curves, None, save_dir,
                                  model, n_input_train, n_input_test, normalize,
                                  train_epochs, batch_size, eval_every, params,
                                  early_stopping=early_stopping, patience=patience, model_desc=model_desc,
                                  n_folds=3, tf_seed=1123, numpy_seed=1123, verbose=False, reload_existing=True,
                                  post_fold_func=functools.partial(post_fold_func, results_list=each_fold_results))
                results.append((desc_short, n_input_train, cv_test, each_fold_results))
                print('{0}: {1}'.format(desc_short, cv_test))
            tf.reset_default_graph()
            gc.collect()

    name, _ = os.path.splitext(tasks_file)
    with open('{0}_results.txt'.format(name), 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)
