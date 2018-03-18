from __future__ import print_function

import os

import gc
import tensorflow as tf
import numpy as np
import json

from sklearn.model_selection import KFold

from models.mlp.mlp import MLP
from models.mlp.mlp_bn import MLP_BN
from models.mlp.mlp_decov import MLP_DeCov
from models.mlp.mlp_dropout import MLP_Dropout
from models.mlp.mlp_exp_decay import MLP_EXP_DECAY
from models.mlp.mlp_l1 import MLP_L1
from models.mlp.mlp_l1_elu import MLP_L1_ELU
from models.mlp.mlp_l1_exp_decay import MLP_L1_EXP_DECAY
from models.mlp.mlp_l2 import MLP_L2
from models.mlp.mlp_l2_elu import MLP_L2_ELU
from util.loader import load_data_as_numpy
from util.common import ensure_dir, model_from_name, parse_task_args
from util.common import normalized, fill_batch


def test_model(session, configs, learning_curves, log_dir,
               model_class, normalize, train_epochs, batch_size, eval_every, params,
               tf_seed=1123, numpy_seed=1123, verbose=True):
    if batch_size is None:
        batch_size = params['batch_size']
        params = dict(params)
        del params['batch_size']
    # this is wrong the whole time
    # needs to be train_configs.shape[0]
    # will not change not to break repeatability of optimization results
    num_train_samples = configs.shape[0]
    epoch_steps = num_train_samples / batch_size

    tf.set_random_seed(tf_seed)

    input_tensor = tf.placeholder(tf.float32, [None, 5])
    target = tf.placeholder(tf.float32, [None, 1])
    phase = tf.placeholder(tf.bool, name='phase')

    mlp = model_class(input_tensor, target, phase, **params)

    x = np.zeros((batch_size, 5), dtype=np.float32)
    y = np.zeros((batch_size, 1), dtype=np.float32)

    k_fold = KFold(n_splits=3, shuffle=True, random_state=1)
    performances = np.zeros(3)
    y_y_hat = np.zeros((2, learning_curves.shape[0]))

    current_fold = 0
    rs_ = np.random.RandomState(numpy_seed)
    for train_indices, test_indices in k_fold.split(configs):
        session.run(tf.global_variables_initializer())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        train_configs = configs[train_indices]
        train_curves = learning_curves[train_indices]
        test_configs = configs[test_indices]
        test_curves = learning_curves[test_indices]

        if normalize:
            train_configs, mean, std = normalized(train_configs)
            test_configs, _, _ = normalized(test_configs, mean, std)

        total_epochs = 0
        curr_steps = 0
        while total_epochs < train_epochs:
            for _ in range(int(epoch_steps)):
                x, y = fill_batch(x, y, train_configs, train_curves, rs_)
                loss, _, _, = session.run([mlp.loss, mlp.optimize, update_ops], {mlp.input_tensor: x,
                                                                                 mlp.target: y,
                                                                                 phase: 1})
                curr_steps += 1

            total_epochs += 1

        performances[current_fold] = session.run(mlp.loss_pure, {mlp.input_tensor: test_configs,
                                                                 mlp.target: test_curves[:, -1].reshape(-1, 1),
                                                                 phase: 0})
        y_hat = session.run(mlp.prediction, {mlp.input_tensor: test_configs,
                                             phase: 0})
        y_y_hat[0, test_indices] = test_curves[:, -1]
        y_y_hat[1, test_indices] = y_hat[:, 0]
        current_fold += 1
    if verbose:
        print('mean cross-validation loss: {0}, params: {1}'.format(performances.mean(), params))
    return performances.mean(), list([list(v) for v in y_y_hat])



configs, learning_curves = load_data_as_numpy()

models = [MLP, MLP_BN, MLP_DeCov, MLP_Dropout, MLP_EXP_DECAY, MLP_L1, MLP_L1_ELU,
          MLP_L1_EXP_DECAY, MLP_L2, MLP_L2_ELU]

args = parse_task_args()
tasks_file = args.tasks_file

with open(tasks_file, 'r') as f:
    tasks = json.load(f)

results = []
for task in tasks[:5]:
    model = model_from_name(models, task['name'])
    batch_size = task['batch_size'] if 'batch_size' in task else 12
    train_epochs = task['train_epochs'] if 'train_epochs' in task else 300
    eval_every = 4
    normalize = task['normalize'] if 'normalize' in task else True
    params = task['params']

    with tf.Session() as session:
        loss, y_y_hat = test_model(
            session, configs, learning_curves, None,
            model, normalize, train_epochs, batch_size, eval_every, params,
            tf_seed=1123, numpy_seed=1123, verbose=True
        )
        results.append((task['name'], loss, y_y_hat))
        print('{0}: {1}'.format(task['name'], loss))
    tf.reset_default_graph()
    gc.collect()

name, _ = os.path.splitext(tasks_file)
with open('{0}_results.txt'.format(name), 'w') as f:
    json.dump(results, f)
