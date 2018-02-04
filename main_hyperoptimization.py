from __future__ import print_function

import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from models.mlp_l1_elu import MLP_L1_ELU
from util.loader import load_data


def fill_batch(x, y, configs_, curves_):
    for i in range(batch_size):
        index = np.random.randint(0, configs_.shape[0])
        x[i] = configs_[index]
        y[i] = curves_[index, -1]
    return x, y


def normalized(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def sample_params(rs):
    return {
        'learning_rate': 10 ** rs.uniform(-5, -1),
        # 'drop_rate': rs.uniform(0, 1)
        'reg_weight': 10 ** rs.uniform(-5, -2.5)
    }


# TODO: add checkpoints to restore best weights
def run_model(session, log_dir, model_class, normalize, params):
    input_tensor = tf.placeholder(tf.float32, [None, 5])
    target = tf.placeholder(tf.float32, [None, 1])
    phase = tf.placeholder(tf.bool, name='phase')

    mlp = model_class(input_tensor, target, phase, **params)

    if log_dir is not None:
        train_summary_writer = tf.summary.FileWriter(
            '{3}/{0}_{1}_{2}'.format(
                model_class.__name__,
                '_'.join(['{0}={1}'.format(a, b) for a, b in zip(params.keys(), params.values())]),
                datetime.now(),
                log_dir
            ),
            session.graph
        )

    x = np.zeros((batch_size, 5), dtype=np.float32)
    y = np.zeros((batch_size, 1), dtype=np.float32)

    k_fold = KFold(n_splits=3, shuffle=True, random_state=1)
    performances = np.zeros(3)

    current_fold = 0
    for train_indices, test_indices in k_fold.split(configs):
        if log_dir is not None:
            t_loss_summary = tf.summary.scalar('losses/TrainingLoss_fold:{0}'.format(current_fold),
                                               mlp.loss)
            v_loss_summary = tf.summary.scalar('losses/ValidationLoss_fold:{0}'.format(current_fold),
                                               mlp.loss_pure)

        session.run(tf.global_variables_initializer())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        train_configs = configs[train_indices]
        train_curves = learning_curves[train_indices]
        test_configs = configs[test_indices]
        test_curves = learning_curves[test_indices]

        if normalize:
            # TODO: ask about how to do this normalization: together or separately
            train_configs = normalized(train_configs)
            test_configs = normalized(test_configs)

        total_epochs = 0
        curr_steps = 0
        while total_epochs < train_epochs:
            for _ in range(epoch_steps):
                x, y = fill_batch(x, y, train_configs, train_curves)
                loss, _, _ = session.run([mlp.loss, mlp.optimize, update_ops], {mlp.input_tensor: x,
                                                                                mlp.target: y,
                                                                                phase: 1})
                curr_steps += 1

            total_epochs += 1

            if log_dir is not None:
                if total_epochs % eval_every == 0:
                    sm, t_loss = session.run([t_loss_summary, mlp.loss],
                                             {mlp.input_tensor: train_configs,
                                              mlp.target: train_curves[:, -1].reshape(-1, 1),
                                              phase: 0})
                    train_summary_writer.add_summary(sm, total_epochs)

                    sm, ev_loss, pure_loss = session.run([v_loss_summary, mlp.loss, mlp.loss_pure],
                                                         {mlp.input_tensor: test_configs,
                                                          mlp.target: test_curves[:, -1].reshape(-1, 1),
                                                          phase: 0})
                    train_summary_writer.add_summary(sm, total_epochs)

        performances[current_fold] = session.run(mlp.loss_pure, {mlp.input_tensor: test_configs,
                                                                 mlp.target: test_curves[:, -1].reshape(-1, 1),
                                                                 phase: 0})
        # print(session.run(mlp.prediction, {mlp.input_tensor: test_configs}) - test_curves[:, -1].reshape(-1, 1))
        current_fold += 1
    print('mean cross-validation loss: {0}, params: {1}'.format(performances.mean(), params))
    return performances.mean()


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    log_dir = 'logs'
    res_dir = 'optimization_results'
    ensure_dir(log_dir)
    ensure_dir(res_dir)

    configs, learning_curves = load_data(source_dir='./data')

    configs = np.array(list(map(lambda x: list(x.values()), configs)))
    learning_curves = np.array(learning_curves)

    batch_size = 12
    train_epochs = 300
    patience = 40
    eval_every = 4
    normalize = True
    run_time = 3600 * 8

    num_train_samples = configs.shape[0]
    epoch_steps = num_train_samples / batch_size

    model = MLP_L1_ELU
    rs = np.random.RandomState(1)
    results = []

    start = datetime.now()

    while (datetime.now() - start).total_seconds() < run_time:
        graph = tf.Graph()
        session = tf.Session(graph=graph)
        with graph.as_default():
            params = sample_params(rs)
            cv_loss = run_model(session, None, model, normalize, params)
            results.append((cv_loss, params))
        session.close()

    with open(os.path.join(res_dir, '{0}_{1}'.format(model.__name__, datetime.now())), 'w') as f:
        json.dump(results, f)

    # mlp = MLP(input_tensor, target, phase, learning_rate=0.00005)
    # mlp = MLP_Dropout(input_tensor, target, phase, learning_rate=0.00005, drop_rate=0.2)
    # mlp = MLP_BN(input_tensor, target, phase, learning_rate=0.00005)
