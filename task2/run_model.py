from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from util.common import normalized, fill_batch


# TODO: add checkpoints to restore best weights
def run_model(session, configs, learning_curves, log_dir,
              model_class, normalize, train_epochs, batch_size, eval_every, params,
              tf_seed=1123, numpy_seed=1123, verbose=True):
    if batch_size is None:
        batch_size = params['batch_size']
        params = dict(params)
        del params['batch_size']
    # TODO: this is wrong the whole time
    # needs to be train_configs.shape[0]
    # will not change not to break repeatability of optimization results
    num_train_samples = configs.shape[0]
    epoch_steps = num_train_samples / batch_size

    tf.set_random_seed(tf_seed)

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
    rs_ = np.random.RandomState(numpy_seed)
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
    if verbose:
        print('mean cross-validation loss: {0}, params: {1}'.format(performances.mean(), params))
    return performances.mean()
