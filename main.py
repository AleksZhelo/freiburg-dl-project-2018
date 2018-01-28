from __future__ import print_function

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from models.mlp_bn import MLP_BN
from util.loader import load_data

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

configs, learning_curves = load_data(source_dir='./data')

configs = np.array(list(map(lambda x: list(x.values()), configs)))
learning_curves = np.array(learning_curves)

# TODO: necessary?
# idx = np.array(list(range(configs.shape[0])))
# np.random.shuffle(idx)
# configs = configs[idx]
# learning_curves = learning_curves[idx]

batch_size = 32
train_epochs = 1000
patience = 40
eval_every = 4

num_train_samples = configs.shape[0]
epoch_steps = configs.shape[0] / batch_size


def fill_batch(x, y, configs_, curves_):
    for i in range(batch_size):
        index = np.random.randint(0, configs_.shape[0])
        x[i] = configs_[index]
        y[i] = curves_[index, -1]
    return x, y


# TODO: add checkpoints to restore best weights
with tf.Session() as session:
    input_tensor = tf.placeholder(tf.float32, [None, 5])
    target = tf.placeholder(tf.float32, [None, 1])
    phase = tf.placeholder(tf.bool, name='phase')

    # mlp = MLP(input_tensor, target, learning_rate=0.00005)
    mlp = MLP_BN(input_tensor, target, phase, learning_rate=0.00005)

    train_summary_writer = tf.summary.FileWriter('logs/train_{0}'.format(datetime.now()), session.graph)

    x = np.zeros((batch_size, 5), dtype=np.float32)
    y = np.zeros((batch_size, 1), dtype=np.float32)

    k_fold = KFold(n_splits=3)
    performances = np.zeros(3)

    current_fold = 0
    for train_indices, test_indices in k_fold.split(configs):
        loss_summary = tf.summary.scalar('losses/ValidationLoss' + str(current_fold), mlp.loss)

        session.run(tf.global_variables_initializer())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        train_configs = configs[train_indices]
        train_curves = learning_curves[train_indices]
        test_configs = configs[test_indices]
        test_curves = learning_curves[test_indices]

        print('Starting fold', current_fold)
        total_epochs = 0
        curr_steps = 0
        best_loss = float('inf')
        while curr_steps < patience:
            for _ in range(epoch_steps):
                x, y = fill_batch(x, y, train_configs, train_curves)
                loss, _, _ = session.run([mlp.loss, mlp.optimize, update_ops], {mlp.input_tensor: x,
                                                                                mlp.target: y,
                                                                                phase: 1})
            curr_steps += 1
            total_epochs += 1

            if total_epochs % eval_every == 0:
                sm, ev_loss = session.run([loss_summary, mlp.loss], {mlp.input_tensor: test_configs,
                                                                     mlp.target: test_curves[:, -1].reshape(-1, 1),
                                                                     phase: 0})
                train_summary_writer.add_summary(sm, total_epochs)

                if ev_loss < best_loss:
                    best_loss = ev_loss
                    curr_steps = 0

                print(total_epochs, ev_loss)
        print('best loss:', best_loss)
        performances[current_fold] = best_loss
        # performances[current_fold] = session.run(mlp.loss, {mlp.input_tensor: test_configs,
        #                                                     mlp.target: test_curves[:, -1].reshape(-1, 1),})
        # print(session.run(mlp.prediction, {mlp.input_tensor: test_configs}) - test_curves[:, -1].reshape(-1, 1))
        current_fold += 1
    print('mean cross-validation loss: ', performances.mean())
