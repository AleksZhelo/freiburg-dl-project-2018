from __future__ import print_function

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from models.mlp_l1 import MLP_L1
from util.loader import load_data

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

configs, learning_curves = load_data(source_dir='./data')

configs = np.array(list(map(lambda x: list(x.values()), configs)))
learning_curves = np.array(learning_curves)

batch_size = 12
train_epochs = 300
patience = 40
eval_every = 4
normalize = True
repeats = 10

num_train_samples = configs.shape[0]
epoch_steps = configs.shape[0] / batch_size


def fill_batch(x, y, configs_, curves_):
    for i in range(batch_size):
        index = np.random.randint(0, configs_.shape[0])
        x[i] = configs_[index]
        y[i] = curves_[index, -1]
    return x, y


def normalized(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


# TODO: add checkpoints to restore best weights
with tf.Session() as session:
    input_tensor = tf.placeholder(tf.float32, [None, 5])
    target = tf.placeholder(tf.float32, [None, 1])
    phase = tf.placeholder(tf.bool, name='phase')

    # mlp = MLP(input_tensor, target, learning_rate=0.00005)
    # mlp = MLP_Dropout(input_tensor, target, phase, learning_rate=0.00005, drop_rate=0.2)
    # mlp = MLP_BN(input_tensor, target, phase, learning_rate=0.00005)
    mlp = MLP_L1(input_tensor, target, phase, learning_rate=0.0012907988958460479, reg_weight=0.0010125179006881145)

    # TODO: write hyperparams, regularization and so on in the name
    train_summary_writer = tf.summary.FileWriter('logs/train_{0}'.format(datetime.now()), session.graph)

    x = np.zeros((batch_size, 5), dtype=np.float32)
    y = np.zeros((batch_size, 1), dtype=np.float32)

    k_fold = KFold(n_splits=3, shuffle=True, random_state=1)
    performances = np.zeros((repeats, 3))

    for r in range(repeats):
        current_fold = 0
        for train_indices, test_indices in k_fold.split(configs):
            t_loss_summary = tf.summary.scalar('losses/TrainingLoss_repeat:{0}_fold:{1}'.format(r, current_fold),
                                               mlp.loss)
            v_loss_summary = tf.summary.scalar('losses/ValidationLoss_repeat:{0}_fold:{1}'.format(r, current_fold),
                                               mlp.loss_pure)

            session.run(tf.global_variables_initializer())
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            train_configs = configs[train_indices]
            train_curves = learning_curves[train_indices]
            test_configs = configs[test_indices]
            test_curves = learning_curves[test_indices]

            if normalize:
                train_configs = normalized(train_configs)
                test_configs = normalized(test_configs)

            print('Starting fold', current_fold)
            total_epochs = 0
            curr_steps = 0
            # best_loss = float('inf')
            # while curr_steps < patience:
            while total_epochs < train_epochs:
                for _ in range(epoch_steps):
                    x, y = fill_batch(x, y, train_configs, train_curves)
                    loss, _, _ = session.run([mlp.loss, mlp.optimize, update_ops], {mlp.input_tensor: x,
                                                                                    mlp.target: y,
                                                                                    phase: 1})
                    curr_steps += 1

                total_epochs += 1

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

                    # if pure_loss < best_loss:
                    #     best_loss = pure_loss
                    #     curr_steps = 0

                    # print(total_epochs, ev_loss)
            # print('best loss:', best_loss)
            # performances[r, current_fold] = best_loss
            performances[r, current_fold] = session.run(mlp.loss_pure, {mlp.input_tensor: test_configs,
                                                                        mlp.target: test_curves[:, -1].reshape(-1, 1),
                                                                        phase: 0})
            # print(session.run(mlp.prediction, {mlp.input_tensor: test_configs}) - test_curves[:, -1].reshape(-1, 1))
            current_fold += 1
        print('mean cross-validation loss: {0}, total epochs: {1}'.format(performances[r].mean(), total_epochs))
    print('total mean cross-validation loss: ', performances.mean())
