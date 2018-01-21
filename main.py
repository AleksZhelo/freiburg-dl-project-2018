from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from models.mlp import MLP
from util.loader import load_data

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

configs, learning_curves = load_data(source_dir='./data')

configs = np.array(list(map(lambda x: list(x.values()), configs)))
learning_curves = np.array(learning_curves)

batch_size = 32
train_steps = 1000
num_train_samples = configs.shape[0]


def fill_batch(x, y, configs_, curves_):
    for i in range(batch_size):
        index = np.random.randint(0, configs_.shape[0])
        x[i] = configs_[index]
        y[i] = curves_[index, -1]
    return x, y


with tf.Session() as session:
    input_tensor = tf.placeholder(tf.float32, [None, 5])
    target = tf.placeholder(tf.float32, [None, 1])

    mlp = MLP(input_tensor, target, learning_rate=0.00005)

    train_summary_writer = tf.summary.FileWriter('logs/train', session.graph)

    x = np.zeros((batch_size, 5), dtype=np.float32)
    y = np.zeros((batch_size, 1), dtype=np.float32)

    k_fold = KFold(n_splits=3)
    performances = np.zeros(3)

    current_fold = 0
    for train_indices, test_indices in k_fold.split(configs):
        loss_summary = tf.summary.scalar('losses/ValidationLoss' + str(current_fold), mlp.loss)

        session.run(tf.global_variables_initializer())

        train_configs = configs[train_indices]
        train_curves = learning_curves[train_indices]
        test_configs = configs[test_indices]
        test_curves = learning_curves[test_indices]

        print('Starting fold', current_fold)
        for step in range(train_steps):
            x, y = fill_batch(x, y, train_configs, train_curves)
            loss, _ = session.run([mlp.loss, mlp.optimize], {mlp.input_tensor: x,
                                                             mlp.target: y})
            if step % 10 == 0:
                sm, loss = session.run([loss_summary, mlp.loss],
                                       {mlp.input_tensor: test_configs,
                                        mlp.target: test_curves[:, -1].reshape(-1, 1)})
                train_summary_writer.add_summary(sm, step)
                print(loss)
        performances[current_fold] = session.run(mlp.loss, {mlp.input_tensor: test_configs,
                                                            mlp.target: test_curves[:, -1].reshape(-1, 1)})
        # print(session.run(mlp.prediction, {mlp.input_tensor: test_configs}) - test_curves[:, -1].reshape(-1, 1))
        current_fold += 1
    print('mean cross-validation loss: ', performances.mean())
