from __future__ import print_function

import os

import tensorflow as tf

from models.mlp_decov import MLP_DeCov
from models.mlp_l1 import MLP_L1
from task2.run_model import run_model
from util.common import ensure_dir
from util.loader import load_data_as_numpy

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
ensure_dir(log_dir)

configs, learning_curves = load_data_as_numpy()

batch_size = 12
train_epochs = 300
patience = 40
eval_every = 4
normalize = True

model = MLP_DeCov

with tf.Session() as session:
    params = dict(learning_rate=0.0012907988958460479, reg_weight=0.0010125179006881145)
    # params['exponential_decay'] = True
    # params['decay_rate'] = 0.25
    # params['decay_steps'] = configs.shape[0] / batch_size
    run_model(session, configs, learning_curves, log_dir,
              model, normalize, train_epochs, batch_size, eval_every, params,
              tf_seed=1337, numpy_seed=1123, verbose=True)
