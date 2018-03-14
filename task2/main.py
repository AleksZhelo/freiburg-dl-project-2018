from __future__ import print_function

import os

import tensorflow as tf

from models.mlp_decov import MLP_DeCov
from models.mlp_exp_decay import MLP_EXP_DECAY
from models.mlp_l1 import MLP_L1
from task2.run_model import run_model
from util.common import ensure_dir
from util.loader import load_data_as_numpy

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
ensure_dir(log_dir)

configs, learning_curves = load_data_as_numpy()

# batch_size = 12
batch_size = 6
train_epochs = 300
patience = 40
eval_every = 4
normalize = True

# model = MLP_DeCov
model = MLP_EXP_DECAY

with tf.Session() as session:
    # params = {'learning_rate': 0.0019412167434611945, 'reg_weight': 0.001663258647698526}
    params = {'learning_rate': 0.001}
    params['exponential_decay'] = True
    params['learning_rate_end'] = 0.00001
    params['decay_in_epochs'] = 250
    params['decay_steps'] = configs.shape[0] / batch_size
    run_model(session, configs, learning_curves, None,
              model, normalize, train_epochs, batch_size, eval_every, params,
              tf_seed=1123, numpy_seed=1123, verbose=True)
