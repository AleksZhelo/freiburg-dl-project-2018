from __future__ import print_function

import os

import tensorflow as tf

from models.mlp.mlp_decov import MLP_DeCov
from models.mlp.mlp_exp_decay import MLP_EXP_DECAY
from models.mlp.mlp_l1_elu import MLP_L1_ELU
from models.mlp.mlp_l2 import MLP_L2
from task2.run_model import run_model
from util.common import ensure_dir
from util.loader import load_data_as_numpy

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
ensure_dir(log_dir)
save_dir = os.path.join(os.path.dirname(__file__), 'mlp_checkpoints')
ensure_dir(save_dir)

configs, learning_curves = load_data_as_numpy()

# batch_size = 12
batch_size = None
train_epochs = 100
patience = 200
eval_every = 1
normalize = True

model = MLP_L2
# model = MLP_L1_ELU
# model = MLP_DeCov
# model = MLP_EXP_DECAY

with tf.Session() as session:
    params = {'learning_rate': 0.0023973566100493346, 'reg_weight': 0.0010437651588141193, 'batch_size': 3}
    # params = {'learning_rate': 0.0019496383939788997, 'reg_weight': 0.000680544162698294}
    # params['exponential_decay'] = True
    # params['learning_rate_end'] = 0.00001
    # params['decay_in_epochs'] = 250
    # params['decay_steps'] = configs.shape[0] / batch_size
    run_model(session, configs, learning_curves, None,
              model, normalize, train_epochs, batch_size, eval_every, params,
              early_stopping=False, patience=patience, model_desc=None, save_dir=save_dir,
              tf_seed=1123, numpy_seed=1123, verbose=True)
