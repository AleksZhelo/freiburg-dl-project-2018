import tensorflow as tf

from models.mlp_exp_decay import MLP_EXP_DECAY
from util.decorators import define_scope


class MLP_L1_EXP_DECAY(MLP_EXP_DECAY):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, phase,
                 learning_rate=0.001, reg_weight=0.001,
                 exponential_decay=False, learning_rate_end=None,
                 decay_steps=None, decay_in_epochs=None):
        self.reg_weight = reg_weight
        super(MLP_L1_EXP_DECAY, self).__init__(input_tensor, target, phase, learning_rate,
                                               exponential_decay, learning_rate_end,
                                               decay_steps, decay_in_epochs)

    @define_scope
    def loss(self):
        l1_regularizer = tf.contrib.layers.l1_regularizer(
            scale=self.reg_weight, scope=None
        )
        weights = [var for var in tf.trainable_variables() if 'kernel' in var.name
                   and 'dense_2' not in var.name]
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

        return tf.losses.mean_squared_error(self.target, self.prediction) + regularization_penalty

    @staticmethod
    def sample_params(rs):
        return {
            'learning_rate': 10 ** rs.uniform(-2, -1),
            'reg_weight': 10 ** rs.uniform(-5, -2.5)
        }
