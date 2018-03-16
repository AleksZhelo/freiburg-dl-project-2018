import tensorflow as tf

from models.mlp.mlp_dropout import MLP_Dropout
from util.decorators import define_scope


class MLP_L1_Dropout(MLP_Dropout):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, phase,
                 learning_rate=0.001, reg_weight=0.001, drop_rate=0.1,
                 exponential_decay=False, decay_steps=None, decay_rate=0.99):
        self.reg_weight = reg_weight
        super(MLP_L1_Dropout, self).__init__(input_tensor, target, phase,
                                             learning_rate, drop_rate,
                                             exponential_decay, decay_steps, decay_rate)

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
            'learning_rate': 10 ** rs.uniform(-5, -1),
            'reg_weight': 10 ** rs.uniform(-5, -2.5),
            'drop_rate': rs.uniform(0, 1)
        }
