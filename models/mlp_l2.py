import tensorflow as tf

from models.mlp import MLP
from util.decorators import define_scope


class MLP_L2(MLP):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, phase, learning_rate=0.001, reg_weight=0.001):
        self.reg_weight = reg_weight
        super(MLP_L2, self).__init__(input_tensor, target, phase, learning_rate)

    @define_scope
    def loss(self):
        l2_regularizer = tf.contrib.layers.l2_regularizer(
            scale=self.reg_weight, scope=None
        )
        weights = [var for var in tf.trainable_variables() if 'kernel' in var.name
                   and 'dense_2' not in var.name]
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)

        return tf.losses.mean_squared_error(self.target, self.prediction) + regularization_penalty

    @staticmethod
    def sample_params(rs):
        return {
            'learning_rate': 10 ** rs.uniform(-5, -1),
            'reg_weight': 10 ** rs.uniform(-5, -2.5)
        }
