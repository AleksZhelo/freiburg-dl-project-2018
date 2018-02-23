import tensorflow as tf

from models.mlp import MLP
from util.decorators import define_scope


class MLP_DeCov(MLP):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, phase,
                 learning_rate=0.001, reg_weight=0.001,
                 exponential_decay=False, decay_steps=None, decay_rate=0.99):
        self.reg_weight = reg_weight
        super(MLP_DeCov, self).__init__(input_tensor, target, phase, learning_rate,
                                        exponential_decay, decay_steps, decay_rate)

    @define_scope
    def loss(self):
        # TODO: try on first hidden?
        mean_activation = tf.reduce_mean(self.last_hidden, axis=0)
        v = self.last_hidden - mean_activation
        cov = tf.matmul(tf.transpose(v), v)
        regularization_penalty = 0.5 * (tf.square(tf.norm(cov)) - tf.square(tf.norm(tf.diag_part(cov))))

        # TODO: small values of reg_weight lead to NaNs
        # actually not necessarily small values
        # https://github.com/tensorflow/tensorflow/issues/12071 - ?
        if self.reg_weight > 0:
            return tf.losses.mean_squared_error(self.target, self.prediction) + self.reg_weight * regularization_penalty
        else:
            return tf.losses.mean_squared_error(self.target, self.prediction)

    @staticmethod
    def sample_params(rs):
        return {
            'learning_rate': 10 ** rs.uniform(-5, -1),
            'reg_weight': 10 ** rs.uniform(-5, -2.5)
        }
