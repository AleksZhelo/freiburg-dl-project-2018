import tensorflow as tf

from util.decorators import define_scope


class MLP(object):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, phase, learning_rate=0.001):
        self.input_tensor = input_tensor
        self.target = target
        self.phase = phase
        self.learning_rate = learning_rate
        self.prediction, self.loss, self.loss_pure, self.optimize  # lazy initialization

    # TODO: figure out scopes, how this annotation works, and what is the proper initializer
    # for both weights and biases!
    @define_scope(initializer=tf.contrib.slim.xavier_initializer(seed=1))
    # @define_scope(initializer=tf.constant_initializer(0.01))  # significantly worse, wow
    def prediction(self):
        x = tf.layers.dense(inputs=self.input_tensor, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x, units=1, activation=None)
        return x

    @define_scope
    def optimize(self):
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss)

    @define_scope
    def loss(self):
        return tf.losses.mean_squared_error(self.target, self.prediction)

    @define_scope
    def loss_pure(self):
        return tf.losses.mean_squared_error(self.target, self.prediction)

    @staticmethod
    def sample_params(rs):
        return {
            'learning_rate': 10 ** rs.uniform(-5, -1)
        }
