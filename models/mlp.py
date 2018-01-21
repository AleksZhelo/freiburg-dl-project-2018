import tensorflow as tf

from util.decorators import define_scope


class MLP(object):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, learning_rate=0.001):
        self.input_tensor = input_tensor
        self.target = target
        self.learning_rate = learning_rate
        self.prediction, self.optimize, self.loss  # lazy initialization

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
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
