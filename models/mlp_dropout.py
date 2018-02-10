import tensorflow as tf

from models.mlp import MLP
from util.decorators import define_scope


class MLP_Dropout(MLP):

    def __init__(self, input_tensor, target, phase, learning_rate=0.001, drop_rate=0.1):
        self.drop_rate = drop_rate
        super(MLP_Dropout, self).__init__(input_tensor, target, phase, learning_rate)

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        x = tf.layers.dense(inputs=self.input_tensor, units=64, activation=tf.nn.relu)
        x = tf.layers.dropout(x, rate=self.drop_rate, training=self.phase)
        x = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu)
        x = tf.layers.dropout(x, rate=self.drop_rate, training=self.phase)
        x = tf.layers.dense(inputs=x, units=1, activation=None)
        return x