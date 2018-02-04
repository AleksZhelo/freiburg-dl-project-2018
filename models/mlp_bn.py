import tensorflow as tf

from models.mlp import MLP
from util.decorators import define_scope


class MLP_BN(MLP):

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        x = tf.layers.dense(inputs=self.input_tensor, units=64, activation=None)
        x = tf.layers.batch_normalization(x, center=True, scale=True, training=self.phase)
        x = tf.nn.relu(x)
        x = tf.layers.dense(inputs=x, units=64, activation=None)
        x = tf.layers.batch_normalization(x, center=True, scale=True, training=self.phase)
        x = tf.nn.relu(x)
        x = tf.layers.dense(inputs=x, units=1, activation=None)
        return x
