from models.mlp_l1 import MLP_L1
import tensorflow as tf


class MLP_L1_SGD(MLP_L1):

    def _optimizer(self, lr):
        return tf.train.GradientDescentOptimizer(lr)
