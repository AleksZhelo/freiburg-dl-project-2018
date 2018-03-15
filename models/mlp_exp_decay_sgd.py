from models.mlp_exp_decay import MLP_EXP_DECAY
import tensorflow as tf


class MLP_EXP_DECAY_SGD(MLP_EXP_DECAY):

    def _optimizer(self, lr):
        return tf.train.GradientDescentOptimizer(lr)
