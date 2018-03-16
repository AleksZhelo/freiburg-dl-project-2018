import tensorflow as tf

from util.decorators import define_scope
from util.lr_decay import exponential_decay as exp_decay


class MLP_EXP_DECAY(object):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, phase, learning_rate=0.001,
                 exponential_decay=False, learning_rate_end=None,
                 decay_steps=None, decay_in_epochs=None):
        self.input_tensor = input_tensor
        self.target = target
        self.phase = phase
        self.learning_rate = learning_rate
        self.global_step = None
        self.exponential_decay = exponential_decay
        self.learning_rate_end = learning_rate_end
        self.decay_steps = decay_steps
        self.decay_in_epochs = decay_in_epochs
        self.first_hidden = None
        self.last_hidden = None
        self.prediction, self.loss, self.loss_pure, self.optimize  # lazy initialization

    @define_scope(initializer=tf.contrib.slim.xavier_initializer(seed=1))
    def prediction(self):
        self.first_hidden = tf.layers.dense(inputs=self.input_tensor, units=64, activation=tf.nn.relu)
        self.last_hidden = tf.layers.dense(inputs=self.first_hidden, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=self.last_hidden, units=1, activation=None)
        return x

    @define_scope
    def optimize(self):
        if self.exponential_decay:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = exp_decay(self.learning_rate, self.learning_rate_end,
                                           self.global_step, self.decay_steps,
                                           self.decay_in_epochs)

        optimizer = self._optimizer(self.learning_rate)
        return optimizer.minimize(self.loss, global_step=self.global_step)

    def _optimizer(self, lr):
        return tf.train.RMSPropOptimizer(lr)

    @define_scope
    def loss(self):
        return tf.losses.mean_squared_error(self.target, self.prediction)

    @define_scope
    def loss_pure(self):
        return tf.losses.mean_squared_error(self.target, self.prediction)

    @staticmethod
    def sample_params(rs):
        return {
            'learning_rate': 10 ** rs.uniform(-2, -1)
        }

    @staticmethod
    def append_decay_params(params, rs, decay_steps=None):
        params['exponential_decay'] = True
        params['learning_rate_end'] = 10 ** rs.uniform(-5, -2)
        params['decay_steps'] = decay_steps if decay_steps is not None else rs.randint(1, 300)
        params['decay_in_epochs'] = 300  # rs.randint(30, 300)
