import tensorflow as tf

from util.decorators import define_scope


class MLP(object):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, phase, params):
        self.input_tensor = input_tensor
        self.target = target
        self.phase = phase
        self.layers = [params["l1"], params["l2"], params["l3"], params["l4"], params["l5"]]
        self.dropout = params["dropout"]
        self.learning_rate = params["learning_rate"]
        self.global_step = None
        self.exponential_decay = False
        self.decay_steps = None
        self.decay_rate = 0.99
        self.prediction, self.loss, self.loss_pure, self.optimize  # lazy initialization

    @define_scope(initializer=tf.contrib.slim.xavier_initializer(seed=1))
    def prediction(self):
        x = self.input_tensor
        for i in range(5):
            x = self.build_layer(x, self.layers[i])
        x = tf.layers.dense(inputs=x, units=1, activation=None)
        return x

    def build_layer(self, inp, opts):
        if opts[0] == 0:
            return inp
        x = tf.layers.dense(inp, units=opts[0], activation=tf.nn.relu)
        if self.dropout:
            x = tf.layers.dropout(x, rate=opts[1], training=self.phase)
        return x

    @define_scope
    def optimize(self):
        if self.exponential_decay:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                            self.decay_steps, self.decay_rate,
                                                            staircase=True)  # TODO: compare
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

    # @staticmethod
    # def sample_params(rs):
    #     return {
    #         'learning_rate': 10 ** rs.uniform(-5, -1)
    #     }
    #
    # @staticmethod
    # def append_decay_params(params, rs, decay_steps=None):
    #     params['exponential_decay'] = True
    #     params['decay_rate'] = rs.uniform(0.25, 1)
    #     params['decay_steps'] = decay_steps if decay_steps is not None else rs.randint(1, 300)
