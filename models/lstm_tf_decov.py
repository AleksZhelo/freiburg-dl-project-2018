import tensorflow as tf
import numpy as np

from util.decorators import define_scope
from util.loader import load_data_as_numpy


class LSTM_TF_DeCov(object):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, lstm_initial_state, phase,
                 learning_rate=0.001, batch_size=12, reg_weight=0.001,
                 exponential_decay=False, decay_steps=None, decay_rate=0.99):
        self.input_tensor = input_tensor
        self.target = target
        self.phase = phase
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_weight = reg_weight

        self.global_step = None
        self.exponential_decay = exponential_decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.multi_lstm = None
        self.lstm_outputs = None
        self.lstm_initial_state = lstm_initial_state
        self.lstm_final_state = None
        self.first_dense = None
        self.last_dense = None
        self.prediction, self.loss, self.loss_pure, self.optimize  # lazy initialization

    @define_scope(initializer=tf.contrib.slim.xavier_initializer(seed=1))
    def prediction(self):
        self.multi_lstm = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(size) for size in [64, 64]]
        )

        self.lstm_outputs, self.lstm_final_state = tf.nn.dynamic_rnn(
            cell=self.multi_lstm, inputs=self.input_tensor,
            initial_state=self.lstm_initial_state, dtype=tf.float32
        )

        # drop = tf.layers.dropout(self.lstm_outputs, training=self.phase, rate=0.5)
        # self.first_dense = tf.layers.dense(inputs=drop, units=64, activation=tf.nn.relu)
        self.first_dense = tf.layers.dense(inputs=self.lstm_outputs, units=64, activation=tf.nn.relu)
        self.last_dense = tf.layers.dense(inputs=self.first_dense, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=self.last_dense, units=1, activation=None)
        return x

    @define_scope
    def optimize(self):
        if self.exponential_decay:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                            self.decay_steps, self.decay_rate,
                                                            staircase=True)  # TODO: compare
        optimizer = self._optimizer(self.learning_rate)
        self.gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        # self.gradients, _ = tf.clip_by_global_norm(self.gradients, 0.15)
        return optimizer.apply_gradients(zip(self.gradients, variables), global_step=self.global_step)

    def init_lstm_state(self):
        return self.multi_lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

    def _optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)

    @define_scope
    def loss(self):
        out_reshaped = tf.transpose(self.lstm_outputs, [1, 0, 2])
        initializer = np.zeros((64, 64), dtype=np.float32)
        covariances = \
            tf.scan(
                lambda _, x: tf.matmul(
                    tf.transpose(x - tf.reduce_mean(x, axis=0)),
                    x - tf.reduce_mean(x, axis=0)
                ),
                out_reshaped, initializer, infer_shape=False
            )
        initializer = np.array(0.0, dtype=np.float32)
        decov_losses = \
            tf.scan(lambda _, x: 0.5 * (tf.reduce_sum(tf.square(x)) - tf.reduce_sum(tf.square(tf.diag_part(x)))),
                    covariances, initializer)
        regularization_penalty = tf.reduce_sum(decov_losses)

        if self.reg_weight > 0:
            return tf.losses.mean_squared_error(self.target, self.prediction) + self.reg_weight * regularization_penalty
        else:
            return tf.losses.mean_squared_error(self.target, self.prediction)

    @define_scope
    def loss_pure(self):
        return tf.losses.mean_squared_error(self.target, self.prediction)

    @staticmethod
    def sample_params(rs):
        return {
            'learning_rate': 10 ** rs.uniform(-5, -1),
            'reg_weight': 10 ** rs.uniform(-3.5, 1)
        }

    @staticmethod
    def append_decay_params(params, rs, decay_steps=None):
        params['exponential_decay'] = True
        params['decay_rate'] = rs.uniform(0.25, 1)
        params['decay_steps'] = decay_steps if decay_steps is not None else rs.randint(1, 300)


if __name__ == '__main__':
    batch_size = 12
    n_input = 5

    configs, learning_curves = load_data_as_numpy()

    with tf.Session() as session:
        input_tensor = tf.placeholder(tf.float32, [None, None, 6])
        target = tf.placeholder(tf.float32, [None, None, 1])
        c1 = tf.placeholder(tf.float32, [None, 64])
        h1 = tf.placeholder(tf.float32, [None, 64])
        c2 = tf.placeholder(tf.float32, [None, 64])
        h2 = tf.placeholder(tf.float32, [None, 64])
        initial_state = (tf.nn.rnn_cell.LSTMStateTuple(c1, h1),
                         tf.nn.rnn_cell.LSTMStateTuple(c2, h2))
        phase = tf.placeholder(tf.bool, name='phase')

        rnn_test = LSTM_TF_DeCov(input_tensor, target, initial_state, phase, batch_size=batch_size)

        session.run(tf.global_variables_initializer())

        test_batch = np.zeros((batch_size, n_input, 6))
        test_batch[:, :, :5] = np.repeat(configs[:12].reshape(12, 1, 5), 5, axis=1)
        test_batch[:, :, 5] = np.repeat(learning_curves[:12, 0].reshape(12, 1), 5, axis=1)

        pred, final_state = session.run([rnn_test.prediction, rnn_test.lstm_final_state],
                                        feed_dict={c1: np.zeros((batch_size, 64), dtype=np.float32),
                                                   h1: np.zeros((batch_size, 64), dtype=np.float32),
                                                   c2: np.zeros((batch_size, 64), dtype=np.float32),
                                                   h2: np.zeros((batch_size, 64), dtype=np.float32),
                                                   input_tensor: test_batch})
        print(pred)
        print(final_state)
