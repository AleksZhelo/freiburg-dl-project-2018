import tensorflow as tf

from models.lstm_tf_decov import LSTM_TF_DeCov
from util.decorators import define_scope


class LSTM_TF_Dropout(LSTM_TF_DeCov):

    # noinspection PyStatementEffect
    def __init__(self, input_tensor, target, lstm_initial_state, phase, learning_rate=0.001, batch_size=12,
                 reg_weight=0.001, drop_rate=0.5, exponential_decay=False, decay_steps=None, decay_rate=0.99):
        self.drop_rate = drop_rate
        super(LSTM_TF_Dropout, self).__init__(
            input_tensor, target, lstm_initial_state, phase, learning_rate, batch_size,
            reg_weight, exponential_decay, decay_steps, decay_rate
        )

    @define_scope(initializer=tf.contrib.slim.xavier_initializer(seed=1))
    def prediction(self):
        self.multi_lstm = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(size) for size in [64, 64]]
        )

        self.lstm_outputs, self.lstm_final_state = tf.nn.dynamic_rnn(
            cell=self.multi_lstm, inputs=self.input_tensor,
            initial_state=self.lstm_initial_state, dtype=tf.float32
        )

        self.first_dense = tf.layers.dense(inputs=self.lstm_outputs, units=64, activation=tf.nn.relu)
        drop = tf.layers.dropout(self.first_dense, training=self.phase, rate=self.drop_rate)
        self.last_dense = tf.layers.dense(inputs=drop, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=self.last_dense, units=1, activation=None)
        return x

    @define_scope
    def loss(self):
        return tf.losses.mean_squared_error(self.target, self.prediction)

    @staticmethod
    def sample_params(rs):
        return {
            'learning_rate': 10 ** rs.uniform(-5, -1),
            'drop_rate': rs.uniform(0.0, 0.8)
        }
