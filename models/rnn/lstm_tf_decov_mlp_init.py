import tensorflow as tf

from models.rnn.lstm_tf_decov import LSTM_TF_DeCov
from util.decorators import define_scope


class LSTM_TF_DeCov_MLP_init(LSTM_TF_DeCov):

    @define_scope(initializer=tf.contrib.slim.xavier_initializer(seed=1))
    def prediction(self):
        self.multi_lstm = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(size) for size in [64, 64]]
        )

        self.first_cell_init_h_1 = tf.layers.dense(inputs=self.input_tensor[:, 0, :5], units=64, activation=tf.nn.relu)
        self.first_cell_init_h_2 = tf.layers.dense(inputs=self.first_cell_init_h_1, units=64, activation=tf.nn.relu)

        self.deep_initial_state = tf.cond(
            tf.equal(tf.reduce_sum(tf.abs(self.input_tensor[:, 0, :5])), 0.0),
            lambda: self.lstm_initial_state,
            lambda: (tf.nn.rnn_cell.LSTMStateTuple(self.lstm_initial_state[0].c,
                                                   self.first_cell_init_h_2),
                     tf.nn.rnn_cell.LSTMStateTuple(self.lstm_initial_state[1].c,
                                                   self.lstm_initial_state[1].h))
        )

        self.lstm_outputs, self.lstm_final_state = tf.nn.dynamic_rnn(
            cell=self.multi_lstm, inputs=tf.expand_dims(self.input_tensor[:, :, 5], -1),
            initial_state=self.deep_initial_state, dtype=tf.float32
        )

        self.first_dense = tf.layers.dense(inputs=self.lstm_outputs, units=64, activation=tf.nn.relu)
        self.last_dense = tf.layers.dense(inputs=self.first_dense, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=self.last_dense, units=1, activation=None)
        return x
