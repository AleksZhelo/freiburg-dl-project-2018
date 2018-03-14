import tensorflow as tf

from models.lstm_tf_decov import LSTM_TF_DeCov
from util.decorators import define_scope


class LSTM_TF(LSTM_TF_DeCov):

    def __init__(self, input_tensor, target, lstm_initial_state, phase, learning_rate=0.001, batch_size=12,
                 exponential_decay=False, decay_steps=None, decay_rate=0.99):
        super(LSTM_TF, self).__init__(
            input_tensor, target, lstm_initial_state, phase, learning_rate, batch_size,
            0, exponential_decay, decay_steps, decay_rate
        )

    @define_scope
    def loss(self):
        return tf.losses.mean_squared_error(self.target, self.prediction)

    @staticmethod
    def sample_params(rs):
        return {
            'learning_rate': 10 ** rs.uniform(-5, -1)
        }
