import tensorflow as tf

from models.rnn.lstm_tf_decov import LSTM_TF_DeCov
from models.rnn.lstm_tf_decov_mlp_init import LSTM_TF_DeCov_MLP_init
from util.decorators import define_scope


class LSTM_TF_MLP_init(LSTM_TF_DeCov_MLP_init):

    def __init__(self, input_tensor, target, lstm_initial_state, phase, learning_rate=0.001, batch_size=12,
                 exponential_decay=False, decay_steps=None, decay_rate=0.99):
        super(LSTM_TF_MLP_init, self).__init__(
            input_tensor, target, lstm_initial_state, phase, learning_rate, batch_size,
            0, exponential_decay, decay_steps, decay_rate
        )

    @define_scope
    def loss(self):
        return tf.losses.mean_squared_error(self.target, self.prediction)

    @staticmethod
    def sample_params(rs):
        return {
            'learning_rate': 10 ** rs.uniform(-3.5, -1)
        }
