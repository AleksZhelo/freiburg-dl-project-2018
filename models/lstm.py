from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.regularizers import l1
from keras.optimizers import Adam, SGD

nodes = 64

def lstm(d, lr, decay = 0, many2many = False, regularize = False, alpha = 0.01, batchsize = 1):
    """
    Returns a compiled model with 2 LSTM layers and 2 Dense layers,
    with 64 units each.
    
    d: number of variables representing one time step (should be set to the
        number of configuration parameters plus 1 for the learning curve's values)
    """
    model = Sequential()
    model.add(LSTM(nodes,
                   input_shape=(None,d),
                   batch_size=batchsize,
                   return_sequences=True,
                   stateful=False,
                   kernel_regularizer = l1(alpha) if regularize else None,
                   recurrent_regularizer = l1(alpha) if regularize else None))
    model.add(LSTM(nodes,
                   return_sequences=many2many,
                   stateful=False,
                   kernel_regularizer = l1(alpha) if regularize else None,
                   recurrent_regularizer = l1(alpha) if regularize else None))
    model.add(Dense(nodes,
                    activation="relu",
                    kernel_regularizer = l1(alpha) if regularize else None))
    model.add(Dense(nodes,
                    activation="relu",
                    kernel_regularizer = l1(alpha) if regularize else None))
    model.add(Dense(1,
                    activation="sigmoid",
                    kernel_regularizer = l1(alpha) if regularize else None))
    optimizer = SGD(lr=lr, decay = decay)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def small_lstm(d, lr, decay = 0, many2many = False, regularize = False, alpha = 0.01, batchsize = 1):
    model = Sequential()
    model.add(LSTM(1,
                   input_shape=(None,d),
                   batch_size=batchsize,
                   return_sequences=many2many,
                   stateful=False,
                   kernel_regularizer = l1(alpha) if regularize else None,
                   recurrent_regularizer = l1(alpha) if regularize else None))
    model.add(Dense(1,
                    activation="sigmoid",
                    kernel_regularizer = l1(alpha) if regularize else None))
    optimizer = SGD(lr=lr, decay = decay)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model