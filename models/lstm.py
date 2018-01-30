from keras.models import Sequential
from keras.layers import Dense, LSTM

nodes = 64

def lstm(d):
    """
    Returns a compiled model with 2 LSTM layers and 2 Dense layers,
    with 64 units each.
    
    d: number of variables representing one time step (should be set to the
        number of configuration parameters plus 1 for the learning curve's values)
    """
    model = Sequential()
    model.add(LSTM(nodes,
                   input_shape=(None,d),
                   batch_size=1,
                   return_sequences=True,
                   stateful=False))
    model.add(LSTM(nodes,
                   return_sequences=False,
                   stateful=False))
    model.add(Dense(nodes,
                    activation="sigmoid"))
    model.add(Dense(nodes,
                    activation="sigmoid"))
    model.add(Dense(1,
                    activation="linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model