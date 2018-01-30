import numpy as np

def get_time_series(configs, learning_curves, repeat_config=False):
    """
    Creates list of arrays representing the configuration and learning curves
    in a way that it can be used for time series learning.
    A list of 2d arrays is returned instead of a 3d array, because the time series
    may have different length.
    
    configs: list of configurations
    learning_curves: list of learning curves
    repeat_config: if True, the configuration parameters are repeated each time step,
        otherwise only in the first time step
    
    returns: a list of 2d arrays of shape (time_steps, config_params + 1)
    """
    n = len(configs)
    n_params = len(configs[0])
    params = sorted([param for param in configs[0]])
    X = []
    for i in range(n):
        n_steps = len(learning_curves[i])
        x = np.zeros((n_steps, n_params + 1))
        for j in range(n_steps):
            if j == 0 or repeat_config:
                for k, param in enumerate(params):
                    x[j][k] = configs[i][param]
            x[j][-1] = learning_curves[i][j]
        X.append(x)
    return X