import numpy as np

def add_nontraining_time_series(configs, time_series, n):
    indices = [i for i in range(len(time_series)) if time_series[i][-1] > 0.8]
    steps = len(time_series[0])
    for i in range(n):
        ix = indices[int(np.random.rand() * len(indices))]
        config = configs[ix]
        for key in config:
            config[key] = config[key] + np.random.normal(0, 0.01)
        curve = list(np.random.normal(0.9, 0.005, size=steps))
        configs.append(config)
        time_series.append(curve)
    return configs, time_series

def add_perturbed_time_series(configs, time_series, n):
    indices = range(len(time_series))
    steps = len(time_series[0])
    for i in range(n):
        ix = indices[int(np.random.rand() * len(indices))]
        config = configs[ix]
        for key in config:
            config[key] = config[key] + np.random.normal(0, 0.01)
        curve = list(np.random.normal(time_series[ix], 0.005, size=steps))
        configs.append(config)
        time_series.append(curve)
    return configs, time_series