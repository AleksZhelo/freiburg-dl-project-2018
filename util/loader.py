import glob
import json
import os

import numpy as np


def load_data(source_dir='./data'):
    configs = []
    learning_curves = []

    for fn in glob.glob(os.path.join(source_dir, "*.json")):
        with open(fn, 'r') as fh:
            tmp = json.load(fh)
            configs.append(tmp['config'])
            learning_curves.append(tmp['learning_curve'])
    return configs, learning_curves


def load_data_as_numpy(source_dir=os.path.join(os.path.dirname(__file__), '..', 'data')):
    configs, learning_curves = load_data(source_dir)

    configs = np.array(list(map(lambda x: list(x.values()), configs)))
    learning_curves = np.array(learning_curves)

    return configs, learning_curves
