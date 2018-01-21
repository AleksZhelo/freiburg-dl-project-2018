import glob
import json
import os


def load_data(source_dir='./data'):
    configs = []
    learning_curves = []

    for fn in glob.glob(os.path.join(source_dir, "*.json")):
        with open(fn, 'r') as fh:
            tmp = json.load(fh)
            configs.append(tmp['config'])
            learning_curves.append(tmp['learning_curve'])
    return configs, learning_curves
