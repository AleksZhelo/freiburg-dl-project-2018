from json import JSONDecodeError

import numpy as np
import json
import argparse
import os

from util.common import get_pd_frame_task2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        required=True,
        help='The optimization log files to process.'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Shows overall best configurations if this flag is set.'
    )

    parser.add_argument(
        '--table',
        action='store_true',
        help='Compiles a best results per model table.'
    )

    return parser.parse_args()


def parse_old_format(ls):
    dat = []
    for l in ls:
        split = l.split('loss:')[1].split('params:')
        dat.append((float(split[0][:-2]), eval(split[1])))
    return dat


if __name__ == '__main__':
    args = parse_args()
    for path in args.files:
        if os.path.isdir(path):
            args.files.remove(path)
            args.files.extend([os.path.join(path, file) for file in os.listdir(path)])

    total_data = []
    for file in args.files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
            except JSONDecodeError:
                f.seek(0)
                try:
                    data = parse_old_format(f.readlines())
                except IndexError or IOError:
                    print('Failed to read file {0}'.format(file))
                    continue

        training_settings = None
        if 'hyperband' in os.path.basename(file) or 'random_search' in os.path.basename(file):
            if 'LSTM' in os.path.basename(file):
                training_settings = data[-1]
                data = data[:-1]
            data = [(d['loss'], d, file) for d in data] if args.summary or args.table else [(d['loss'], d) for d in
                                                                                            data]
        else:
            data = [(d[0], d[1], file) for d in data] if args.summary or args.table else data

        if args.summary or args.table:
            total_data.extend(data)
        else:
            data = np.array(data)
            data = data[np.argsort(data[:, 0])]
            print('----------{0}----------'.format(file))
            print('model evaluations: {0}'.format(data.shape[0]))
            if training_settings is not None:
                print('training settings: {0}'.format(training_settings))
            print(data[:10])

    if args.summary:
        total_data = np.array(total_data)
        total_data = total_data[np.argsort(total_data[:, 0])]

        for k, entry in enumerate(total_data[:30]):
            print('top {1} loss: {0:.6f}'.format(entry[0], k + 1))
            if 'extra' in entry[1]:
                if 'cv_valid' in entry[1]['extra']:
                    print('cv_valid: {0}'.format(entry[1]['extra']['cv_valid']))
                    del entry[1]['extra']['cv_valid']
                if 'cv_test' in entry[1]['extra']:
                    print('cv_test: {0}'.format(entry[1]['extra']['cv_test']))
                    del entry[1]['extra']['cv_test']
            print('file: {0}'.format(entry[2]))
            print('config: {0}'.format(entry[1]['config'] if 'config' in entry[1] else entry[1]))
            if 'extra' in entry[1]:
                print('extra: {0}'.format(entry[1]['extra']))
            print()

    if args.table:
        total_data = np.array(total_data)
        total_data = total_data[np.argsort(total_data[:, 0])]

        model_to_result = dict()
        for entry in total_data:
            file_name = os.path.basename(entry[2])
            file_name, _ = os.path.splitext(file_name)
            model = file_name.split('2018')[0]
            if 'hyperband' in model:
                model = model.split('hyperband')[0]
            if model[-1] == '_':
                model = model[:-1]

            if 'config' in entry[1]:
                entry[1] = entry[1]['config']

            if 'exponential_decay' in entry[1] and entry[1]['exponential_decay']:
                model += ' with lr decay'
                if 'decay_rate' in entry[1]:
                    model += ' tf'
            model = model.lower()

            if model in model_to_result:
                if model_to_result[model][0] > entry[0]:
                    model_to_result[model] = entry
            else:
                model_to_result[model] = entry

        losses = [entry[0] for entry in model_to_result.values()]
        params = [entry[1] for entry in model_to_result.values()]
        for config in params:
            if 'decay_steps' in config:
                del config['decay_steps']
            if 'exponential_decay' in config:
                del config['exponential_decay']
            if 'decay_in_epochs' in config:
                del config['decay_in_epochs']
            if 'learning_rate' in config:
                value = config['learning_rate']
                del config['learning_rate']
                config['lr'] = value
            if 'learning_rate_end' in config:
                value = config['learning_rate_end']
                del config['learning_rate_end']
                config['lr_end'] = value
            if 'batch_size' in config:
                del config['batch_size']
            for key, value in config.items():
                if isinstance(value, float):
                    config[key] = np.round(value, 6)

        params = [", ".join(["{0}: {1}".format(key, value) for key, value in config.items()])
                  for config in params]

        estimators = [key for key in model_to_result.keys()]
        frame = get_pd_frame_task2(losses, params, estimators)
        print(frame)
