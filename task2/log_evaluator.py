from json import JSONDecodeError

import numpy as np
import json
import argparse
import os


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
        if 'hyperband' in os.path.basename(file):
            if 'LSTM' in os.path.basename(file):
                training_settings = data[-1]
                data = data[:-1]
            data = [(d['loss'], d, file) for d in data] if args.summary else [(d['loss'], d) for d in data]
        else:
            data = [(d[0], d[1], file) for d in data] if args.summary else data

        if args.summary:
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
            print('file: {0}'.format(entry[2]))
            print('config: {0}'.format(entry[1]))
            print()
