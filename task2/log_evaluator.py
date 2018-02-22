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

    for file in args.files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
            except JSONDecodeError:
                f.seek(0)
                data = parse_old_format(f.readlines())

        data = np.array(data) if 'hyperband' not in os.path.basename(file) else np.array([(d['loss'], d) for d in data])
        data = data[np.argsort(data[:, 0])]
        print('----------{0}----------'.format(file))
        print('model evaluations: {0}'.format(data.shape[0]))
        print(data[:10])
