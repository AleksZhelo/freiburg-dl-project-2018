from json import JSONDecodeError

import numpy as np
import pandas as pd
import json
import argparse
import os

from util.common import get_pd_frame_task2, get_pd_frame_task3


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

    parser.add_argument(
        '--rnn',
        action='store_true',
        help='Indicates whether the results were obtained from the RNN model.'
    )

    return parser.parse_args()


def parse_old_format(ls):
    dat = []
    for l in ls:
        split = l.split('loss:')[1].split('params:')
        dat.append((float(split[0][:-2]), eval(split[1])))
    return dat


def collect_best_results_per_model(total_data):
    model_to_result = dict()
    model_to_result_metadata = dict()
    for entry in total_data:
        model = get_model_name(entry[2])

        entry_1_full = entry[1].copy()
        if not args.rnn:
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
                model_to_result_metadata[model] = entry_1_full
        else:
            model_to_result[model] = entry
            model_to_result_metadata[model] = entry_1_full
    return model_to_result, model_to_result_metadata


def get_model_name(file_name):
    base_name = os.path.basename(file_name)
    base_name, _ = os.path.splitext(base_name)
    model = base_name.split('2018')[0]
    if 'hyperband' in model:
        model = model.split('hyperband')[0]
    if model[-1] == '_':
        model = model[:-1]
    return model


if __name__ == '__main__':
    args = parse_args()
    for path in args.files:
        if os.path.isdir(path):
            args.files.remove(path)
            args.files.extend([os.path.join(path, file) for file in os.listdir(path)])

    total_data = []
    for file in args.files:
        if os.path.isdir(file):
            continue
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
                data = [(d['loss'], d, file, training_settings) for d in data] if args.summary or args.table else [
                    (d['loss'], d) for d in
                    data]
            else:
                data = [(d['loss'], d, file) for d in data] if args.summary or args.table else [
                    (d['loss'], d) for d in
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

        model_to_result, model_to_result_metadata = \
            collect_best_results_per_model(total_data)

        tasks = []
        if not args.rnn:
            for key, entry in model_to_result.items():
                task = dict()
                task['name'] = get_model_name(entry[2])
                metadata = model_to_result_metadata[key]
                if 'epochs' in metadata:
                    task['train_epochs'] = metadata['epochs']
                task['params'] = entry[1].copy()
                if 'batch_size' in task['params']:
                    task['batch_size'] = task['params']['batch_size']
                    del task['params']['batch_size']
                tasks.append(task)
            with open('task2_best_models.txt', 'w') as f:
                json.dump(tasks, f)
        else:
            for key, entry in model_to_result.items():
                task = dict()
                task['name'] = get_model_name(entry[2])
                task['params'] = entry[1]['config'].copy()
                task['settings'] = entry[3]
                if 'epochs' in entry[1]:
                    task['settings']['train_epochs'] = entry[1]['epochs']
                task['model_desc'] = '{0}_{1}_{2}'.format(task['name'],
                                                          '_'.join(['{0}={1}'.format(a, b) for a, b in
                                                                    zip(task['params'].keys(),
                                                                        task['params'].values())]),
                                                          '_'.join(['{0}={1}'.format(a, b) for a, b in
                                                                    zip(task['settings'].keys(),
                                                                        task['settings'].values())]))
                tasks.append(task)
            with open('task3_best_models.txt', 'w') as f:
                json.dump(tasks, f)

        losses = [entry[0] for entry in model_to_result.values()]
        extras = None

        if not args.rnn:
            params = [entry[1] for entry in model_to_result.values()]
        else:
            params = [entry[1]['config'] for entry in model_to_result.values()]
            extras = [entry[1]['extra'] for entry in model_to_result.values()]

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

        if args.rnn:
            var_input_losses = [e['cv_test'] for e in extras]
            frame = get_pd_frame_task3(losses, var_input_losses,
                                       ['rnd' for _ in range(len(var_input_losses))],
                                       params, estimators)
            frame.to_csv('task3_table.csv')
            print(frame)
        else:
            frame = get_pd_frame_task2(losses, params, estimators)
            pd.set_option('display.height', 1000)
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_colwidth', 1000)
            frame.to_latex('../reports/task2_table.tex')
            print(frame)
