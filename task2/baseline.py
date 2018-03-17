import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

# noinspection PyUnresolvedReferences
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, \
    BaggingRegressor
# noinspection PyUnresolvedReferences
from sklearn.svm import SVR
# noinspection PyUnresolvedReferences
from sklearn import linear_model

from sklearn.preprocessing import StandardScaler
# from autosklearn.regression import AutoSklearnRegressor
import util
from util.common import loss, ensure_dir
from util.loader import load_data
from util.plots import scatter, boxplot


def main(estimators, n_folds=3):
    # read data and transform it to numpy arrays
    configs, learning_curves = load_data(source_dir='../data')
    configs = np.array(list(map(lambda x: list(x.values()), configs)))
    learning_curves = np.array(learning_curves)

    # initialise CV
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    # store predicted and true y
    y_y_hat = np.zeros((len(estimators), 2, learning_curves.shape[0]))

    performances = np.zeros((len(estimators), 6))

    for m_idx, model_desc in enumerate(estimators):
        current_fold = 0

        print(model_desc)
        for preprocessing in [False, True]:
            # CV folds
            for train_indices, test_indices in k_fold.split(configs):
                # split into training and test data
                train_configs = configs[train_indices]
                train_curves = learning_curves[train_indices]
                test_configs = configs[test_indices]
                test_curves = learning_curves[test_indices]

                # preprocessing
                if preprocessing:
                    scaler = StandardScaler()
                    train_configs = scaler.fit_transform(train_configs)
                    test_configs = scaler.transform(test_configs)

                # train model
                model = eval(model_desc)

                model.fit(train_configs, train_curves[:, -1])

                # evaluate model
                y = test_curves[:, -1]
                y_hat = model.predict(test_configs)
                test_loss = loss(y_hat, y)
                performances[m_idx, current_fold] = test_loss
                print("fold test loss = %f" % test_loss)

                # store prediction
                if preprocessing:
                    y_y_hat[m_idx, 0, test_indices] = y
                    y_y_hat[m_idx, 1, test_indices] = y_hat
                current_fold += 1

        print("mean CV loss w/o prep = {0:.5f}, w prep = {1:.5f}".format(
            np.mean(performances[m_idx, :3]), np.mean(performances[m_idx, 3:6])
        ))

    data = {'no prep': np.mean(performances[:, :3], axis=1),
            'prep': np.mean(performances[:, 3:6], axis=1)}
    frame = pd.DataFrame(data, index=estimators)
    print(frame)

    return performances, y_y_hat


def save_scatter_plot(y, y_hat, loss, name):
    scatter(
        y, y_hat,
        '{0}: MSE {1}'.format(name, loss),
        os.path.join(plots_dir, '{0}_scatter.png'.format(name))
    )


if __name__ == "__main__":
    estimators = [
        'GradientBoostingRegressor()',
        # 'linear_model.LinearRegression()',
        # 'linear_model.Ridge(alpha=10)',
        # 'RandomForestRegressor(n_estimators=30)',
        # 'SVR(kernel=\'linear\')',
        # 'BaggingRegressor()',
        # 'AutoSklearnRegressor(time_left_for_this_task=60)'
    ]
    estimators_short = [
        'GBR',
        # 'linear_model.LinearRegression()',
        # 'linear_model.Ridge(alpha=10)',
        # 'RandomForestRegressor(n_estimators=30)',
        # 'SVR(kernel=\'linear\')',
        # 'BaggingRegressor()',
        # 'AutoSklearnRegressor(time_left_for_this_task=60)'
    ]
    performances, y_y_hat_baselines = main(estimators)

    with open('task2_best_models_results.txt', 'r') as f:
        task2_results = json.load(f)

    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    ensure_dir(plots_dir)

    errors = []

    for m_idx, estimator in enumerate(estimators):
        y = y_y_hat_baselines[m_idx, 0]
        y_hat = y_y_hat_baselines[m_idx, 1]
        loss = np.round(np.mean(performances[m_idx, 3:6]), 6)
        name = estimator.split('(')[0]

        errors.append((estimators_short[m_idx], y - y_hat))
        save_scatter_plot(y, y_hat, loss, name)
    for res in task2_results:
        y = np.array(res[2][0])
        y_hat = np.array(res[2][1])
        loss = np.round(res[1], 6)
        name = res[0]

        errors.append((name, y - y_hat))
        save_scatter_plot(y, y_hat, loss, name)
    boxplot(errors, squared=True, logarithmic=True,
            file_name=os.path.join(plots_dir, 'task2_boxplot.png'))
