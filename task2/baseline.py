import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from util.common import loss
from util.loader import load_data


def plot_yhat_over_y(y_hat, y):
    plt.plot(y, y_hat, 'x')
    plt.xlabel("y")
    plt.ylabel("y_hat")
    plt.axis("equal")
    plt.plot([0, 1], [0, 1], 'r')
    plt.show()


def main(estimators, n_folds=3):
    # read data and transform it to numpy arrays
    configs, learning_curves = load_data(source_dir='../data')
    configs = np.array(list(map(lambda x: list(x.values()), configs)))
    learning_curves = np.array(learning_curves)

    # initialise CV
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    # store predicted and true y
    all_y = []
    all_y_hat = []

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
                all_y = np.append(all_y, y)
                all_y_hat = np.append(all_y_hat, y_hat)
                current_fold += 1

        print("mean CV loss w/o prep = {0:.5f}, w prep = {1:.5f}".format(
            np.mean(performances[m_idx, :3]), np.mean(performances[m_idx, 3:6])
        ))
        # plot_yhat_over_y(y_hat, y)

    data = {'no prep': np.mean(performances[:, :3], axis=1),
            'prep': np.mean(performances[:, 3:6], axis=1)}
    frame = pd.DataFrame(data, index=estimators)
    print(frame)


if __name__ == "__main__":
    estimators = [
        'GradientBoostingRegressor()',
        'linear_model.LinearRegression()',
        'linear_model.Ridge(alpha=10)',
        'RandomForestRegressor(n_estimators=30)',
        'SVR(kernel=\'linear\')',
        'BaggingRegressor()',
        # 'AutoSklearnRegressor(time_left_for_this_task=60)'
    ]
    main(estimators)
