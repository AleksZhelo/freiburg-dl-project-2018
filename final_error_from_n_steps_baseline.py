from __future__ import print_function

import numpy as np

# noinspection PyUnresolvedReferences
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, \
    BaggingRegressor
# noinspection PyUnresolvedReferences
from sklearn.svm import SVR
# noinspection PyUnresolvedReferences
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

from util.common import loss, print_pd_frame_from_multi_input_performances
from util.loader import load_data_as_numpy
from sklearn.preprocessing import StandardScaler

normalize = True
use_config = False
configs, learning_curves = load_data_as_numpy()

estimators = [
    'GradientBoostingRegressor(learning_rate=0.033, n_estimators=300)',
    # 'linear_model.SGDRegressor(loss=\'squared_loss\')',
    # 'linear_model.SGDRegressor(loss=\'epsilon_insensitive\', epsilon=0.005)',
    'LinearRegression()',
    # 'linear_model.Ridge(alpha=0.1)',
    'RandomForestRegressor(n_estimators=30)',
    # 'SVR(C=2.0, kernel=\'linear\', epsilon=0.005)',
    'BaggingRegressor()'
]

k_fold = KFold(n_splits=3, shuffle=True, random_state=1)

performances = np.zeros((len(estimators), 3, 4))

for m_idx, model_desc in enumerate(estimators):

    current_fold = 0

    print(model_desc)
    for train_indices, test_indices in k_fold.split(configs):
        # split into training and test data
        train_configs = configs[train_indices]
        train_curves = learning_curves[train_indices]
        test_configs = configs[test_indices]
        test_curves = learning_curves[test_indices]

        if normalize:
            scaler = StandardScaler()
            train_configs = scaler.fit_transform(train_configs)
            test_configs = scaler.transform(test_configs)

        for k, input_points in enumerate([5, 10, 20, 30]):
            if use_config:
                train_x = np.zeros((train_configs.shape[0], 5 + input_points))
                test_x = np.zeros((test_configs.shape[0], 5 + input_points))

                train_x[:, :5] = train_configs[:]
                train_x[:, 5:5 + input_points] = train_curves[:, :input_points]

                test_x[:, :5] = test_configs[:]
                test_x[:, 5:5 + input_points] = test_curves[:, :input_points]
            else:
                train_x = train_curves[:, :input_points]
                test_x = test_curves[:, :input_points]

            train_y = train_curves[:, -1]

            clf = eval(model_desc)
            clf.fit(train_x, train_y)

            pred_y = clf.predict(test_x)
            fold_loss = loss(pred_y, test_curves[:, -1])
            performances[m_idx, current_fold, k] = fold_loss
        print('fold {0} loss: {1:}'.format(current_fold, np.round(performances[m_idx, current_fold], 6)))
        current_fold += 1

    print('mean CV performance: {0} \n'.format(np.round(performances[m_idx].mean(axis=0), 6)))

frame = print_pd_frame_from_multi_input_performances(performances, [e.split('(')[0] for e in estimators])
frame = frame.sort_values('loss_mean')
frame.to_latex('out/task3_point2_baselines.tex')
