from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, \
    BaggingRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from util.common import loss
from util.loader import load_data_as_numpy
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


def predict_curve(model, config, start_points, prediction_length):
    x = np.zeros((1, 9))
    x[0, :5][:] = config
    predicted_curve = np.zeros(prediction_length)
    predicted_curve[:start_points.shape[0]][:] = start_points

    curr_point = start_points.shape[0]
    x[0, 5:9][:] = start_points[curr_point - 4:curr_point]
    while curr_point < prediction_length:
        point = model.predict(x).reshape(-1)
        predicted_curve[curr_point] = point
        curr_point += 1

        x[0, 5:9][:] = np.roll(x[0, 5:9], -1)
        x[0, 8] = point

    return predicted_curve


normalize = True
configs, learning_curves = load_data_as_numpy()

best_model = None

k_fold = KFold(n_splits=3, shuffle=True, random_state=1)
performances = np.zeros((3, 4))

current_fold = 0
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

    train_x = np.zeros((train_configs.shape[0] * 36, 9))
    train_y = np.zeros(train_configs.shape[0] * 36)

    for i in range(train_configs.shape[0]):
        for j in range(4, train_curves.shape[1]):
            train_x[i * 36 + j - 4, :5] = train_configs[i]
            train_x[i * 36 + j - 4, 5:9] = train_curves[i, j - 4:j]
            train_y[i * 36 + j - 4] = train_curves[i, j]

    # clf = SVR(C=2.0, kernel='linear', epsilon=0.005)
    # clf = RandomForestRegressor(n_estimators=30)
    clf = GradientBoostingRegressor(learning_rate=0.033, n_estimators=300)
    # clf = linear_model.SGDRegressor(loss='squared_loss')
    # clf = linear_model.SGDRegressor(loss='epsilon_insensitive')  # seems to be much worse
    clf.fit(train_x, train_y)

    for k, start_point in enumerate([4, 9, 19, 29]):
        pred_curves = np.array([predict_curve(clf, test_configs[t], test_curves[t, :start_point], test_curves.shape[1])
                                for t in range(test_configs.shape[0])])
        fold_loss = loss(pred_curves[:, -1], test_curves[:, -1])
        performances[current_fold, k] = fold_loss
    print('fold {0} loss: {1}'.format(current_fold, performances[current_fold]))
    current_fold += 1

print('mean CV performance: {0}'.format(performances.mean(axis=0)))

steps = test_curves.shape[1]
fig = plt.figure(figsize=(10, 10))
for i, idx in enumerate(np.random.choice(np.arange(0, test_configs.shape[0]), 20)):
    y_hat = predict_curve(clf, test_configs[idx], test_curves[idx, :4], steps)

    plt.subplot(4, 5, i + 1)
    plt.plot(range(40), test_curves[idx], "g")
    plt.plot(range(steps), y_hat, "r")
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()
