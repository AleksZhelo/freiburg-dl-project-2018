from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from util.common import loss
from util.loader import load_data_as_numpy
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVR
from sklearn import linear_model

normalize = True
configs, learning_curves = load_data_as_numpy()

if normalize:
    configs = StandardScaler().fit_transform(configs)

train_x = np.zeros((configs.shape[0] * 35, 9))
train_y = np.zeros(configs.shape[0] * 35)

test_x = np.concatenate((configs, learning_curves[:, -5:-1]), axis=1)
test_y = learning_curves[:, -1]

for i in range(configs.shape[0]):
    for j in range(4, learning_curves.shape[1] - 1):
        train_x[i * 35 + j - 4, :5] = configs[i]
        train_x[i * 35 + j - 4, 5:9] = learning_curves[i, j - 4:j]
        train_y[i * 35 + j - 4] = learning_curves[i, j]

# clf = SVR(C=1.0, kernel='linear', epsilon=0.1)  # well this won't work with ~85k samples
clf = linear_model.SGDRegressor(loss='squared_loss')
# clf = linear_model.SGDRegressor(loss='epsilon_insensitive')  # seems to be much worse
clf.fit(train_x, train_y)

print(loss(clf.predict(test_x), test_y))

for i, idx in enumerate(np.random.choice(np.arange(0, configs.shape[0]), 20)):
    y_hat = clf.predict(
        np.concatenate((train_x[idx * 35:(idx + 1) * 35], test_x[idx].reshape(1, -1)), axis=0)
    ).reshape(-1)

    y_hat = np.concatenate((learning_curves[idx, :4], y_hat))

    plt.subplot(4, 5, i + 1)
    plt.plot(range(40), learning_curves[idx], "g")
    plt.plot(range(40), y_hat, "r")
    plt.ylim(0, 1)
plt.show()
