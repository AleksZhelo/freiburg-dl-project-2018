import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge as Regression
#from sklearn.kernel_ridge import KernelRidge as Regression
from sklearn.pipeline import make_pipeline

from util.common import loss
from util.loader import load_data

class MyFeatures():
    def __init__(self, degree, logarithmic):
        self.degree = degree  # not used
        self.logarithmic = logarithmic
    
    def transform(self, X):
        n = X.shape[0]
        m = X.shape[1]
        assert(m == 1)
        x = X[:, 0]
        Xt = np.zeros((n, 1))
        #Xt[:, 0] = x
        #Xt[:, 1] = np.power(x, 2)
        Xt[:, 0] = np.log10(x + 1) if self.logarithmic else x
        return Xt
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def fit(self, X, y=None):
        return self


configs, learning_curves = load_data(source_dir='./data')

degree = 1  # degree of polynomial when using PolynomialFeatures, not used in MyFeatures
train_steps_range = (10, 20)  # range of time steps that are used to fit a function
do_weighting = False # whether to repeat later time steps multiple times to give more weight on them when fitting the function
alpha = 0.1  # alpha value for Ridge regression
logarithmic = True  # whether to fit a logarithmic or linear function

n = len(learning_curves)
y_final = [curve[-1] for curve in learning_curves]
y_hat_final = []

for i in range(n):
    model = make_pipeline(MyFeatures(degree, logarithmic), Regression(alpha=alpha))
    
    if not do_weighting:
        x_train = np.array(list(range(train_steps_range[0], train_steps_range[1]))).reshape(-1, 1)
        y_train = np.array(learning_curves[i][train_steps_range[0]:train_steps_range[1]]).reshape(-1, 1)
    else:
        x_train = []
        y_train = []
        for j in range(train_steps_range[0], train_steps_range[1]):
            for k in range(int(np.power(j + 1, 2))):
                x_train.append(j)
                y_train.append(learning_curves[i][j])
        x_train = np.array(x_train).reshape(-1, 1)
        y_train = np.array(y_train).reshape(-1, 1)
    
    model.fit(x_train, y_train)
    
    x_test = np.array(list(range(40))).reshape(-1, 1)
    y_hat = model.predict(x_test).reshape(-1)
    y_hat_final.append(y_hat[-1])
    
    if i < 20:
        plt.subplot(4, 5, i + 1)
        plt.plot(range(40), learning_curves[i], "g")
        plt.plot(range(40), y_hat, "r")
        plt.ylim(0, 1)

print("test MSE @ epoch 40 = %f" % loss(np.array(y_final), np.array(y_hat_final)))