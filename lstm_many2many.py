

import numpy as np
from sklearn.model_selection import KFold

import matplotlib.pyplot as  plt

from util.loader import load_data
from util.time_series_data import get_time_series
from models.lstm import lstm
from baseline import loss

repeat_config = False
n_steps = 10
epochs = 100
initialise_model = True

configs, learning_curves = load_data(source_dir='./data')
X = get_time_series(configs, learning_curves, repeat_config=repeat_config)
Y = [curve[1:(n_steps+1)] for curve in learning_curves]

n_params = len(configs[0])
d = n_params + 1

X = [x[:n_steps,:] for x in X]

# 3 fold CV:
n_folds = 3
k_fold = KFold(n_splits=n_folds)
fold_mses = []
fold = 0

for training_indices, test_indices in k_fold.split(X):
    fold = fold + 1
    
    X_train = [X[index] for index in training_indices]
    Y_train = [Y[index] for index in training_indices]
    X_test = [X[index] for index in test_indices]
    Y_test = [Y[index] for index in test_indices]
    n_train = len(training_indices)
    n_test = len(test_indices)
    
    if initialise_model:
        model = lstm(d, many2many = True, regularize = False, alpha = 1E-5) 
    
    for epoch in range(epochs):
        print("epoch = %i" % epoch)
        
        training_losses = []
        for i in range(n_train):
            x = X_train[i].reshape(1,-1,d)
            y = np.array(Y_train[i]).reshape(1,-1,1)
            y_hat = model.predict(x)
            model.train_on_batch(x, y)
            training_losses.append(loss(y, y_hat))
        print("training loss =   %f" % np.mean(training_losses))
        
        # validation
        validation_loss = []
        for i in range(n_test):
            x = X_test[i].reshape(1,-1,d)
            y = np.array(Y_test[i]).reshape(1,-1,1)
            y_hat = model.predict(x)
            validation_loss.append(loss(y, y_hat))
        print("validation loss = %f" % np.mean(validation_loss))
    break

#%%

n_steps_test = 10

final_y = []
final_y_hat = []

plt.plot()

for i in range(n_test):
    index = test_indices[i]
    x = X[index][:n_steps_test, :].reshape(1,-1,d)
    #y = Y[index]
    for j in range(40 - n_steps_test):
        pred = model.predict(x).reshape(-1)
        xx = np.zeros((x.shape[0], x.shape[1] + 1, x.shape[2]))
        xx[:,:-1,:] = x
        xx[:,-1,-1] = pred[-1]
        if repeat_config:
            xx[:, -1, :-1] = x[:, -1, :-1]
        x = xx
    if i < 10:
        plt.subplot(2, 5, i + 1)
        plt.plot(learning_curves[test_indices[i]], "g")
        plt.plot(range(1, 40), pred, "r")
    final_y.append(learning_curves[test_indices[i]][-1])
    final_y_hat.append(pred[-1])

plt.show()
print("test MSE @ epoch 40 = %f" % loss(np.array(final_y), np.array(final_y_hat)))