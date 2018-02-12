import numpy as np
from sklearn.model_selection import KFold

import matplotlib.pyplot as  plt

from util.common import loss
from util.loader import load_data
from util.time_series_data import get_time_series
from models.lstm import lstm

repeat_config = False
randomize_length = False
n_steps = 20
n_steps_valid = n_steps
n_steps_test = n_steps
epochs = 200
initialise_model = True

configs, learning_curves = load_data(source_dir='./data')
X = get_time_series(configs, learning_curves, repeat_config=repeat_config)
Y = [curve for curve in learning_curves]

n_params = len(configs[0])
d = n_params + 1

def plot_predicted_curves(model, X, test_indices, n_steps_test, filename = None):
    plt.figure(figsize=(20, 10))
    for i in range(20):
        index = test_indices[i]
        x = X[index][:n_steps_test, :].reshape(1,-1,d)
        for j in range(40 - n_steps_test):
            pred = model.predict(x).reshape(-1)
            xx = np.zeros((x.shape[0], x.shape[1] + 1, x.shape[2]))
            xx[:,:-1,:] = x
            xx[:,-1,-1] = pred[-1]
            if repeat_config:
                xx[:, -1, :-1] = x[:, -1, :-1]
            x = xx
        plt.subplot(4, 5, i + 1)
        plt.plot(learning_curves[test_indices[i]], "g")
        plt.plot(range(1, 40), pred, "r")
    if filename != None:
        plt.savefig(filename)

def predict_whole_sequence(model, x):
    n_existing_steps = x.shape[0]
    x = x.reshape(1, -1, d)
    for j in range(40 - n_existing_steps):
        pred = model.predict(x).reshape(-1)
        xx = np.zeros((x.shape[0], x.shape[1] + 1, x.shape[2]))
        xx[:,:-1,:] = x
        xx[:,-1,-1] = pred[-1]
        if repeat_config:
            xx[:, -1, :-1] = x[:, -1, :-1]
        x = xx
    return pred

def evaluate_step40_loss(model, test_indices, n_steps):
    n_test = len(test_indices)
    final_y = []
    final_y_hat = []
    for i in range(n_test):
        index = test_indices[i]
        x = X[index][:n_steps, :]
        pred = predict_whole_sequence(model, x)
        final_y.append(learning_curves[index][-1])
        final_y_hat.append(pred[-1])
    return loss(np.array(final_y), np.array(final_y_hat))

# 3 fold CV:
n_folds = 3
k_fold = KFold(n_splits=n_folds)
fold_mses = []
fold = 0

for training_indices, test_indices in k_fold.split(X):
    fold = fold + 1
    
    X_train = [X[index] for index in training_indices]
    Y_train = [Y[index] for index in training_indices]
    X_test = [X[index][:n_steps_valid,:] for index in test_indices]
    Y_test = [Y[index][1:(n_steps_valid+1)] for index in test_indices]
    n_train = len(training_indices)
    n_test = len(test_indices)
    
    if initialise_model:
        model = lstm(d, many2many = True, regularize = False, alpha = 5E-5) 
    
    best_epoch40_loss = float("inf")
    best_epoch = -1
    
    for epoch in range(epochs):
        print("epoch = %i" % epoch)
        
        training_losses = []
        for i in range(n_train):
            if randomize_length:
                n_steps = int(np.random.uniform(5, 21))
            x = X_train[i][:n_steps,:].reshape(1,-1,d)
            y = np.array(Y_train[i][1:(n_steps+1)]).reshape(1,-1,1)
            y_hat = model.predict(x)
            model.train_on_batch(x, y)
            training_losses.append(loss(y, y_hat))
        print("training loss =   %f" % np.mean(training_losses))
        
        # validation
        if (epoch + 1) % 1 == 0:
            validation_loss = []
            for i in range(n_test):
                x = X_test[i].reshape(1,-1,d)
                y = np.array(Y_test[i]).reshape(1,-1,1)
                y_hat = model.predict(x)
                validation_loss.append(loss(y, y_hat))
            print("validation loss = %f" % np.mean(validation_loss))
        
        if (epoch + 1) % 10 == 0:
          test_loss = evaluate_step40_loss(model, test_indices, n_steps_test)
          print("test MSE@40 = %f" % test_loss)
          if test_loss < best_epoch40_loss:
              best_epoch = epoch
              best_epoch40_loss = test_loss
              filename = "figures/LSTM_m2m_%i_steps_epoch_%i.png" % (n_steps, epoch)
              plot_predicted_curves(model, X, test_indices, n_steps_test, filename = filename)
    break

#%%

test_loss = evaluate_step40_loss(model, test_indices, n_steps_test)
print("final test MSE@40 = %f" % test_loss)

print("best MSE@40 at epoch %i = %f" % (best_epoch, best_epoch40_loss))

filename = "figures/LSTM_m2m_%i_steps_epoch_%i.png" % (n_steps, epoch)
plot_predicted_curves(model, X, test_indices, n_steps_test, filename = filename)