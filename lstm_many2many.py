import sys
import numpy as np
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as  plt

from util.loader import load_data
from util.time_series_data import get_time_series, reshape_X, reshape_y
from models.lstm import lstm, small_lstm
from baseline import loss
from preprocessing.standard_scaler import StandardScaler
from preprocessing.augmentation import add_nontraining_time_series, add_perturbed_time_series

randomize_length = False
n_steps = 5
n_steps_valid = n_steps
n_steps_test = n_steps

epochs = 1000
initialise_model = True
lr = 0.1
decay = 0
batchsize = 1

use_configs = True
config_step = True
repeat_config = False
scale_configs = True

regularize = False
alpha = 1E-4

remove_nonlearning = False
add_perturbed = 0
add_nontraining = 0

validation_split = 0.3
evaluate_each = 10

tmp_file_name = "tmp/model01_5s"


# read command args
"""
randomize_length = sys.argv[1] == "1"
n_steps = int(sys.argv[2])
lr = float(sys.argv[3])
decay = float(sys.argv[4])
tmp_file_name = sys.argv[5]
batchsize = int(sys.argv[6])
"""

configs, learning_curves = load_data(source_dir='./data')

if remove_nonlearning:
    keep_indices = [i for i in range(len(learning_curves)) if learning_curves[i][-1] < 0.8]
    configs = [configs[i] for i in keep_indices]
    learning_curves = [learning_curves[i] for i in keep_indices]

n_params = len(configs[0]) if use_configs else 0
d = n_params + 1

def plot_predicted_curves(model, X_test, test_indices, filename = None):
    plt.figure(figsize=(20, 10))
    n_plots = 20
    pred = predict_whole_sequences(model, X_test[:n_plots, :n_steps_test, :])
    for i in range(n_plots):
        plt.subplot(4, 5, i + 1)
        plt.plot(learning_curves[test_indices[i]], "g")
        if config_step:
            plt.plot(range(40), pred[i, :, :], "r")
        else:
            plt.plot(range(1, 40), pred[i, :, :], "r")
    if filename != None:
        plt.savefig(filename)
        plt.close()

def predict_whole_sequences(model, X):
    n = X.shape[0]
    true_steps = X.shape[1]
    d = X.shape[2]
    final_step = 41 if config_step else 40
    XX = np.zeros((n, final_step, d))
    XX[:, :true_steps, :] = X
    for j in range(true_steps, final_step):
        pred = model.predict(XX[:, :j, :])
        XX[:, j, -1] = pred[:, -1, 0]
        if repeat_config:
            XX[:, j, :-1] = XX[:, j-1, :-1]
    return pred

def evaluate_step40_loss(model, X_test, test_indices, n_steps_test):
    final_y = [learning_curves[index][-1] for index in test_indices]
    pred = predict_whole_sequences(model, X_test[:, :n_steps_test, :])
    final_y_hat = pred[:, -1, 0]
    return loss(np.array(final_y), final_y_hat)

# 3 fold CV:
n_folds = 3
k_fold = KFold(n_splits=n_folds)
fold_mses = []
fold = 0

fold_test_errors = []

for training_indices, test_indices in k_fold.split(learning_curves):
    fold = fold + 1
    
    # split into training and validation
    training_indices = np.random.permutation(training_indices)
    valid_split_index = int(validation_split * len(training_indices))
    validation_indices = training_indices[:valid_split_index]
    training_indices = training_indices[valid_split_index:]
    
    # prepare training data:
    configs_train = [configs[index] for index in training_indices]
    learning_curves_train = [learning_curves[index] for index in training_indices]
    if scale_configs:
        scaler = StandardScaler()
        configs_train = scaler.fit_transform(configs_train)
    if add_perturbed > 0:
        configs_train, learning_curves_train = add_perturbed_time_series(configs_train,
                                                                           learning_curves_train,
                                                                           add_perturbed)
    if add_nontraining > 0:
        configs_train, learning_curves_train = add_nontraining_time_series(configs_train,
                                                                           learning_curves_train,
                                                                           add_nontraining)
    n_train = len(configs_train)
    X_train = get_time_series(configs_train, learning_curves_train,
                              use_configs=use_configs,
                              repeat_config=repeat_config,
                              config_step=config_step)
    X_train = reshape_X(X_train)
    Y_train = learning_curves_train
    
    # prepare validation data:
    configs_valid = [configs[index] for index in validation_indices]
    learning_curves_valid = [learning_curves[index] for index in validation_indices]
    if scale_configs:
        configs_valid = scaler.transform(configs_valid)
    X_valid = get_time_series(configs_valid, learning_curves_valid,
                              use_configs=use_configs,
                              repeat_config=repeat_config,
                              config_step=config_step)
    X_valid = reshape_X(X_valid)
    
    # prepare test data:
    configs_test = [configs[index] for index in test_indices]
    learning_curves_test = [learning_curves[index] for index in test_indices]
    if scale_configs:
        configs_test = scaler.transform(configs_test)
    X_test = get_time_series(configs_test, learning_curves_test,
                             use_configs=use_configs,
                             repeat_config=repeat_config,
                             config_step=config_step)
    X_test = reshape_X(X_test)
    
    n_valid = len(validation_indices)
    n_test = len(test_indices)
    
    Y_train = reshape_y(Y_train)
    Y_valid = [learning_curves_valid[i][1:(n_steps_valid+1)] for i in range(n_valid)]
    Y_test = [learning_curves_test[i][1:(n_steps_test+1)] for i in range(n_test)]
    
    n_batches = int(np.ceil(n_train / batchsize))
    
    if initialise_model:
        model = lstm(d, lr, decay = decay, many2many = True, regularize = regularize,
                     alpha = alpha, batchsize = None)
        #model = small_lstm(d, lr, decay = decay, many2many = True, regularize = regularize,
        #                   alpha = alpha, batchsize = None)
    
    best_valid_e40 = {}
    best_valid_e40[5] = float("inf")
    best_valid_e40[10] = float("inf")
    best_valid_e40[20] = float("inf")
    best_valid_e40[30] = float("inf")
    best_mean_valid_e40 = float("inf")
    best_valid_e40_epoch = -1
    
    for epoch in range(epochs):
        print("epoch = %i" % epoch)
        
        # random permutation of training data
        permutation = np.random.permutation(range(n_train))
        X_train_permuted = X_train[permutation, :, :]
        Y_train_permuted = Y_train[permutation, :, :]
        
        training_losses = []
        for batch in range(n_batches):
            if randomize_length:
                n_steps = int(np.random.uniform(5, 21))
            batch_begin = batch * batchsize
            batch_end = batch_begin + batchsize
            x = X_train_permuted[batch_begin:batch_end, :n_steps, :]
            y = Y_train_permuted[batch_begin:batch_end, 1:(n_steps+1)]
            y_hat = model.predict(x)
            model.train_on_batch(x, y)
            training_losses.append(loss(y, y_hat))
        print("training loss =   %f" % np.mean(training_losses))
        
        # validation
        if (epoch + 1) % 1 == 0:
            y_hat = model.predict(X_valid[:, :n_steps_valid, :])[:, :, 0]
            validation_loss = loss(Y_valid, y_hat)
            print("validation loss = %f" % np.mean(validation_loss))
        
        if (epoch + 1) % evaluate_each == 0:
            print(lr, decay, batchsize)
            print("best[:5]  = %f @ %i" % (best_valid_e40[5], best_valid_e40_epoch))
            print("best[:10] = %f @ %i" % (best_valid_e40[10], best_valid_e40_epoch))
            print("best[:20] = %f @ %i" % (best_valid_e40[20], best_valid_e40_epoch))
            print("best[:30] = %f @ %i" % (best_valid_e40[30], best_valid_e40_epoch))
            
            valid_e40_5 = evaluate_step40_loss(model, X_valid, validation_indices, 5)
            print("validation MSE[:5]@40  = %f" % valid_e40_5)
            valid_e40_10 = evaluate_step40_loss(model, X_valid, validation_indices, 10)
            print("validation MSE[:10]@40 = %f" % valid_e40_10)
            valid_e40_20 = evaluate_step40_loss(model, X_valid, validation_indices, 20)
            print("validation MSE[:20]@40 = %f" % valid_e40_20)
            valid_e40_30 = evaluate_step40_loss(model, X_valid, validation_indices, 30)
            print("validation MSE[:30]@40 = %f" % valid_e40_30)
            
            mean_valid_e40 = np.mean([valid_e40_5, valid_e40_10, valid_e40_20, valid_e40_30])
            if mean_valid_e40 < best_mean_valid_e40:
                print("new best model")
                
                best_valid_e40_epoch = epoch
                best_valid_e40[5] = valid_e40_5
                best_valid_e40[10] = valid_e40_10
                best_valid_e40[20] = valid_e40_20
                best_valid_e40[30] = valid_e40_30
                best_mean_valid_e40 = mean_valid_e40
                
                # evaluation on test data
                test_e40 = {}
                test_e40[5] = evaluate_step40_loss(model, X_test, test_indices, 5)
                test_e40[10] = evaluate_step40_loss(model, X_test, test_indices, 10)
                test_e40[20] = evaluate_step40_loss(model, X_test, test_indices, 20)
                test_e40[30] = evaluate_step40_loss(model, X_test, test_indices, 30)
                
                filename = tmp_file_name + "_f%i_e%i" % (fold, epoch)
                print(filename)
                plot_predicted_curves(model, X_test, test_indices, filename = filename)
            print(test_e40)
    fold_test_errors.append(test_e40)

for steps in [5, 10, 20, 30]:
    print("MSE@40 for %i input steps:" % steps)
    e40_folds = [fold_res[steps] for fold_res in fold_test_errors]
    print(e40_folds)
    print("mean = %f" % np.mean(e40_folds))