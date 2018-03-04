import datetime
import multiprocessing as mp

import numpy as np
from sklearn.model_selection import KFold
from keras.models import clone_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as  plt

from util.loader import load_data
from util.time_series_data import get_time_series, reshape_X, reshape_y
from models.lstm import lstm
from util.common import loss
from util.tensorboard import tensorboard_log_values
from preprocessing.standard_scaler import StandardScaler
from preprocessing.augmentation import add_nontraining_time_series, add_perturbed_time_series

def current_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_default_config():
    return {"lr" : 0.01,
            "batchsize" : 1,
            "lr_decay" : False,
            "config_step" : True,
            "repeat_config" : False,
            "augment" : False,
            "weight_decay" : False}

def random_boolean(p=0.5):
    return np.random.binomial(1, p) == 1

def get_random_config():
    return {"lr" : np.power(10.0, np.random.uniform(-4, -2)),
            "batchsize" : int(np.power(2.0, np.random.uniform(0, 6))),
            "lr_decay" : random_boolean(),
            "decay" : np.power(10.0, np.random.uniform(-8, -1)),
            "config_step" : random_boolean(),
            "repeat_config" : random_boolean(),
            "augment" : random_boolean(),
            "add_perturbed" : int(np.power(10.0, np.random.uniform(0, 3))),
            "add_nontraining" : int(np.power(10.0, np.random.uniform(0, 2))),
            "weight_decay" : random_boolean(),
            "alpha" : np.power(10.0, np.random.uniform(-8, -2))}

def task3(config,
          randomize_length,
          n_steps,
          epochs,
          log_dir="logs"):
    n_steps_valid = n_steps
    n_steps_test = n_steps
    
    use_configs = True
    config_step = config["config_step"]
    repeat_config = config["repeat_config"]
    scale_configs = True
    
    validation_split = 0.3
    evaluate_each = 1
    
    lr = config["lr"]
    batchsize = config["batchsize"]
    
    lr_decay = config["lr_decay"]
    decay = 0 if not lr_decay else config["decay"]
    
    regularize = config["weight_decay"]
    alpha = 0 if not regularize else config["alpha"]
    
    remove_nonlearning = False
    
    augment = config["augment"]
    add_perturbed = 0 if not augment else config["add_perturbed"]
    add_nontraining = 0 if not augment else config["add_nontraining"]
    
    # title of current run
    run_name = current_time_str()
    if not randomize_length:
        run_name += "_%is" % n_steps
    else:
        run_name += "_rnd"
    run_name += "_lr%f" % lr
    run_name += "_bs%i" % batchsize
    if lr_decay:
        run_name += "_dc%f" % decay
    if regularize:
        run_name += "_a%f" % alpha
    run_name += "_cstp" if config_step else ""
    run_name += "_rptcnfg" if repeat_config else ""
    if augment:
        run_name += "_augm_%i_%i" % (add_perturbed, add_nontraining)
    print(run_name)
    
    # functions
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
        if config_step:
            n_steps_test += 1
        final_y = [learning_curves[index][-1] for index in test_indices]
        pred = predict_whole_sequences(model, X_test[:, :n_steps_test, :])
        final_y_hat = pred[:, -1, 0]
        return loss(np.array(final_y), final_y_hat)
    
    # file name for plots
    tmp_file_name = "tmp/model_%s" % run_name
    
    if config_step:
        n_steps_train = n_steps
        n_steps_valid += 1
        n_steps_test += 1
    else:
        n_steps_train = n_steps - 1
    
    # read data
    configs, learning_curves = load_data(source_dir='./data')
    
    if remove_nonlearning:
        keep_indices = [i for i in range(len(learning_curves)) if learning_curves[i][-1] < 0.8]
        configs = [configs[i] for i in keep_indices]
        learning_curves = [learning_curves[i] for i in keep_indices]
    
    n_params = len(configs[0]) if use_configs else 0
    d = n_params + 1
    
    # 3 fold CV:
    n_folds = 3
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
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
        
        model = lstm(d, lr, decay = decay, many2many = True, regularize = regularize,
                     alpha = alpha, batchsize = None)
        
        best_valid_e40 = {}
        for k in [5, 10, 20, 30]:
            best_valid_e40[k] = float("inf")
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
                    n_steps_train = int(np.random.uniform(5, 21))
                    if config_step:
                        n_steps_train += 1
                batch_begin = batch * batchsize
                batch_end = batch_begin + batchsize
                x = X_train_permuted[batch_begin:batch_end, :n_steps_train, :]
                y = Y_train_permuted[batch_begin:batch_end, 1:(n_steps_train+1)]
                y_hat = model.predict(x)
                model.train_on_batch(x, y)
                training_losses.append(loss(y, y_hat))
            training_loss = np.mean(training_losses)
            print("training loss =   %f" % training_loss)
            
            # validation
            if (epoch + 1) % 1 == 0:
                y_hat = model.predict(X_valid[:, :n_steps_valid, :])[:, :, 0]
                validation_loss = np.mean(loss(Y_valid, y_hat))
                print("validation loss = %f" % validation_loss)
            
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
                
                prefix = "losses_f%i/" % fold
                tensorboard_log_values(log_dir, run_name, epoch, {prefix + "training" : training_loss,
                                                         prefix + "validation" : validation_loss,
                                                         prefix + "validation_E40_5" : valid_e40_5,
                                                         prefix + "validation_E40_10" : valid_e40_10,
                                                         prefix + "validation_E40_20" : valid_e40_20,
                                                         prefix + "validation_E40_30" : valid_e40_30,
                                                         prefix + "validation_E40_mean" : mean_valid_e40})
                
                if mean_valid_e40 < best_mean_valid_e40:
                    print("* new best model *")
                    
                    best_valid_e40_epoch = epoch
                    best_valid_e40[5] = valid_e40_5
                    best_valid_e40[10] = valid_e40_10
                    best_valid_e40[20] = valid_e40_20
                    best_valid_e40[30] = valid_e40_30
                    best_mean_valid_e40 = mean_valid_e40
                    
                    best_model = clone_model(model)
                    best_model.set_weights(model.get_weights())
            
            """if (epoch + 1) % 10 == 0:
                filename = tmp_file_name + "_f%i_e%i.png" % (fold, epoch)
                print(filename)
                plot_predicted_curves(model, X_test, test_indices, filename = filename)"""
        
        # evaluation on test data
        test_e40 = {}
        test_e40[5] = evaluate_step40_loss(best_model, X_test, test_indices, 5)
        test_e40[10] = evaluate_step40_loss(best_model, X_test, test_indices, 10)
        test_e40[20] = evaluate_step40_loss(best_model, X_test, test_indices, 20)
        test_e40[30] = evaluate_step40_loss(best_model, X_test, test_indices, 30)
        fold_test_errors.append(test_e40)
        print(test_e40)
        
        #filename = tmp_file_name + "_f%i_best.png" % fold
        #print(filename)
        #plot_predicted_curves(best_model, X_test, test_indices, filename = filename)
    
    means_e40 = {}
    for steps in [5, 10, 20, 30]:
        print("MSE@40 for %i input steps:" % steps)
        e40_folds = [fold_res[steps] for fold_res in fold_test_errors]
        print(e40_folds)
        mean_e40 = np.mean(e40_folds)
        print("mean = %f" % mean_e40)
        means_e40[steps] = mean_e40
    return means_e40

if __name__ == "__main__":
    randomize_length = True
    n_steps = 20
    experiment = "successive_halving"  # choose from {"default", "random_search", "successive_halving"}
    tensorboard_log_dir = "logs/successive_halving_rnd_01"
    
    ###############
    ### DEFAULT ###
    ###############
    if experiment == "default":
        epochs = 1000
        logfile = "logs/default_%s_%ie_%s.log" % ("rnd" if randomize_length else (str(n_steps) + "s"),
                                                       epochs,
                                                       current_time_str())
        config = get_default_config()
        results = task3(config, randomize_length, n_steps, epochs, log_dir=tensorboard_log_dir)
        with open(logfile, "w") as f:
            f.write(str(config) + "\n")
            f.write(str(results) + "\n")
            f.write(str(np.mean([results[s] for s in results])) + "\n")
    
    #####################
    ### RANDOM SEARCH ###
    #####################
    elif experiment == "random_search":
        n_configs = 100
        epochs = 200
        logfile = "logs/random_search_%s_%ie_%s.log" % ("rnd" if randomize_length else (str(n_steps) + "s"),
                                                       epochs,
                                                       current_time_str())
        pool = mp.Pool()
        for i in range(n_configs):
            config = get_random_config()
            results = pool.apply(task3,
                                 args=(config, randomize_length, n_steps, epochs, tensorboard_log_dir))
            with open(logfile, "a") as f:
                f.write(str(config) + "\n")
                f.write(str(results) + "\n")
                f.write(str(np.mean([results[s] for s in results])) + "\n")
                f.write("\n")
                f.write(current_time_str() + "\n")
    
    ##########################
    ### SUCCESSIVE HALVING ###
    ##########################
    elif experiment == "successive_halving":
        n_configs = 256
        epochs = 1
        iterations = 8
    
        logfile = "logs/successive_halving_%s_%s.log" % ("rnd" if randomize_length else (str(n_steps) + "s"),
                                                       current_time_str())
        
        pool = mp.Pool()
        configs = [get_random_config() for i in range(n_configs)]
        for i in range(iterations):
            with open(logfile, "a") as f:
                f.write("\n### ITERATION %i ###\n\n" % i)
                f.write("%i epochs\n\n" % epochs)
            
            config_results = []
            for config in configs:
                results = pool.apply(task3,
                                     args=(config, randomize_length, n_steps, epochs, tensorboard_log_dir))
                mean_result = np.mean([results[s] for s in results])
                config_results.append(mean_result)
                with open(logfile, "a") as f:
                    f.write(str(config) + "\n")
                    f.write(str(results) + "\n")
                    f.write(str(mean_result) + "\n")
                    f.write("\n")
                    f.write(current_time_str() + "\n")
            
            n_configs = n_configs // 2
            configs = [pair[1] for pair in sorted(zip(config_results, configs))[:n_configs]]
            epochs = epochs * 2