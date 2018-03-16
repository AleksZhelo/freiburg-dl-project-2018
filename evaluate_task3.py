import numpy as np
from keras.models import load_model
from sklearn.model_selection import KFold
from util.loader import load_data
from util.time_series_data import get_time_series, reshape_X
from preprocessing.standard_scaler import StandardScaler
from util.plots import scatter, extrapolation

def predict_whole_sequences(model, X, config_step, until=40):
    n = X.shape[0]
    true_steps = X.shape[1]
    d = X.shape[2]
    final_step = until + 1 if config_step else until
    XX = np.zeros((n, final_step, d))
    XX[:, :true_steps, :] = X
    for j in range(true_steps, final_step):
        pred = model.predict(XX[:, :j, :])
        XX[:, j, -1] = pred[:, -1, 0]
        """if repeat_config:
            XX[:, j, :-1] = XX[:, j-1, :-1]"""
    return pred[:, (true_steps - 1):, 0]

configs, learning_curves = load_data(source_dir='./data')
until = 40

for n_steps in [-1, 5, 10, 20]:
    randomize_length = n_steps == -1
    
    config_step = True if randomize_length or n_steps == 10 else False
    
    n_folds = 3
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold = 0
    
    fold_test_errors = []
    
    y_e40 = []
    y_hat_e40 = {5 : [],
                 10 : [],
                 20 : [],
                 30 : []}
    
    for training_indices, test_indices in k_fold.split(learning_curves):
        fold = fold + 1
        
        # load model
        model_file_name = "trained_models/best_model_%s_fold%i.h5" % ("rnd" if randomize_length else (str(n_steps) + "s"),
                                                                      fold)
        model = load_model(model_file_name)
        model.compile("adam", "mse")
        
        # prepare training data:
        configs_train = [configs[index] for index in training_indices]
        learning_curves_train = [learning_curves[index] for index in training_indices]
        scaler = StandardScaler()
        configs_train = scaler.fit_transform(configs_train)
        
        # prepare test data:
        configs_test = [configs[index] for index in test_indices]
        learning_curves_test = [learning_curves[index] for index in test_indices]
        configs_test = scaler.transform(configs_test)
        X_test = get_time_series(configs_test, learning_curves_test,
                                 use_configs=True,
                                 repeat_config=False,
                                 config_step=config_step)
        X_test = reshape_X(X_test)
        n_test = len(test_indices)
        
        preds = {}
        
        for test_steps in [5, 10, 20, 30]:
            if config_step:
                x_steps = test_steps + 1
            else:
                x_steps = test_steps
            preds[test_steps] = predict_whole_sequences(model, X_test[:, :x_steps, :], config_step, until=until)
        
        for i in range(n_test):
            file_name = "task3_plots/%s_fold%i_curve%i_until%i.png" % ("rnd" if randomize_length else (str(n_steps) + "s"),
                                                                       fold,
                                                                       i,
                                                                       until)
            extrapolation(learning_curves_test[i],
                          [("5 steps", preds[5][i]),
                           ("10 steps", preds[10][i]),
                           ("20 steps", preds[20][i]),
                           ("30 steps", preds[30][i])],
                          file_name = file_name,
                          n_steps=until)
            y_e40.append(learning_curves_test[i][-1])
            for test_steps in [5, 10, 20, 30]:
                y_hat_e40[test_steps].append(preds[test_steps][i, -1])
    
    if until == 40:
        for test_steps in [5, 10, 20, 30]:
            mse = np.mean(np.power(np.array(y_e40) - np.array(y_hat_e40[test_steps]), 2))
            title = "tr = %s, te = %i, MSE = %f" % ("rnd" if randomize_length else str(n_steps),
                                                    test_steps,
                                                    mse)
            file_name = "task3_plots/%s_pred%i_scatter.png" % ("rnd" if randomize_length else (str(n_steps) + "s"),
                                                               test_steps)
            scatter(y_e40,
                    y_hat_e40[test_steps],
                    title,
                    file_name)