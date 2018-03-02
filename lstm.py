import numpy as np
from sklearn.model_selection import KFold

from util.loader import load_data
from util.time_series_data import get_time_series
from models.lstm import lstm
from preprocessing.standard_scaler import StandardScaler

lr = 0.1
scale_configs = True
regularize = False
alpha = 0.001
repeat_config = False

def predict(model, X):
    """
    Given a LSTM model and data matrix X,
    returns a list of predictions for each entry in X.
    """
    d = X[0].shape[1]
    return [model.predict(x.reshape(1,-1, d))[0][0] for x in X]

def evaluate(model, X, y):
    """
    Returns the MSE of a given LSTM model.
    """
    y_hat = predict(model, X)
    mse = np.mean(np.power(np.array(y) - np.array(y_hat), 2))
    return mse

def main(n_steps=10, epochs=200):
    """
    Trains and evaluates the MSE of a LSTM model in a 3-fold CV.
    
    n_steps: how many steps of the learning curve are used for training and predicting
    epochs: number of training epochs
    repeat_config: if True the configuration is fed into the network in each
        time step, otherwise only in the first time step
    """
    configs, learning_curves = load_data(source_dir='./data')
    Y = [curve[-1] for curve in learning_curves]
    
    n_params = len(configs[0])
    d = n_params + 1
    
    # 3 fold CV:
    n_folds = 3
    k_fold = KFold(n_splits=n_folds)
    fold_mses = []
    fold = 0
    for training_indices, test_indices in k_fold.split(Y):
        fold = fold + 1
        print("***** FOLD %i *****" % fold)
        
        # prepare training data:
        configs_train = [configs[index] for index in training_indices]
        learning_curves_train = [learning_curves[index][:n_steps] for index in training_indices]
        if scale_configs:
            scaler = StandardScaler()
            configs_train = scaler.fit_transform(configs_train)
        X_train = get_time_series(configs_train, learning_curves_train, repeat_config=repeat_config)
        Y_train = [Y[index] for index in training_indices]
        
        # prepare test data:
        configs_test = [configs[index] for index in test_indices]
        learning_curves_test = [learning_curves[index][:n_steps] for index in test_indices]
        if scale_configs:
            configs_test = scaler.transform(configs_test)
        X_test = get_time_series(configs_test, learning_curves_test, repeat_config=repeat_config)
        Y_test = [Y[index] for index in test_indices]

        n_train = len(training_indices)
        
        model = lstm(d, lr, regularize = regularize, alpha = alpha)
        
        # training:
        for epoch in range(epochs):
            print("epoch = %i" % epoch)
            for i in range(n_train):
                x = X_train[i].reshape(1,-1,d)
                y = Y_train[i]
                model.train_on_batch(x, np.array([[y]]))
                model.reset_states()
            
            # validation output:
            mse_train = evaluate(model, X_train, Y_train)
            mse_test = evaluate(model, X_test, Y_test)
            print("training mse = %f" % mse_train)
            print("test mse = %f" % mse_test)
            #if (epoch + 1) % 10 == 0:
            #    predictions = predict(model, X_test)
            #    print("y_hat", "y")
            #    for y_hat, y in zip(predictions, Y_test):
            #        print(y_hat, y)
        
        # evaluation:
        fold_mses.append(evaluate(model, X_test, Y_test))
    
    print("\nmse per fold:")
    print(fold_mses)
    print("mean mse:")
    print(np.mean(fold_mses))


if __name__ == "__main__":
    main()
