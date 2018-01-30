import numpy as np
from sklearn.model_selection import KFold

from util.loader import load_data
from util.time_series_data import get_time_series
from models.lstm import lstm

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

def main(n_steps=5, epochs=100, repeat_config=False):
    """
    Trains and evaluates the MSE of a LSTM model in a 3-fold CV.
    
    n_steps: how many steps of the learning curve are used for training and predicting
    epochs: number of training epochs
    repeat_config: if True the configuration is fed into the network in each
        time step, otherwise only in the first time step
    """
    configs, learning_curves = load_data(source_dir='./data')
    X = get_time_series(configs, learning_curves, repeat_config=repeat_config)
    Y = [curve[-1] for curve in learning_curves]
    
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
        print("***** FOLD %i *****" % fold)
        
        X_train = [X[index] for index in training_indices]
        Y_train = [Y[index] for index in training_indices]
        X_test = [X[index] for index in test_indices]
        Y_test = [Y[index] for index in test_indices]
        n_train = len(training_indices)
        
        model = lstm(d)
        
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
            if (epoch + 1) % 10 == 0:
                predictions = predict(model, X_test)
                print("y_hat", "y")
                for y_hat, y in zip(predictions, Y_test):
                    print(y_hat, y)
        
        # evaluation:
        fold_mses.append(evaluate(model, X_test, Y_test))
    
    print("\nmse per fold:")
    print(fold_mses)
    print("mean mse:")
    print(np.mean(fold_mses))


if __name__ == "__main__":
    main()
