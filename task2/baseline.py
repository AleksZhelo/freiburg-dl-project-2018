import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from autosklearn.regression import AutoSklearnRegressor

from util.common import loss
from util.loader import load_data

def plot_yhat_over_y(y_hat, y):
    plt.plot(y, y_hat, 'x')
    plt.xlabel("y")
    plt.ylabel("y_hat")
    plt.axis("equal")
    plt.plot([0,1], [0,1], 'r')
    plt.show()

def main(n_folds=3, preprocessing=False):
    # read data and transform it to numpy arrays
    configs, learning_curves = load_data(source_dir='../data')
    configs = np.array(list(map(lambda x: list(x.values()), configs)))
    learning_curves = np.array(learning_curves)
    
    # initialise CV
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    performances = []
    
    # store predicted and true y
    all_y = []
    all_y_hat = []    
    
    # CV folds
    for train_indices, test_indices in k_fold.split(configs):
        # split into training and test data
        train_configs = configs[train_indices]
        train_curves = learning_curves[train_indices]
        test_configs = configs[test_indices]
        test_curves = learning_curves[test_indices]
        
        # preprocessing
        if preprocessing:
            scaler = StandardScaler()
            train_configs = scaler.fit_transform(train_configs)
            print(scaler.mean_)
            test_configs = scaler.transform(test_configs)
        
        # train model
        #model = LinearRegression()
        model = AutoSklearnRegressor(time_left_for_this_task=60)
        
        model.fit(train_configs, train_curves[:, -1])
        
        # evaluate model
        y = test_curves[:, -1]
        y_hat = model.predict(test_configs)
        test_loss = loss(y_hat, y)
        performances.append(test_loss)
        print("fold test loss = %f" % test_loss)
        
        # store prediction
        all_y = np.append(all_y, y)
        all_y_hat = np.append(all_y_hat, y_hat)
    
    print("mean CV loss = %f" % np.mean(performances))
    plot_yhat_over_y(y_hat, y)

if __name__ == "__main__":
    main()