import numpy as np


def train_test_split(X, Y, fraction, seed=None):
    """Returns a training and a testing dataset

    X and Y are numpy arrays with as many rows as data records, 
    fraction (=0..1) defines the relative amount of training data, 
    and seed is the initialization of the random number generator.
    """
    rng = np.random.default_rng(seed)          # initialize random number generator
    n_total_data = X.shape[0]                  # number of data in the full DS  
    n_training = int(fraction * n_total_data)  # use fraction of data for training

    indices = np.arange(n_total_data)          # range of integers for indexing X and Y
    rng.shuffle(indices)                       # shuffle elements of "indices" in place
    X = X[indices]                             # X and y are "shuffled" such that each X[i] 
    Y = Y[indices]                             # ...  still is associated with the same Y[i]
       
    X_training = X[:n_training]                # use the first "n_training" points for the
    Y_training = Y[:n_training]                # training dataset and ...
    X_testing = X[n_training:]                 # the rest of the data for testing.
    Y_testing = Y[n_training:]

    return X_training, Y_training, X_testing, Y_testing



# ____________________________________________________________________________
# Define a set of activation functions and their derivatives
def sigmoidal_function(x):
    return 1 / (1 + np.exp(-x))


def sigmoidal_derivative(x):
    return sigmoidal_function(x)*(1.0 - sigmoidal_function(x))
    
    
def tanh_function(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# ____________________________________________________________________________
# Define the cost function and its derivative 
def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def dMSE_dy(y_true, y_pred):
    return 2 / y_pred.size * (y_pred - y_true) 

