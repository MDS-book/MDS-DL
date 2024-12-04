import pandas as pd
import numpy as np
from abc import ABC
from tqdm import trange
from mdsdl.utilities import tanh_function, tanh_derivative, MSE, MSE_derivative



# ____________________________________________________________________________
# Create an abstract base class for a generic layer of a neural network.
# This defines the "structure" of any layer of the neural net -- any 
# specialized layer should be derived from `NNLayer`: 
class NNLayer(ABC):
    """Abstract base class which serves as a "template" for fully connected
    layers and for activation layers.

    Each derived class has the input x and output y both of which are numpy arrays.
    Also, each derived class should implement the two methods below (with the
    same signature)
    """
    def __init__(self):
        self.x = np.empty(0)
        self.y = np.empty(0)

    def feed_forward(self, x):
        """Perform a forward step: compute output for given input     
        and additionally store the input and output vectors
        
        :param x: numpy array of all inputs for one record
        :returns: numpy array of all output for one record
        """
        raise NotImplementedError("Any layer needs to implement the \
                                  feed_forward method.")

    def backward_propagation(self, dJdy, learning_rate):
        """Back-propagate the error sensitivity to the input
        
        :returns: numpy array of dJ/dy of the previous layer
        """
        raise NotImplementedError("Any layer needs to implement the \
                                  backward_propagation method.")


# ____________________________________________________________________________
# Do the concrete implementation of a FCL and a AL
# Both of them are derived from the `NNLayer` ABC
class FullyConnectedLayer(NNLayer):
    """A fully connected layer

    Does the weight initialization and provides methods for the feed forward and
    the backward propagation.
    """

    def __init__(self, n_inputs, n_outputs, seed=None):
        """Initialize the state of the FC layer.

        :param n_inputs: number of input variable (= size of x = number of input neurons)
        :param n_outputs: number of output variable (= size of y)
        :param seed: Used for seeding the random number generator.
        """
        # initialize the random number generator
        super().__init__()
        rng = np.random.default_rng(seed)

        # initialize weights and bias values by sampling from in between -0.5 and 0.5
        self.weights = rng.random(size=(n_inputs, n_outputs)) - 0.5
        self.biases = rng.random(size=(1, n_outputs)) - 0.5

    def feed_forward(self, x):
        # `numpy.atleast_2d` to ensure that np.dot(self.x.T, dJdy) also works for 1d array from the input layer
        self.x = np.atleast_2d(x)  
        self.y = np.dot(self.x, self.weights) + self.biases
        return self.y

    def backward_propagation(self, dJdy, learning_rate):
        dJdW = np.dot(self.x.T, dJdy)
        dJdb = np.dot(np.ones(self.x.shape[0]), dJdy)  # results in a 1D array
        self.weights -= learning_rate * dJdW
        self.biases -= learning_rate * dJdb

        dJdy_prev = np.dot(dJdy, self.weights.T)  
        return dJdy_prev


class ActivationLayer(NNLayer):
    def __init__(self, activation_function=tanh_function, activation_derivative=tanh_derivative):
        """Store the activation function and its derivative for reuse

        You can change the activation function types for each layer but typically every
        AL uses the same activation function.
        """
        super().__init__()
        self.phi = activation_function
        self.dphidx = activation_derivative

    def feed_forward(self, x):
        self.x = np.atleast_2d(x)
        self.y = self.phi(self.x)
        return self.y

    def backward_propagation(self, dJdy, learning_rate):
        return self.dphidx(self.x) * dJdy


# ____________________________________________________________________________
# The `Network` class provides functionality for assembling the network, for 
# training, and for predicting
class FCNetwork:
    def __init__(self, cost_function, derivative_of_cost):
        self.layers = []
        self.cost_function = cost_function
        self.derivative_of_cost = derivative_of_cost

    def add_layer(self, layer):
        self.layers.append(layer)
        
    def train(self, X_train, Y_train, epochs, learning_rate):
        all_costs = []
        for i in trange(epochs):
            cost = 0
            for x_train, y_train in zip(X_train, Y_train):
                # compute prediction by forward propagations through all layers
                y_pred = x_train
                for layer in self.layers:
                    y_pred = layer.feed_forward(y_pred)

                cost += self.cost_function(y_train, y_pred)

                # backprop. of errors (changes weights and biases)
                error = self.derivative_of_cost(y_train, y_pred)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            all_costs.append(cost / X_train.shape[0])
        return all_costs

    def predict(self, X):
        """Predict target vector/value for given input vector

        :param X: feature matrix with records in rows and features in columns 
        :returns Y: target matrix with records in rows and target variables in columns
        """
        Y = []
        for x in np.atleast_2d(X):
            y = np.atleast_2d(x)
            for layer in self.layers:
                y = layer.feed_forward(y)
            Y.append(y)
        return np.array(Y)
