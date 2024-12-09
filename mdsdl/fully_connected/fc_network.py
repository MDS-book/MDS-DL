import numpy as np
from tqdm import trange
from mdsdl.utilities import MSE, MSE_derivative, sigmoidal_derivative, sigmoidal_function
from mdsdl.fully_connected import FullyConnectedLayer, ActivationLayer


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
                    error, dJdW, dJdb = layer.backward_propagation(error, learning_rate)

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

