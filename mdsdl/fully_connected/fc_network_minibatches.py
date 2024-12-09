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

    def train(self, X_train, Y_train, epochs, learning_rate, batch_size):
        all_costs = []

        # Determine the number of samples and batches
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        for epoch in trange(epochs):
            cost = 0

            # Shuffle data for each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train, Y_train = X_train[indices], Y_train[indices]

            # Iterate over all batches (including the last partial batch if needed)
            for batch_idx in range(n_batches + 1):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)  # Handle last batch size

                # Extract mini-batch
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                # Forward propagation for the batch
                y_pred = X_batch
                for layer in self.layers:
                    y_pred = layer.feed_forward(y_pred)

                # Compute cost for the batch
                cost += self.cost_function(Y_batch, y_pred)

                # Backward propagation for the batch
                error = self.derivative_of_cost(Y_batch, y_pred)
                for layer in reversed(self.layers):
                    error, dJdW, dJdb = layer.backward_propagation(error, learning_rate)

            # Store average cost per epoch
            all_costs.append(cost / n_batches)

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
        # we already know that each y is only a single data record. But since y as the input to
        # feed_forward had to be 2d we have now 1 extra dimension that we need to get rid of:
        return  np.array(Y).reshape(X.shape[0], -1)

