import numpy as np
from abc import ABC
from numpy.typing import NDArray
from typing import Union, Tuple
from mdsdl.utilities import tanh_function, tanh_derivative



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

        :todo: x should always have shape = n_records x n_features, even if it is just a single record. Consequence: we don't need np.at_least2d() in each implementation of predict/feed_forward

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
        """Performs a feed forward step and stores the input data

        The returned value has the same shape as the input `x` but
        is at least a 2D array of shape (n_records, n_inputs), where
        n_inputs is the number of inputs to this layer.
        """
        # `numpy.atleast_2d` to ensure that np.dot(self.x.T, dJdy) also works for 1d array from the input layer
        # Then, the output is a 2D array
        self.x = np.atleast_2d(x)
        self.y = np.dot(self.x, self.weights) + self.biases
        return self.y

    def backward_propagation(self, dJdy, learning_rate):
        """Also returns dJdW and dJdb for debugging purposes.
        dJdW: (n_inputs, n_outputs)
        dJdb: (1, n_outputs)
        """
        dJdW = np.dot(self.x.T, dJdy)

        # axis=0: Sums across the mini-batch dimension.
        # keepdims=True: Ensures the resulting shape is consistent with the biases' shape.
        dJdb = np.sum(dJdy, axis=0, keepdims=True)
        #dJdb = np.dot(np.ones(self.x.shape[0]), dJdy)  # results in a 1D array

        # Compute dJdx before updating weights
        dJdy_prev = np.dot(dJdy, self.weights.T)  # also called dJdx

        self.weights -= learning_rate * dJdW
        self.biases -= learning_rate * dJdb

        return dJdy_prev, dJdW, dJdb


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
        """Performs a feed forward step and stores the input data

        The returned value has the same shape as the input `x` but
        is at least a 2D array of shape (n_records, n_inputs), where
        n_inputs is the number of inputs to this layer.
        """
        self.x = np.atleast_2d(x)
        self.y = self.phi(self.x)
        return self.y

    def backward_propagation(self, dJdy, learning_rate):
        dJdy_prev = self.dphidx(self.x) * dJdy
        dJdW, dJdb = None, None  # the AL has no trainable weights
        return dJdy_prev, dJdW, dJdb

