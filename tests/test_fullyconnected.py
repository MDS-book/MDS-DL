import pytest
import numpy as np
from mdsdl.utilities import tanh_function, tanh_derivative
from mdsdl.fully_connected import FullyConnectedLayer, ActivationLayer


######################################################################################################
# These are some helper functions for computing the forwards and backward pass
# for a network of 2 neurons, 2 inputs and a single output. As activation
# function, the sigmoidal function is hard-coded.
def forward_pass(W, B, X1, X2):
    def sigmoidal_function(x):
        return 1 / (1 + np.exp(-x))

    (w1, w2, w3, w4, w5, w6), (b1, b2, b3) = W, B

    a1 = b1 + X1 * w1 + X2 * w2
    a2 = b2 + X1 * w3 + X2 * w4
    y1 = sigmoidal_function(a1)
    y2 = sigmoidal_function(a2)
    a3 = y1 * w5 + y2 * w6 + b3
    return a1, a2, y1, y2, a3


def predict(W, B, X1, X2):
    _, _, _, _, Y_pred = forward_pass(W, B, X1, X2)
    return Y_pred


def backward_pass(W, B, X1, X2, Y):
    w1, w2, w3, w4, w5, w6 = W
    a1, a2, y1, y2, a3 = forward_pass(W, B, X1, X2)

    dJdb1 = -(Y - a3) * w5 * y1 * (1 - y1)
    dJdw1 = dJdb1 * X1
    dJdw2 = dJdb1 * X2
    dJdb2 = -(Y - a3) * w6 * y2 * (1 - y2)
    dJdw3 = dJdb2 * X1
    dJdw4 = dJdb2 * X2
    dJdb3 = -(Y - a3)
    dJdw5 = -(Y - a3) * y1
    dJdw6 = -(Y - a3) * y2

    dJdW, dJdB = (dJdw1, dJdw2, dJdw3, dJdw4, dJdw5, dJdw6), (dJdb1, dJdb2, dJdb3)
    return dJdW, dJdB
######################################################################################################



class TestFullyConnectedLayer:
    def test_fully_connected_layer_initialization(self):
        n_inputs = 3
        n_outputs = 2
        layer = FullyConnectedLayer(n_inputs, n_outputs, seed=42)

        assert layer.weights.shape == (n_inputs, n_outputs), "Weights shape mismatch"
        assert layer.biases.shape == (1, n_outputs), "Biases shape mismatch"
        assert np.all(layer.weights >= -0.5) and np.all(layer.weights <= 0.5), "Weights not in expected range"
        assert np.all(layer.biases >= -0.5) and np.all(layer.biases <= 0.5), "Biases not in expected range"

    def test_fully_connected_layer_feed_forward(self):
        n_inputs = 2
        n_outputs = 1
        layer = FullyConnectedLayer(n_inputs, n_outputs)
        # Set weights and biases to known values
        layer.weights = np.array([[0.5], [0.5]])
        layer.biases = np.array([[0.1]])
        x = np.array([[1, 2]])
        expected_output = np.dot(x, layer.weights) + layer.biases  # [[1*0.5 + 2*0.5 + 0.1]] = [[1.6]]
        print(expected_output)
        output = layer.feed_forward(x)
        np.testing.assert_array_almost_equal(output, expected_output, err_msg="Feed forward output mismatch")

    def test_fully_connected_layer_feed_forward_2(self):
        w1, w2, b1 = 0.1, 0.4, 0.5  # first unit
        w3, w4, b2 = -0.7, 0.3, 0.1  # second unit
        w5, w6, b3 = 0.5, -0.3, 0.1  # output node
        W = (w1, w2, w3, w4, w5, w6)
        B = (b1, b2, b3)
        X = np.array([
            [1.1, -2.5],
            [5.0, 1.0],
        ])
        X1 = X[:, 0]
        X2 = X[:, 1]

        y1 = b1 + w1 * X1 + w2 * X2
        y2 = b2 + w3 * X1 + w4 * X2
        #print(np.stack([y1, y2], axis=1))

        # A hidden layer with two inputs per unit (n_inputs=2) and
        # two units, each of which has 1 output (n_outputs=2)
        layer = FullyConnectedLayer(n_inputs=2, n_outputs=2)
        layer.weights = np.array([
            [w1, w3],
            [w2, w4]
        ])
        layer.biases = np.array([[b1, b2]])
        y = layer.feed_forward(X)

        np.testing.assert_allclose(actual=y,
                                   desired=np.stack([y1, y2], axis=1),
                                   err_msg="forward pass difference")


    # def test_fully_connected_layer_feed_forward_with_sigmoid(self):
    #     w1, w2, b1 =  0.1, 0.4, 0.5  # first unit
    #     w3, w4, b2 = -0.7, 0.3, 0.1  # second unit
    #     w5, w6, b3 = 0.5, -0.3, 0.1  # output node
    #     W = (w1, w2, w3, w4, w5, w6)
    #     B = (b1, b2, b3)
    #     X = np.array([
    #         [1.1, -2.5],
    #         [5.0, 1.0],
    #     ])
    #     X1 = X[:, 0]
    #     X2 = X[:, 1]
    #
    #     a1, a2, y1, y2, a3 = forward_pass(W, B, X1, X2)
    #     print("\nana. output:", a1, a2)
    #     print("ana. output:", y1, y2)
    #     print(np.stack([y1, y2], axis=1))
    #
    #     # A hidden layer with two inputs per unit (n_inputs=2) and
    #     # two units, each of which has 1 output (n_outputs=2)
    #     layer = FullyConnectedLayer(n_inputs=2, n_outputs=2)
    #     layer.weights = np.array([
    #         [w1, w3],
    #         [w2, w4]
    #     ])
    #     layer.biases = np.array([[b1, b2]])
    #     y = layer.feed_forward(X)
    #     print("\nFCN output:", y)
    #     print("x:", layer.x, "\nw:", layer.weights, "\nb:", layer.biases)
    #
    #     np.testing.assert_allclose(actual=y,
    #                                desired=np.stack([y1, y2], axis=1),
    #                                err_msg="forward pass difference")


    def test_fully_connected_layer_backward_propagation(self):
        n_inputs = 2
        n_outputs = 1
        layer = FullyConnectedLayer(n_inputs, n_outputs)

        # Set weights and biases to known values
        layer.weights = np.array([[0.5], [0.5]])
        layer.biases = np.array([[0.1]])
        x = np.array([[1, 2]])
        layer.feed_forward(x)
        dJdy = np.array([[1]])
        learning_rate = 0.1

        # Manually compute gradients
        expected_dJdW = np.dot(x.T, dJdy)  # [[1], [2]]
        expected_dJdb = dJdy  # [[1]]
        expected_dJdx = np.dot(dJdy, layer.weights.T)  # [[0.5, 0.5]]

        # Perform backward propagation
        dJdx = layer.backward_propagation(dJdy, learning_rate)

        # Check weights and biases update
        np.testing.assert_array_almost_equal(layer.weights, np.array([[0.5 - 0.1 * 1], [0.5 - 0.1 * 2]]),
                                             err_msg="Weights not updated correctly")
        np.testing.assert_array_almost_equal(layer.biases, np.array([[0.1 - 0.1 * 1]]),
                                             err_msg="Biases not updated correctly")
        # Check dJdx
        print(dJdx)
        print(expected_dJdx)
        #np.testing.assert_array_almost_equal(dJdx, expected_dJdx, err_msg="Backward propagation output mismatch")


class TestActivationLayer:
    def test_activation_layer_feed_forward(self):
        activation_layer = ActivationLayer(activation_function=tanh_function,
                                           activation_derivative=tanh_derivative)
        x = np.array([[0, 1]])
        expected_output = np.tanh(x)
        output = activation_layer.feed_forward(x)
        np.testing.assert_array_almost_equal(output, expected_output, err_msg="Activation feed forward output mismatch")

    def test_activation_layer_backward_propagation(self):
        activation_layer = ActivationLayer(activation_function=tanh_function,
                                           activation_derivative=tanh_derivative)
        x = np.array([[0, 1]])
        activation_layer.feed_forward(x)
        dJdy = np.array([[1, 1]])
        expected_dJdx = (1 - np.tanh(x) ** 2) * dJdy
        dJdx = activation_layer.backward_propagation(dJdy, learning_rate=None)
        np.testing.assert_array_almost_equal(dJdx, expected_dJdx, err_msg="Activation backward propagation output mismatch")
