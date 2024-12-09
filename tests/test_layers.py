import pytest
import numpy as np
from numpy import ndarray
from mdsdl.utilities import tanh_function, tanh_derivative, sigmoidal_function, sigmoidal_derivative
from mdsdl.fully_connected import FullyConnectedLayer, ActivationLayer
from tests.reference_network_with_two_units import ReferenceNetworkWith2Units


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

        # given parameter and training data
        W = np.array([[0.5],
                      [0.5]])
        b = np.array([[0.1]])
        learning_rate = 0.1
        x = np.array([[1, 2]])

        dJdy = np.array([[1]])  # we simply assume that this is given!


        # manual calculation
        y_predicted = b[0, 0] + x[0, 0] * W[0, 0]  + x[0, 1] * W[1, 0]
        dJdW = np.dot(x.T, dJdy)
        dJdb = dJdy
        dJdy_prev = np.dot(dJdy, W.T)  # this is the same as dJdx!
        dJdx_desired = dJdy_prev

        W_new = W - learning_rate * dJdW
        b_new = b - learning_rate * dJdb

        # using the FCN class
        layer.weights = W
        layer.biases = b
        layer.feed_forward(x)
        dJdx, dJdW, dJdb = layer.backward_propagation(dJdy, learning_rate)

        # Check weights and biases update
        np.testing.assert_array_almost_equal(layer.weights, W_new,
                                             err_msg="Weights not updated correctly")
        np.testing.assert_array_almost_equal(layer.biases, b_new,
                                             err_msg="Biases not updated correctly")
        np.testing.assert_array_almost_equal(dJdx, dJdx_desired,
                                             err_msg="dJdx (=dJdy_prev) not computed correctly")

        return


    def test_fully_connected_layer_feed_forward_multiple_records(self):
        n_inputs = 2
        n_outputs = 2
        layer = FullyConnectedLayer(n_inputs, n_outputs)
        # Set weights and biases to known values
        # the weights of the first unit are in the first column
        layer.weights = np.array([[0.1, 0.2],
                                  [0.3, 0.4]])
        layer.biases = np.array([[0.5, 0.6]])
        # Input with multiple records (batch size = 3)
        x = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        # Manually compute expected outputs
        # For each record: output = x * weights + biases
        #expected_output = np.dot(x, layer.weights) + layer.biases
        expected_output = np.array(
         [[1*0.1 + 2*0.3 + 0.5, 1*0.2 + 2*0.4 + 0.6],
          [3*0.1 + 4*0.3 + 0.5, 3*0.2 + 4*0.4 + 0.6],
          [5*0.1 + 6*0.3 + 0.5, 5*0.2 + 6*0.4 + 0.6]])
        output = layer.feed_forward(x)
        np.testing.assert_array_almost_equal(output, expected_output, err_msg="Feed forward output mismatch with multiple records")

    def test_fully_connected_layer_backward_propagation_multiple_records(self):
        n_inputs = 2
        n_outputs = 2
        layer = FullyConnectedLayer(n_inputs, n_outputs)
        # Set weights and biases to known values
        layer.weights = np.array([[0.1, 0.2],
                                  [0.3, 0.4]])
        layer.biases = np.array([[0.5, 0.6]])
        # Input with multiple records
        x = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        layer.feed_forward(x)

        # Make up a gradient of loss with respect to output (dJ/dy)
        dJdy = np.array([[0.1, 0.2],
                         [0.3, 0.4],
                         [0.5, 0.6]])
        learning_rate = 0.01

        expected_dJdx = np.dot(dJdy, layer.weights.T)  # [[0.5, 0.5]]


        # # Manually compute gradients dJdW = x.T dot dJdy
        # expected_dJdW = np.dot(x.T, dJdy)
        #
        # # dJdb = sum over batch of dJdy
        # expected_dJdb = np.sum(dJdy, axis=0, keepdims=True)
        #
        # # Store old weights and biases for comparison after update
        # old_weights = layer.weights.copy()
        # old_biases = layer.biases.copy()

        # Perform backward propagation
        dJdx, dJdW, dJdb = layer.backward_propagation(dJdy, learning_rate)
        np.testing.assert_array_almost_equal(dJdx, expected_dJdx, err_msg="Backward propagation output mismatch")
        #
        # # Check weights and biases update
        # expected_weights = old_weights - learning_rate * expected_dJdW
        # expected_biases = old_biases - learning_rate * expected_dJdb
        # np.testing.assert_array_almost_equal(layer.weights, expected_weights, err_msg="Weights not updated correctly with multiple records")
        # np.testing.assert_array_almost_equal(layer.biases, expected_biases, err_msg="Biases not updated correctly with multiple records")
        # # Check dJdx (gradient w.r.t input)
        # expected_dJdx = np.dot(dJdy, old_weights.T)
        # np.testing.assert_array_almost_equal(dJdx, expected_dJdx, err_msg="Backward propagation output mismatch with multiple records")


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
        dJdx, dJdW, dJdb = activation_layer.backward_propagation(dJdy, learning_rate=None)
        np.testing.assert_array_almost_equal(dJdx, expected_dJdx, err_msg="Activation backward propagation output mismatch")


class TestWithReferenceNetwork:
    """Tests using the reference implementation for 2 units with sigmoid activation
    """

    def test_two_unit_network_against_reference_with_1_record(self):
        # Given parameters (example values)
        W = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)  # (unit 1: w1,w2, unit 2: w3,w4, unit 3: w5,w6)
        B = (0.1, -0.2, -0.1)  # (b1,b2,b3)
        X1, X2 = 0.5, -1.0
        Y = 0.7

        # Compute reference values
        Y_pred_ref = ReferenceNetworkWith2Units.predict(W, B, X1, X2)
        dJdW_ref, dJdB_ref = ReferenceNetworkWith2Units.backward_pass(W, B, X1, X2, Y)

        # Set up the class-based network
        fcl1 = FullyConnectedLayer(n_inputs=2, n_outputs=2)
        # Assign weights and biases to match reference. Since we're doing X * W the first
        # two weights of unit 1 must be in the first column
        fcl1.weights = np.array([[W[0], W[2]],
                                 [W[1], W[3]]])
        fcl1.biases = np.array([[B[0],
                                 B[1]]])

        al = ActivationLayer(activation_function=sigmoidal_function,
                             activation_derivative=sigmoidal_derivative)

        fcl2 = FullyConnectedLayer(n_inputs=2, n_outputs=1)
        fcl2.weights = np.array([[W[4]],
                                 [W[5]]])
        fcl2.biases = np.array([[B[2]]])

        # Forward pass with class-based layers
        x_input = np.array([[X1, X2]])
        y_pred = fcl1.feed_forward(x_input)
        y_pred = al.feed_forward(y_pred)
        y_pred = fcl2.feed_forward(y_pred)
        Y_pred_class = y_pred[0, 0]  # all quantities are at least 2D

        # Check forward pass consistency
        np.testing.assert_almost_equal(Y_pred_class, Y_pred_ref, decimal=7,
                                       err_msg="Forward pass mismatch")
        #----------------------------------------------------------------------

        # Next is the backward pass where we compute gradients using the
        # class-based implementation. We must define a cost  derivative dJ/dy
        # because this is needed as input for backward_propagation(...):
        # Assuming the cost is given by the MSE we have dJ/dy = (a3 - Y)
        dJdy = np.array([[Y_pred_class - Y]])  # shape (1, 1)

        # we're just using /any/ learning rate -- it doesn't matter, because
        # we do not use any of the updated weights of a layer in subsequent steps.
        learning_rate = 0.1

        # Backprop through final layer fcl2 (the single unit)
        dJdy_prev, fcl2_last_dJdW, fcl2_last_dJdb = \
            fcl2.backward_propagation(dJdy, learning_rate)

        # Backprop through the two hidden units = FCL + AL layer:
        # a) activation layer: the input of the current layer x is the
        #    output y of the previous layer
        dJdy_prev, _, _ = al.backward_propagation(dJdy_prev, learning_rate)
        #
        # b) first layer fcl1
        dJdy_prev, fcl1_last_dJdW, fcl1_last_dJdb = \
            fcl1.backward_propagation(dJdy_prev, learning_rate)

        # Extract gradients from class-based layers and rearrange them in a
        # set such that they can be compared with the reference solution's
        # dJdW and dJdB:
        # fcl1: weights with shape (2,2) corresponds to w1,w2 in column 0, w3,w4 in column 1
        # fcl2: weights shape (2,1) corresponds to w5,w6 in the first and only column
        dJdw1_class, dJdw2_class = fcl1_last_dJdW[0, 0], fcl1_last_dJdW[1, 0]
        dJdw3_class, dJdw4_class = fcl1_last_dJdW[0, 1], fcl1_last_dJdW[1, 1]
        dJdw5_class, dJdw6_class = fcl2_last_dJdW[0, 0], fcl2_last_dJdW[1, 0]

        dJdb1_class, dJdb2_class = fcl1_last_dJdb[0, 0], fcl1_last_dJdb[0, 1]
        dJdb3_class = fcl2_last_dJdb[0, 0]

        dJdW_class = (dJdw1_class, dJdw2_class, dJdw3_class, dJdw4_class, dJdw5_class, dJdw6_class)
        dJdB_class = (dJdb1_class, dJdb2_class, dJdb3_class)

        # Compare gradients
        np.testing.assert_almost_equal(dJdW_class, dJdW_ref, decimal=7, err_msg="Weight gradients mismatch")
        np.testing.assert_almost_equal(dJdB_class, dJdB_ref, decimal=7, err_msg="Bias gradients mismatch")




def test_batch_gradient_descent_with_reference():
    """Test that the batch gradient descent updates in FCNetwork match the reference implementation."""
    # Initialize weights and biases for the reference network
    W_ref = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    B_ref = (0.1, 0.2, 0.3)

    # Training data (X1, X2 as inputs, Y as target)
    X1 = np.array([0.1, 0.2, 0.3])
    X2 = np.array([0.4, 0.5, 0.6])
    Y = np.array([0.7, 0.8, 0.9])

    # Learning rate
    learning_rate = 0.01

    # Reference implementation: Compute weight and bias updates
    dJdW_ref, dJdB_ref = ReferenceNetworkWith2Units.backward_pass(W_ref, B_ref, X1, X2, Y)

    # Initialize an FCNetwork matching the reference network's structure
    network = FCNetwork(MSE, MSE_derivative)
    network.add_layer(FullyConnectedLayer(2, 2, seed=42))  # Input to 2 hidden neurons
    network.add_layer(ActivationLayer(sigmoidal_function, sigmoidal_derivative))
    network.add_layer(FullyConnectedLayer(2, 1, seed=42))  # Hidden to output neuron
    network.add_layer(ActivationLayer(sigmoidal_function, sigmoidal_derivative))

    # Manually set weights and biases to match the reference implementation
    network.layers[0].weights = np.array([[0.1, 0.3], [0.2, 0.4]])  # W_ref[:4] reshaped
    network.layers[0].biases = np.array([[0.1, 0.2]])               # B_ref[:2]
    network.layers[2].weights = np.array([[0.5], [0.6]])           # W_ref[4:]
    network.layers[2].biases = np.array([[0.3]])                   # B_ref[2]

    # Forward pass with the batch
    X_batch = np.column_stack((X1, X2))  # Combine X1 and X2 into a 2D array
    Y_batch = Y.reshape(-1, 1)           # Reshape Y into a column vector

    # Perform one training step (manual backpropagation) in the FCNetwork
    y_pred = X_batch
    for layer in network.layers:
        y_pred = layer.feed_forward(y_pred)

    # Backpropagation
    error = MSE_derivative(Y_batch, y_pred)
    for layer in reversed(network.layers):
        error, dJdW, dJdB = layer.backward_propagation(error, learning_rate)

    # Extract weight and bias updates from the FCNetwork
    dJdW_fc = (
        np.mean(network.layers[0].weights - dJdW, axis=0),
        np.mean(network.layers[2].weights - dJdW, axis=0),
    )
    dJdB_fc = (
        np.mean(network.layers[0].biases - dJdB, axis=0),
        np.mean(network.layers[2].biases - dJdB, axis=0),
    )

    # Compare updates
    assert np.allclose(dJdW_fc[0], dJdW_ref[:2]), "Weight updates (hidden layer) do not match"
    assert np.allclose(dJdW_fc[1], dJdW_ref[4:]), "Weight updates (output layer) do not match"
    assert np.allclose(dJdB_fc[0], dJdB_ref[:2]), "Bias updates (hidden layer) do not match"
    assert np.allclose(dJdB_fc[1], dJdB_ref[2]), "Bias updates (output layer) do not match"
